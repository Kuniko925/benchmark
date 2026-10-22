"""Adapter for the official TalalWasim/Vita-CLIP ViT-B/16 implementation.

No torch imports at module scope: the launcher must isolate each GPU first.
The external repository and backbone remain user-managed dependencies.
"""
import importlib
import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

# Hub reads this at import time; set it before importing Transformers.
# Use the same HTTP fallback as the benchmark launcher for Windows downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from bench_vlm_worker import write_json


def convert_hf_clip_state(state, vision_layers=12, text_layers=12):
    """Map Transformers CLIP tensors to the official Vita-CLIP backbone names.

    Text attention packs Q/K/V, while the vision encoder keeps them separate.
    CLIP's bias-free patch convolution needs an explicit zero bias in Vita-CLIP.
    New video prompts/temporal parameters keep the upstream initialization.
    """
    import torch

    converted = {}
    direct = {
        "visual.patch_embed.proj.weight": "vision_model.embeddings.patch_embedding.weight",
        "visual.cls_token": "vision_model.embeddings.class_embedding",
        "visual.pos_embed": "vision_model.embeddings.position_embedding.weight",
        "textual.token_embedding.weight": "text_model.embeddings.token_embedding.weight",
        "textual.positional_embedding": "text_model.embeddings.position_embedding.weight",
        "logit_scale": "logit_scale",
    }
    for destination, source in direct.items():
        converted[destination] = state[source]
    patch = converted["visual.patch_embed.proj.weight"]
    converted["visual.patch_embed.proj.bias"] = patch.new_zeros(patch.shape[0])
    converted["visual.proj"] = state["visual_projection.weight"].T.contiguous()
    converted["textual.text_projection"] = state["text_projection.weight"].T.contiguous()
    for destination, source in (("visual.ln_pre", "vision_model.pre_layrnorm"),
                                ("visual.ln_post", "vision_model.post_layernorm"),
                                ("textual.ln_final", "text_model.final_layer_norm")):
        for suffix in ("weight", "bias"):
            converted[f"{destination}.{suffix}"] = state[f"{source}.{suffix}"]
    for visual, count in ((True, vision_layers), (False, text_layers)):
        for index in range(count):
            source = f"{'vision_model' if visual else 'text_model'}.encoder.layers.{index}"
            destination = f"visual.blocks.{index}" if visual else f"textual.transformer.resblocks.{index}"
            modules = {"layer_norm1": "norm1" if visual else "ln_1",
                       "layer_norm2": "norm2" if visual else "ln_2",
                       "mlp.fc1": "mlp.fc1" if visual else "mlp.c_fc",
                       "mlp.fc2": "mlp.fc2" if visual else "mlp.c_proj",
                       "self_attn.out_proj": "attn.out_proj"}
            for hf_name, vita_name in modules.items():
                for suffix in ("weight", "bias"):
                    converted[f"{destination}.{vita_name}.{suffix}"] = state[f"{source}.{hf_name}.{suffix}"]
            for suffix in ("weight", "bias"):
                tensors = [state[f"{source}.self_attn.{part}_proj.{suffix}"] for part in ("q", "k", "v")]
                if visual:
                    for part, tensor in zip(("q", "k", "v"), tensors):
                        converted[f"{destination}.attn.{part}_proj.{suffix}"] = tensor
                else:
                    converted[f"{destination}.attn.in_proj_{suffix}"] = torch.cat(tensors, dim=0)
    return converted


def prepare_backbone(output_path):
    """Download the public HF CLIP-B/16 backbone on CPU and save Vita-compatible weights."""
    import torch
    from transformers import CLIPModel

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}; choose a new path")
    source = "openai/clip-vit-base-patch16"
    print(f"Downloading {source} from Hugging Face on CPU...", flush=True)
    model = CLIPModel.from_pretrained(source).cpu().float().eval()
    vision, text = model.config.vision_config, model.config.text_config
    if ((vision.hidden_size, vision.patch_size, vision.image_size, vision.num_hidden_layers,
         vision.num_attention_heads, text.hidden_size, text.num_hidden_layers,
         text.num_attention_heads, model.config.projection_dim, text.vocab_size,
         text.max_position_embeddings) != (768, 16, 224, 12, 12, 512, 12, 8, 512, 49408, 77)):
        raise ValueError("Expected the standard CLIP ViT-B/16 architecture")
    state = convert_hf_clip_state(model.state_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting an existing checkpoint.
    with output_path.open("xb") as file:
        torch.save(state, file)
    write_json(output_path.with_suffix(output_path.suffix + ".json"), {
        "source": source, "revision": getattr(model.config, "_commit_hash", None),
        "conversion": "Transformers CLIP to Vita-CLIP backbone", "tensors": len(state),
    })
    print(f"Saved Vita-CLIP backbone: {output_path.resolve()}", flush=True)


def validate_vita_files(repo, backbone):
    training = Path(repo) / "training"
    for name in ("VitaCLIP_model.py", "VitaCLIP_vision_encoder.py",
                 "VitaCLIP_vision_encoder_utils.py", "VitaCLIP_text_encoder.py",
                 "VitaCLIP_text_encoder_utils.py", "bpe_simple_vocab_16e6.txt.gz"):
        if not (training / name).is_file():
            raise FileNotFoundError(f"Missing official Vita-CLIP file: {training / name}; check --vita_repo")
    if not Path(backbone).is_file():
        raise FileNotFoundError(f"Converted CLIP backbone not found: {backbone}; check --backbone_path")


def load_backbone(model, path):
    """Reject incomplete/incompatible backbones rather than train frozen random weights."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Expected the official converted CLIP backbone state dictionary")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = model.state_dict()
    # cls_proj belongs to the new video prompting blocks, not the CLIP backbone.
    missing = [name for name, param in model.named_parameters()
               if not param.requires_grad and ".cls_proj." not in name and name not in state]
    mismatched = [name for name, value in state.items()
                  if name in expected and getattr(value, "shape", None) != expected[name].shape]
    if missing or mismatched:
        raise ValueError("Incompatible CLIP backbone: use the official converted ViT-B/16 .pth. "
                         f"Missing frozen weights: {missing[:5]}; shape mismatches: {mismatched[:5]}")
    model.load_state_dict(state, strict=False)


class VitaVideoProcessor:
    """Uniformly sampled frames, CLIP resize/center-crop/normalization, B,T,C,H,W."""

    def __init__(self):
        from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                            Normalize, Resize, ToTensor)
        self.transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
            ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073),
                                  (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, videos):
        import torch
        from PIL import Image

        return torch.stack([torch.stack([self.transform(Image.fromarray(frame)) for frame in frames])
                            for frames in videos])

    def save_pretrained(self, directory):
        write_json(Path(directory) / "preprocessor_config.json", {
            "backend": "vita", "resize_short_edge": 224, "center_crop": 224,
            "interpolation": "bicubic", "rescale_factor": 1 / 255,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        })


def build_vita(args, classes):
    import torch

    validate_vita_files(args.vita_repo, args.backbone_path)
    if args.num_frames < 1:
        raise ValueError("num_frames must be positive")
    sys.path.insert(0, str(Path(args.vita_repo).resolve() / "training"))
    VitaCLIP = importlib.import_module("VitaCLIP_model").VitaCLIP
    TextPromptLearner = importlib.import_module("VitaCLIP_text_encoder").TextPromptLearner

    class_names = [re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label) for label in classes]
    class_file = args.output_dir / "vita_classes.txt"
    # The upstream implementation reads class names using the OS default encoding.
    class_file.write_text("\n".join(class_names) + "\n")
    options = {
        "num_frames": args.num_frames, "use_summary_token": True,
        "use_local_prompts": True, "use_global_prompts": True, "num_global_prompts": 8,
        "use_text_prompt_learning": True, "text_num_prompts": 8, "text_prompt_CSC": True,
    }
    core = VitaCLIP(**options, text_prompt_classes_path=str(class_file))
    load_backbone(core, args.backbone_path)
    # Class token buffers must be generated AFTER pretrained token embeddings load.
    core.prompt_learner = TextPromptLearner(
        classnames=class_names, text_model=core.textual, num_prompts=8, CSC=True, ctx_pos="end")
    core.tokenized_prompts = core.prompt_learner.tokenized_prompts.to("cuda:0")

    class VitaModelAdapter(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vita = model

        def forward(self, pixel_values):
            # Shared benchmark inputs are B,T,C,H,W; Vita-CLIP requires B,C,T,H,W.
            logits = self.vita(pixel_values.permute(0, 2, 1, 3, 4).contiguous())
            return SimpleNamespace(logits_per_video=logits)

        def save_pretrained(self, directory):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(self.vita.state_dict(), directory / "vita_model.pth")
            write_json(directory / "vita_config.json", {
                "architecture": "Vita-CLIP-B/16", "options": options, "classes": classes,
                "class_names": class_names, "backbone_path": args.backbone_path,
                "source": "https://github.com/TalalWasim/Vita-CLIP",
                "training_mode": "frozen_backbone_multimodal_prompt_learning",
            })

    return VitaModelAdapter(core).to("cuda:0"), VitaVideoProcessor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Vita-CLIP backbone weights from Hugging Face.")
    parser.add_argument("--prepare_backbone", type=Path, required=True,
                        help="Output .pth path; downloads openai/clip-vit-base-patch16")
    arguments = parser.parse_args()
    prepare_backbone(arguments.prepare_backbone)
