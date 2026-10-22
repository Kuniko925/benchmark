"""Use the official EVL architecture with converted CLIP ViT-B/16 backbone weights.

Only import torch inside the worker, after the launcher isolates its CUDA device.
Source: https://github.com/OpenGVLab/efficient-video-recognition
"""
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from bench_vlm_worker import write_json


def validate_evl_files(repo, backbone):
    for name in ("model.py", "vision_transformer.py", "weight_loaders.py"):
        path = Path(repo) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing official EVL file: {path}; check --evl_repo")
    if not Path(backbone).is_file():
        raise FileNotFoundError(f"CLIP backbone missing: {backbone}; prepare it with vita_backend.py --prepare_backbone")


def load_evl_modules(repo):
    """Resolve upstream absolute imports without colliding with other model modules."""
    def read_module(name, filename):
        spec = importlib.util.spec_from_file_location(name, Path(repo) / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    vision = read_module("_benchmark_evl_vision", "vision_transformer.py")
    weights = read_module("_benchmark_evl_weights", "weight_loaders.py")
    aliases = {"vision_transformer": vision, "weight_loaders": weights}
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules.update(aliases)
        model = read_module("_benchmark_evl_model", "model.py")
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return model, vision


def evl_backbone_state(path):
    """Reuse the HF-to-Vita CLIP conversion; EVL needs only intermediate vision features."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("EVL expects a converted CLIP .pth from vita_backend.py --prepare_backbone")
    result = {}
    for name, tensor in state.items():
        name = name.removeprefix("module.")
        if not name.startswith("visual."):
            continue
        name = name.removeprefix("visual.")
        if name in {"cls_token", "pos_embed"} or name.startswith(("patch_embed.proj.", "ln_pre.")) or re.fullmatch(
                r"blocks\.\d+\.(?:norm[12]|attn\.(?:[qkv]_proj|out_proj)|mlp\.fc[12])\.(?:weight|bias)", name):
            result[name] = tensor
    if "patch_embed.proj.weight" not in result:
        raise ValueError("No converted CLIP visual backbone found; use vita_backend.py --prepare_backbone")
    # The official EVL backbone extracts patches explicitly and uses a linear layer.
    result["patch_embed.proj.weight"] = result["patch_embed.proj.weight"].flatten(1)
    return result


def build_evl(args, classes, device="cuda:0"):
    import torch
    from vita_backend import VitaVideoProcessor

    validate_evl_files(args.evl_repo, args.backbone_path)
    if args.num_frames < 1:
        raise ValueError("num_frames must be positive")
    upstream, vision = load_evl_modules(args.evl_repo)

    class DeviceTemporalCrossAttention(upstream.TemporalCrossAttention):
        def forward(self, q, k):
            # Upstream hardcodes device='cuda'. Allocate on the input device so
            # CPU checks and the isolated GPU use the same temporal computation.
            n, t, length, _, _ = q.size()
            result = q.new_zeros((n, t, length, self.w1.size(-1)))
            result[:, 1:, 1:, :] += self.forward_half(q[:, 1:], k[:, :-1], self.w1)
            result[:, :-1, 1:, :] += self.forward_half(q[:, :-1], k[:, 1:], self.w2)
            return result

    upstream.TemporalCrossAttention = DeviceTemporalCrossAttention
    # The upstream constructor performs strict backbone validation and freezing.
    upstream.weight_loader_fn_dict["benchmark_converted_clip"] = evl_backbone_state
    preset = vision.vit_presets["ViT-B/16-lnpre"]
    decoder_layers = min(4, preset["num_layers"])
    core = upstream.EVLTransformer(
        num_frames=args.num_frames, backbone_name="ViT-B/16-lnpre",
        backbone_type="benchmark_converted_clip", backbone_path=str(args.backbone_path),
        backbone_mode="freeze_fp32", decoder_num_layers=decoder_layers,
        decoder_qkv_dim=preset["feature_dim"], decoder_num_heads=preset["num_heads"],
        num_classes=len(classes), enable_temporal_conv=True,
        enable_temporal_pos_embed=True, enable_temporal_cross_attention=True,
    )

    class EVLModelAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.evl = core
            # Upstream hides the frozen backbone in a list. Register it here so
            # device moves, parameter counts, profiling and checkpoints include it.
            self.backbone = core.backbone[0]

        def train(self, mode=True):
            super().train(mode)
            self.backbone.eval()
            return self

        def forward(self, pixel_values):
            return SimpleNamespace(logits_per_video=self.evl(pixel_values.permute(0, 2, 1, 3, 4).contiguous()))

        def save_pretrained(self, directory):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), directory / "evl_model.pth")
            write_json(directory / "evl_config.json", {
                "architecture": "EVL-B/16", "classes": classes, "num_frames": args.num_frames,
                "decoder_layers": decoder_layers, "backbone_path": args.backbone_path,
                "training_mode": "frozen_backbone_evl_decoder",
                "source": "https://github.com/OpenGVLab/efficient-video-recognition",
            })

    return EVLModelAdapter().to(device), VitaVideoProcessor()
