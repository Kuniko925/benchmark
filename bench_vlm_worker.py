"""Single-GPU video classification worker supporting X-CLIP, Vita-CLIP and EVL.

X-CLIP uses full-parameter fine-tuning; Vita-CLIP uses frozen-backbone prompt
learning; EVL trains a decoder on frozen image features. These benchmark loops
do not reproduce the full paper training recipes.
Launch through run_vlm.py --backend xclip|vita|evl to isolate GPUs and record telemetry.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_manifest(path, limit):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"video_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Video CSV must contain video_path,label columns")
        rows = list(reader)
    if not rows or any(not row["label"].strip() or not row["video_path"].strip() for row in rows):
        raise ValueError("Video CSV must contain nonempty video paths and class-name labels")
    classes = sorted({row["label"].strip() for row in rows})
    if len(classes) < 2:
        raise ValueError("Classification requires at least two class names in the CSV")
    samples = []
    for row in (rows if limit == -1 else rows[:limit]):
        video = Path(row["video_path"])
        video = video if video.is_absolute() else path.parent / video
        if not video.is_file():
            raise FileNotFoundError(video)
        samples.append((video.resolve(), classes.index(row["label"].strip())))
    return samples, classes


def read_ucf101(video_root, splits_dir, split, limit, seed):
    """Use only the official training list; retain all 101 class candidates."""
    classes_by_id = {}
    for line in (splits_dir / "classInd.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            identifier, label = line.split(maxsplit=1)
            identifier = int(identifier)
            if identifier in classes_by_id:
                raise ValueError(f"Duplicate UCF101 class ID: {identifier}")
            classes_by_id[identifier] = label.strip()
    if set(classes_by_id) != set(range(1, 102)) or len(set(classes_by_id.values())) != 101:
        raise ValueError("UCF101 classInd.txt must contain 101 unique classes, IDs 1 through 101")
    classes = [classes_by_id[index] for index in range(1, 102)]
    candidates = []
    list_path = splits_dir / f"trainlist{split:02d}.txt"
    for line in list_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        relative, identifier = line.rsplit(maxsplit=1)
        relative = Path(relative.replace("\\", "/"))
        identifier = int(identifier)
        if identifier not in classes_by_id or relative.parts[0] != classes_by_id[identifier]:
            raise ValueError(f"Invalid class/path pair in {list_path}: {line}")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a relative UCF101 video path: {relative}")
        candidates.append(((video_root / relative).resolve(), identifier - 1))
    if not candidates:
        raise ValueError(f"Empty training split: {list_path}")
    samples = candidates if limit == -1 else random.Random(seed).sample(candidates, min(limit, len(candidates)))
    for video, _ in samples:
        if not video.is_file():
            raise FileNotFoundError(f"UCF101 video missing: {video}; check --ucf101_root")
    return samples, classes


def read_clip(path, num_frames):
    """Uniformly sample the whole video, retaining only the requested frames."""
    import av
    import numpy as np

    with av.open(str(path)) as container:
        frame_count = container.streams.video[0].frames
        if not frame_count:
            frame_count = sum(1 for _ in container.decode(video=0))
    if frame_count < 1:
        raise ValueError(f"No decodable video frames: {path}")
    indices = np.linspace(0, frame_count - 1, num_frames).astype(int).tolist()
    wanted = set(indices)
    frames = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frames[index] = frame.to_ndarray(format="rgb24")
            if index >= indices[-1]:
                break
    if not wanted.issubset(frames):
        raise ValueError(f"Video frame metadata does not match decoded frames: {path}")
    return [frames[index] for index in indices]


def profile_forward(model, inputs, output_dir, output_path, precision, class_count=None):
    import torch
    from flops_profiler.profiler import FlopsProfiler

    previous_state = model.training
    profiler = FlopsProfiler(model)
    try:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            model(**inputs)
            torch.cuda.synchronize(0)
            profiler.start_profile()
            start = time.perf_counter()
            model(**inputs)
            torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
        profile = {
            "forward_flops": float(profiler.get_total_flops(as_string=False)),
            "forward_macs": float(profiler.get_total_macs(as_string=False)),
            "forward_latency_sec": elapsed,
            "parameters": float(profiler.get_total_params(as_string=False)),
            "video_input_shape": list(inputs["pixel_values"].shape),
            "class_prompt_count": class_count if class_count is not None else inputs["input_ids"].shape[0],
            "notice": "Forward only, one training batch and all output classes; not training FLOPs",
        }
        profile["estimated_forward_flops_per_sec"] = profile["forward_flops"] / elapsed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.print_model_profile(profile_step=0, module_depth=3, top_modules=10,
                                     detailed=True, output_file=str(output_path))
        write_json(output_dir / "forward_flops.json", profile)
    finally:
        try:
            profiler.stop_profile()
        finally:
            try:
                profiler.end_profile()
            finally:
                model.train(previous_state)


def parse_worker_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["xclip", "vita", "evl"], default="xclip")
    parser.add_argument("--model-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-csv", type=Path)
    source.add_argument("--ucf101-root", type=Path)
    parser.add_argument("--ucf101-splits", type=Path)
    parser.add_argument("--ucf101-split", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--flops-log-path", type=Path)
    parser.add_argument("--vita-repo", type=Path, help="Official repository; required for --backend vita")
    parser.add_argument("--backbone-path", type=Path, help="Converted CLIP backbone; required for vita/evl")
    parser.add_argument("--evl-repo", type=Path, help="Official EVL repository; required for --backend evl")
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args(argv)
    if args.backend == "vita":
        if args.vita_repo is None or args.backbone_path is None:
            parser.error("--vita-repo and --backbone-path are required with --backend vita")
    elif args.backend == "evl" and (args.evl_repo is None or args.backbone_path is None):
        parser.error("--evl-repo and --backbone-path are required with --backend evl")
    if args.backend in {"vita", "evl"} and args.num_frames < 1:
        parser.error("num-frames must be positive")
    if min(args.batch_size, args.epochs, args.logging_steps) < 1 or args.learning_rate <= 0:
        parser.error("batch size, epochs, logging steps and learning rate must be positive")
    if args.samples == 0 or args.samples < -1:
        parser.error("samples must be -1 (all) or positive")
    if args.max_steps < -1 or args.max_steps == 0:
        parser.error("max_steps must be -1 or positive")
    if args.ucf101_root is not None and args.ucf101_splits is None:
        parser.error("--ucf101-splits is required with --ucf101-root")
    return args


def main():
    args = parse_worker_args()
    backend = args.backend
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The parent restricts CUDA visibility before this interpreter imports torch.
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU; launch with run_vlm.py")
    torch.cuda.set_device(0)
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 unavailable; use --precision fp32")
    if args.ucf101_root is not None:
        samples, classes = read_ucf101(args.ucf101_root, args.ucf101_splits,
                                      args.ucf101_split, args.samples, args.seed)
    else:
        samples, classes = read_manifest(args.video_csv, args.samples)
    if backend == "vita":
        from vita_backend import build_vita
        model, processor = build_vita(args, classes)
        num_frames = args.num_frames
        training_mode = "frozen_backbone_multimodal_prompt_learning"
    elif backend == "evl":
        from evl_backend import build_evl
        model, processor = build_evl(args, classes)
        num_frames = args.num_frames
        training_mode = "frozen_backbone_evl_decoder"
    else:
        from transformers import AutoConfig, AutoProcessor, XCLIPModel
        config = AutoConfig.from_pretrained(args.model_id)
        if config.model_type != "xclip":
            raise ValueError(f"Video backend not implemented for model_type={config.model_type}; currently supports xclip")
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = XCLIPModel.from_pretrained(args.model_id).to("cuda:0")
        num_frames = config.vision_config.num_frames
        training_mode = "full_parameter_class_prompt_cross_entropy"
    class_names = [re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', label) for label in classes]
    prompts = ([" ".join(["X"] * 8) + " " + label.replace("_", " ") + "." for label in class_names]
               if backend == "vita" else [f"a video of {label}" for label in class_names])
    if backend == "evl":
        prompts = []

    def make_inputs(batch):
        if backend in {"vita", "evl"}:
            return {"pixel_values": processor([read_clip(path, num_frames) for path, _ in batch]).to("cuda:0")}
        inputs = processor(videos=[read_clip(path, num_frames) for path, _ in batch], text=prompts,
                           padding=True, truncation=True,
                           max_length=config.text_config.max_position_embeddings,
                           return_tensors="pt")
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        expected = (len(batch), num_frames)
        if inputs["pixel_values"].ndim != 5 or tuple(inputs["pixel_values"].shape[:2]) != expected:
            raise ValueError(f"Unexpected pixel_values shape: {inputs['pixel_values'].shape}, expected {expected} + (C,H,W)")
        return inputs

    physical_gpu = os.environ.get("BENCH_PHYSICAL_GPU", "unknown")
    write_json(args.output_dir / "effective_config.json", {
        "arguments": vars(args), "physical_gpu": physical_gpu,
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backend": backend, "training_mode": training_mode,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset": "UCF101" if args.ucf101_root is not None else "local_csv",
        "subset_policy": "all training rows" if args.samples == -1 else (
            "seeded random sample from official training split" if args.ucf101_root is not None else "first CSV rows"),
        "classes": classes, "class_prompts": prompts, "num_frames": num_frames,
        "sampling_policy": "uniform frames across the whole video; repeat for short clips; decoded on demand per batch",
        "samples": [{"video_path": str(path), "label_index": label} for path, label in samples],
        "pytorch_version": str(torch.__version__),
    })
    print(f"Physical GPU {physical_gpu}; model={args.model_id}; samples={len(samples)}; classes={len(classes)}", flush=True)
    if args.profile_flops:
        # Fail visibly rather than continue training after a profiler may have modified hooks.
        try:
            profile_forward(model, make_inputs(samples[:args.batch_size]), args.output_dir,
                            args.flops_log_path or args.output_dir / "flops_profile.txt", args.precision,
                            class_count=len(prompts))
        except Exception as exc:
            (args.output_dir / "forward_flops_error.txt").write_text(str(exc), encoding="utf-8")
            raise

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(samples) / args.batch_size)
    target_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    step = 0
    completed_epochs = 0
    loss_sum = 0.0
    model.train()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    start_epoch, start_perf = time.time(), time.perf_counter()
    status = "failed"
    try:
        while step < target_steps:
            order = torch.randperm(len(samples)).tolist()
            for offset in range(0, len(order), args.batch_size):
                batch = [samples[index] for index in order[offset:offset + args.batch_size]]
                inputs = make_inputs(batch)
                labels = torch.tensor([label for _, label in batch], device="cuda:0")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    logits = model(**inputs).logits_per_video
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                loss_sum += loss.item()
                if step == 1 or step % args.logging_steps == 0 or step == target_steps:
                    print(f"step={step}/{target_steps} loss={loss.item():.6f}", flush=True)
                if offset + len(batch) == len(order):
                    completed_epochs += 1
                    print(f"Completed epoch {completed_epochs}", flush=True)
                if step >= target_steps:
                    break
        torch.cuda.synchronize(0)
        status = "completed"
    finally:
        elapsed = time.perf_counter() - start_perf
        write_json(args.output_dir / "train_window.json", {
            "start_epoch": start_epoch, "end_epoch": time.time(),
            "elapsed_seconds": elapsed, "status": status,
        })

    model.save_pretrained(args.output_dir / "model")
    processor.save_pretrained(args.output_dir / "model")
    write_json(args.output_dir / "worker_summary.json", {
        "status": status, "gpu": physical_gpu, "model": args.model_id,
        "samples": len(samples), "optimizer_steps": step, "num_frames": num_frames,
        "completed_epochs": completed_epochs,
        "training_seconds": elapsed,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
        "train_metrics": {"mean_step_loss": loss_sum / step},
        "training_mode": training_mode,
    })
    print(f"DONE: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
