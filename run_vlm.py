"""Launch two independent video-model training jobs with shared host telemetry.

Supports X-CLIP, Vita-CLIP and EVL classification using UCF101 or a local video CSV.
"""
import argparse
import os
from pathlib import Path

from run import launch_benchmark
from bench_vlm_worker import read_manifest, read_ucf101


DEFAULT_UCF101_ROOT = Path(r"D:\Bench\dataset\UCF-101" if os.name == "nt"
                           else "/mnt/d/Bench/dataset/UCF-101")


def video_worker_options(args, outdir):
    source_options = (["--ucf101-root", str(args.ucf101_root),
                       "--ucf101-splits", str(args.ucf101_splits),
                       "--ucf101-split", str(args.ucf101_split)]
                      if args.ucf101_root is not None else ["--video-csv", str(args.video_csv)])
    backend_options = (["--vita-repo", str(args.vita_repo),
                        "--backbone-path", str(args.backbone_path), "--num-frames", str(args.num_frames)]
                       if args.backend == "vita" else [])
    if args.backend == "evl":
        backend_options = ["--evl-repo", str(args.evl_repo), "--backbone-path", str(args.backbone_path),
                           "--num-frames", str(args.num_frames)]
    return [
        "--backend", args.backend, "--model-id", args.model_id, *source_options,
        "--output-dir", str(outdir), "--samples", str(args.samples),
        "--max-steps", str(args.max_steps), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--epochs", str(args.epochs),
        "--seed", str(args.seed), "--precision", args.precision,
        "--logging-steps", str(args.logging_steps), *backend_options,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["xclip", "vita", "evl"], default="xclip")
    parser.add_argument("--model_id", help="Required for X-CLIP; result-folder label for vita/evl")
    parser.add_argument("--vita_repo", type=Path,
                        help="Local clone of https://github.com/TalalWasim/Vita-CLIP; required for vita")
    parser.add_argument("--backbone_path", type=Path,
                        help="Converted CLIP backbone .pth; required for vita/evl")
    parser.add_argument("--evl_repo", type=Path,
                        help="Local clone of OpenGVLab/efficient-video-recognition; required for evl")
    parser.add_argument("--num_frames", type=int, default=8)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video_csv", type=Path,
                        help="CSV with video_path,label columns; paths relative to the CSV")
    source.add_argument("--ucf101_root", type=Path,
                        help="Extracted UCF-101 folder containing action-class subfolders; "
                             f"defaults to {DEFAULT_UCF101_ROOT} when --video_csv is omitted")
    parser.add_argument("--ucf101_splits", type=Path,
                        help="Official ucfTrainTestlist folder containing classInd.txt and trainlist01.txt")
    parser.add_argument("--ucf101_split", type=int, choices=[1, 2, 3], default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max_samples", dest="samples", type=int, default=4)
    selection.add_argument("--all_samples", dest="samples", action="store_const", const=-1)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Optimizer steps per worker; -1 runs the requested epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--output_dir", dest="output_root", type=Path)
    parser.add_argument("--gpu_log_path", type=Path)
    parser.add_argument("--cpu_log_path", type=Path)
    parser.add_argument("--flops_log_path", type=Path)
    parser.add_argument("--log_interval_sec", dest="interval_sec", type=float, default=1.0)
    parser.add_argument("--csv_max_rows", type=int, default=100000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument("--carbon-intensity-g-per-kwh", type=float)
    args = parser.parse_args()
    if args.backend == "vita":
        if args.vita_repo is None or args.backbone_path is None:
            parser.error("--vita_repo and --backbone_path are required with --backend vita")
        if args.model_id is None:
            args.model_id = "Vita-CLIP-B"
    elif args.backend == "evl":
        if args.evl_repo is None or args.backbone_path is None:
            parser.error("--evl_repo and --backbone_path are required with --backend evl")
        if args.model_id is None:
            args.model_id = "EVL-B16"
    elif not args.model_id:
        parser.error("--model_id is required with --backend xclip")
    if args.ucf101_root is None and args.video_csv is None:
        args.ucf101_root = DEFAULT_UCF101_ROOT
    if min(args.batch_size, args.epochs, args.csv_max_rows, args.logging_steps) < 1 or args.learning_rate <= 0:
        parser.error("batch size, epochs, row limit, logging steps and learning rate must be positive")
    if args.samples == 0 or args.samples < -1:
        parser.error("max_samples must be -1 (all) or positive")
    if args.max_steps < -1 or args.max_steps == 0 or args.interval_sec <= 0:
        parser.error("max_steps must be -1 or positive; log_interval_sec must be positive")
    if args.carbon_intensity_g_per_kwh is not None and args.carbon_intensity_g_per_kwh < 0:
        parser.error("carbon intensity must be nonnegative")
    try:
        if args.backend == "vita":
            from vita_backend import validate_vita_files
            args.vita_repo = args.vita_repo.resolve()
            args.backbone_path = args.backbone_path.resolve()
            validate_vita_files(args.vita_repo, args.backbone_path)
        elif args.backend == "evl":
            from evl_backend import validate_evl_files
            args.evl_repo = args.evl_repo.resolve()
            args.backbone_path = args.backbone_path.resolve()
            validate_evl_files(args.evl_repo, args.backbone_path)
        if args.backend in {"vita", "evl"} and args.num_frames < 1:
            parser.error("num_frames must be positive")
        if args.ucf101_root is not None:
            if args.ucf101_splits is None:
                parser.error("--ucf101_splits is required with --ucf101_root")
            args.ucf101_root = args.ucf101_root.resolve()
            args.ucf101_splits = args.ucf101_splits.resolve()
            read_ucf101(args.ucf101_root, args.ucf101_splits,
                        args.ucf101_split, args.samples, args.seed)
        else:
            args.video_csv = args.video_csv.resolve()
            read_manifest(args.video_csv, args.samples)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    launch_benchmark(args,
                     worker_filename="bench_vlm_worker.py",
                     worker_options=video_worker_options)


if __name__ == "__main__":
    main()
