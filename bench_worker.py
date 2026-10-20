"""Single-GPU causal-language-model LoRA benchmark worker.

Launched by run.py with CUDA_VISIBLE_DEVICES=<physical GPU UUID>.
Run it via the launcher, not directly, for correct GPU isolation and telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def format_alpaca(example: dict) -> str:
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def profile_forward_flops(model, tokenizer, output_path: Path, sequence_length: int) -> dict:
    """Optional FORWARD-PASS profile; not equivalent to training FLOPs."""
    from flops_profiler.profiler import FlopsProfiler
    import torch

    sample_text = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\nExplain why cooling is important for GPU training.\n\n"
        "### Response:\n"
    )
    encoded = tokenizer(
        sample_text, return_tensors="pt", truncation=True,
        padding="max_length", max_length=sequence_length,
    )
    inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
    previous_training_state = model.training
    model.eval()
    profiler = None
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
        torch.cuda.synchronize(0)
        profiler = FlopsProfiler(model)
        profiler.start_profile()
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs, use_cache=False)
        torch.cuda.synchronize(0)
        latency = time.perf_counter() - t0
        flops = profiler.get_total_flops(as_string=False)
        macs = profiler.get_total_macs(as_string=False)
        params = profiler.get_total_params(as_string=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.print_model_profile(
            profile_step=0, module_depth=3, top_modules=10,
            detailed=True, output_file=str(output_path),
        )
        return {
            "forward_flops": float(flops),
            "forward_macs": float(macs),
            "parameters": float(params),
            "forward_latency_sec": latency,
            "estimated_forward_flops_per_sec": float(flops / latency) if latency else None,
            "notice": "Forward-only, single GPU, one padded input; NOT training FLOPs",
        }
    finally:
        if profiler is not None:
            try:
                profiler.stop_profile()
                profiler.end_profile()
            except Exception as exc:
                print(f"FLOPs cleanup warning: {exc}", flush=True)
        model.train(previous_training_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--profile-sequence-length", type=int, default=512)
    parser.add_argument("--flops-log-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", required=True)
    args = parser.parse_args()

    if args.samples == 0 or args.samples < -1 or args.sequence_length < 16 or (args.max_steps < -1 or args.max_steps == 0):
        parser.error("--samples -1 (all) or positive, --sequence-length >= 16, --max-steps -1 or positive")
    if args.logging_steps < 1:
        parser.error("--logging-steps must be positive")
    if min(args.batch_size, args.gradient_accumulation_steps, args.lora_r, args.lora_alpha) < 1:
        parser.error("batch size, accumulation steps, LoRA rank and alpha must be positive")
    if args.learning_rate <= 0 or args.epochs <= 0 or not 0 <= args.lora_dropout < 1:
        parser.error("learning rate and epochs must be positive; LoRA dropout must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import torch only after the launcher has set CUDA_VISIBLE_DEVICES.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible CUDA GPU, got {torch.cuda.device_count()}; "
            "launch with run.py"
        )
    torch.cuda.set_device(0)
    set_seed(args.seed)
    gpu_label = os.environ.get("BENCH_PHYSICAL_GPU", "unknown")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"Physical GPU: {gpu_label}; visible UUID: {uuid}", flush=True)
    print(f"Logical device: cuda:0 / {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}; bf16: {torch.cuda.is_bf16_supported()}", flush=True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/driver does not support bf16 as required for the benchmark")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,  # retained from the working original code
        device_map={"": 0},  # logical CUDA 0 is the ONE GPU visible to this process
    )
    model.config.use_cache = False

    print(f"Loading dataset: {args.dataset}", flush=True)
    dataset = load_dataset(args.dataset, split="train")
    if args.samples != -1:
        dataset = dataset.select(range(min(args.samples, len(dataset))))
    dataset = dataset.map(
        lambda example: {"text": format_alpaca(example)},
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,  # avoids two workers writing the same mapped cache
    )
    print(f"Training samples: {len(dataset)}", flush=True)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    training_config = SFTConfig(
        output_dir=str(args.output_dir / "trainer"),
        dataset_text_field="text",
        max_length=args.sequence_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        disable_tqdm=True,
        save_strategy="no",
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,  # safer on Windows; no subprocess re-import
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        loss_type="nll",
    )
    write_json(args.output_dir / "effective_config.json", {
        "arguments": vars(args),
        "physical_gpu": gpu_label,
        "gpu_uuid": uuid,
        "training_config": training_config.to_dict(),
        "lora_config": lora_config.to_dict(),
    })
    trainer = SFTTrainer(
        model=model, args=training_config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora_config,
    )

    if args.profile_flops:
        print("Profiling forward FLOPs (optional)...", flush=True)
        try:
            profile = profile_forward_flops(
                trainer.model, tokenizer, args.flops_log_path or args.output_dir / "flops_profile.txt",
                args.profile_sequence_length,
            )
            write_json(args.output_dir / "forward_flops.json", profile)
            print(f"Forward FLOPs: {profile['forward_flops']:,.0f}", flush=True)
        except Exception as exc:
            # A profiler failure must not make a viable training benchmark fail.
            msg = f"Optional FLOPs profiler failed ({type(exc).__name__}): {exc}"
            print(msg, flush=True)
            (args.output_dir / "forward_flops_error.txt").write_text(msg, encoding="utf-8")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    start_epoch = time.time()
    start_perf = time.perf_counter()
    print("TRAINING START", flush=True)
    result = None
    status = "failed"
    try:
        result = trainer.train()
        torch.cuda.synchronize(0)
        status = "completed"
    finally:
        end_epoch = time.time()
        elapsed = time.perf_counter() - start_perf
        write_json(
            args.output_dir / "train_window.json",
            {"start_epoch": start_epoch, "end_epoch": end_epoch,
             "elapsed_seconds": elapsed, "status": status},
        )
        print(f"TRAINING END: {status}, {elapsed:.2f} sec", flush=True)

    peak_vram_gib = torch.cuda.max_memory_allocated(0) / 1024**3
    print("Saving LoRA adapter...", flush=True)
    trainer.save_model(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))
    write_json(
        args.output_dir / "worker_summary.json",
        {
            "status": status,
            "gpu": gpu_label, "gpu_uuid": uuid,
            "model": args.model_id, "pytorch_version": str(torch.__version__),
            "samples": len(dataset), "sequence_length": args.sequence_length,
            "max_steps": args.max_steps,
            "training_seconds": elapsed,
            "peak_allocated_vram_gib": round(peak_vram_gib, 3),
            "train_metrics": result.metrics if result is not None else None,
            "completed_utc": timestamp(),
        },
    )
    print(f"DONE: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
