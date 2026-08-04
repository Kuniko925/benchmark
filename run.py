import os

os.environ["USE_LIBUV"] = "0"
os.environ["ACCELERATE_USE_GLOO"] = "1"

import threading
import time
from pathlib import Path
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from codecarbon import EmissionsTracker
from gpulog import gpu_logger
from cpulog import cpu_logger
from flopslog import profile_forward_flops

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

def main(arg) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("PyTorch:", torch.__version__)
    print("GPU log:", args.gpu_log_path.resolve())

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, use_cache=False,)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    dataset = load_dataset(args.dataset, split="train",)
    dataset = dataset.select(range(min(1000, len(dataset))))
    dataset = dataset.map(lambda example: {"text": format_alpaca(example)},remove_columns=dataset.column_names,)

    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_config = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        packing=False,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(model=model, args=training_config, train_dataset=dataset, processing_class=tokenizer, peft_config=lora_config,)

    profile_forward_flops(
        model=trainer.model, tokenizer=tokenizer, output_path=args.flops_log_path, device=torch.device(f"cuda:0"), sequence_length=args.profile_sequence_length,
    )

    stop_event = threading.Event()

    stop_event = threading.Event()
    gpu_logger_threads = []

    for gpu_index in range(torch.cuda.device_count()):
        gpu_log_path = args.gpu_log_path.with_name(
            f"{args.gpu_log_path.stem}_gpu{gpu_index}"
            f"{args.gpu_log_path.suffix}"
        )

        thread = threading.Thread(
            target=gpu_logger,
            args=(
                stop_event,
                gpu_log_path,
                gpu_index,
                args.log_interval_sec,
            ),
            daemon=True,
        )

        gpu_logger_threads.append(thread)

    cpu_logger_thread = threading.Thread(
        target=cpu_logger,
        args=(stop_event, args.cpu_log_path, args.log_interval_sec,),
        daemon=True,
    )

    carbon_tracker = EmissionsTracker(
        log_level="error",
        save_to_file=False,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for thread in gpu_logger_threads:
        thread.start()

    cpu_logger_thread.start()
    carbon_tracker.start()
    start = time.perf_counter()

    try:
        trainer.train()

    finally:
        torch.cuda.synchronize()

        stop_event.set()

        for thread in gpu_logger_threads:
            thread.join(timeout=10)
        cpu_logger_thread.join(timeout=10)

    elapsed = time.perf_counter() - start
    emissions = carbon_tracker.stop()
    energy_wh = carbon_tracker._total_energy.kWh * 1000

    peak_vram_gib = (torch.cuda.max_memory_allocated() / 1024**3)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training time: {elapsed:.2f} sec")
    print(f"Peak allocated VRAM: {peak_vram_gib:.2f} GiB")
    print(f"LoRA adapter saved to: {args.output_dir.resolve()}")
    print(f"GPU metrics saved to: {args.gpu_log_path.resolve()}")
    print(f"CPU metrics saved to: {args.cpu_log_path.resolve()}")
    print(f"Energy: {energy_wh:.4f} Wh")
    print(f"CO2 emissions: {emissions * 1000:.4f} gCO2eq")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", type=str, default=Path("./mistral-7b-v03-alpaca-lora"))
    parser.add_argument("--gpu_log_path", type=str, default=Path("./gpu_metrics_mistralai.csv"))
    parser.add_argument("--cpu_log_path", type=str, default=Path("./cpu_metrics_mistralai.csv"),)
    parser.add_argument("--log_interval_sec", type=float, default=1.0)
    parser.add_argument("--profile_flops", action="store_true",)
    parser.add_argument("--flops_log_path", type=Path, default=Path("./flops_profile.txt"),)
    parser.add_argument("--profile_sequence_length",type=int, default=512,)
    args = parser.parse_args()
    main(args)