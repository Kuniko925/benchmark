import csv
import threading
import time
from datetime import datetime
from pathlib import Path
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from pynvml import (
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetClockInfo,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
    NVML_CLOCK_MEM,
    NVML_CLOCK_SM,
)
from PyLibreHardwareMonitor import Computer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from flops_profiler.profiler import FlopsProfiler
from codecarbon import EmissionsTracker



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


def gpu_logger(
    stop_event: threading.Event,
    csv_path: Path,
    gpu_index: int = 0,
    interval_sec: float = 1.0,
) -> None:

    nvmlInit()

    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        gpu_name = nvmlDeviceGetName(handle)

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "timestamp",
                    "elapsed_sec",
                    "gpu_index",
                    "gpu_name",
                    "temperature_c",
                    "gpu_utilization_percent",
                    "memory_utilization_percent",
                    "power_w",
                    "memory_used_mib",
                    "memory_total_mib",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                ]
            )

            monitor_start = time.perf_counter()

            while not stop_event.is_set():
                timestamp = datetime.now().isoformat(
                    sep=" ",
                    timespec="milliseconds",
                )
                elapsed_sec = time.perf_counter() - monitor_start

                temperature_c = nvmlDeviceGetTemperature(
                    handle,
                    NVML_TEMPERATURE_GPU,
                )

                utilization = nvmlDeviceGetUtilizationRates(handle)

                power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0

                memory = nvmlDeviceGetMemoryInfo(handle)
                memory_used_mib = memory.used / 1024**2
                memory_total_mib = memory.total / 1024**2

                sm_clock_mhz = nvmlDeviceGetClockInfo(
                    handle,
                    NVML_CLOCK_SM,
                )
                memory_clock_mhz = nvmlDeviceGetClockInfo(
                    handle,
                    NVML_CLOCK_MEM,
                )

                writer.writerow(
                    [
                        timestamp,
                        f"{elapsed_sec:.3f}",
                        gpu_index,
                        gpu_name,
                        temperature_c,
                        utilization.gpu,
                        utilization.memory,
                        f"{power_w:.3f}",
                        f"{memory_used_mib:.2f}",
                        f"{memory_total_mib:.2f}",
                        sm_clock_mhz,
                        memory_clock_mhz,
                    ]
                )

                csv_file.flush()

                stop_event.wait(interval_sec)

    except Exception as exc:
        print(f"GPU monitoring error: {exc}")

    finally:
        nvmlShutdown()

def find_cpu_sensor(
    cpu_data: dict,
    sensor_type: str,
    preferred_names: list[str],
) -> float | None:
    for hardware_name, hardware_data in cpu_data.items():
        if not isinstance(hardware_data, dict):
            continue

        sensor_group = hardware_data.get(sensor_type, {})

        if not isinstance(sensor_group, dict):
            continue

        for preferred_name in preferred_names:
            preferred_lower = preferred_name.lower()

            for sensor_name, value in sensor_group.items():
                if (
                    preferred_lower in sensor_name.lower()
                    and value is not None
                ):
                    return float(value)

    return None


def average_cpu_sensor(
    cpu_data: dict,
    sensor_type: str,
    name_fragment: str,
) -> float | None:
    values: list[float] = []
    fragment_lower = name_fragment.lower()

    for hardware_data in cpu_data.values():
        if not isinstance(hardware_data, dict):
            continue

        sensor_group = hardware_data.get(sensor_type, {})

        if not isinstance(sensor_group, dict):
            continue

        for sensor_name, value in sensor_group.items():
            if (
                fragment_lower in sensor_name.lower()
                and value is not None
            ):
                values.append(float(value))

    if not values:
        return None

    return sum(values) / len(values)

def cpu_logger(
    stop_event: threading.Event,
    csv_path: Path,
    interval_sec: float = 1.0,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        computer = Computer(
            IsCpuEnabled=True,
        )

        initial_cpu_data = computer.cpu

        print("CPU monitoring started.")
        print("Detected CPU sensor data:")
        print(initial_cpu_data)

        with csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow(
                [
                    "timestamp",
                    "elapsed_sec",
                    "cpu_package_temperature_c",
                    "cpu_core_average_temperature_c",
                    "cpu_package_power_w",
                    "cpu_total_load_percent",
                    "cpu_average_clock_mhz",
                ]
            )

            monitor_start = time.perf_counter()

            while not stop_event.is_set():
                timestamp = datetime.now().isoformat(
                    sep=" ",
                    timespec="milliseconds",
                )

                elapsed_sec = (
                    time.perf_counter() - monitor_start
                )

                cpu_data = computer.cpu

                package_temperature = find_cpu_sensor(
                    cpu_data,
                    "Temperature",
                    [
                        "CPU Package",
                        "Package",
                        "CPU Die",
                        "Tctl/Tdie",
                        "Core Max",
                    ],
                )

                core_average_temperature = find_cpu_sensor(
                    cpu_data,
                    "Temperature",
                    [
                        "Core Average",
                        "CPU Core Average",
                    ],
                )

                if core_average_temperature is None:
                    core_average_temperature = average_cpu_sensor(
                        cpu_data,
                        "Temperature",
                        "CPU Core #",
                    )

                package_power = find_cpu_sensor(
                    cpu_data,
                    "Power",
                    [
                        "CPU Package",
                        "Package",
                        "CPU PPT",
                    ],
                )

                total_load = find_cpu_sensor(
                    cpu_data,
                    "Load",
                    [
                        "CPU Total",
                        "Total CPU",
                    ],
                )

                average_clock = find_cpu_sensor(
                    cpu_data,
                    "Clock",
                    [
                        "CPU Core Average",
                        "Core Average",
                    ],
                )

                if average_clock is None:
                    average_clock = average_cpu_sensor(
                        cpu_data,
                        "Clock",
                        "CPU Core #",
                    )

                writer.writerow(
                    [
                        timestamp,
                        f"{elapsed_sec:.3f}",
                        (
                            f"{package_temperature:.2f}"
                            if package_temperature is not None
                            else ""
                        ),
                        (
                            f"{core_average_temperature:.2f}"
                            if core_average_temperature is not None
                            else ""
                        ),
                        (
                            f"{package_power:.2f}"
                            if package_power is not None
                            else ""
                        ),
                        (
                            f"{total_load:.2f}"
                            if total_load is not None
                            else ""
                        ),
                        (
                            f"{average_clock:.2f}"
                            if average_clock is not None
                            else ""
                        ),
                    ]
                )

                csv_file.flush()
                stop_event.wait(interval_sec)

    except Exception as exc:
        print(f"CPU monitoring error: {exc}")

def profile_forward_flops(
    model,
    tokenizer,
    output_path: Path,
    device: torch.device,
    sequence_length: int = 512,
) -> dict:
    sample_text = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\n"
        "Explain why cooling is important for GPU training.\n\n"
        "### Response:\n"
    )

    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
        padding="max_length",
    )

    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    previous_training_state = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(3):
            model(
                **inputs,
                use_cache=False,
            )

    torch.cuda.synchronize()

    profiler = FlopsProfiler(model)
    profiler.start_profile()

    start = time.perf_counter()

    try:
        with torch.no_grad():
            model(
                **inputs,
                use_cache=False,
            )

        torch.cuda.synchronize()

        elapsed_sec = time.perf_counter() - start

        flops = profiler.get_total_flops(
            as_string=False
        )
        macs = profiler.get_total_macs(
            as_string=False
        )
        params = profiler.get_total_params(
            as_string=False
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        profiler.print_model_profile(
            profile_step=0,
            module_depth=3,
            top_modules=10,
            detailed=True,
            output_file=str(output_path),
        )

    finally:
        profiler.stop_profile()
        profiler.end_profile()

        if previous_training_state:
            model.train()

    flops_per_sec = (
        flops / elapsed_sec
        if elapsed_sec > 0
        else 0.0
    )

    print(f"Forward FLOPs: {flops:,.0f}")
    print(f"Forward MACs: {macs:,.0f}")
    print(f"Parameters: {params:,.0f}")
    print(f"Forward latency: {elapsed_sec:.6f} sec")
    print(f"Estimated FLOP/s: {flops_per_sec:,.0f}")
    print(f"Profile saved to: {output_path.resolve()}")

    return {
        "forward_flops": float(flops),
        "forward_macs": float(macs),
        "parameters": float(params),
        "forward_latency_sec": elapsed_sec,
        "estimated_flops_per_sec": flops_per_sec,
    }


def main(arg) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(GPU_INDEX))
    print("PyTorch:", torch.__version__)
    print("GPU log:", args.gpu_log_path.resolve())

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": GPU_INDEX},
        use_cache=False,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    dataset = load_dataset("tatsu-lab/alpaca", split="train",)
    dataset = dataset.select(range(min(1000, len(dataset))))
    dataset = dataset.map(lambda example: {"text": format_alpaca(example)},remove_columns=dataset.column_names,)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
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

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    profile_forward_flops(
        model=trainer.model,
        tokenizer=tokenizer,
        output_path=args.flops_log_path,
        device=torch.device(f"cuda:{GPU_INDEX}"),
        sequence_length=args.profile_sequence_length,
    )

    stop_event = threading.Event()

    gpu_logger_thread = threading.Thread(
        target=gpu_logger,
        args=(
            stop_event,
            args.gpu_log_path,
            GPU_INDEX,
            args.log_interval_sec,
        ),
        daemon=True,
    )

    cpu_logger_thread = threading.Thread(
        target=cpu_logger,
        args=(
            stop_event,
            args.cpu_log_path,
            args.log_interval_sec,
        ),
        daemon=True,
    )

    carbon_tracker = EmissionsTracker(
        log_level="error",
        save_to_file=False,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    gpu_logger_thread.start()
    cpu_logger_thread.start()
    carbon_tracker.start()
    start = time.perf_counter()

    try:
        trainer.train()

    finally:
        torch.cuda.synchronize()

        stop_event.set()
        gpu_logger_thread.join(timeout=10)
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

GPU_INDEX = 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--output_dir", type=str, default=Path("./mistral-7b-v03-alpaca-lora"))
    parser.add_argument("--gpu_log_path", type=str, default=Path("./gpu_metrics_mistralai.csv"))
    parser.add_argument("--cpu_log_path", type=str, default=Path("./cpu_metrics_mistralai.csv"),)
    parser.add_argument("--log_interval_sec", type=float, default=1.0)
    parser.add_argument("--profile_flops", action="store_true",)
    parser.add_argument("--flops_log_path", type=Path, default=Path("./flops_profile.txt"),)
    parser.add_argument("--profile_sequence_length",type=int, default=512,)
    args = parser.parse_args()
    main(args)