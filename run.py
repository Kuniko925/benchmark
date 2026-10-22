"""Independent, simultaneous single-GPU LoRA benchmarks on Windows.

Launches one worker per *physical GPU UUID*, avoiding DataParallel/DDP.
GPU power is monitored once per physical GPU; CPU power is monitored once
for the host. Does not double-count host CPU or CodeCarbon system power.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from telemetry import RowRotatingCsv, csv_parts


def name(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def safe_nvml(function, default=None):
    try:
        return function()
    except Exception:
        return default


def cpu_sensor(cpu_data, group: str, name_parts: list[str]):
    """Best-effort host sensor discovery using supplied code's dictionary format."""
    if not isinstance(cpu_data, dict):
        return None
    for hardware_data in cpu_data.values():
        if not isinstance(hardware_data, dict):
            continue
        sensors = hardware_data.get(group, {})
        if not isinstance(sensors, dict):
            continue
        for fragment in name_parts:
            for sensor_name, value in sensors.items():
                if fragment.lower() in str(sensor_name).lower() and value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
    return None


CPU_FIELDS = [
    "cpu_package_temperature_c", "cpu_core_average_temperature_c",
    "cpu_package_power_w", "cpu_total_load_percent", "cpu_average_clock_mhz",
]


def cpu_core_values(hardware_data, group: str, pattern: str):
    values = []
    sensors = hardware_data.get(group, {}) if isinstance(hardware_data, dict) else {}
    if isinstance(sensors, dict):
        for sensor_name, value in sensors.items():
            if re.fullmatch(pattern, str(sensor_name), flags=re.IGNORECASE) and value is not None:
                try:
                    number = float(value)
                    if math.isfinite(number):
                        values.append(number)
                except (TypeError, ValueError):
                    pass
    return values


def mean(values):
    return sum(values) / len(values) if values else None


def cpu_measurements(cpu_data, cpu_names):
    """Return socket rows and a host aggregate; never count partial power as total."""
    data = cpu_data if isinstance(cpu_data, dict) else {}
    rows = []
    for cpu_name in cpu_names:
        hardware = data.get(cpu_name, {})
        socket = {cpu_name: hardware}
        temperatures = cpu_core_values(hardware, "Temperature", r"CPU Core #\d+")
        clocks = cpu_core_values(hardware, "Clock", r"CPU Core #\d+")
        loads = cpu_core_values(hardware, "Load", r"CPU Core #\d+(?: Thread #\d+)?")
        core_temp = cpu_sensor(socket, "Temperature", ["Core Average", "CPU Core Average"])
        reported_load = cpu_sensor(socket, "Load", ["CPU Total", "Total CPU"])
        rows.append({
            "cpu_name": cpu_name,
            "cpu_package_temperature_c": cpu_sensor(socket, "Temperature", [
                "CPU Package", "Package", "CPU Die", "Tctl/Tdie",
            ]),
            "cpu_core_average_temperature_c": mean(temperatures) if temperatures else core_temp,
            "cpu_package_power_w": cpu_sensor(socket, "Power", ["CPU Package", "Package", "CPU PPT"]),
            "cpu_total_load_percent": mean(loads) if loads else reported_load,
            "cpu_average_clock_mhz": mean(clocks),
            "reported_cpu_total_load_percent": reported_load,
            "load_source": "mean_core_thread_sensors" if loads else "reported_cpu_total",
            "core_temperature_count": len(temperatures),
            "clock_count": len(clocks),
            "load_count": len(loads),
        })

    aggregate = {}
    for field in CPU_FIELDS:
        # Missing an entire socket must not silently produce a partial host result.
        if not rows or any(row[field] is None for row in rows):
            aggregate[field] = None
            continue
        if field == "cpu_package_power_w":
            aggregate[field] = sum(row[field] for row in rows)
        elif field == "cpu_package_temperature_c":
            aggregate[field] = max(row[field] for row in rows)
        else:
            count_key = {
                "cpu_core_average_temperature_c": "core_temperature_count",
                "cpu_average_clock_mhz": "clock_count",
                "cpu_total_load_percent": "load_count",
            }[field]
            weights = [max(row[count_key], 1) for row in rows]
            aggregate[field] = sum(row[field] * weight for row, weight in zip(rows, weights)) / sum(weights)
    return rows, aggregate


def read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def gpu_log_path(base: Path, index: int) -> Path:
    return base.with_name(f"{base.stem}_gpu{index}{base.suffix}")


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max_samples", "--samples", dest="samples", type=int, default=4,
                           help="Maximum training samples per GPU; -1 uses all")
    selection.add_argument("--all_samples", dest="samples", action="store_const", const=-1)
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--sequence_length", "--sequence-length", dest="sequence_length", type=int, default=512)
    parser.add_argument("--profile_sequence_length", type=int, default=512)
    parser.add_argument("--model_id", "--model-id", dest="model_id", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", "--output-root", dest="output_root", type=Path, default=None,
                        help="Results root; default: ./<model name> (contains gpu0/, gpu1/)")
    parser.add_argument("--gpu_log_path", type=Path, default=None)
    parser.add_argument("--cpu_log_path", type=Path, default=None)
    parser.add_argument("--flops_log_path", type=Path, default=None,
                        help="FLOPs path base; GPU index is appended to the filename")
    parser.add_argument("--log_interval_sec", "--interval-sec", dest="interval_sec", type=float, default=1.0)
    parser.add_argument("--csv_max_rows", type=int, default=100000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--profile_flops", "--profile-flops", dest="profile_flops", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=50)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    parser.add_argument("--carbon-intensity-g-per-kwh", type=float, default=None,
                        help="Optional supplied grid emission factor. No factor assumed.")
    args = parser.parse_args()
    if args.interval_sec <= 0 or args.samples == 0 or args.samples < -1:
        parser.error("interval must be positive; samples must be -1 (all) or positive")
    if min(args.csv_max_rows, args.logging_steps) < 1:
        parser.error("csv_max_rows and logging_steps must be positive")
    if args.sequence_length < 16 or args.profile_sequence_length < 1:
        parser.error("--sequence_length must be >= 16; --profile_sequence_length must be positive")
    if args.max_steps < -1 or args.max_steps == 0:
        parser.error("--max_steps must be -1 or positive")
    if min(args.batch_size, args.gradient_accumulation_steps, args.lora_r, args.lora_alpha) < 1:
        parser.error("batch size, accumulation steps, LoRA rank and alpha must be positive")
    if args.learning_rate <= 0 or args.epochs <= 0 or not 0 <= args.lora_dropout < 1:
        parser.error("learning rate and epochs must be positive; LoRA dropout must be in [0, 1)")
    if args.carbon_intensity_g_per_kwh is not None and args.carbon_intensity_g_per_kwh < 0:
        parser.error("carbon intensity must be >= 0")

    launch_benchmark(args)


def llm_worker_options(args, outdir):
    return [
        "--model-id", args.model_id, "--dataset", args.dataset,
        "--output-dir", str(outdir), "--samples", str(args.samples),
        "--max-steps", str(args.max_steps), "--sequence-length", str(args.sequence_length),
        "--profile-sequence-length", str(args.profile_sequence_length), "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--learning-rate", str(args.learning_rate), "--epochs", str(args.epochs),
        "--logging-steps", str(args.logging_steps),
        "--lora-r", str(args.lora_r), "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout), "--target-modules", *args.target_modules,
    ]


def launch_benchmark(args, worker_filename="bench_worker.py", worker_options=llm_worker_options):
    """Share physical GPU isolation, telemetry and energy summaries across workloads."""
    worker = Path(__file__).resolve().with_name(worker_filename)
    if not worker.is_file():
        raise FileNotFoundError(f"Missing worker: {worker}")

    model_name = Path(args.model_id.rstrip("/")).name
    if not model_name or model_name in {".", ".."}:
        raise ValueError("--model_id must contain a model name")
    if args.output_root is None:
        args.output_root = Path(model_name)
    root = args.output_root.resolve()
    args.gpu_log_path = (args.gpu_log_path or root / "gpu_metrics.csv").resolve()
    args.cpu_log_path = (args.cpu_log_path or root / "cpu_metrics.csv").resolve()
    if args.flops_log_path is not None:
        args.flops_log_path = args.flops_log_path.resolve()

    from pynvml import (
        NVML_CLOCK_MEM, NVML_CLOCK_SM, NVML_TEMPERATURE_GPU,
        nvmlDeviceGetClockInfo, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlDeviceGetPowerUsage,
        nvmlDeviceGetTemperature, nvmlDeviceGetUUID,
        nvmlDeviceGetUtilizationRates, nvmlInit, nvmlShutdown,
    )

    nvmlInit()
    jobs = []
    files = []
    cpu_monitor = None
    gpu_logs = []
    cpu_log = None
    cpu_writer = None
    start_epoch = time.time()
    try:
        total = nvmlDeviceGetCount()
        if total < 2:
            raise RuntimeError(f"Expected at least 2 NVIDIA GPUs, NVML detected {total}")
        root.mkdir(parents=True, exist_ok=True)
        # Reuse the model folder, but never overwrite a previous worker's results.
        for idx in (0, 1):
            if (root / f"gpu{idx}").exists():
                raise FileExistsError(
                    f"Existing results: {root / f'gpu{idx}'}. "
                    "Choose another --output_dir or move the previous results."
                )
        print(f"Results folder: {root}", flush=True)
        print("Mode: two independent single-GPU training jobs; no gradient synchronization", flush=True)

        # Use UUIDs so NVML's GPU 0/1 correspond exactly to the worker's physical GPU,
        # regardless of CUDA device enumeration order.
        gpu_descriptions = []
        for idx in (0, 1):
            handle = nvmlDeviceGetHandleByIndex(idx)
            uuid = name(nvmlDeviceGetUUID(handle))
            gpu_name = name(nvmlDeviceGetName(handle))
            gpu_descriptions.append((idx, uuid, gpu_name, handle))
            print(f"Physical GPU {idx}: {gpu_name} ({uuid})", flush=True)

        for idx, uuid, gpu_name, handle in gpu_descriptions:
            gpu_csv = gpu_log_path(args.gpu_log_path, idx)
            gpu_csv.parent.mkdir(parents=True, exist_ok=True)
            fp = RowRotatingCsv(gpu_csv, args.csv_max_rows)
            files.append(fp)
            writer = fp
            writer.writerow([
                "timestamp", "epoch_seconds", "gpu_index", "gpu_uuid", "gpu_name",
                "temperature_c", "gpu_utilization_percent", "memory_utilization_percent",
                "power_w", "memory_used_mib", "memory_total_mib",
                "sm_clock_mhz", "memory_clock_mhz",
            ])
            gpu_logs.append((idx, uuid, gpu_name, handle, fp, writer, gpu_csv))

        cpu_csv = args.cpu_log_path
        cpu_csv.parent.mkdir(parents=True, exist_ok=True)
        cpu_log = RowRotatingCsv(cpu_csv, args.csv_max_rows)
        files.append(cpu_log)
        cpu_writer = cpu_log
        cpu_writer.writerow(["timestamp", "epoch_seconds", *CPU_FIELDS])
        cpu_diagnostics = {"initial_raw_sensors": None, "errors": []}
        cpu_diagnostics_path = root / "cpu_sensor_diagnostics.json"
        try:
            from PyLibreHardwareMonitor import Computer
            cpu_monitor = Computer(IsCpuEnabled=True)
            cpu_diagnostics["initial_raw_sensors"] = cpu_monitor.cpu
            print("CPU sensor monitor initialized", flush=True)
        except Exception as exc:
            cpu_monitor = None
            cpu_diagnostics["errors"].append(f"{type(exc).__name__}: {exc}")
            print(f"CPU sensors unavailable ({type(exc).__name__}: {exc}); GPU benchmark continues", flush=True)
        cpu_diagnostics_path.write_text(
            json.dumps(cpu_diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        initial_cpu_data = cpu_diagnostics["initial_raw_sensors"]
        cpu_names = list(initial_cpu_data) if isinstance(initial_cpu_data, dict) else []
        socket_logs = {}

        # Both workers use the same downloaded model and selected dataset, but have
        # independent CUDA contexts, LoRA adapters, Trainer dirs and stdout logs.
        for idx, uuid, gpu_name, handle in gpu_descriptions:
            outdir = root / f"gpu{idx}"
            outdir.mkdir()
            logfile = (outdir / "worker.log").open("w", encoding="utf-8", buffering=1)
            files.append(logfile)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = uuid
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["HF_HUB_DISABLE_XET"] = "1"  # worked on this server
            env["PYTHONUNBUFFERED"] = "1"
            env["BENCH_PHYSICAL_GPU"] = str(idx)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            command = [
                sys.executable, "-u", str(worker),
                *worker_options(args, outdir),
            ]
            if args.profile_flops:
                command.append("--profile-flops")
                if args.flops_log_path is not None:
                    command.extend(["--flops-log-path", str(gpu_log_path(args.flops_log_path, idx))])
            job = subprocess.Popen(
                command, cwd=str(worker.parent), env=env,
                stdout=logfile, stderr=subprocess.STDOUT,
            )
            jobs.append((idx, job, outdir))
            print(f"Started GPU {idx}: PID={job.pid} (log: {outdir / 'worker.log'})", flush=True)

        last_status = 0.0
        cpu_failed = False
        cpu_missing_reported = False
        while True:
            now = time.time()
            stamp = datetime.now().isoformat(timespec="milliseconds")
            for idx, uuid, gpu_name, handle, fp, writer, gpu_csv in gpu_logs:
                util = safe_nvml(lambda h=handle: nvmlDeviceGetUtilizationRates(h))
                mem = safe_nvml(lambda h=handle: nvmlDeviceGetMemoryInfo(h))
                temperature = safe_nvml(lambda h=handle: nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU))
                power_mw = safe_nvml(lambda h=handle: nvmlDeviceGetPowerUsage(h))
                sm = safe_nvml(lambda h=handle: nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM))
                memclock = safe_nvml(lambda h=handle: nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM))
                writer.writerow([
                    stamp, now, idx, uuid, gpu_name,
                    temperature if temperature is not None else "",
                    util.gpu if util is not None else "",
                    util.memory if util is not None else "",
                    round(power_mw / 1000, 3) if power_mw is not None else "",
                    round(mem.used / 1024**2, 2) if mem is not None else "",
                    round(mem.total / 1024**2, 2) if mem is not None else "",
                    sm if sm is not None else "",
                    memclock if memclock is not None else "",
                ])
                fp.flush()

            cpu_data = None
            if cpu_monitor is not None and not cpu_failed:
                try:
                    cpu_data = cpu_monitor.cpu
                except Exception as exc:
                    cpu_failed = True
                    cpu_diagnostics["errors"].append(f"{type(exc).__name__}: {exc}")
                    cpu_diagnostics_path.write_text(
                        json.dumps(cpu_diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
                    )
                    print(f"CPU sensor stopped: {exc}", flush=True)
            if isinstance(cpu_data, dict):
                for cpu_name in cpu_data:
                    if cpu_name not in cpu_names:
                        cpu_names.append(cpu_name)
            socket_rows, host_cpu = cpu_measurements(cpu_data, cpu_names)
            for index, row in enumerate(socket_rows):
                cpu_name = row["cpu_name"]
                if cpu_name not in socket_logs:
                    path = cpu_csv.with_name(f"{cpu_csv.stem}_cpu{index}{cpu_csv.suffix}")
                    fp = RowRotatingCsv(path, args.csv_max_rows)
                    files.append(fp)
                    writer = fp
                    writer.writerow([
                        "timestamp", "epoch_seconds", "cpu_index", "cpu_name", *CPU_FIELDS,
                        "reported_cpu_total_load_percent", "load_source",
                    ])
                    socket_logs[cpu_name] = (fp, writer, path)
                fp, writer, _ = socket_logs[cpu_name]
                writer.writerow([
                    stamp, now, index, cpu_name,
                    *(row[field] if row[field] is not None else "" for field in CPU_FIELDS),
                    row["reported_cpu_total_load_percent"] if row["reported_cpu_total_load_percent"] is not None else "",
                    row["load_source"],
                ])
                fp.flush()
            if not cpu_missing_reported:
                missing = [key for key, value in host_cpu.items() if value is None]
                if missing:
                    cpu_missing_reported = True
                    cpu_diagnostics["missing_fields_at_first_detection"] = missing
                    cpu_diagnostics["raw_sensors_at_first_detection"] = cpu_data
                    cpu_diagnostics_path.write_text(
                        json.dumps(cpu_diagnostics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
                    )
                    print(f"CPU sensor fields unavailable: {', '.join(missing)}. "
                          f"See {cpu_diagnostics_path}", flush=True)
            cpu_writer.writerow([
                stamp, now,
                *(host_cpu[field] if host_cpu[field] is not None else "" for field in CPU_FIELDS),
            ])
            cpu_log.flush()

            statuses = [(idx, proc.poll()) for idx, proc, _ in jobs]
            if now - last_status >= 30:
                print("Worker status: " + ", ".join(
                    f"GPU{idx}={'running' if code is None else ('done' if code == 0 else 'FAILED('+str(code)+')')}"
                    for idx, code in statuses
                ), flush=True)
                last_status = now
            if all(code is not None for _, code in statuses):
                break
            time.sleep(args.interval_sec)

    except KeyboardInterrupt:
        print("Interrupted: terminating both workers...", flush=True)
        for _, proc, _ in jobs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in jobs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise SystemExit(130)
    except Exception:
        # A launch/monitor failure must not leave unmonitored training running.
        for _, proc, _ in jobs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in jobs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        for fp in files:
            try:
                fp.close()
            except Exception:
                pass
        nvmlShutdown()

    results = []
    for idx, proc, outdir in jobs:
        gpu_csv = gpu_log_path(args.gpu_log_path, idx)
        details = read_json(outdir / "worker_summary.json")
        results.append({
            "gpu_index": idx,
            "exit_code": proc.returncode,
            "training": details,
            "telemetry_files": [str(path) for path in csv_parts(gpu_csv)],
            "log": str(outdir / "worker.log"),
        })

    summary = {
        "status": "completed" if all(proc.returncode == 0 for _, proc, _ in jobs) else "some_workers_failed",
        "model": args.model_id,
        "samples_per_gpu": "all" if args.samples == -1 else args.samples,
        "requested_epochs": args.epochs,
        "csv_max_rows": args.csv_max_rows,
        "host_cpu_telemetry_files": [str(path) for path in csv_parts(args.cpu_log_path)],
        "mode": "two independent one-GPU training jobs (NOT distributed training)",
        "elapsed_session_seconds": round(time.time() - start_epoch, 2),
        "workers": results,
        "energy_calculation": "deferred_to_analysis_environment",
        "cpu_sockets": [
            {"cpu_name": cpu_name, "log": str(path),
             "telemetry_files": [str(part) for part in csv_parts(path)]}
            for cpu_name, (_, _, path) in socket_logs.items()
        ],
        "notes": [
            "No energy integration or emissions calculation is performed on the training server.",
            "Copy raw CSV parts, summary.json and gpu*/train_window.json to the analysis environment.",
            "The CPU package power (when available) is logged ONCE for the entire host; cannot be attributed to a worker.",
            "Host CPU package power is the sum across all detected sockets; missing socket power leaves the total blank.",
            "Host package temperature is the hottest socket; core temperature/clock are weighted by available core sensor counts.",
            "CPU load is the mean of available core/thread load sensors, falling back to reported CPU Total; socket CSV preserves that raw total.",
            "CPU and GPU sensor readings are sampling-based estimates, not wall-outlet energy.",
            "Forward FLOPs profiling (if enabled) is not the same as training FLOPs.",
        ],
    }
    if args.carbon_intensity_g_per_kwh is not None:
        summary["carbon_intensity_g_per_kwh"] = args.carbon_intensity_g_per_kwh
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Summary: {root / 'summary.json'}", flush=True)
    for r in results:
        print(f"GPU{r['gpu_index']} exit={r['exit_code']}", flush=True)
        print(f"  log: {r['log']}", flush=True)
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
