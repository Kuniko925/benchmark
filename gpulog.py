import csv
import threading
import time
from datetime import datetime
from pathlib import Path

from pynvml import (
    NVML_CLOCK_MEM,
    NVML_CLOCK_SM,
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
