import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from PyLibreHardwareMonitor import Computer


def find_cpu_sensor(cpu_data: dict, sensor_type: str, preferred_names: list[str],) -> float | None:
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





def average_cpu_sensor(cpu_data: dict, sensor_type: str, name_fragment: str,) -> float | None:
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

def cpu_logger(stop_event: threading.Event, csv_path: Path, interval_sec: float = 1.0,) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        computer = Computer(IsCpuEnabled=True,)
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
