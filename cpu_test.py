import ctypes
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from pprint import pprint

from PyLibreHardwareMonitor import Computer


def main() -> None:
    report = {
        "package_version": version("PyLibreHardwareMonitor"),
        "is_admin": bool(ctypes.windll.shell32.IsUserAnAdmin()) if os.name == "nt" else None,
        "samples": [],
        "error": None,
    }
    output = Path("cpu_sensor_diagnostics.json")
    try:
        computer = Computer(IsCpuEnabled=True)
        for index in range(3):
            cpu_data = computer.cpu
            report["samples"].append(cpu_data)
            print(f"CPU sensor sample {index + 1}:")
            pprint(cpu_data)
            if index < 2:
                time.sleep(1)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Administrator: {report['is_admin']}; diagnostics: {output.resolve()}")


if __name__ == "__main__":
    main()
