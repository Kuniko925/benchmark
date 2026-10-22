"""Calculate energy offline from copied telemetry; Python standard library only."""
import argparse
import json
import math
from pathlib import Path, PureWindowsPath

from telemetry import integrate_energy


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def copied_parts(root, recorded):
    """Resolve by filename so Windows server paths work on another OS/machine.

    Copy all listed CSV parts directly into the copied results root. Never
    silently drop missing parts, as this would undercount measured energy.
    """
    parts = [root / PureWindowsPath(value).name for value in recorded]
    if not parts:
        raise ValueError("No telemetry files recorded")
    if len(set(parts)) != len(parts):
        raise ValueError("Duplicate telemetry filenames; preserve distinct log names when copying")
    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(f"Missing telemetry part: {part}")
    return parts


def energy(root, recorded, column="power_w", window=None):
    parts = copied_parts(root, recorded)
    return integrate_energy(
        parts[0], column=column, paths=parts,
        start_epoch=window["start_epoch"] if window else None,
        end_epoch=window["end_epoch"] if window else None,
    )


def analyze(root, intensity=None):
    summary = read_json(root / "summary.json")
    workers = []
    for worker in summary["workers"]:
        index = worker["gpu_index"]
        recorded = worker.get("telemetry_files")
        if not recorded:
            raise ValueError(f"GPU {index} has no telemetry manifest; use summary.json from the updated launcher")
        window_path = root / f"gpu{index}" / "train_window.json"
        window = read_json(window_path) if window_path.is_file() else None
        workers.append({
            "gpu_index": index, "exit_code": worker["exit_code"],
            "gpu_session_energy": energy(root, recorded),
            "gpu_training_energy": energy(root, recorded, window=window) if window else None,
            "training_window_status": window.get("status") if window else "not_recorded",
        })
    cpu_energy = energy(root, summary["host_cpu_telemetry_files"], column="cpu_package_power_w")
    total_gpu = (
        round(sum(worker["gpu_session_energy"]["energy_wh"] for worker in workers), 5)
        if workers and all(worker["gpu_session_energy"] is not None for worker in workers) else None
    )
    report = {
        "source_summary": str(root / "summary.json"),
        "training_status": summary["status"], "model": summary["model"],
        "workers": workers, "total_gpu_session_energy_wh": total_gpu,
        "host_cpu_session_energy": cpu_energy,
        "cpu_sockets": [
            {"cpu_name": socket["cpu_name"],
             "session_energy": energy(root, socket["telemetry_files"], column="cpu_package_power_w")}
            for socket in summary.get("cpu_sockets", [])
        ],
        "notes": [
            "Energy is integrated from sampled power, not measured at the wall outlet.",
            "Session energy includes loading/training/saving. Training windows are GPU-specific.",
            "Missing power breaks integration. Inspect covered_seconds before comparing runs.",
            "Host CPU energy already includes all measured sockets; do not add socket energy again.",
        ],
    }
    if intensity is None:
        intensity = summary.get("carbon_intensity_g_per_kwh")
    if intensity is not None:
        if not math.isfinite(intensity) or intensity < 0:
            raise ValueError("Carbon intensity must be finite and nonnegative")
        report["carbon_intensity_g_per_kwh"] = intensity
        if total_gpu is not None:
            report["gpu_only_estimated_emissions_g"] = round(total_gpu / 1000 * intensity, 6)
            if cpu_energy is not None:
                report["gpu_plus_cpu_estimated_emissions_g"] = round(
                    (total_gpu + cpu_energy["energy_wh"]) / 1000 * intensity, 6
                )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--carbon-intensity-g-per-kwh", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    output = args.output or root / "energy_summary.json"
    if output.exists():
        parser.error(f"Output exists: {output}; choose another --output")
    if not (root / "summary.json").is_file():
        parser.error(f"Missing summary.json in {root}")
    report = analyze(root, args.carbon_intensity_g_per_kwh)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"Energy report: {output.resolve()}")


if __name__ == "__main__":
    main()
