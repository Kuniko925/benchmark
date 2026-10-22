"""Bounded CSV parts and streaming energy integration; no GPU dependencies."""
import csv
import math
from pathlib import Path


def csv_parts(base):
    base = Path(base)
    parts = []
    if base.exists():
        parts.append(base)
    parts.extend(sorted(base.parent.glob(f"{base.stem}.part[0-9][0-9][0-9][0-9][0-9][0-9]{base.suffix}")))
    return parts


class RowRotatingCsv:
    """The first writerow supplies the header; every part repeats it.

    All parts are retained. max_rows counts data rows, excluding the header.
    """
    def __init__(self, path, max_rows=100000):
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.path = Path(path)
        self.max_rows = max_rows
        self.header = None
        self.part = 0
        self.rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if csv_parts(self.path):
            raise FileExistsError(f"Existing telemetry: {self.path}; choose a new output location")
        self.file = self.path.open("x", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)

    def writerow(self, row):
        row = list(row)
        if self.header is None:
            self.header = row
            self.writer.writerow(row)
            return
        if self.rows >= self.max_rows:
            self.file.close()
            self.part += 1
            path = self.path.with_name(f"{self.path.stem}.part{self.part:06d}{self.path.suffix}")
            self.file = path.open("x", newline="", encoding="utf-8-sig")
            self.writer = csv.writer(self.file)
            self.writer.writerow(self.header)
            self.rows = 0
        self.writer.writerow(row)
        self.rows += 1

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def integrate_energy(csv_file, start_epoch=None, end_epoch=None, column="power_w", paths=None):
    """Integrate across ordered parts in constant memory, including part boundaries.

    Clip segments to the requested interval. Missing/invalid samples break a
    segment rather than estimating power through an interval with missing data.
    """
    previous = None
    energy_wh = 0.0
    covered = 0.0
    for path in (csv_parts(csv_file) if paths is None else paths):
        with path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                try:
                    current = (float(row["epoch_seconds"]), float(row[column]))
                    if not all(math.isfinite(value) for value in current) or current[1] < 0:
                        raise ValueError("Invalid timestamp or power")
                except (KeyError, TypeError, ValueError):
                    previous = None
                    continue
                if previous is not None:
                    t0, p0 = previous
                    t1, p1 = current
                    if t1 > t0:
                        a = max(t0, start_epoch) if start_epoch is not None else t0
                        b = min(t1, end_epoch) if end_epoch is not None else t1
                        if b > a:
                            pa = p0 + (p1 - p0) * (a - t0) / (t1 - t0)
                            pb = p0 + (p1 - p0) * (b - t0) / (t1 - t0)
                            energy_wh += (pa + pb) / 2 * (b - a) / 3600
                            covered += b - a
                previous = current
    return {"energy_wh": round(energy_wh, 5), "covered_seconds": round(covered, 2)} if covered > 0 else None
