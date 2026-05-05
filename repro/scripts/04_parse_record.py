#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_latest(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(encoding="utf-8")
    records = re.split(r"\n?Scene ID: ", text)
    if len(records) < 2:
        raise SystemExit(f"No record block found in {path}")
    block = "Scene ID: " + records[1]

    def grab(pattern: str, cast):
        match = re.search(pattern, block)
        if not match:
            raise SystemExit(f"Could not parse pattern: {pattern}")
        return cast(match.group(1))

    return {
        "scene": grab(r"Scene ID:\s*(.+)", str),
        "episode": grab(r"Episode ID:\s*(.+)", str),
        "num_total": grab(r"No\.(\d+) task is finished", int),
        "success": grab(r"Total Success\s+\|\s+(\d+)", int),
        "spl_sum": grab(r"Total SPL\s+\|\s+([\d.]+)", float),
        "soft_spl_sum": grab(r"Total Soft SPL\s+\|\s+([\d.]+)", float),
        "dtg_sum": grab(r"Total Distance to Goal\s+\|\s+([\d.]+)", float),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: 04_parse_record.py videos/test_hm3dv2_val/continue.txt")
    path = Path(sys.argv[1])
    data = extract_latest(path)
    n = int(data["num_total"])
    success = int(data["success"])
    spl_sum = float(data["spl_sum"])
    soft_spl_sum = float(data["soft_spl_sum"])
    dtg_sum = float(data["dtg_sum"])
    print(f"episodes: {n}")
    print(f"success: {success}")
    print(f"SR: {100.0 * success / n:.2f}")
    print(f"SPL: {100.0 * spl_sum / n:.2f}")
    print(f"SoftSPL: {100.0 * soft_spl_sum / n:.2f}")
    print(f"DTG: {dtg_sum / n:.4f}")


if __name__ == "__main__":
    main()

