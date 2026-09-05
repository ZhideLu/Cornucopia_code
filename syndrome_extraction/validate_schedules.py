#!/usr/bin/env python3
"""Build and validate Cornucopia syndrome schedules with Stim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_construction.affine_codes import build_code
from code_construction.cornucopia_codes import SPECS
from syndrome_extraction.syndrome_schedule import (
    StimNoiseConfig,
    validate_syndrome_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("syndrome_extraction/results"),
    )
    parser.add_argument("--write-stim", action="store_true")
    parser.add_argument("--p-cx", type=float, default=0.0)
    parser.add_argument("--p-measure-flip", type=float, default=0.0)
    parser.add_argument("--p-final-measure-flip", type=float, default=0.0)
    parser.add_argument("--p-reset-flip", type=float, default=0.0)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stim_dir = args.output_dir / "stim_circuits"
    noise = StimNoiseConfig(
        p_cx=args.p_cx,
        p_measure_flip=args.p_measure_flip,
        p_final_measure_flip=args.p_final_measure_flip,
        p_reset_flip=args.p_reset_flip,
    )

    records = []
    for spec in SPECS:
        code = build_code(spec.to_code_spec())
        stim_path = None
        if args.write_stim:
            stim_path = stim_dir / f"{spec.name}_r{args.rounds}.stim"
        record = validate_syndrome_schedule(
            code,
            rounds=args.rounds,
            noise=noise,
            write_stim_path=stim_path,
        )
        if stim_path is not None:
            record["stim_path"] = stim_path.relative_to(args.output_dir).as_posix()
        records.append(record)
        status = (
            "OK"
            if record.get("data_disjoint") and record.get("deterministic_detectors")
            else "FAIL"
        )
        print(
            f"{status} {record['name']} {record['parameter_label']} "
            f"detectors={record['num_detectors']} dem_errors={record.get('dem_num_errors')} "
            f"stim={record.get('stim_path')}"
        )

    summary_path = args.output_dir / "schedule_validation_summary.json"
    summary_path.write_text(
        json.dumps(records, indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary_json={summary_path}")
    if args.write_stim:
        print(f"stim_dir={stim_dir}")

    failed = [
        record["name"]
        for record in records
        if not record.get("data_disjoint") or not record.get("deterministic_detectors")
    ]
    if failed:
        raise SystemExit(f"schedule validation failed: {failed}")


if __name__ == "__main__":
    main()
