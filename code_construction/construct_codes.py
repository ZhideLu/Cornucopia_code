"""Construct the eight Cornucopia codes and verify their algebraic invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_construction.cornucopia_codes import SPECS, construct_one, write_text_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("code_construction/results"),
    )
    parser.add_argument("--write-matrices", action="store_true")
    parser.add_argument("--strict", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = args.output_dir / "matrices"
    records = [
        construct_one(spec, write_matrices=args.write_matrices, matrix_dir=matrix_dir)
        for spec in SPECS
    ]
    json_path = args.output_dir / "constructed_codes_summary.json"
    txt_path = args.output_dir / "constructed_codes_summary.txt"
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    write_text_summary(records, txt_path)

    for record in records:
        status = "OK" if record["checks"]["all_checks_ok"] else "FAIL"
        print(
            f"{status} {record['name']} {record['parameter_label']} "
            f"[[n,k,d]]=[[{record['n']},{record['k']},{record['expected_d']}]] "
            f"ranks=({record['rank_HX']},{record['rank_HZ']}) "
            f"source={record['source_apm']}"
        )
    print(f"summary_json={json_path}")
    print(f"summary_txt={txt_path}")
    if args.write_matrices:
        print(f"matrix_dir={matrix_dir}")

    failed = [record["name"] for record in records if not record["checks"]["all_checks_ok"]]
    if failed and args.strict:
        raise SystemExit(f"construction checks failed: {failed}")


if __name__ == "__main__":
    main()
