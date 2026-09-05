#!/usr/bin/env python3
"""Circuit-level noise simulation for the two BB codes.

Uses the shared sampling and decoding engine with bivariate-bicycle
parity checks and an eight-layer syndrome-extraction schedule.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bivariate_bicycle.code_construction import (
    BBCodeSpec,
    code_summary,
    parse_code_names,
)
from bivariate_bicycle.syndrome_circuit import (
    BB_X_ORDER,
    BB_Z_ORDER,
    StimNoiseConfig,
    build_code_and_validate,
    build_memory_circuit,
    logical_rows_for_basis,
)
from circuit_simulation import memory_experiment as sim

Basis = Literal["Z", "X"]
DecodingMode = Literal["xyz", "xz"]


def parse_bases(value: str) -> list[Basis]:
    if value == "both":
        return ["Z", "X"]
    if value in ("Z", "X"):
        return [value]  # type: ignore[list-item]
    raise SystemExit("--basis must be Z, X, or both")


def parse_decoding_modes(value: str) -> list[DecodingMode]:
    if value == "both":
        return ["xyz", "xz"]
    if value in ("xyz", "xz"):
        return [value]  # type: ignore[list-item]
    raise SystemExit("--decoding-mode must be xyz, xz, or both")


def make_conditions(
    specs: Sequence[BBCodeSpec],
    bases: Sequence[Basis],
    modes: Sequence[DecodingMode],
    p_list: Sequence[float],
    cycles: int,
) -> list[sim.Condition]:
    conditions: list[sim.Condition] = []
    for spec in specs:
        for basis in bases:
            for mode in modes:
                for p_noise in p_list:
                    conditions.append(
                        sim.Condition(
                            code_name=spec.name,
                            parameter_label=spec.parameter_label,
                            p_code=spec.block_size,
                            l=spec.ell,
                            j=spec.m,
                            expected_d=spec.expected_d,
                            basis=basis,
                            decoding_mode=mode,
                            p_noise=float(p_noise),
                            cycles=int(cycles),
                        )
                    )
    return conditions


def build_and_save_artifacts(
    spec: BBCodeSpec,
    condition: sim.Condition,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict:
    from relay_bp.stim import CheckMatrices  # type: ignore

    paths = sim.artifact_paths(output_dir, condition)
    if not force and all(path.exists() for path in paths.values()):
        return sim.load_artifact_meta(output_dir, condition)

    code, reconstruction, overlap_ok, overlap_rows = build_code_and_validate(spec)
    if not overlap_ok:
        raise ValueError(f"{spec.name}: BB schedule overlap: {overlap_rows}")

    logical_rows = logical_rows_for_basis(code, condition.basis)
    noise = StimNoiseConfig(
        p_cx=condition.p_noise,
        p_measure_flip=condition.p_noise,
        p_final_measure_flip=condition.p_noise,
        p_reset_flip=condition.p_noise,
    )
    circuit = build_memory_circuit(
        code,
        basis=condition.basis,
        logical_rows=logical_rows,
        rounds=condition.cycles,
        noise=noise,
    )
    detectors_removed = 0
    if condition.decoding_mode == "xz":
        circuit, _detectors_kept, detectors_removed = sim.filter_detectors_for_xz(
            circuit, condition.basis
        )

    paths["stim"].parent.mkdir(parents=True, exist_ok=True)
    paths["stim"].write_text(str(circuit), encoding="utf-8")
    dem = circuit.detector_error_model(decompose_errors=False)
    paths["dem"].write_text(str(dem), encoding="utf-8")

    matrices = CheckMatrices.from_dem(dem)
    check_matrix = matrices.check_matrix.tocsr().astype(np.uint8)
    observables_matrix = matrices.observables_matrix.tocsr().astype(np.uint8)
    check_matrix.data %= 2
    observables_matrix.data %= 2
    check_matrix.eliminate_zeros()
    observables_matrix.eliminate_zeros()
    error_priors = np.asarray(matrices.error_priors, dtype=np.float64)
    sim.save_check_matrices(paths["matrices"], check_matrix, observables_matrix, error_priors)

    if circuit.num_detectors <= 0 or circuit.num_observables <= 0 or check_matrix.shape[1] <= 0:
        raise ValueError(
            f"{condition.label}: invalid DEM dimensions "
            f"detectors={circuit.num_detectors} observables={circuit.num_observables} "
            f"columns={check_matrix.shape[1]}"
        )

    ranks = code_summary(spec)
    meta = {
        "script_version": 1,
        "label": condition.label,
        "code_name": condition.code_name,
        "parameter_label": condition.parameter_label,
        "P": condition.p_code,
        "L": condition.l,
        "J": condition.j,
        "ell": int(spec.ell),
        "m": int(spec.m),
        "a": list(spec.a),
        "b": list(spec.b),
        "expected_d": condition.expected_d,
        "basis": condition.basis,
        "decoding_mode": condition.decoding_mode,
        "p": condition.p_noise,
        "cycles": condition.cycles,
        "noise": noise.to_dict(),
        "schedule": {"Z": list(BB_Z_ORDER), "X": list(BB_X_ORDER)},
        "layer_reconstruction": reconstruction,
        "n": int(code.n),
        "k": int(code.num_logicals()),
        "rank_hx": int(ranks["rank_hx"]),
        "rank_hz": int(ranks["rank_hz"]),
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
        "num_dem_errors": int(dem.num_errors),
        "num_error_columns": int(check_matrix.shape[1]),
        "detectors_removed_for_xz": int(detectors_removed),
        "stim_path": str(paths["stim"]),
        "dem_path": str(paths["dem"]),
        "matrices_path": str(paths["matrices"]),
    }
    sim.write_json_atomic(paths["meta"], meta)
    return meta


def write_bb_summary(output_dir: Path, records: list[dict[str, object]]) -> None:
    sim.write_summary(output_dir, records)
    summary_txt = output_dir / "summary.txt"
    if not summary_txt.exists():
        return
    text = summary_txt.read_text(encoding="utf-8")
    text = text.replace(
        "Cornucopia circuit-level noise simulation summary",
        "BB code circuit-level noise simulation summary",
        1,
    )
    text = text.replace(" P=", " block_size=")
    summary_txt.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_workers = max(1, (os.cpu_count() or 1) - 10)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["generate", "decode", "both"], default="both")
    parser.add_argument("--decoder", choices=["relaybp"], default="relaybp")
    parser.add_argument("--codes", default="all")
    parser.add_argument("--basis", choices=["Z", "X", "both"], default="both")
    parser.add_argument("--decoding-mode", choices=["xyz", "xz", "both"], default="xyz")
    parser.add_argument("--p-list", default="0.002,0.003,0.004")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--shot-chunk", type=int, default=100)
    parser.add_argument("--decode-batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--sample-dir", type=Path, default=Path("bivariate_bicycle/results/samples")
    )
    parser.add_argument("--decode-dir", type=Path, default=Path("bivariate_bicycle/results/decode"))
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--force-artifacts", action="store_true")
    parser.add_argument("--force-shots", action="store_true")
    parser.add_argument("--force-decode", action="store_true")
    relay_defaults = sim.RelayBPConfig()
    parser.add_argument("--gamma0", type=float, default=relay_defaults.gamma0)
    parser.add_argument("--pre-iter", type=int, default=relay_defaults.pre_iter)
    parser.add_argument("--num-sets", type=int, default=relay_defaults.num_sets)
    parser.add_argument("--set-max-iter", type=int, default=relay_defaults.set_max_iter)
    parser.add_argument("--gamma-dist-min", type=float, default=relay_defaults.gamma_dist_min)
    parser.add_argument("--gamma-dist-max", type=float, default=relay_defaults.gamma_dist_max)
    parser.add_argument("--stop-nconv", type=int, default=relay_defaults.stop_nconv)
    relaybp_fallback_defaults = sim.RelayBPFallbackConfig(
        gamma0=0.1,
        pre_iter=500,
        num_sets=200,
        set_max_iter=200,
        gamma_dist_min=-0.24,
        gamma_dist_max=0.6,
        stop_nconv=4,
    )
    parser.add_argument(
        "--relaybp-fallback",
        action="store_true",
        help=("Retry primary RelayBP non-converged shots with a second stronger RelayBP config."),
    )
    parser.add_argument(
        "--relaybp-fallback-retry-unconverged",
        action="store_true",
        help=(
            "Do not rerun primary RelayBP. Read saved unconverged details from "
            "a previous BB RelayBP run and run only the strong RelayBP fallback "
            "on those shots. Requires --stage decode and --relaybp-fallback."
        ),
    )
    parser.add_argument(
        "--relaybp-fallback-source-hash",
        default="",
        help=(
            "Optional source BB RelayBP decode hash whose saved unconverged "
            "details should be retried by strong RelayBP fallback. Omit to "
            "auto-select the newest compatible source record."
        ),
    )
    parser.add_argument(
        "--relaybp-fallback-gamma0",
        type=float,
        default=relaybp_fallback_defaults.gamma0,
    )
    parser.add_argument(
        "--relaybp-fallback-pre-iter",
        type=int,
        default=relaybp_fallback_defaults.pre_iter,
    )
    parser.add_argument(
        "--relaybp-fallback-num-sets",
        type=int,
        default=relaybp_fallback_defaults.num_sets,
    )
    parser.add_argument(
        "--relaybp-fallback-set-max-iter",
        type=int,
        default=relaybp_fallback_defaults.set_max_iter,
    )
    parser.add_argument(
        "--relaybp-fallback-gamma-dist-min",
        type=float,
        default=relaybp_fallback_defaults.gamma_dist_min,
    )
    parser.add_argument(
        "--relaybp-fallback-gamma-dist-max",
        type=float,
        default=relaybp_fallback_defaults.gamma_dist_max,
    )
    parser.add_argument(
        "--relaybp-fallback-stop-nconv",
        type=int,
        default=relaybp_fallback_defaults.stop_nconv,
    )
    bposd_defaults = sim.BpOsdRetryConfig()
    parser.add_argument(
        "--bposd-retry-unconverged",
        action="store_true",
        help=(
            "Do not rerun primary RelayBP. Read saved RelayBP unconverged "
            "details from a previous BB decode record and run BPOSD only on "
            "the rows that are still unconverged."
        ),
    )
    parser.add_argument(
        "--bposd-retry-source-hash",
        default="",
        help=(
            "Optional source BB RelayBP decode hash whose saved still-unconverged "
            "details should be retried by BPOSD. Omit to auto-select the newest "
            "compatible source record."
        ),
    )
    parser.add_argument("--bposd-max-iter", type=int, default=bposd_defaults.max_iter)
    parser.add_argument("--bposd-bp-method", default=bposd_defaults.bp_method)
    parser.add_argument(
        "--bposd-ms-scaling-factor",
        type=float,
        default=bposd_defaults.ms_scaling_factor,
    )
    parser.add_argument("--bposd-schedule", default=bposd_defaults.schedule)
    parser.add_argument("--bposd-osd-method", default=bposd_defaults.osd_method)
    parser.add_argument("--bposd-osd-order", type=int, default=bposd_defaults.osd_order)
    parser.add_argument("--bposd-threads", type=int, default=bposd_defaults.threads)
    parser.add_argument(
        "--bposd-workers",
        type=int,
        default=default_workers,
        help=(
            "Process-level worker count for BPOSD retry. Each worker builds "
            "its own BpOsdDecoder and decodes a subset of saved-unconverged rows."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim.require_runtime_dependencies()
    if args.cycles <= 0 or args.shots <= 0 or args.shot_chunk <= 0 or args.workers <= 0:
        raise SystemExit("--cycles, --shots, --shot-chunk, and --workers must be positive")
    if args.decode_batch_size < 0:
        raise SystemExit("--decode-batch-size must be nonnegative")
    if args.relaybp_fallback_retry_unconverged:
        if args.stage != "decode":
            raise SystemExit("--relaybp-fallback-retry-unconverged requires --stage decode")
        if not args.relaybp_fallback:
            raise SystemExit("--relaybp-fallback-retry-unconverged requires --relaybp-fallback")
    if args.bposd_retry_unconverged:
        if args.stage != "decode":
            raise SystemExit("--bposd-retry-unconverged requires --stage decode")
        if args.relaybp_fallback_retry_unconverged:
            raise SystemExit(
                "--bposd-retry-unconverged cannot be combined with "
                "--relaybp-fallback-retry-unconverged"
            )
        if args.bposd_workers <= 0:
            raise SystemExit("--bposd-workers must be positive")
        if args.bposd_threads <= 0:
            raise SystemExit("--bposd-threads must be positive")

    specs = parse_code_names(args.codes)
    spec_by_name = {spec.name: spec for spec in specs}
    bases = parse_bases(args.basis)
    modes = parse_decoding_modes(args.decoding_mode)
    p_list = sim.parse_p_list(args.p_list)
    conditions = make_conditions(specs, bases, modes, p_list, args.cycles)
    relay_config = sim.RelayBPConfig(
        args.gamma0,
        args.pre_iter,
        args.num_sets,
        args.set_max_iter,
        args.gamma_dist_min,
        args.gamma_dist_max,
        args.stop_nconv,
    )
    relaybp_fallback_config = sim.RelayBPFallbackConfig(
        enabled=bool(args.relaybp_fallback),
        gamma0=float(args.relaybp_fallback_gamma0),
        pre_iter=int(args.relaybp_fallback_pre_iter),
        num_sets=int(args.relaybp_fallback_num_sets),
        set_max_iter=int(args.relaybp_fallback_set_max_iter),
        gamma_dist_min=float(args.relaybp_fallback_gamma_dist_min),
        gamma_dist_max=float(args.relaybp_fallback_gamma_dist_max),
        stop_nconv=int(args.relaybp_fallback_stop_nconv),
    )
    bposd_retry_config = sim.BpOsdRetryConfig(
        enabled=bool(args.bposd_retry_unconverged),
        max_iter=int(args.bposd_max_iter),
        bp_method=str(args.bposd_bp_method),
        ms_scaling_factor=float(args.bposd_ms_scaling_factor),
        schedule=str(args.bposd_schedule),
        osd_method=str(args.bposd_osd_method),
        osd_order=int(args.bposd_osd_order),
        threads=int(args.bposd_threads),
    )

    args.sample_dir.mkdir(parents=True, exist_ok=True)
    args.decode_dir.mkdir(parents=True, exist_ok=True)

    print(f"Python expected: {sim.RELAYBP_PYTHON}", flush=True)
    print(
        f"stage={args.stage} decoder={args.decoder} "
        f"codes={[spec.name for spec in specs]} bases={bases} "
        f"modes={modes} p_list={p_list} cycles={args.cycles} shots={args.shots} "
        f"shot_chunk={args.shot_chunk} decode_batch_size={args.decode_batch_size} "
        f"workers={args.workers} bposd_retry_unconverged={args.bposd_retry_unconverged} "
        f"bposd_retry_source_hash={args.bposd_retry_source_hash} "
        f"bposd_workers={args.bposd_workers}",
        flush=True,
    )
    print(f"sample_dir={args.sample_dir}", flush=True)
    print(f"decode_dir={args.decode_dir}", flush=True)
    print(f"relaybp_config={asdict(relay_config)}", flush=True)
    if relaybp_fallback_config.enabled:
        print(
            f"relaybp_fallback_config={asdict(relaybp_fallback_config)}",
            flush=True,
        )
    if args.relaybp_fallback_retry_unconverged:
        print(
            "relaybp_fallback_retry_unconverged=True "
            f"source_hash={args.relaybp_fallback_source_hash or 'auto'}",
            flush=True,
        )
    if bposd_retry_config.enabled:
        print(f"bposd_retry_config={asdict(bposd_retry_config)}", flush=True)
        print(f"bposd_workers={args.bposd_workers}", flush=True)
        print(
            f"bposd_retry_unconverged=True source_hash={args.bposd_retry_source_hash or 'auto'}",
            flush=True,
        )

    sample_summary_records = (
        sim.load_existing_summary_records(args.sample_dir)
        if args.stage in ("generate", "both")
        else []
    )
    decode_summary_records = (
        sim.rebuild_decode_summary_from_jsonl(args.sample_dir, args.decode_dir, "relaybp")
        if args.stage in ("decode", "both")
        else []
    )

    for idx, condition in enumerate(conditions):
        print(f"\n[{idx + 1}/{len(conditions)}] {condition.label}", flush=True)
        spec = spec_by_name[condition.code_name]
        condition_seed = int((args.seed + 1000003 * idx) % (2**64 - 1))
        if args.bposd_retry_unconverged:
            meta = sim.load_artifact_meta(args.sample_dir, condition)
            record = sim.run_bposd_retry_condition(
                condition,
                args.sample_dir,
                args.decode_dir,
                args.shots,
                args.shot_chunk,
                relay_config,
                meta,
                force=bool(args.force_decode),
                progress_seconds=args.progress_seconds,
                bposd_config=bposd_retry_config,
                bposd_workers=int(args.bposd_workers),
                source_hash=str(args.bposd_retry_source_hash).strip() or None,
            )
            record["shots_requested"] = record.get("shots_decoded")
            decode_summary_records = sim.rebuild_decode_summary_from_jsonl(
                args.sample_dir, args.decode_dir, "relaybp"
            )
            write_bb_summary(args.decode_dir, decode_summary_records)
            continue
        if args.relaybp_fallback_retry_unconverged:
            meta = sim.load_artifact_meta(args.sample_dir, condition)
            record = sim.run_relaybp_fallback_retry_condition(
                condition,
                args.sample_dir,
                args.decode_dir,
                args.shots,
                args.shot_chunk,
                relay_config,
                meta,
                force=bool(args.force_decode),
                progress_seconds=args.progress_seconds,
                decode_batch_size=args.decode_batch_size,
                relaybp_fallback_config=relaybp_fallback_config,
                source_hash=str(args.relaybp_fallback_source_hash).strip() or None,
            )
            record["shots_requested"] = record.get("shots_decoded")
            decode_summary_records = sim.rebuild_decode_summary_from_jsonl(
                args.sample_dir, args.decode_dir, "relaybp"
            )
            write_bb_summary(args.decode_dir, decode_summary_records)
            continue

        if args.stage in ("generate", "both"):
            meta = build_and_save_artifacts(
                spec, condition, args.sample_dir, force=bool(args.force_artifacts)
            )
            sim.run_generate_condition(
                condition,
                args.sample_dir,
                args.shots,
                args.shot_chunk,
                args.workers,
                condition_seed,
                meta,
                force=bool(args.force_shots),
                progress_seconds=args.progress_seconds,
            )
            summary = sim.condition_generation_summary(
                condition, args.sample_dir, args.shots, args.shot_chunk, meta
            )
            sim.upsert_summary_record(sample_summary_records, summary)
            write_bb_summary(args.sample_dir, sample_summary_records)

        if args.stage in ("decode", "both"):
            meta = sim.load_artifact_meta(args.sample_dir, condition)
            record = sim.run_decode_condition(
                condition,
                args.sample_dir,
                args.decode_dir,
                args.shots,
                args.shot_chunk,
                args.workers,
                "relaybp",
                relay_config,
                meta,
                force=bool(args.force_decode),
                progress_seconds=args.progress_seconds,
                decode_batch_size=args.decode_batch_size,
                mip_fallback_config=None,
                relaybp_fallback_config=relaybp_fallback_config,
            )
            record["shots_requested"] = record.get("shots_decoded")
            decode_summary_records = sim.rebuild_decode_summary_from_jsonl(
                args.sample_dir, args.decode_dir, "relaybp"
            )
            write_bb_summary(args.decode_dir, decode_summary_records)

    print(f"sample_summary_json={args.sample_dir / 'summary.json'}", flush=True)
    print(f"decode_summary_json={args.decode_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
