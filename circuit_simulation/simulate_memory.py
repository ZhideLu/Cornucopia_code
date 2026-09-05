"""Sample and decode Cornucopia logical-memory experiments."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit_simulation.memory_experiment import (
    RELAYBP_PYTHON,
    BpOsdRetryConfig,
    DecoderName,
    MIPFallbackConfig,
    RelayBPConfig,
    RelayBPFallbackConfig,
    build_and_save_artifacts,
    condition_generation_summary,
    load_artifact_meta,
    load_existing_summary_records,
    make_conditions,
    parse_bases,
    parse_codes,
    parse_decoding_modes,
    parse_p_list,
    rebuild_decode_summary_from_jsonl,
    require_runtime_dependencies,
    run_bposd_retry_condition,
    run_decode_condition,
    run_generate_condition,
    run_mip_retry_unsolved_condition,
    run_relaybp_fallback_retry_condition,
    upsert_summary_record,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    default_workers = max(1, (os.cpu_count() or 1) - 10)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["generate", "decode", "both"], default="both")
    parser.add_argument("--codes", default="all")
    parser.add_argument("--basis", choices=["Z", "X", "both"], default="both")
    parser.add_argument("--decoding-mode", choices=["xyz", "xz", "both"], default="xyz")
    parser.add_argument("--decoder", choices=["relaybp"], default="relaybp")
    parser.add_argument("--p-list", default="0.002,0.003,0.004")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--shot-chunk", type=int, default=100)
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=1000,
        help="Shots decoded per RelayBP batch. Use 0 to decode all saved shots in one call.",
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=None,
        help="Directory for circuit/DEM/CheckMatrices artifacts and sampled shot chunks. Default: circuit_simulation/results/samples.",
    )
    parser.add_argument(
        "--decode-dir",
        type=Path,
        default=None,
        help="Directory for decode JSONL files and decode summary. Default: circuit_simulation/results/decode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root: store sample outputs under OUTPUT_DIR/sample_results and decode outputs under OUTPUT_DIR/decode_results unless overridden.",
    )
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--min-failures",
        type=int,
        default=0,
        help=(
            "If positive, run --stage both repeatedly, increasing the total "
            "requested shots by --shots each round, until total failures "
            "reaches this threshold for each condition. For RelayBP+MIP this "
            "uses the post-fallback all-shots failures."
        ),
    )
    parser.add_argument(
        "--min-converged-failures",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. "
            "If positive, run --stage both repeatedly, increasing the total "
            "requested shots by --shots each round, until c_failures reaches "
            "this threshold for each condition."
        ),
    )
    parser.add_argument("--force-artifacts", action="store_true")
    parser.add_argument("--force-shots", action="store_true")
    parser.add_argument("--force-decode", action="store_true")
    relay_defaults = RelayBPConfig()
    parser.add_argument("--gamma0", type=float, default=relay_defaults.gamma0)
    parser.add_argument("--pre-iter", type=int, default=relay_defaults.pre_iter)
    parser.add_argument("--num-sets", type=int, default=relay_defaults.num_sets)
    parser.add_argument("--set-max-iter", type=int, default=relay_defaults.set_max_iter)
    parser.add_argument("--gamma-dist-min", type=float, default=relay_defaults.gamma_dist_min)
    parser.add_argument("--gamma-dist-max", type=float, default=relay_defaults.gamma_dist_max)
    parser.add_argument("--stop-nconv", type=int, default=relay_defaults.stop_nconv)
    relaybp_fallback_defaults = RelayBPFallbackConfig()
    parser.add_argument(
        "--relaybp-fallback",
        action="store_true",
        help=(
            "For RelayBP only: retry primary RelayBP non-converged shots with "
            "a second stronger RelayBP config before optional MIP fallback."
        ),
    )
    parser.add_argument(
        "--relaybp-fallback-retry-unconverged",
        action="store_true",
        help=(
            "For RelayBP only: do not rerun primary RelayBP. Read saved "
            "unconverged details from a previous conventional RelayBP+MIP run "
            "and run only the strong RelayBP fallback on those shots. Requires "
            "--stage decode and --relaybp-fallback."
        ),
    )
    parser.add_argument(
        "--relaybp-fallback-source-hash",
        default="",
        help=(
            "Optional source conventional RelayBP+MIP decode hash whose saved "
            "unconverged details should be retried by strong RelayBP fallback. "
            "Omit to auto-select the newest compatible source record."
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
    bposd_defaults = BpOsdRetryConfig()
    parser.add_argument(
        "--bposd-retry-unconverged",
        action="store_true",
        help=(
            "For RelayBP only: do not rerun primary RelayBP. Read saved "
            "RelayBP unconverged details from a previous decode record and run "
            "BPOSD only on the rows that are still unconverged."
        ),
    )
    parser.add_argument(
        "--bposd-retry-source-hash",
        default="",
        help=(
            "Optional source RelayBP decode hash whose saved still-unconverged "
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
            "Process-level worker count for BPOSD retry. Each worker builds its "
            "own BpOsdDecoder and decodes a subset of saved-unconverged rows. "
            "This affects scheduling only, not the decode cache hash."
        ),
    )
    mip_defaults = MIPFallbackConfig()
    parser.add_argument(
        "--mip-fallback",
        action="store_true",
        help="For RelayBP only: decode non-converged shots with an exact MIP fallback.",
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        default=mip_defaults.time_limit,
        help="Per-shot MIP time limit in seconds. 0 means no limit.",
    )
    parser.add_argument(
        "--mip-wall-time-limit",
        type=float,
        default=mip_defaults.wall_time_limit,
        help=(
            "Total deferred MIP fallback wall-time limit in seconds per decode "
            "record. 0 means no global MIP wall-time limit."
        ),
    )
    parser.add_argument(
        "--mip-rel-gap",
        type=float,
        default=mip_defaults.mip_rel_gap,
    )
    parser.add_argument(
        "--mip-feasible-only",
        action="store_true",
        help="Stop each fallback MIP after the first feasible syndrome-matching solution.",
    )
    parser.add_argument(
        "--mip-workers",
        type=int,
        default=default_workers,
        help=(
            "Parallel worker count for deferred MIP fallback. "
            "This affects scheduling only, not the decode cache hash."
        ),
    )
    parser.add_argument(
        "--mip-threads",
        type=int,
        default=mip_defaults.threads,
        help=(
            "CBC threads per individual MIP solve. Increase this when there "
            "are few non-converged shots and many idle CPU cores."
        ),
    )
    parser.add_argument(
        "--mip-unsolved-not-failure",
        action="store_true",
        help="Do not count unsolved MIP fallback shots as failures.",
    )
    parser.add_argument(
        "--mip-retry-unsolved",
        action="store_true",
        help=(
            "Do not rerun RelayBP. Read MIP fallback detail files from a previous "
            "RelayBP+MIP run and retry only the shots whose MIP fallback was "
            "unsolved. Requires --stage decode and --mip-fallback. If "
            "--mip-retry-source-hash is omitted, the newest compatible source "
            "record is selected automatically."
        ),
    )
    parser.add_argument(
        "--mip-retry-source-hash",
        default="",
        help=(
            "Optional source RelayBP+MIP decode hash whose saved unsolved MIP "
            "details will be retried. Omit to auto-select the newest compatible source."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    if args.cycles <= 0 or args.shots <= 0 or args.shot_chunk <= 0 or args.workers <= 0:
        raise SystemExit("--cycles, --shots, --shot-chunk, and --workers must be positive")
    if args.decode_batch_size < 0:
        raise SystemExit("--decode-batch-size must be nonnegative")
    if args.mip_workers <= 0:
        raise SystemExit("--mip-workers must be positive")
    if args.mip_threads <= 0:
        raise SystemExit("--mip-threads must be positive")
    if args.mip_retry_unsolved:
        if args.stage != "decode":
            raise SystemExit("--mip-retry-unsolved requires --stage decode")
        if not args.mip_fallback:
            raise SystemExit("--mip-retry-unsolved requires --mip-fallback")
        if args.relaybp_fallback:
            raise SystemExit("--mip-retry-unsolved cannot be combined with --relaybp-fallback")
        if args.min_failures > 0 or args.min_converged_failures > 0:
            raise SystemExit("--mip-retry-unsolved cannot be combined with adaptive extension")
    if args.relaybp_fallback_retry_unconverged:
        if args.stage != "decode":
            raise SystemExit("--relaybp-fallback-retry-unconverged requires --stage decode")
        if not args.relaybp_fallback:
            raise SystemExit("--relaybp-fallback-retry-unconverged requires --relaybp-fallback")
        if args.mip_retry_unsolved:
            raise SystemExit(
                "--relaybp-fallback-retry-unconverged cannot be combined with --mip-retry-unsolved"
            )
        if args.min_failures > 0 or args.min_converged_failures > 0:
            raise SystemExit(
                "--relaybp-fallback-retry-unconverged cannot be combined with adaptive extension"
            )
    if args.bposd_retry_unconverged:
        if args.stage != "decode":
            raise SystemExit("--bposd-retry-unconverged requires --stage decode")
        if args.decoder != "relaybp":
            raise SystemExit("--bposd-retry-unconverged requires --decoder relaybp")
        if args.mip_retry_unsolved or args.relaybp_fallback_retry_unconverged:
            raise SystemExit(
                "--bposd-retry-unconverged cannot be combined with other retry-only modes"
            )
        if args.min_failures > 0 or args.min_converged_failures > 0:
            raise SystemExit("--bposd-retry-unconverged cannot be combined with adaptive extension")
        if args.bposd_threads <= 0:
            raise SystemExit("--bposd-threads must be positive")
        if args.bposd_workers <= 0:
            raise SystemExit("--bposd-workers must be positive")
    if args.min_failures < 0:
        raise SystemExit("--min-failures must be nonnegative")
    if args.min_converged_failures < 0:
        raise SystemExit("--min-converged-failures must be nonnegative")
    if args.min_failures > 0 and args.min_converged_failures > 0:
        raise SystemExit(
            "Use only one adaptive target: --min-failures or --min-converged-failures."
        )
    adaptive_target = int(args.min_failures or args.min_converged_failures)
    adaptive_metric = "failures" if args.min_failures > 0 else "c_failures"
    if adaptive_target > 0 and args.stage != "both":
        adaptive_flag = "--min-failures" if args.min_failures > 0 else "--min-converged-failures"
        raise SystemExit(
            f"{adaptive_flag} requires --stage both so the script can "
            "generate additional samples before decoding them"
        )
    specs = parse_codes(args.codes)
    spec_by_name = {spec.name: spec for spec in specs}
    bases = parse_bases(args.basis)
    modes = parse_decoding_modes(args.decoding_mode)
    p_list = parse_p_list(args.p_list)
    conditions = make_conditions(specs, bases, modes, p_list, args.cycles)
    relay_config = RelayBPConfig(
        args.gamma0,
        args.pre_iter,
        args.num_sets,
        args.set_max_iter,
        args.gamma_dist_min,
        args.gamma_dist_max,
        args.stop_nconv,
    )
    bposd_retry_config = BpOsdRetryConfig(
        enabled=bool(args.bposd_retry_unconverged),
        max_iter=int(args.bposd_max_iter),
        bp_method=str(args.bposd_bp_method),
        ms_scaling_factor=float(args.bposd_ms_scaling_factor),
        schedule=str(args.bposd_schedule),
        osd_method=str(args.bposd_osd_method),
        osd_order=int(args.bposd_osd_order),
        threads=int(args.bposd_threads),
    )
    relaybp_fallback_config = RelayBPFallbackConfig(
        enabled=bool(args.relaybp_fallback),
        gamma0=float(args.relaybp_fallback_gamma0),
        pre_iter=int(args.relaybp_fallback_pre_iter),
        num_sets=int(args.relaybp_fallback_num_sets),
        set_max_iter=int(args.relaybp_fallback_set_max_iter),
        gamma_dist_min=float(args.relaybp_fallback_gamma_dist_min),
        gamma_dist_max=float(args.relaybp_fallback_gamma_dist_max),
        stop_nconv=int(args.relaybp_fallback_stop_nconv),
    )
    mip_fallback_config = MIPFallbackConfig(
        enabled=bool(args.mip_fallback),
        time_limit=float(args.mip_time_limit),
        wall_time_limit=float(args.mip_wall_time_limit),
        mip_rel_gap=float(args.mip_rel_gap),
        feasible_only=bool(args.mip_feasible_only),
        count_unsolved_as_failure=not bool(args.mip_unsolved_not_failure),
        workers=int(args.mip_workers),
        threads=int(args.mip_threads),
    )
    decoder: DecoderName = args.decoder
    if mip_fallback_config.enabled and decoder != "relaybp":
        raise SystemExit("--mip-fallback is only supported with --decoder relaybp")
    if relaybp_fallback_config.enabled and decoder != "relaybp":
        raise SystemExit("--relaybp-fallback is only supported with --decoder relaybp")
    decoder_config: RelayBPConfig = relay_config
    decode_summary_stem = "summary"
    if args.output_dir is not None:
        if args.sample_dir is None:
            args.sample_dir = args.output_dir / "sample_results"
        if args.decode_dir is None:
            args.decode_dir = args.output_dir / "decode_results"
    if args.sample_dir is None:
        args.sample_dir = Path("circuit_simulation/results/samples")
    if args.decode_dir is None:
        args.decode_dir = Path("circuit_simulation/results/decode")
    args.sample_dir.mkdir(parents=True, exist_ok=True)
    args.decode_dir.mkdir(parents=True, exist_ok=True)

    print(f"Python executable: {Path(RELAYBP_PYTHON).name}", flush=True)
    print(
        f"stage={args.stage} decoder={decoder} codes={[spec.name for spec in specs]} bases={bases} modes={modes} p_list={p_list} cycles={args.cycles} shots={args.shots} shot_chunk={args.shot_chunk} decode_batch_size={args.decode_batch_size} workers={args.workers} min_failures={args.min_failures} min_converged_failures={args.min_converged_failures} mip_retry_unsolved={args.mip_retry_unsolved} mip_retry_source_hash={args.mip_retry_source_hash} relaybp_fallback_retry_unconverged={args.relaybp_fallback_retry_unconverged} relaybp_fallback_source_hash={args.relaybp_fallback_source_hash} bposd_retry_unconverged={args.bposd_retry_unconverged} bposd_retry_source_hash={args.bposd_retry_source_hash} bposd_workers={args.bposd_workers}",
        flush=True,
    )
    print(f"sample_dir={args.sample_dir}", flush=True)
    print(f"decode_dir={args.decode_dir}", flush=True)
    print(f"relaybp_config={asdict(decoder_config)}", flush=True)
    if mip_fallback_config.enabled:
        print(f"mip_fallback_config={asdict(mip_fallback_config)}", flush=True)
    if relaybp_fallback_config.enabled:
        print(
            f"relaybp_fallback_config={asdict(relaybp_fallback_config)}",
            flush=True,
        )
    if bposd_retry_config.enabled:
        print(f"bposd_retry_config={asdict(bposd_retry_config)}", flush=True)
        print(f"bposd_workers={args.bposd_workers}", flush=True)
    if args.mip_retry_unsolved:
        print(
            f"mip_retry_unsolved=True source_hash={args.mip_retry_source_hash or 'auto'}",
            flush=True,
        )
    if args.relaybp_fallback_retry_unconverged:
        print(
            "relaybp_fallback_retry_unconverged=True "
            f"source_hash={args.relaybp_fallback_source_hash or 'auto'}",
            flush=True,
        )
    if args.bposd_retry_unconverged:
        print(
            f"bposd_retry_unconverged=True source_hash={args.bposd_retry_source_hash or 'auto'}",
            flush=True,
        )

    sample_summary_records: list[dict[str, object]] = (
        load_existing_summary_records(args.sample_dir) if args.stage in ("generate", "both") else []
    )
    decode_summary_records: list[dict[str, object]] = (
        rebuild_decode_summary_from_jsonl(args.sample_dir, args.decode_dir, decoder)
        if args.stage in ("decode", "both")
        else []
    )

    for idx, condition in enumerate(conditions):
        print(f"\n[{idx + 1}/{len(conditions)}] {condition.label}", flush=True)
        spec = spec_by_name[condition.code_name]
        condition_seed = int((args.seed + 1000003 * idx) % (2**64 - 1))
        if args.bposd_retry_unconverged:
            meta = load_artifact_meta(args.sample_dir, condition)
            record = run_bposd_retry_condition(
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
            decode_summary_records = rebuild_decode_summary_from_jsonl(
                args.sample_dir, args.decode_dir, decoder
            )
            write_summary(args.decode_dir, decode_summary_records, decode_summary_stem)
            continue
        if args.relaybp_fallback_retry_unconverged:
            meta = load_artifact_meta(args.sample_dir, condition)
            record = run_relaybp_fallback_retry_condition(
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
            decode_summary_records = rebuild_decode_summary_from_jsonl(
                args.sample_dir, args.decode_dir, decoder
            )
            write_summary(args.decode_dir, decode_summary_records, decode_summary_stem)
            continue
        if args.mip_retry_unsolved:
            meta = load_artifact_meta(args.sample_dir, condition)
            record = run_mip_retry_unsolved_condition(
                condition,
                args.sample_dir,
                args.decode_dir,
                args.shots,
                args.shot_chunk,
                relay_config,
                meta,
                force=bool(args.force_decode),
                progress_seconds=args.progress_seconds,
                mip_fallback_config=mip_fallback_config,
                source_hash=str(args.mip_retry_source_hash).strip() or None,
            )
            record["shots_requested"] = record.get("shots_decoded")
            decode_summary_records = rebuild_decode_summary_from_jsonl(
                args.sample_dir, args.decode_dir, decoder
            )
            write_summary(args.decode_dir, decode_summary_records, decode_summary_stem)
            continue
        if args.stage in ("generate", "both"):
            meta = build_and_save_artifacts(
                spec, condition, args.sample_dir, force=args.force_artifacts
            )
        else:
            meta = load_artifact_meta(args.sample_dir, condition)

        requested_shots = int(args.shots)
        round_index = 0
        while True:
            round_index += 1
            if adaptive_target > 0:
                print(
                    f"  adaptive round {round_index}: target_total_shots={requested_shots} "
                    f"target_{adaptive_metric}>={adaptive_target}",
                    flush=True,
                )
            loop_force_shots = bool(args.force_shots and round_index == 1)
            loop_force_decode = bool(args.force_decode and round_index == 1)
            if args.stage in ("generate", "both"):
                run_generate_condition(
                    condition,
                    args.sample_dir,
                    requested_shots,
                    args.shot_chunk,
                    args.workers,
                    condition_seed,
                    meta,
                    force=loop_force_shots,
                    progress_seconds=args.progress_seconds,
                )
                upsert_summary_record(
                    sample_summary_records,
                    condition_generation_summary(
                        condition,
                        args.sample_dir,
                        requested_shots,
                        args.shot_chunk,
                        meta,
                    ),
                )
                write_summary(args.sample_dir, sample_summary_records)
            if args.stage in ("decode", "both"):
                record = run_decode_condition(
                    condition,
                    args.sample_dir,
                    args.decode_dir,
                    requested_shots,
                    args.shot_chunk,
                    args.workers,
                    decoder,
                    decoder_config,
                    meta,
                    force=loop_force_decode,
                    progress_seconds=args.progress_seconds,
                    decode_batch_size=args.decode_batch_size,
                    mip_fallback_config=mip_fallback_config,
                    relaybp_fallback_config=relaybp_fallback_config,
                )
                record["shots_requested"] = record.get("shots_decoded")
                decode_summary_records = rebuild_decode_summary_from_jsonl(
                    args.sample_dir, args.decode_dir, decoder
                )
                write_summary(args.decode_dir, decode_summary_records, decode_summary_stem)
                if adaptive_target > 0:
                    metric_value = int(record.get(adaptive_metric) or 0)
                    converged = int(record.get("converged") or 0)
                    decoded = int(record.get("shots_decoded") or 0)
                    print(
                        f"  adaptive status: decoded={decoded} converged={converged} "
                        f"{adaptive_metric}={metric_value}/{adaptive_target}",
                        flush=True,
                    )
                    if metric_value >= adaptive_target:
                        print(
                            f"  adaptive stop: reached {adaptive_metric}={metric_value} "
                            f"for {condition.label}",
                            flush=True,
                        )
                        break
                    requested_shots += int(args.shots)
                    print(
                        f"  adaptive continue: {adaptive_metric}={metric_value} < "
                        f"{adaptive_target}; extending total shots "
                        f"to {requested_shots}",
                        flush=True,
                    )
                    continue
            break

    if args.stage in ("generate", "both"):
        write_summary(args.sample_dir, sample_summary_records)
        print("sample_summary_json=" + str(args.sample_dir / "summary.json"), flush=True)
        print("sample_summary_txt=" + str(args.sample_dir / "summary.txt"), flush=True)
    if args.stage in ("decode", "both"):
        decode_summary_records = rebuild_decode_summary_from_jsonl(
            args.sample_dir, args.decode_dir, decoder
        )
        write_summary(args.decode_dir, decode_summary_records, decode_summary_stem)
        print(
            "decode_summary_json=" + str(args.decode_dir / f"{decode_summary_stem}.json"),
            flush=True,
        )
        print(
            "decode_summary_txt=" + str(args.decode_dir / f"{decode_summary_stem}.txt"),
            flush=True,
        )


if __name__ == "__main__":
    main()
