#!/usr/bin/env python3
"""Logical-memory sampling, decoding, and retries for the three code families.

Detector and observable samples are stored in packed NumPy arrays. Decoding
reads these samples and records block failures and logical-observable errors.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy import sparse

from circuit_simulation.decoders import (
    _init_mip_worker,
    _mip_decode_worker,
    build_bposd_retry_decoder,
    build_relaybp_runner,
    decode_bposd_retry_row,
    init_bposd_retry_worker,
    load_mip_fallback_details,
    load_relaybp_unconverged_details,
    make_bposd_passthrough_record,
    relaybp_bposd_retry_config_hash,
    relaybp_fallback_retry_config_hash,
    relaybp_mip_config_hash,
    relaybp_mip_retry_config_hash,
    run_bposd_retry_worker_task,
    save_bposd_retry_details,
    save_mip_fallback_details,
    save_relaybp_unconverged_details,
)
from circuit_simulation.logical_error_statistics import (
    aggregate_decode_records,
    summary_record_key,
)
from circuit_simulation.logical_error_statistics import (
    condition_generation_summary as condition_generation_summary,
)
from circuit_simulation.logical_error_statistics import (
    load_existing_summary_records as load_existing_summary_records,
)
from circuit_simulation.logical_error_statistics import (
    upsert_summary_record as upsert_summary_record,
)
from circuit_simulation.logical_error_statistics import write_summary as write_summary
from circuit_simulation.shot_data import (
    artifact_paths,
    bposd_retry_details_path,
    decode_jsonl_path,
    load_check_matrices,
    load_shots_manifest,
    mip_fallback_details_path,
    relaybp_unconverged_details_path,
    save_check_matrices,
    shot_chunk_path,
    shot_file_info,
    shot_matches_task,
    shots_manifest_path,
    stable_hash,
    write_json_atomic,
    write_shots_manifest,
)
from circuit_simulation.shot_data import file_fingerprint as file_fingerprint
from circuit_simulation.simulation_parameters import (
    DECODE_SCHEMA_VERSION,
    SCRIPT_VERSION,
    BpOsdRetryConfig,
    Condition,
    DecoderName,
    DecodeTask,
    GenerateTask,
    MIPFallbackConfig,
    RelayBPConfig,
    RelayBPFallbackConfig,
    format_p,
)
from code_construction.affine_codes import build_code
from code_construction.cornucopia_codes import SPECS
from syndrome_extraction.memory_circuit import (
    build_memory_circuit,
    logical_rows_for_basis,
)
from syndrome_extraction.syndrome_schedule import StimNoiseConfig

Basis = Literal["Z", "X"]
DecodingMode = Literal["xyz", "xz"]
RELAYBP_PYTHON = sys.executable


_GEN_DEM = None
_DECODER_RUNNER = None


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("stim", "relay_bp", "ldpc", "bposd", "scipy", "numpy"):
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if missing:
        raise ImportError(
            "Missing RelayBP runtime dependencies in the current Python "
            "environment. Missing: " + "; ".join(missing)
        )


def parse_p_list(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise SystemExit("--p-list must contain at least one probability")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in out):
        raise SystemExit("--p-list probabilities must be finite and lie in [0, 1]")
    labels = [format_p(p) for p in out]
    if len(labels) != len(set(labels)):
        raise SystemExit(
            "--p-list contains probabilities with the same output label. "
            "Use distinct values that remain distinguishable at six significant digits."
        )
    return out


def parse_codes(value: str | None):
    if value is None or value.strip().lower() in ("", "all"):
        return list(SPECS)
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    specs = [spec for spec in SPECS if spec.name in wanted]
    missing = sorted(wanted - {spec.name for spec in specs})
    if missing:
        raise SystemExit(
            f"Unknown code names: {missing}. Available: {[spec.name for spec in SPECS]}"
        )
    return specs


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
    specs,
    bases: Sequence[Basis],
    modes: Sequence[DecodingMode],
    p_list: Sequence[float],
    cycles: int,
) -> list[Condition]:
    return [
        Condition(
            spec.name,
            spec.parameter_label,
            int(spec.p),
            int(spec.l),
            int(spec.j),
            int(spec.expected_d),
            basis,
            mode,
            float(p_noise),
            int(cycles),
        )
        for spec in specs
        for basis in bases
        for mode in modes
        for p_noise in p_list
    ]


def filter_detectors_for_xz(circuit, basis: Basis):
    import stim  # type: ignore

    keep_coord0 = {0.0, 2.0} if basis == "Z" else {1.0, 3.0}
    filtered = stim.Circuit()
    kept = 0
    removed = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            coords = inst.gate_args_copy()
            if coords and float(coords[0]) not in keep_coord0:
                removed += 1
                continue
            kept += 1
        filtered.append(inst)
    return filtered, kept, removed


def build_and_save_artifacts(
    spec, condition: Condition, output_dir: Path, *, force: bool = False
) -> dict:
    from relay_bp.stim import CheckMatrices  # type: ignore

    paths = artifact_paths(output_dir, condition)
    if not force and all(path.exists() for path in paths.values()):
        return load_artifact_meta(output_dir, condition)
    code = build_code(spec.to_code_spec())
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
        circuit, _detectors_kept, detectors_removed = filter_detectors_for_xz(
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
    save_check_matrices(paths["matrices"], check_matrix, observables_matrix, error_priors)
    if circuit.num_detectors <= 0 or circuit.num_observables <= 0 or check_matrix.shape[1] <= 0:
        raise ValueError(
            f"{condition.label}: invalid DEM dimensions detectors={circuit.num_detectors} observables={circuit.num_observables} columns={check_matrix.shape[1]}"
        )
    meta = {
        "script_version": SCRIPT_VERSION,
        "label": condition.label,
        "code_name": condition.code_name,
        "parameter_label": condition.parameter_label,
        "P": condition.p_code,
        "L": condition.l,
        "J": condition.j,
        "expected_d": condition.expected_d,
        "basis": condition.basis,
        "decoding_mode": condition.decoding_mode,
        "p": condition.p_noise,
        "cycles": condition.cycles,
        "noise": noise.to_dict(),
        "n": int(code.n),
        "k": int(code.num_logicals()),
        "num_detectors": int(circuit.num_detectors),
        "num_observables": int(circuit.num_observables),
        "num_dem_errors": int(dem.num_errors),
        "num_error_columns": int(check_matrix.shape[1]),
        "detectors_removed_for_xz": int(detectors_removed),
        "stim_path": str(paths["stim"]),
        "dem_path": str(paths["dem"]),
        "matrices_path": str(paths["matrices"]),
    }
    write_json_atomic(paths["meta"], meta)
    return meta


def load_artifact_meta(output_dir: Path, condition: Condition) -> dict:
    paths = artifact_paths(output_dir, condition)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts for {condition.label}. Run --stage generate first. Missing: {missing}"
        )
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    expected = {
        "code_name": condition.code_name,
        "parameter_label": condition.parameter_label,
        "P": condition.p_code,
        "L": condition.l,
        "J": condition.j,
        "expected_d": condition.expected_d,
        "basis": condition.basis,
        "decoding_mode": condition.decoding_mode,
        "p": condition.p_noise,
        "cycles": condition.cycles,
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"{condition.label}: saved artifacts describe a different physical condition. "
            "Choose another sample directory or explicitly regenerate with --force-artifacts."
        )
    return meta


def expected_generate_tasks(
    output_dir: Path,
    condition: Condition,
    shots: int,
    shot_chunk: int,
    seed: int,
    meta: dict,
) -> list[GenerateTask]:
    tasks: list[GenerateTask] = []
    for chunk_index in range(math.ceil(shots / shot_chunk)):
        start = chunk_index * shot_chunk
        count = min(shot_chunk, shots - start)
        chunk_seed = int((seed + 0x9E3779B97F4A7C15 * (chunk_index + 1)) % (2**64 - 1))
        tasks.append(
            GenerateTask(
                chunk_index,
                start,
                count,
                chunk_seed,
                str(shot_chunk_path(output_dir, condition, chunk_index)),
                int(meta["num_detectors"]),
                int(meta["num_observables"]),
            )
        )
    return tasks


def shot_file_ok(path: Path, expected_shots: int, num_detectors: int, num_observables: int) -> bool:
    return shot_file_info(path, expected_shots, num_detectors, num_observables) is not None


def init_generate_worker(dem_path: str) -> None:
    global _GEN_DEM
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import stim  # type: ignore

    _GEN_DEM = stim.DetectorErrorModel(Path(dem_path).read_text(encoding="utf-8"))


def run_generate_task(task: GenerateTask) -> dict[str, object]:
    if _GEN_DEM is None:
        raise RuntimeError("generate worker was not initialized")
    t0 = time.monotonic()
    sampler = _GEN_DEM.compile_sampler(seed=int(task.seed))
    detectors, observables, _errors = sampler.sample(
        int(task.num_shots), bit_packed=True, return_errors=False
    )
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as fout:
        np.savez_compressed(
            fout,
            detector_data=detectors.astype(np.uint8),
            observable_data=observables.astype(np.uint8),
            detector_shape=np.asarray([task.num_shots, task.num_detectors], dtype=np.int64),
            observable_shape=np.asarray([task.num_shots, task.num_observables], dtype=np.int64),
            num_shots=np.asarray(task.num_shots, dtype=np.int64),
            start_shot=np.asarray(task.start_shot, dtype=np.int64),
            chunk_index=np.asarray(task.chunk_index, dtype=np.int64),
            seed=np.asarray(task.seed, dtype=np.uint64),
            bitorder=np.asarray("little"),
        )
    tmp.replace(output_path)
    return {
        "chunk_index": task.chunk_index,
        "start_shot": task.start_shot,
        "num_shots": task.num_shots,
        "seed": task.seed,
        "shot_path": str(output_path),
        "seconds": time.monotonic() - t0,
    }


def run_generate_condition(
    condition: Condition,
    output_dir: Path,
    shots: int,
    shot_chunk: int,
    workers: int,
    seed: int,
    meta: dict,
    *,
    force: bool,
    progress_seconds: float,
) -> None:
    tasks = expected_generate_tasks(output_dir, condition, shots, shot_chunk, seed, meta)
    if not force and any(Path(task.output_path).exists() for task in tasks):
        try:
            load_shots_manifest(output_dir, condition)
        except (OSError, ValueError, KeyError, TypeError):
            # Regenerate samples whose source circuit cannot be verified.
            force = True
    pending = [task for task in tasks if force or not shot_matches_task(task)]
    if not pending:
        write_shots_manifest(output_dir, condition, tasks, meta, shots, shot_chunk, seed)
        print(
            f"  generate {condition.label}: all {len(tasks)} chunks already present",
            flush=True,
        )
        return
    print(
        f"  generate {condition.label}: {len(pending)}/{len(tasks)} chunks pending",
        flush=True,
    )
    dem_path = artifact_paths(output_dir, condition)["dem"]
    completed = 0
    completed_shots = 0
    t0 = time.monotonic()
    last_progress = t0
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers, initializer=init_generate_worker, initargs=(str(dem_path),)
    ) as pool:
        for result in pool.imap_unordered(run_generate_task, pending, chunksize=1):
            completed += 1
            completed_shots += int(result["num_shots"])
            now = time.monotonic()
            if now - last_progress >= progress_seconds or completed == len(pending):
                print(
                    f"    generated chunks={completed}/{len(pending)} shots={completed_shots} rate={completed_shots / max(now - t0, 1e-9):.2f} shots/s",
                    flush=True,
                )
                last_progress = now
    write_shots_manifest(output_dir, condition, tasks, meta, shots, shot_chunk, seed)


def unpack_bits(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    if cols == 0:
        return np.zeros((rows, 0), dtype=np.uint8)
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, bitorder="little")[
        :, :cols
    ].astype(np.uint8, copy=False)


def run_deferred_mip_fallback(
    detectors: np.ndarray,
    predicted: np.ndarray,
    unconverged_indices: np.ndarray,
    check_matrix: sparse.csr_matrix,
    observables_matrix: sparse.csr_matrix,
    error_priors: np.ndarray,
    config: MIPFallbackConfig,
    *,
    progress_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[int, dict[str, object]]]:
    """Decode all RelayBP non-converged shots with a deferred MIP pool."""
    num_unconverged = int(len(unconverged_indices))
    actual_unsolved_mask = np.zeros(predicted.shape[0], dtype=bool)
    unsolved_mask = np.zeros(predicted.shape[0], dtype=bool)
    if num_unconverged <= 0:
        return (
            predicted,
            actual_unsolved_mask,
            unsolved_mask,
            {
                "called": 0,
                "solved": 0,
                "unsolved": 0,
                "seconds": 0.0,
                "workers": 0,
                "hard_timeouts": 0,
                "hard_time_limit_exceeded": False,
                "wall_time_limit_exceeded": False,
            },
            {},
        )

    workers = min(max(int(config.workers), 1), num_unconverged)
    hard_time_limit = float(config.time_limit)
    wall_time_limit = float(config.wall_time_limit)
    hard_limit_text = "none" if hard_time_limit <= 0 else f"{hard_time_limit:g}s/shot"
    wall_limit_text = "none" if wall_time_limit <= 0 else f"{wall_time_limit:g}s/condition"
    print(
        f"    deferred MIP fallback: shots={num_unconverged} workers={workers} "
        f"hard_time_limit={hard_limit_text} wall_time_limit={wall_limit_text}",
        flush=True,
    )
    t0 = time.monotonic()
    last_progress = t0
    solved = 0
    unsolved = 0
    hard_timeouts = 0
    mip_seconds = 0.0
    wall_time_exceeded = False
    mip_info_by_index: dict[int, dict[str, object]] = {}
    completed_indices: set[int] = set()

    def consume_result(
        shot_index: int, obs_prediction: np.ndarray | None, mip_info: dict[str, object]
    ) -> None:
        nonlocal solved
        nonlocal unsolved
        nonlocal hard_timeouts
        nonlocal mip_seconds
        shot_index = int(shot_index)
        if shot_index in completed_indices:
            return
        completed_indices.add(shot_index)
        mip_seconds += float(mip_info.get("mip_seconds", 0.0))
        mip_info_by_index[shot_index] = {
            "mip_status": str(mip_info.get("mip_status", "")),
            "mip_success": bool(mip_info.get("mip_success", False)),
            "mip_message": str(mip_info.get("mip_message", "")),
            "mip_feasible_only": bool(mip_info.get("mip_feasible_only", config.feasible_only)),
            "mip_num_solutions": int(mip_info.get("mip_num_solutions", 0) or 0),
            "mip_fun": mip_info.get("mip_fun"),
            "mip_seconds": float(mip_info.get("mip_seconds", 0.0)),
        }
        if str(mip_info.get("mip_status", "")) == "hard_time_limit":
            hard_timeouts += 1
        if obs_prediction is None:
            unsolved += 1
            actual_unsolved_mask[shot_index] = True
            unsolved_mask[shot_index] = bool(config.count_unsolved_as_failure)
        else:
            solved += 1
            predicted[shot_index] = np.asarray(obs_prediction, dtype=np.uint8)

    def timeout_info(reason: str, seconds: float, status: str) -> dict[str, object]:
        return {
            "mip_status": status,
            "mip_success": False,
            "mip_message": reason,
            "mip_feasible_only": bool(config.feasible_only),
            "mip_num_solutions": 0,
            "mip_fun": None,
            "mip_seconds": float(max(seconds, 0.0)),
        }

    def print_progress(now: float) -> None:
        done = solved + unsolved
        rate = done / max(now - t0, 1e-9)
        print(
            f"    MIP fallback done={done}/{num_unconverged} "
            f"solved={solved} unsolved={unsolved} "
            f"hard_timeouts={hard_timeouts} rate={rate:.2f} shots/s",
            flush=True,
        )

    def make_pool():
        ctx = mp.get_context("spawn")
        return ctx.Pool(
            processes=workers,
            initializer=_init_mip_worker,
            initargs=(check_matrix, observables_matrix, error_priors, asdict(config)),
        )

    pending = [
        (int(index), detectors[int(index)].astype(np.uint8, copy=True))
        for index in np.asarray(unconverged_indices, dtype=np.int64)
    ]
    active: dict[int, tuple[tuple[int, np.ndarray], object, float]] = {}
    pool = make_pool()
    try:
        while pending or active:
            while pending and len(active) < workers:
                task = pending.pop(0)
                shot_index = int(task[0])
                active[shot_index] = (
                    task,
                    pool.apply_async(_mip_decode_worker, (task,)),
                    time.monotonic(),
                )

            made_progress = False
            for shot_index, (task, result, _started_at) in list(active.items()):
                if result.ready():
                    try:
                        consume_result(*result.get(timeout=0))
                    except BaseException as exc:
                        consume_result(
                            int(shot_index),
                            None,
                            timeout_info(f"MIP worker exception: {exc!r}", 0.0, "exception"),
                        )
                    del active[shot_index]
                    made_progress = True

            now = time.monotonic()
            elapsed_total = now - t0
            if wall_time_limit > 0 and elapsed_total >= wall_time_limit:
                wall_time_exceeded = True
                print(
                    f"    MIP fallback wall_time_limit reached: "
                    f"elapsed={elapsed_total:.1f}s limit={wall_time_limit:.1f}s",
                    flush=True,
                )
                pool.terminate()
                pool.join()
                for shot_index, (task, _result, started_at) in list(active.items()):
                    consume_result(
                        int(shot_index),
                        None,
                        timeout_info(
                            "MIP fallback wall_time_limit exceeded",
                            now - started_at,
                            "wall_time_limit",
                        ),
                    )
                active.clear()
                for shot_index, _syndrome in pending:
                    consume_result(
                        int(shot_index),
                        None,
                        timeout_info(
                            "MIP fallback wall_time_limit exceeded before launch",
                            0.0,
                            "wall_time_limit",
                        ),
                    )
                pending.clear()
                break

            timed_out = []
            if hard_time_limit > 0:
                timed_out = [
                    shot_index
                    for shot_index, (_task, _result, started_at) in active.items()
                    if now - started_at >= hard_time_limit
                ]
            if timed_out:
                timed_out_set = set(int(index) for index in timed_out)
                print(
                    f"    MIP hard timeout: shots={sorted(timed_out_set)[:5]}"
                    f"{'...' if len(timed_out_set) > 5 else ''} "
                    f"limit={hard_time_limit:.1f}s; restarting MIP workers",
                    flush=True,
                )
                pool.terminate()
                pool.join()
                requeue: list[tuple[int, np.ndarray]] = []
                for shot_index, (task, _result, started_at) in list(active.items()):
                    if int(shot_index) in timed_out_set:
                        consume_result(
                            int(shot_index),
                            None,
                            timeout_info(
                                "single-shot MIP hard time limit exceeded",
                                now - started_at,
                                "hard_time_limit",
                            ),
                        )
                    else:
                        requeue.append(task)
                active.clear()
                pending = requeue + pending
                if pending:
                    pool = make_pool()
                else:
                    pool = None
                last_progress = now
                made_progress = True

            if now - last_progress >= progress_seconds:
                print_progress(now)
                last_progress = now

            if not made_progress and (pending or active):
                time.sleep(0.1)

            if pool is None and pending:
                pool = make_pool()

        if pool is not None:
            pool.close()
            pool.join()
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    return (
        predicted,
        actual_unsolved_mask,
        unsolved_mask,
        {
            "called": int(num_unconverged),
            "solved": int(solved),
            "unsolved": int(unsolved),
            "seconds": float(mip_seconds),
            "workers": int(workers),
            "hard_timeouts": int(hard_timeouts),
            "hard_time_limit_exceeded": bool(hard_timeouts > 0),
            "wall_time_limit_exceeded": bool(wall_time_exceeded),
        },
        mip_info_by_index,
    )


def run_relaybp_batch_with_deferred_mip(
    tasks: Sequence["DecodeTask"],
    detectors: np.ndarray,
    observables: np.ndarray,
    chunk_records: list[dict[str, object]],
    *,
    runner: object,
    relaybp_config: RelayBPConfig,
    check_matrix: sparse.csr_matrix,
    observables_matrix: sparse.csr_matrix,
    error_priors: np.ndarray,
    mip_fallback_config: MIPFallbackConfig | None,
    relaybp_fallback_runner: object | None = None,
    relaybp_fallback_config: RelayBPFallbackConfig | None = None,
    decode_batch_size: int,
    progress_seconds: float,
    load_seconds: float,
    t0: float,
    mip_fallback_details_path: Path | None = None,
    mip_fallback_detail_metadata: dict[str, object] | None = None,
    relaybp_unconverged_details_path: Path | None = None,
    relaybp_unconverged_detail_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    num_shots = int(observables.shape[0])
    batch_size = int(decode_batch_size)
    if batch_size <= 0 or batch_size > num_shots:
        batch_size = num_shots
    mip_enabled = mip_fallback_config is not None and mip_fallback_config.enabled
    relaybp_fallback_enabled = (
        relaybp_fallback_config is not None and relaybp_fallback_config.enabled
    )

    print(
        f"    decode batch prepared: shots={num_shots} chunks={len(tasks)} "
        f"detectors={detectors.shape[1]} observables={observables.shape[1]} "
        f"decode_batch_size={batch_size} decoder=relaybp mip_mode=deferred "
        f"relaybp_fallback={relaybp_fallback_enabled} "
        f"mip_workers={mip_fallback_config.workers if mip_enabled and mip_fallback_config is not None else 0}",
        flush=True,
    )

    decode_t0 = time.monotonic()
    last_progress = decode_t0
    predicted = np.zeros_like(observables, dtype=np.uint8)
    converged_mask = np.zeros(num_shots, dtype=bool)
    iterations = np.zeros(num_shots, dtype=np.int64)
    relaybp_batch_seconds: dict[int, float] = {}
    relaybp_fallback_attempted_mask = np.zeros(num_shots, dtype=bool)
    relaybp_fallback_converged_mask = np.zeros(num_shots, dtype=bool)
    relaybp_fallback_iterations = np.zeros(num_shots, dtype=np.int64)
    relaybp_fallback_seconds = 0.0
    primary_converged_mask = np.zeros(num_shots, dtype=bool)
    primary_iterations = np.zeros(num_shots, dtype=np.int64)

    decoded = 0
    relaybp_progress_converged = 0
    relaybp_progress_failures = 0
    relaybp_progress_c_failures = 0
    relaybp_progress_observable_mismatches = 0
    for batch_index, start in enumerate(range(0, num_shots, batch_size)):
        stop = min(start + batch_size, num_shots)
        batch_t0 = time.monotonic()
        detailed_results = runner.decode_observables_detailed_batch(
            detectors[start:stop].astype(np.uint8, copy=False),
            parallel=True,
            progress_bar=False,
        )
        predicted[start:stop] = np.vstack(
            [np.asarray(result.observables, dtype=np.uint8) for result in detailed_results]
        )
        converged_mask[start:stop] = np.asarray(
            [bool(result.converged) for result in detailed_results], dtype=bool
        )
        iterations[start:stop] = np.asarray(
            [int(result.iterations) for result in detailed_results], dtype=np.int64
        )
        primary_converged_mask[start:stop] = converged_mask[start:stop]
        primary_iterations[start:stop] = iterations[start:stop]
        batch_converged_mask = converged_mask[start:stop]
        batch_relay_mismatch_mask = predicted[start:stop] != observables[start:stop].astype(
            np.uint8, copy=False
        )
        batch_relay_failure_mask = np.any(batch_relay_mismatch_mask, axis=1)
        relaybp_progress_converged += int(np.sum(batch_converged_mask))
        relaybp_progress_failures += int(np.sum(batch_relay_failure_mask))
        relaybp_progress_c_failures += int(np.sum(batch_relay_failure_mask & batch_converged_mask))
        relaybp_progress_observable_mismatches += int(np.sum(batch_relay_mismatch_mask))
        relaybp_batch_seconds[int(batch_index)] = time.monotonic() - batch_t0
        decoded += stop - start
        now = time.monotonic()
        if now - last_progress >= progress_seconds or decoded == num_shots:
            rate = decoded / max(now - decode_t0, 1e-9)
            print(
                f"    RelayBP decoded shots={decoded}/{num_shots} "
                f"converged={relaybp_progress_converged}/{decoded} "
                f"relaybp_failures={relaybp_progress_failures} "
                f"c_failures={relaybp_progress_c_failures} "
                f"observable_mismatches={relaybp_progress_observable_mismatches} "
                f"rate={rate:.2f} shots/s",
                flush=True,
            )
            last_progress = now

    primary_unconverged_indices = np.flatnonzero(~primary_converged_mask)
    if relaybp_fallback_enabled and len(primary_unconverged_indices):
        if relaybp_fallback_runner is None:
            raise RuntimeError("RelayBP fallback was enabled but not initialized")
        fallback_t0 = time.monotonic()
        fallback_decoded = 0
        for start in range(0, len(primary_unconverged_indices), batch_size):
            stop = min(start + batch_size, len(primary_unconverged_indices))
            indices = primary_unconverged_indices[start:stop]
            fallback_results = relaybp_fallback_runner.decode_observables_detailed_batch(
                detectors[indices].astype(np.uint8, copy=False),
                parallel=True,
                progress_bar=False,
            )
            fallback_predicted = np.vstack(
                [np.asarray(result.observables, dtype=np.uint8) for result in fallback_results]
            )
            fallback_converged = np.asarray(
                [bool(result.converged) for result in fallback_results], dtype=bool
            )
            fallback_iters = np.asarray(
                [int(result.iterations) for result in fallback_results],
                dtype=np.int64,
            )
            predicted[indices] = fallback_predicted
            converged_mask[indices] = fallback_converged
            iterations[indices] += fallback_iters
            relaybp_fallback_attempted_mask[indices] = True
            relaybp_fallback_converged_mask[indices] = fallback_converged
            relaybp_fallback_iterations[indices] = fallback_iters
            fallback_decoded += int(len(indices))
            now = time.monotonic()
            print(
                f"    RelayBP fallback decoded shots={fallback_decoded}/"
                f"{len(primary_unconverged_indices)} "
                f"converged={int(np.sum(relaybp_fallback_converged_mask))}/"
                f"{fallback_decoded}",
                flush=True,
            )
        relaybp_fallback_seconds = time.monotonic() - fallback_t0

    unconverged_indices = np.flatnonzero(~converged_mask)
    if mip_enabled:
        if mip_fallback_config is None:
            raise RuntimeError("MIP fallback was enabled but no config was provided")
        (
            predicted,
            mip_actual_unsolved_mask,
            mip_unsolved_mask,
            mip_stats,
            mip_info_by_index,
        ) = run_deferred_mip_fallback(
            detectors,
            predicted,
            unconverged_indices,
            check_matrix,
            observables_matrix,
            error_priors,
            mip_fallback_config,
            progress_seconds=progress_seconds,
        )
    else:
        mip_actual_unsolved_mask = np.zeros(num_shots, dtype=bool)
        mip_unsolved_mask = np.zeros(num_shots, dtype=bool)
        mip_stats = {
            "called": 0,
            "solved": 0,
            "unsolved": 0,
            "seconds": 0.0,
            "workers": 0,
        }
        mip_info_by_index = {}

    mismatch_mask = predicted != observables.astype(np.uint8, copy=False)
    mismatch_counts = np.sum(mismatch_mask, axis=1, dtype=np.int64)
    if np.any(mip_unsolved_mask):
        mismatch_counts[mip_unsolved_mask] = observables.shape[1]
    failure_mask = np.any(mismatch_mask, axis=1) | mip_unsolved_mask

    failures = int(np.sum(failure_mask))
    observable_mismatches = int(np.sum(mismatch_counts))
    converged = int(np.sum(converged_mask))
    unconverged = int(num_shots - converged)
    iterations_total = int(np.sum(iterations))
    max_iterations = int(np.max(iterations)) if len(iterations) else 0
    u_mask = ~converged_mask
    c_failures = int(np.sum(failure_mask & converged_mask))
    c_observable_mismatches = int(np.sum(mismatch_counts[converged_mask]))
    c_iterations_total = int(np.sum(iterations[converged_mask]))
    c_max_iterations = int(np.max(iterations[converged_mask])) if converged else 0
    u_failures = int(np.sum(failure_mask & u_mask))
    u_observable_mismatches = int(np.sum(mismatch_counts[u_mask]))
    u_iterations_total = int(np.sum(iterations[u_mask]))
    u_max_iterations = int(np.max(iterations[u_mask])) if unconverged else 0
    relaybp_fallback_called = int(np.sum(relaybp_fallback_attempted_mask))
    relaybp_fallback_converged = int(np.sum(relaybp_fallback_converged_mask))
    relaybp_fallback_unconverged = relaybp_fallback_called - relaybp_fallback_converged
    relaybp_fallback_iterations_total = int(np.sum(relaybp_fallback_iterations))
    relaybp_fallback_max_iterations = (
        int(np.max(relaybp_fallback_iterations[relaybp_fallback_attempted_mask]))
        if relaybp_fallback_called
        else 0
    )
    relaybp_fallback_success_mask = relaybp_fallback_attempted_mask & converged_mask
    relaybp_fallback_failures = int(np.sum(failure_mask & relaybp_fallback_success_mask))
    relaybp_fallback_observable_mismatches = int(
        np.sum(mismatch_counts[relaybp_fallback_success_mask])
    )
    primary_converged = int(np.sum(primary_converged_mask))
    primary_unconverged = int(num_shots - primary_converged)
    primary_iterations_total = int(np.sum(primary_iterations))
    primary_max_iterations = int(np.max(primary_iterations)) if len(primary_iterations) else 0
    mip_fallback_called = int(mip_stats["called"])
    mip_fallback_solved = int(mip_stats["solved"])
    mip_fallback_unsolved = int(mip_stats["unsolved"])
    mip_fallback_failures = int(u_failures) if mip_enabled else 0
    mip_fallback_observable_mismatches = int(u_observable_mismatches) if mip_enabled else 0
    mip_fallback_seconds = float(mip_stats["seconds"])
    saved_mip_fallback_details_path: str | None = None
    saved_relaybp_unconverged_details_path: str | None = None
    if relaybp_unconverged_details_path is not None and len(primary_unconverged_indices):
        metadata = dict(relaybp_unconverged_detail_metadata or {})
        metadata.update(
            {
                "detail_type": "relaybp_primary_unconverged",
                "num_primary_unconverged": int(len(primary_unconverged_indices)),
            }
        )
        save_relaybp_unconverged_details(
            relaybp_unconverged_details_path,
            metadata=metadata,
            relaybp_config=relaybp_config,
            shot_indices=primary_unconverged_indices.astype(np.int64),
            detectors=detectors[primary_unconverged_indices],
            observables=observables[primary_unconverged_indices],
            old_converged_mask=converged_mask[primary_unconverged_indices],
            old_failure_mask=failure_mask[primary_unconverged_indices],
            old_mismatch_counts=mismatch_counts[primary_unconverged_indices],
            old_iterations=iterations[primary_unconverged_indices],
        )
        saved_relaybp_unconverged_details_path = str(relaybp_unconverged_details_path)
    if mip_fallback_details_path is not None and np.any(mip_actual_unsolved_mask):
        unresolved_indices = np.flatnonzero(mip_actual_unsolved_mask).astype(np.int64)
        metadata = dict(mip_fallback_detail_metadata or {})
        metadata.update(
            {
                "detail_type": "mip_fallback_unsolved",
                "num_unconverged": int(len(unconverged_indices)),
                "num_unsolved": int(len(unresolved_indices)),
                "mip_info_by_index": {
                    str(index): mip_info_by_index.get(int(index), {})
                    for index in unresolved_indices
                },
            }
        )
        save_mip_fallback_details(
            mip_fallback_details_path,
            metadata=metadata,
            config=mip_fallback_config,
            shot_indices=unresolved_indices,
            detectors=detectors[unresolved_indices],
            observables=observables[unresolved_indices],
            old_failure_mask=failure_mask[unresolved_indices],
            old_mismatch_counts=mismatch_counts[unresolved_indices],
        )
        saved_mip_fallback_details_path = str(mip_fallback_details_path)

    batch_records: list[dict[str, object]] = []
    for batch_index, start in enumerate(range(0, num_shots, batch_size)):
        stop = min(start + batch_size, num_shots)
        batch_mask = np.zeros(num_shots, dtype=bool)
        batch_mask[start:stop] = True
        batch_converged_mask = batch_mask & converged_mask
        batch_u_mask = batch_mask & u_mask
        batch_failures = int(np.sum(failure_mask[batch_mask]))
        batch_observable_mismatches = int(np.sum(mismatch_counts[batch_mask]))
        batch_converged = int(np.sum(batch_converged_mask))
        batch_unconverged = int(np.sum(batch_u_mask))
        batch_iterations_total = int(np.sum(iterations[batch_mask]))
        batch_max_iterations = int(np.max(iterations[batch_mask])) if stop > start else 0
        batch_c_iterations_total = int(np.sum(iterations[batch_converged_mask]))
        batch_u_iterations_total = int(np.sum(iterations[batch_u_mask]))
        batch_c_max_iterations = (
            int(np.max(iterations[batch_converged_mask])) if batch_converged else 0
        )
        batch_u_max_iterations = int(np.max(iterations[batch_u_mask])) if batch_unconverged else 0
        batch_relaybp_fallback_called = int(np.sum(relaybp_fallback_attempted_mask[batch_mask]))
        batch_relaybp_fallback_converged = int(np.sum(relaybp_fallback_converged_mask[batch_mask]))
        batch_relaybp_fallback_iterations_total = int(
            np.sum(relaybp_fallback_iterations[batch_mask])
        )
        batch_relaybp_fallback_max_iterations = (
            int(np.max(relaybp_fallback_iterations[batch_mask]))
            if batch_relaybp_fallback_called
            else 0
        )
        batch_relaybp_fallback_success_mask = (
            batch_mask & relaybp_fallback_attempted_mask & converged_mask
        )
        batch_relaybp_fallback_failures = int(
            np.sum(failure_mask[batch_relaybp_fallback_success_mask])
        )
        batch_relaybp_fallback_observable_mismatches = int(
            np.sum(mismatch_counts[batch_relaybp_fallback_success_mask])
        )
        batch_primary_mask = batch_mask & primary_converged_mask
        batch_primary_converged = int(np.sum(batch_primary_mask))
        batch_primary_shots = int(stop - start)
        batch_primary_iterations_total = int(np.sum(primary_iterations[batch_mask]))
        batch_primary_max_iterations = (
            int(np.max(primary_iterations[batch_mask])) if stop > start else 0
        )
        batch_mip_called = batch_unconverged if mip_enabled else 0
        batch_mip_unsolved = int(np.sum(mip_actual_unsolved_mask[batch_mask])) if mip_enabled else 0
        batch_mip_solved = batch_mip_called - batch_mip_unsolved
        batch_records.append(
            {
                "batch_index": int(batch_index),
                "start_shot": int(start),
                "stop_shot": int(stop),
                "num_shots": int(stop - start),
                "failures": batch_failures,
                "observable_mismatches": batch_observable_mismatches,
                "converged": batch_converged,
                "unconverged": batch_unconverged,
                "convergence_rate": batch_converged / max(stop - start, 1),
                "iterations_total": batch_iterations_total,
                "avg_iterations": batch_iterations_total / max(stop - start, 1),
                "max_iterations": batch_max_iterations,
                "c_failures": int(np.sum(failure_mask[batch_converged_mask])),
                "c_observable_mismatches": int(np.sum(mismatch_counts[batch_converged_mask])),
                "c_iterations_total": batch_c_iterations_total,
                "c_avg_iterations": (
                    None if batch_converged <= 0 else batch_c_iterations_total / batch_converged
                ),
                "c_max_iterations": batch_c_max_iterations,
                "u_failures": int(np.sum(failure_mask[batch_u_mask])),
                "u_observable_mismatches": int(np.sum(mismatch_counts[batch_u_mask])),
                "u_iterations_total": batch_u_iterations_total,
                "u_avg_iterations": (
                    None if batch_unconverged <= 0 else batch_u_iterations_total / batch_unconverged
                ),
                "u_max_iterations": batch_u_max_iterations,
                "primary_relaybp_converged": batch_primary_converged,
                "primary_relaybp_unconverged": (batch_primary_shots - batch_primary_converged),
                "primary_relaybp_iterations_total": (batch_primary_iterations_total),
                "primary_relaybp_avg_iterations": (
                    batch_primary_iterations_total / max(batch_primary_shots, 1)
                ),
                "primary_relaybp_max_iterations": (batch_primary_max_iterations),
                "relaybp_fallback_enabled": bool(relaybp_fallback_enabled),
                "relaybp_fallback_called": batch_relaybp_fallback_called,
                "relaybp_fallback_converged": batch_relaybp_fallback_converged,
                "relaybp_fallback_unconverged": (
                    batch_relaybp_fallback_called - batch_relaybp_fallback_converged
                ),
                "relaybp_fallback_iterations_total": (batch_relaybp_fallback_iterations_total),
                "relaybp_fallback_avg_iterations": (
                    None
                    if batch_relaybp_fallback_called <= 0
                    else batch_relaybp_fallback_iterations_total / batch_relaybp_fallback_called
                ),
                "relaybp_fallback_max_iterations": (batch_relaybp_fallback_max_iterations),
                "relaybp_fallback_failures": batch_relaybp_fallback_failures,
                "relaybp_fallback_observable_mismatches": (
                    batch_relaybp_fallback_observable_mismatches
                ),
                "mip_fallback_called": batch_mip_called,
                "mip_fallback_solved": batch_mip_solved,
                "mip_fallback_unsolved": batch_mip_unsolved,
                "mip_fallback_failures": int(np.sum(failure_mask[batch_u_mask])),
                "mip_fallback_observable_mismatches": int(np.sum(mismatch_counts[batch_u_mask])),
                "mip_fallback_seconds": 0.0,
                "seconds": float(relaybp_batch_seconds.get(int(batch_index), 0.0)),
            }
        )

    mip_progress = (
        f"mip={mip_fallback_solved}/{mip_fallback_called} mip_failures={mip_fallback_failures}"
        if mip_enabled
        else "mip=disabled"
    )
    print(
        f"    decoded shots={num_shots}/{num_shots} failures={failures} "
        f"observable_mismatches={observable_mismatches} "
        f"converged={converged}/{num_shots} c_failures={c_failures} "
        f"relaybp_fallback={relaybp_fallback_converged}/{relaybp_fallback_called} "
        f"{mip_progress}",
        flush=True,
    )

    decode_seconds = time.monotonic() - decode_t0
    return {
        "chunk_index": 0,
        "num_chunks": int(len(tasks)),
        "chunks": chunk_records,
        "num_decode_batches": int(len(batch_records)),
        "decode_batch_size": int(batch_size),
        "decode_batches": batch_records,
        "num_shots": num_shots,
        "failures": failures,
        "observable_mismatches": observable_mismatches,
        "converged": converged,
        "unconverged": unconverged,
        "convergence_rate": converged / max(num_shots, 1),
        "iterations_total": iterations_total,
        "avg_iterations": iterations_total / max(num_shots, 1),
        "max_iterations": max_iterations,
        "c_failures": c_failures,
        "c_observable_mismatches": c_observable_mismatches,
        "c_iterations_total": c_iterations_total,
        "c_avg_iterations": None if converged <= 0 else c_iterations_total / converged,
        "c_max_iterations": c_max_iterations,
        "u_failures": u_failures,
        "u_observable_mismatches": u_observable_mismatches,
        "u_iterations_total": u_iterations_total,
        "u_avg_iterations": (None if unconverged <= 0 else u_iterations_total / unconverged),
        "u_max_iterations": u_max_iterations,
        "primary_relaybp_converged": primary_converged,
        "primary_relaybp_unconverged": primary_unconverged,
        "primary_relaybp_convergence_rate": primary_converged / max(num_shots, 1),
        "primary_relaybp_iterations_total": primary_iterations_total,
        "primary_relaybp_avg_iterations": primary_iterations_total / max(num_shots, 1),
        "primary_relaybp_max_iterations": primary_max_iterations,
        "relaybp_fallback_enabled": bool(relaybp_fallback_enabled),
        "relaybp_fallback_config": (
            None if relaybp_fallback_config is None else asdict(relaybp_fallback_config)
        ),
        "relaybp_fallback_called": relaybp_fallback_called,
        "relaybp_fallback_converged": relaybp_fallback_converged,
        "relaybp_fallback_unconverged": relaybp_fallback_unconverged,
        "relaybp_fallback_convergence_rate": (
            None
            if relaybp_fallback_called <= 0
            else relaybp_fallback_converged / relaybp_fallback_called
        ),
        "relaybp_fallback_iterations_total": relaybp_fallback_iterations_total,
        "relaybp_fallback_avg_iterations": (
            None
            if relaybp_fallback_called <= 0
            else relaybp_fallback_iterations_total / relaybp_fallback_called
        ),
        "relaybp_fallback_max_iterations": relaybp_fallback_max_iterations,
        "relaybp_fallback_failures": relaybp_fallback_failures,
        "relaybp_fallback_observable_mismatches": (relaybp_fallback_observable_mismatches),
        "relaybp_fallback_seconds": float(relaybp_fallback_seconds),
        "mip_fallback_enabled": bool(mip_enabled),
        "mip_fallback_mode": "deferred_parallel",
        "mip_fallback_workers": int(mip_stats["workers"]),
        "mip_fallback_called": mip_fallback_called,
        "mip_fallback_solved": mip_fallback_solved,
        "mip_fallback_unsolved": mip_fallback_unsolved,
        "mip_fallback_failures": mip_fallback_failures,
        "mip_fallback_observable_mismatches": mip_fallback_observable_mismatches,
        "mip_fallback_seconds": mip_fallback_seconds,
        "mip_fallback_details_path": saved_mip_fallback_details_path,
        "relaybp_unconverged_details_path": saved_relaybp_unconverged_details_path,
        "logical_error_rate": failures / max(num_shots, 1),
        "load_seconds": float(load_seconds),
        "decode_seconds": float(decode_seconds),
        "seconds": float(time.monotonic() - t0),
        "pid": os.getpid(),
        "decoder": "relaybp",
        "relaybp_internal_parallel": True,
    }


def load_shot_batch(
    tasks: Sequence[DecodeTask],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    detector_batches: list[np.ndarray] = []
    observable_batches: list[np.ndarray] = []
    chunk_records: list[dict[str, object]] = []
    for task in tasks:
        shot_path = Path(task.shot_path)
        with np.load(shot_path, allow_pickle=False) as payload:
            detectors = unpack_bits(payload["detector_data"], payload["detector_shape"])
            observables = unpack_bits(payload["observable_data"], payload["observable_shape"])
            chunk_records.append(
                {
                    "chunk_index": int(task.chunk_index),
                    "shot_path": str(shot_path),
                    "start_shot": int(payload["start_shot"]),
                    "num_shots": int(payload["num_shots"]),
                    "seed": int(payload["seed"]),
                }
            )
        detector_batches.append(detectors.astype(np.uint8, copy=False))
        observable_batches.append(observables.astype(np.uint8, copy=False))
    if not detector_batches:
        raise ValueError("no shot chunks to decode")
    return np.vstack(detector_batches), np.vstack(observable_batches), chunk_records


def partition_decode_tasks(tasks: Sequence[DecodeTask], num_groups: int) -> list[list[DecodeTask]]:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    groups: list[list[DecodeTask]] = [[] for _ in range(min(num_groups, len(tasks)))]
    for index, task in enumerate(tasks):
        groups[index % len(groups)].append(task)
    return [group for group in groups if group]


def run_decode_batch_worker(payload: tuple) -> dict[str, object]:
    (
        tasks,
        matrices_path,
        config,
        decoder,
        decode_batch_size,
        progress_seconds,
        mip_fallback_config,
        relaybp_fallback_config,
        mip_fallback_details_path,
        mip_fallback_detail_metadata,
        relaybp_unconverged_details_path,
        relaybp_unconverged_detail_metadata,
    ) = payload
    result = run_decode_batch(
        tasks,
        Path(matrices_path),
        config,
        decoder=decoder,
        decode_batch_size=int(decode_batch_size),
        progress_seconds=float(progress_seconds),
        mip_fallback_config=mip_fallback_config,
        relaybp_fallback_config=relaybp_fallback_config,
        mip_fallback_details_path=(
            None if mip_fallback_details_path is None else Path(mip_fallback_details_path)
        ),
        mip_fallback_detail_metadata=mip_fallback_detail_metadata,
        relaybp_unconverged_details_path=(
            None
            if relaybp_unconverged_details_path is None
            else Path(relaybp_unconverged_details_path)
        ),
        relaybp_unconverged_detail_metadata=relaybp_unconverged_detail_metadata,
    )
    return {"tasks": tasks, "result": result}


def run_decode_batch(
    tasks: Sequence[DecodeTask],
    matrices_path: Path,
    config: RelayBPConfig,
    *,
    decoder: DecoderName,
    decode_batch_size: int,
    progress_seconds: float,
    mip_fallback_config: MIPFallbackConfig | None = None,
    relaybp_fallback_config: RelayBPFallbackConfig | None = None,
    mip_fallback_details_path: Path | None = None,
    mip_fallback_detail_metadata: dict[str, object] | None = None,
    relaybp_unconverged_details_path: Path | None = None,
    relaybp_unconverged_detail_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if decoder != "relaybp":
        raise ValueError(f"unsupported decoder: {decoder}")
    if not isinstance(config, RelayBPConfig):
        raise TypeError("RelayBP decoder requires RelayBPConfig")

    t0 = time.monotonic()
    detectors, observables, chunk_records = load_shot_batch(tasks)
    load_seconds = time.monotonic() - t0
    runner = build_relaybp_runner(matrices_path, config)
    mip_enabled = bool(mip_fallback_config is not None and mip_fallback_config.enabled)
    relaybp_fallback_enabled = bool(
        relaybp_fallback_config is not None and relaybp_fallback_config.enabled
    )
    relaybp_fallback_runner = None
    mip_check_matrix = None
    mip_observables_matrix = None
    mip_error_priors = None
    if relaybp_fallback_enabled:
        if relaybp_fallback_config is None:
            raise RuntimeError("RelayBP fallback config missing")
        relaybp_fallback_runner = build_relaybp_runner(
            matrices_path, relaybp_fallback_config.relay_config()
        )
    if mip_enabled or relaybp_fallback_enabled:
        (
            mip_check_matrix,
            mip_observables_matrix,
            mip_error_priors,
        ) = load_check_matrices(matrices_path)
        mip_check_matrix = mip_check_matrix.tocsr().astype(np.uint8)
        mip_observables_matrix = mip_observables_matrix.tocsr().astype(np.uint8)

    num_shots = int(observables.shape[0])
    batch_size = int(decode_batch_size)
    if batch_size <= 0 or batch_size > num_shots:
        batch_size = num_shots

    if mip_enabled or relaybp_fallback_enabled:
        if mip_check_matrix is None or mip_observables_matrix is None or mip_error_priors is None:
            raise RuntimeError("RelayBP deferred fallback path was not initialized")
        return run_relaybp_batch_with_deferred_mip(
            tasks,
            detectors,
            observables,
            chunk_records,
            runner=runner,
            relaybp_config=config,
            check_matrix=mip_check_matrix,
            observables_matrix=mip_observables_matrix,
            error_priors=mip_error_priors,
            mip_fallback_config=mip_fallback_config,
            relaybp_fallback_runner=relaybp_fallback_runner,
            relaybp_fallback_config=relaybp_fallback_config,
            decode_batch_size=batch_size,
            progress_seconds=progress_seconds,
            load_seconds=load_seconds,
            t0=t0,
            mip_fallback_details_path=mip_fallback_details_path,
            mip_fallback_detail_metadata=mip_fallback_detail_metadata,
            relaybp_unconverged_details_path=relaybp_unconverged_details_path,
            relaybp_unconverged_detail_metadata=relaybp_unconverged_detail_metadata,
        )

    print(
        f"    decode batch prepared: shots={num_shots} chunks={len(tasks)} "
        f"detectors={detectors.shape[1]} observables={observables.shape[1]} "
        f"decode_batch_size={batch_size} decoder=relaybp",
        flush=True,
    )
    decode_t0 = time.monotonic()
    last_progress = decode_t0
    failures = 0
    observable_mismatches = 0
    converged = 0
    unconverged = 0
    iterations_total = 0
    max_iterations = 0
    c_failures = 0
    c_observable_mismatches = 0
    c_iterations_total = 0
    c_max_iterations = 0
    u_failures = 0
    u_observable_mismatches = 0
    u_iterations_total = 0
    u_max_iterations = 0
    decoded = 0
    batch_records: list[dict[str, object]] = []
    all_converged_mask = np.zeros(num_shots, dtype=bool)
    all_failure_mask = np.zeros(num_shots, dtype=bool)
    all_mismatch_counts = np.zeros(num_shots, dtype=np.int64)
    all_iterations = np.zeros(num_shots, dtype=np.int64)

    for batch_index, start in enumerate(range(0, num_shots, batch_size)):
        stop = min(start + batch_size, num_shots)
        batch_t0 = time.monotonic()
        detailed_results = runner.decode_observables_detailed_batch(
            detectors[start:stop].astype(np.uint8, copy=False),
            parallel=True,
            progress_bar=False,
        )
        predicted = np.vstack(
            [np.asarray(result.observables, dtype=np.uint8) for result in detailed_results]
        )
        converged_mask = np.asarray(
            [bool(result.converged) for result in detailed_results], dtype=bool
        )
        iterations = np.asarray(
            [int(result.iterations) for result in detailed_results], dtype=np.int64
        )
        current_batch_shots = int(stop - start)
        batch_converged = int(np.sum(converged_mask))
        batch_unconverged = int(current_batch_shots - batch_converged)
        batch_iterations_total = int(np.sum(iterations))
        batch_max_iterations = int(np.max(iterations)) if len(iterations) else 0
        predicted = np.asarray(predicted, dtype=np.uint8)
        batch_observables = observables[start:stop].astype(np.uint8, copy=False)
        mismatch_mask = predicted != batch_observables
        mismatch_counts = np.sum(mismatch_mask, axis=1, dtype=np.int64)
        failure_mask = np.any(mismatch_mask, axis=1)
        all_converged_mask[start:stop] = converged_mask
        all_failure_mask[start:stop] = failure_mask
        all_mismatch_counts[start:stop] = mismatch_counts
        all_iterations[start:stop] = iterations
        batch_failures = int(np.sum(failure_mask))
        batch_observable_mismatches = int(np.sum(mismatch_counts))
        u_mask = ~converged_mask
        batch_c_failures = int(np.sum(failure_mask & converged_mask))
        batch_c_observable_mismatches = int(np.sum(mismatch_counts[converged_mask]))
        batch_c_iterations_total = int(np.sum(iterations[converged_mask]))
        batch_c_max_iterations = int(np.max(iterations[converged_mask])) if batch_converged else 0
        batch_u_failures = int(np.sum(failure_mask & u_mask))
        batch_u_observable_mismatches = int(np.sum(mismatch_counts[u_mask]))
        batch_u_iterations_total = int(np.sum(iterations[u_mask]))
        batch_u_max_iterations = int(np.max(iterations[u_mask])) if batch_unconverged else 0
        batch_seconds = time.monotonic() - batch_t0

        failures += batch_failures
        observable_mismatches += batch_observable_mismatches
        converged += batch_converged
        unconverged += batch_unconverged
        iterations_total += batch_iterations_total
        max_iterations = max(max_iterations, batch_max_iterations)
        c_failures += batch_c_failures
        c_observable_mismatches += batch_c_observable_mismatches
        c_iterations_total += batch_c_iterations_total
        c_max_iterations = max(c_max_iterations, batch_c_max_iterations)
        u_failures += batch_u_failures
        u_observable_mismatches += batch_u_observable_mismatches
        u_iterations_total += batch_u_iterations_total
        u_max_iterations = max(u_max_iterations, batch_u_max_iterations)
        decoded += current_batch_shots
        batch_records.append(
            {
                "batch_index": int(batch_index),
                "start_shot": int(start),
                "stop_shot": int(stop),
                "num_shots": int(current_batch_shots),
                "failures": int(batch_failures),
                "observable_mismatches": int(batch_observable_mismatches),
                "converged": int(batch_converged),
                "unconverged": int(batch_unconverged),
                "convergence_rate": batch_converged / max(current_batch_shots, 1),
                "iterations_total": int(batch_iterations_total),
                "avg_iterations": batch_iterations_total / max(current_batch_shots, 1),
                "max_iterations": int(batch_max_iterations),
                "c_failures": int(batch_c_failures),
                "c_observable_mismatches": int(batch_c_observable_mismatches),
                "c_iterations_total": int(batch_c_iterations_total),
                "c_avg_iterations": (
                    None if batch_converged <= 0 else batch_c_iterations_total / batch_converged
                ),
                "c_max_iterations": int(batch_c_max_iterations),
                "u_failures": int(batch_u_failures),
                "u_observable_mismatches": int(batch_u_observable_mismatches),
                "u_iterations_total": int(batch_u_iterations_total),
                "u_avg_iterations": (
                    None if batch_unconverged <= 0 else batch_u_iterations_total / batch_unconverged
                ),
                "u_max_iterations": int(batch_u_max_iterations),
                "mip_fallback_called": 0,
                "mip_fallback_solved": 0,
                "mip_fallback_unsolved": 0,
                "mip_fallback_failures": 0,
                "mip_fallback_observable_mismatches": 0,
                "mip_fallback_seconds": 0.0,
                "seconds": float(batch_seconds),
            }
        )
        now = time.monotonic()
        if now - last_progress >= progress_seconds or decoded == num_shots:
            rate = decoded / max(now - decode_t0, 1e-9)
            print(
                f"    decoded shots={decoded}/{num_shots} failures={failures} "
                f"observable_mismatches={observable_mismatches} "
                f"converged={converged}/{decoded} "
                f"c_failures={c_failures} "
                f"rate={rate:.2f} shots/s",
                flush=True,
            )
            last_progress = now

    decode_seconds = time.monotonic() - decode_t0
    saved_relaybp_unconverged_details_path: str | None = None
    primary_unconverged_indices = np.flatnonzero(~all_converged_mask).astype(np.int64)
    if relaybp_unconverged_details_path is not None and len(primary_unconverged_indices):
        metadata = dict(relaybp_unconverged_detail_metadata or {})
        metadata.update(
            {
                "detail_type": "relaybp_primary_unconverged",
                "num_primary_unconverged": int(len(primary_unconverged_indices)),
            }
        )
        save_relaybp_unconverged_details(
            relaybp_unconverged_details_path,
            metadata=metadata,
            relaybp_config=config,
            shot_indices=primary_unconverged_indices,
            detectors=detectors[primary_unconverged_indices],
            observables=observables[primary_unconverged_indices],
            old_converged_mask=all_converged_mask[primary_unconverged_indices],
            old_failure_mask=all_failure_mask[primary_unconverged_indices],
            old_mismatch_counts=all_mismatch_counts[primary_unconverged_indices],
            old_iterations=all_iterations[primary_unconverged_indices],
        )
        saved_relaybp_unconverged_details_path = str(relaybp_unconverged_details_path)
    return {
        "chunk_index": 0,
        "num_chunks": int(len(tasks)),
        "chunks": chunk_records,
        "num_decode_batches": int(len(batch_records)),
        "decode_batch_size": int(batch_size),
        "decode_batches": batch_records,
        "num_shots": num_shots,
        "failures": int(failures),
        "observable_mismatches": int(observable_mismatches),
        "converged": int(converged),
        "unconverged": int(unconverged),
        "convergence_rate": converged / max(num_shots, 1),
        "iterations_total": int(iterations_total),
        "avg_iterations": iterations_total / max(num_shots, 1),
        "max_iterations": int(max_iterations),
        "c_failures": int(c_failures),
        "c_observable_mismatches": int(c_observable_mismatches),
        "c_iterations_total": int(c_iterations_total),
        "c_avg_iterations": None if converged <= 0 else c_iterations_total / converged,
        "c_max_iterations": int(c_max_iterations),
        "u_failures": int(u_failures),
        "u_observable_mismatches": int(u_observable_mismatches),
        "u_iterations_total": int(u_iterations_total),
        "u_avg_iterations": None if unconverged <= 0 else u_iterations_total / unconverged,
        "u_max_iterations": int(u_max_iterations),
        "primary_relaybp_converged": int(converged),
        "primary_relaybp_unconverged": int(unconverged),
        "primary_relaybp_convergence_rate": converged / max(num_shots, 1),
        "primary_relaybp_iterations_total": int(iterations_total),
        "primary_relaybp_avg_iterations": iterations_total / max(num_shots, 1),
        "primary_relaybp_max_iterations": int(max_iterations),
        "relaybp_unconverged_details_path": saved_relaybp_unconverged_details_path,
        "mip_fallback_enabled": False,
        "mip_fallback_called": 0,
        "mip_fallback_solved": 0,
        "mip_fallback_unsolved": 0,
        "mip_fallback_failures": 0,
        "mip_fallback_observable_mismatches": 0,
        "mip_fallback_seconds": 0.0,
        "logical_error_rate": failures / max(num_shots, 1),
        "load_seconds": float(load_seconds),
        "decode_seconds": float(decode_seconds),
        "seconds": float(time.monotonic() - t0),
        "pid": os.getpid(),
        "decoder": "relaybp",
        "relaybp_internal_parallel": True,
    }


def read_decode_records(
    path: Path,
    *,
    decoder: DecoderName = "relaybp",
    config_hash: str | None = None,
    shot_manifest_hash: str | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if decoder is not None and record.get("decoder", "relaybp") != decoder:
            continue
        hash_key = "relaybp_config_hash"
        if config_hash is not None and record.get(hash_key) != config_hash:
            continue
        if (
            shot_manifest_hash is not None
            and record.get("shot_manifest_hash") != shot_manifest_hash
        ):
            continue
        records.append(record)
    return records


def relaybp_config_from_record(record: dict[str, object]) -> RelayBPConfig:
    payload = record.get("relaybp_config")
    if not isinstance(payload, dict):
        return RelayBPConfig()
    return RelayBPConfig(
        gamma0=float(payload.get("gamma0", RelayBPConfig.gamma0)),
        pre_iter=int(payload.get("pre_iter", RelayBPConfig.pre_iter)),
        num_sets=int(payload.get("num_sets", RelayBPConfig.num_sets)),
        set_max_iter=int(payload.get("set_max_iter", RelayBPConfig.set_max_iter)),
        gamma_dist_min=float(payload.get("gamma_dist_min", RelayBPConfig.gamma_dist_min)),
        gamma_dist_max=float(payload.get("gamma_dist_max", RelayBPConfig.gamma_dist_max)),
        stop_nconv=int(payload.get("stop_nconv", RelayBPConfig.stop_nconv)),
    )


def _record_relaybp_fallback_enabled(record: dict[str, object]) -> bool:
    payload = record.get("relaybp_fallback_config")
    return bool(record.get("relaybp_fallback_enabled")) or (
        isinstance(payload, dict) and bool(payload.get("enabled"))
    )


def relaybp_fallback_config_from_record(record: dict[str, object]) -> RelayBPFallbackConfig | None:
    payload = record.get("relaybp_fallback_config")
    if not isinstance(payload, dict):
        return None
    enabled = bool(payload.get("enabled", record.get("relaybp_fallback_enabled", False)))
    if not enabled:
        return None
    return RelayBPFallbackConfig(
        enabled=True,
        gamma0=float(payload.get("gamma0", RelayBPFallbackConfig.gamma0)),
        pre_iter=int(payload.get("pre_iter", RelayBPFallbackConfig.pre_iter)),
        num_sets=int(payload.get("num_sets", RelayBPFallbackConfig.num_sets)),
        set_max_iter=int(payload.get("set_max_iter", RelayBPFallbackConfig.set_max_iter)),
        gamma_dist_min=float(payload.get("gamma_dist_min", RelayBPFallbackConfig.gamma_dist_min)),
        gamma_dist_max=float(payload.get("gamma_dist_max", RelayBPFallbackConfig.gamma_dist_max)),
        stop_nconv=int(payload.get("stop_nconv", RelayBPFallbackConfig.stop_nconv)),
    )


def _relaybp_fallback_config_matches(
    record: dict[str, object],
    relaybp_fallback_config: RelayBPFallbackConfig | None,
) -> bool:
    target_enabled = relaybp_fallback_config is not None and relaybp_fallback_config.enabled
    if not target_enabled:
        return not _record_relaybp_fallback_enabled(record)
    if not _record_relaybp_fallback_enabled(record):
        return False
    try:
        source_config = relaybp_fallback_config_from_record(record)
    except (TypeError, ValueError):
        return False
    return source_config == relaybp_fallback_config


def _record_mip_fallback_enabled(record: dict[str, object]) -> bool:
    payload = record.get("mip_fallback_config")
    return bool(record.get("mip_fallback_enabled")) or (
        isinstance(payload, dict) and bool(payload.get("enabled"))
    )


def _normalised_mip_fallback_payload(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict) or not bool(payload.get("enabled")):
        return None
    try:
        normalised: dict[str, object] = {
            "enabled": True,
            "time_limit": float(payload.get("time_limit", 0.0) or 0.0),
            "wall_time_limit": float(payload.get("wall_time_limit", 0.0) or 0.0),
            "mip_rel_gap": float(payload.get("mip_rel_gap", 0.0) or 0.0),
            "feasible_only": bool(payload.get("feasible_only", False)),
            "count_unsolved_as_failure": bool(payload.get("count_unsolved_as_failure", True)),
            "threads": int(payload.get("threads", 1) or 1),
        }
    except (TypeError, ValueError):
        return None
    if float(normalised["wall_time_limit"]) <= 0.0:
        normalised.pop("wall_time_limit", None)
    return normalised


def _mip_fallback_config_matches(
    record: dict[str, object], mip_fallback_config: MIPFallbackConfig | None
) -> bool:
    target_enabled = mip_fallback_config is not None and mip_fallback_config.enabled
    if not target_enabled:
        return not _record_mip_fallback_enabled(record)
    if not _record_mip_fallback_enabled(record):
        return False
    source_payload = _normalised_mip_fallback_payload(record.get("mip_fallback_config"))
    target_payload = _normalised_mip_fallback_payload(asdict(mip_fallback_config))
    return source_payload == target_payload


def condition_from_meta(meta: dict[str, object]) -> Condition:
    return Condition(
        code_name=str(meta["code_name"]),
        parameter_label=str(meta["parameter_label"]),
        p_code=int(meta["P"]),
        l=int(meta["L"]),
        j=int(meta["J"]),
        expected_d=int(meta["expected_d"]),
        basis=str(meta["basis"]),  # type: ignore[arg-type]
        decoding_mode=str(meta["decoding_mode"]),  # type: ignore[arg-type]
        p_noise=float(meta["p"]),
        cycles=int(meta["cycles"]),
    )


def read_all_decode_record_groups(
    decode_dir: Path, decoder: DecoderName
) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    decode_subdir = decode_dir / "decode"
    if not decode_subdir.exists():
        return groups
    for path in sorted(decode_subdir.glob("*.jsonl")):
        try:
            mtime_ns = int(path.stat().st_mtime_ns)
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = record.get("label")
            record_decoder = str(record.get("decoder", "relaybp"))
            if record_decoder != decoder:
                continue
            config_hash = record.get("relaybp_config_hash")
            shot_manifest_hash = record.get("shot_manifest_hash")
            if label is None or config_hash is None or shot_manifest_hash is None:
                continue
            if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
                continue
            if "observable_mismatches" not in record:
                continue
            record = dict(record)
            record["_source_path"] = str(path)
            record["_source_mtime_ns"] = mtime_ns
            record["_source_line_index"] = line_index
            groups[(str(label), str(config_hash))] = groups.get(
                (str(label), str(config_hash)), []
            ) + [record]
    return groups


def _record_source_sort_key(record: dict[str, object]) -> tuple[int, int]:
    return (
        int(record.get("_source_mtime_ns", 0)),
        int(record.get("_source_line_index", 0)),
    )


def rebuild_decode_summary_from_jsonl(
    sample_dir: Path, decode_dir: Path, decoder: DecoderName = "relaybp"
) -> list[dict[str, object]]:
    """Summarize saved decode results, keeping the newest run per plot series."""
    groups = read_all_decode_record_groups(decode_dir, decoder)
    best_by_key: dict[tuple[object, ...], tuple[tuple[int, int, int], dict[str, object]]] = {}
    for (label, config_hash), raw_records in groups.items():
        meta_path = sample_dir / "artifacts" / f"{label}.meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_key = max(_record_source_sort_key(record) for record in raw_records)
        clean_records = [
            {key: value for key, value in record.items() if not str(key).startswith("_source_")}
            for record in raw_records
        ]
        latest_record = max(raw_records, key=_record_source_sort_key)
        shot_manifest_hashes = sorted(
            {
                str(record.get("shot_manifest_hash"))
                for record in raw_records
                if record.get("shot_manifest_hash") is not None
            }
        )
        if len(shot_manifest_hashes) == 1:
            shot_manifest_hash = shot_manifest_hashes[0]
        else:
            shot_manifest_hash = f"multiple:{len(shot_manifest_hashes)}"
        condition = condition_from_meta(meta)
        config = relaybp_config_from_record(latest_record)
        summary = aggregate_decode_records(
            condition,
            meta,
            clean_records,
            Path(str(latest_record.get("_source_path", ""))),
            config,
            config_hash,
            shot_manifest_hash,
            shots_manifest_path(sample_dir, condition),
        )
        summary["shot_manifest_hashes"] = shot_manifest_hashes
        summary["shots_requested"] = summary.get("shots_decoded")
        selection_key = (
            source_key[0],
            source_key[1],
            int(summary.get("shots_decoded") or 0),
        )
        summary_key = summary_record_key(summary)
        current = best_by_key.get(summary_key)
        if current is None or selection_key >= current[0]:
            best_by_key[summary_key] = (selection_key, summary)
    return [item[1] for item in best_by_key.values()]


def expected_decode_tasks(condition: Condition, manifest: dict, meta: dict) -> list[DecodeTask]:
    tasks: list[DecodeTask] = []
    missing: list[str] = []
    for chunk in manifest.get("chunks", []):
        path = Path(str(chunk["path"]))
        num_shots = int(chunk["num_shots"])
        if not shot_file_ok(
            path, num_shots, int(meta["num_detectors"]), int(meta["num_observables"])
        ):
            missing.append(str(path))
        else:
            tasks.append(DecodeTask(int(chunk["chunk_index"]), str(path)))
    if missing:
        raise FileNotFoundError(
            f"Missing or invalid shot chunks for {condition.label}. Run --stage generate first. missing_count={len(missing)} sample={missing[:5]}"
        )
    if not tasks:
        raise FileNotFoundError(
            f"Shot manifest has no valid chunks for {condition.label}. Run --stage generate first."
        )
    return sorted(tasks, key=lambda task: task.chunk_index)


def _chunk_int(chunk: dict[str, object], key: str, default: int = 0) -> int:
    value = chunk.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _chunk_path_text(chunk: dict[str, object]) -> str:
    return str(chunk.get("shot_path") or chunk.get("path") or "")


def _manifest_chunks_by_index(manifest: dict) -> dict[int, dict[str, object]]:
    chunks: dict[int, dict[str, object]] = {}
    for chunk in manifest.get("chunks", []):
        if isinstance(chunk, dict):
            chunks[_chunk_int(chunk, "chunk_index")] = chunk
    return chunks


def chunk_fingerprint_from_manifest_chunk(chunk: dict[str, object]) -> dict[str, object]:
    fingerprint = {
        "chunk_index": _chunk_int(chunk, "chunk_index"),
        "path": _chunk_path_text(chunk),
        "start_shot": _chunk_int(chunk, "start_shot"),
        "num_shots": _chunk_int(chunk, "num_shots"),
        "seed": _chunk_int(chunk, "seed"),
    }
    for key in ("size", "mtime_ns"):
        if key in chunk:
            fingerprint[key] = _chunk_int(chunk, key)
    return fingerprint


def _chunk_matches_manifest(
    record_chunk: dict[str, object],
    manifest_chunk: dict[str, object],
    *,
    strict_file_fingerprint: bool,
) -> bool:
    if _chunk_int(record_chunk, "chunk_index") != _chunk_int(manifest_chunk, "chunk_index"):
        return False
    if _chunk_path_text(record_chunk) != _chunk_path_text(manifest_chunk):
        return False
    for key in ("start_shot", "num_shots", "seed"):
        if _chunk_int(record_chunk, key) != _chunk_int(manifest_chunk, key):
            return False
    if strict_file_fingerprint:
        for key in ("size", "mtime_ns"):
            if key not in record_chunk:
                return False
            if _chunk_int(record_chunk, key) != _chunk_int(manifest_chunk, key):
                return False
    return True


def decode_record_covered_chunk_indices(
    record: dict[str, object], manifest_chunks: dict[int, dict[str, object]]
) -> set[int]:
    raw_chunks = record.get("chunk_fingerprints")
    strict_file_fingerprint = True
    if not isinstance(raw_chunks, list):
        raw_chunks = record.get("chunks")
        strict_file_fingerprint = False
    if not isinstance(raw_chunks, list):
        return set()

    covered: set[int] = set()
    decoded_shots = 0
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            return set()
        chunk_index = _chunk_int(raw_chunk, "chunk_index")
        manifest_chunk = manifest_chunks.get(chunk_index)
        if manifest_chunk is None:
            return set()
        if not _chunk_matches_manifest(
            raw_chunk,
            manifest_chunk,
            strict_file_fingerprint=strict_file_fingerprint,
        ):
            return set()
        if chunk_index in covered:
            return set()
        covered.add(chunk_index)
        decoded_shots += _chunk_int(manifest_chunk, "num_shots")

    if not covered:
        return set()
    if int(record.get("num_shots", -1)) != decoded_shots:
        return set()
    return covered


def decode_tasks_for_record(
    record: dict[str, object], manifest_chunks: dict[int, dict[str, object]]
) -> list[DecodeTask]:
    raw_chunks = record.get("chunk_fingerprints")
    if not isinstance(raw_chunks, list):
        raw_chunks = record.get("chunks")
    if not isinstance(raw_chunks, list):
        raise RuntimeError("source decode record has no chunk list")

    tasks: list[DecodeTask] = []
    seen: set[int] = set()
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise RuntimeError("source decode record has a malformed chunk entry")
        chunk_index = _chunk_int(raw_chunk, "chunk_index")
        if chunk_index in seen:
            raise RuntimeError(f"duplicate source chunk index {chunk_index}")
        manifest_chunk = manifest_chunks.get(chunk_index)
        if manifest_chunk is None:
            raise RuntimeError(f"source chunk {chunk_index} is absent from manifest")
        shot_path = _chunk_path_text(manifest_chunk)
        if not shot_path:
            raise RuntimeError(f"manifest chunk {chunk_index} has no shot path")
        tasks.append(DecodeTask(chunk_index, shot_path))
        seen.add(chunk_index)
    return sorted(tasks, key=lambda task: task.chunk_index)


def select_reusable_decode_records(
    records: list[dict[str, object]],
    all_tasks: Sequence[DecodeTask],
    manifest: dict,
    condition: Condition,
    decoder: DecoderName,
    config_hash: str,
    *,
    include_retry_records: bool = False,
) -> tuple[list[dict[str, object]], list[DecodeTask]]:
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    candidates: list[tuple[set[int], int, dict[str, object]]] = []
    hash_key = "relaybp_config_hash"
    for record_order, record in enumerate(records):
        if record.get("decoder", "relaybp") != decoder:
            continue
        if record.get(hash_key) != config_hash:
            continue
        if not include_retry_records and not _is_plain_relaybp_decode_record(record):
            continue
        if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
            continue
        if "observable_mismatches" not in record:
            continue
        if "converged" not in record or "unconverged" not in record:
            continue
        if "c_failures" not in record or "u_failures" not in record:
            continue
        if "c_observable_mismatches" not in record or "u_observable_mismatches" not in record:
            continue
        if record.get("label") != condition.label:
            continue
        covered = decode_record_covered_chunk_indices(record, manifest_chunks)
        if not covered or not covered <= target_indices:
            continue
        candidates.append((covered, record_order, record))

    remaining = set(target_indices)
    selected: list[dict[str, object]] = []
    candidates.sort(
        key=lambda item: (
            -len(item[0]),
            min(item[0]),
            -int(item[1]),
            int(item[2].get("num_shots", 0)),
        )
    )
    for covered, _record_order, record in candidates:
        if covered <= remaining:
            selected.append(record)
            remaining -= covered

    pending = [task for task in all_tasks if int(task.chunk_index) in remaining]
    return selected, pending


def _is_plain_relaybp_decode_record(record: dict[str, object]) -> bool:
    return not (
        bool(record.get("relaybp_fallback_retry_unconverged_enabled"))
        or bool(record.get("mip_retry_unsolved_enabled"))
        or bool(record.get("bposd_retry_unconverged_enabled"))
    )


def _is_reusable_decode_record_for_target(
    record: dict[str, object],
    relaybp_fallback_config: RelayBPFallbackConfig | None,
) -> bool:
    if bool(record.get("mip_retry_unsolved_enabled")):
        return False
    if bool(record.get("bposd_retry_unconverged_enabled")):
        return False
    if not bool(record.get("relaybp_fallback_retry_unconverged_enabled")):
        return True

    # Reuse a fallback retry only when the target enables the same fallback.
    return relaybp_fallback_config is not None and relaybp_fallback_config.enabled


def _normalise_compatible_reuse_record(record: dict[str, object]) -> dict[str, object]:
    normalised = dict(record)
    if bool(normalised.get("relaybp_fallback_retry_unconverged_enabled")):
        normalised["compatible_reuse_source_mode"] = "relaybp_fallback_retry"
        for key in list(normalised):
            if key.startswith("relaybp_fallback_retry_"):
                normalised.pop(key, None)
    else:
        normalised["compatible_reuse_source_mode"] = "decode"
    return normalised


def find_compatible_decode_records_from_other_hashes(
    decode_dir: Path,
    all_tasks: Sequence[DecodeTask],
    manifest: dict,
    condition: Condition,
    decoder: DecoderName,
    target_config_hash: str,
    config: RelayBPConfig,
    *,
    shot_manifest_hash: str,
    mip_fallback_config: MIPFallbackConfig | None,
    relaybp_fallback_config: RelayBPFallbackConfig | None,
) -> list[dict[str, object]]:
    """Find earlier decode records with matching samples and decoder settings.

    Recorded chunks must belong to the current sample set. Decoder and fallback
    parameters must match; records from a separate retry stage are excluded.
    """
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    compatible: list[dict[str, object]] = []
    groups = read_all_decode_record_groups(decode_dir, decoder)
    for (label, source_config_hash), records in groups.items():
        if label != condition.label or source_config_hash == target_config_hash:
            continue
        for record in records:
            if record.get("decoder", "relaybp") != decoder:
                continue
            if record.get("label") != condition.label:
                continue
            if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
                continue
            if not _is_reusable_decode_record_for_target(record, relaybp_fallback_config):
                continue
            try:
                if relaybp_config_from_record(record) != config:
                    continue
            except (TypeError, ValueError):
                continue
            if not _relaybp_fallback_config_matches(record, relaybp_fallback_config):
                continue
            if not _mip_fallback_config_matches(record, mip_fallback_config):
                continue
            covered = decode_record_covered_chunk_indices(record, manifest_chunks)
            if not covered or not covered <= target_indices:
                continue

            migrated = _normalise_compatible_reuse_record(
                {key: value for key, value in record.items() if not str(key).startswith("_source_")}
            )
            original_manifest_hash = migrated.get("shot_manifest_hash")
            migrated["relaybp_config_hash"] = target_config_hash
            migrated["shot_manifest_hash"] = shot_manifest_hash
            migrated["compatible_reuse_source_hash"] = source_config_hash
            if original_manifest_hash != shot_manifest_hash:
                migrated["compatible_reuse_original_shot_manifest_hash"] = original_manifest_hash
            compatible.append(migrated)
    return compatible


def append_compatible_decode_records(
    jsonl_path: Path,
    records: Sequence[dict[str, object]],
    manifest: dict,
) -> int:
    if not records:
        return 0
    existing_ids: set[str] = set()
    for record in read_decode_records(jsonl_path):
        fingerprints = record.get("chunk_fingerprints")
        if isinstance(fingerprints, list):
            existing_ids.add(decode_record_id_from_fingerprints(fingerprints))

    appended = 0
    with jsonl_path.open("a", encoding="utf-8") as fout:
        for record in records:
            fingerprints = record.get("chunk_fingerprints")
            if not isinstance(fingerprints, list):
                covered = decode_record_covered_chunk_indices(
                    record, _manifest_chunks_by_index(manifest)
                )
                if not covered:
                    continue
                fingerprints = chunk_fingerprints_for_tasks(
                    [DecodeTask(index, "") for index in sorted(covered)], manifest
                )
            record_id = decode_record_id_from_fingerprints(fingerprints)
            if record_id in existing_ids:
                continue
            fout.write(json.dumps(record, sort_keys=True) + "\n")
            existing_ids.add(record_id)
            appended += 1
    return appended


def chunk_fingerprints_for_tasks(
    tasks: Sequence[DecodeTask], manifest: dict
) -> list[dict[str, object]]:
    manifest_chunks = _manifest_chunks_by_index(manifest)
    fingerprints = []
    for task in tasks:
        chunk = manifest_chunks.get(int(task.chunk_index))
        if chunk is None:
            raise KeyError(f"missing manifest chunk {task.chunk_index}")
        fingerprints.append(chunk_fingerprint_from_manifest_chunk(chunk))
    return fingerprints


def decode_record_id_from_fingerprints(fingerprints: Sequence[dict[str, object]]) -> str:
    return stable_hash({"chunk_fingerprints": list(fingerprints)})[:12]


def run_decode_condition(
    condition: Condition,
    sample_dir: Path,
    decode_dir: Path,
    shots: int,
    shot_chunk: int,
    workers: int,
    decoder: DecoderName,
    config: RelayBPConfig,
    meta: dict,
    *,
    force: bool,
    progress_seconds: float,
    decode_batch_size: int,
    mip_fallback_config: MIPFallbackConfig | None = None,
    relaybp_fallback_config: RelayBPFallbackConfig | None = None,
) -> dict[str, object]:
    _ = (shots, shot_chunk)
    config_hash = relaybp_mip_config_hash(config, mip_fallback_config, relaybp_fallback_config)
    jsonl_path = decode_jsonl_path(decode_dir, condition, decoder, config_hash)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if force and jsonl_path.exists():
        jsonl_path.unlink()
    manifest = load_shots_manifest(sample_dir, condition)
    shot_manifest_hash = str(manifest["manifest_hash"])
    all_tasks = expected_decode_tasks(condition, manifest, meta)
    all_records = read_decode_records(jsonl_path, decoder=decoder, config_hash=config_hash)
    compatible_records: list[dict[str, object]] = []
    if not force:
        compatible_records = find_compatible_decode_records_from_other_hashes(
            decode_dir,
            all_tasks,
            manifest,
            condition,
            decoder,
            config_hash,
            config,
            shot_manifest_hash=shot_manifest_hash,
            mip_fallback_config=mip_fallback_config,
            relaybp_fallback_config=relaybp_fallback_config,
        )
    existing, pending = select_reusable_decode_records(
        [] if force else all_records + compatible_records,
        all_tasks,
        manifest,
        condition,
        decoder,
        config_hash,
    )
    if compatible_records:
        compatible_record_ids = {
            decode_record_id_from_fingerprints(record["chunk_fingerprints"])
            for record in compatible_records
            if isinstance(record.get("chunk_fingerprints"), list)
        }
        selected_compatible_records = [
            record
            for record in existing
            if isinstance(record.get("chunk_fingerprints"), list)
            and decode_record_id_from_fingerprints(record["chunk_fingerprints"])
            in compatible_record_ids
        ]
        appended = append_compatible_decode_records(
            jsonl_path, selected_compatible_records, manifest
        )
        if appended:
            print(
                f"  decode {condition.label}: migrated {appended} compatible "
                "cached decode record(s) from older config hash",
                flush=True,
            )
            all_records = read_decode_records(jsonl_path, decoder=decoder, config_hash=config_hash)
            existing, pending = select_reusable_decode_records(
                all_records,
                all_tasks,
                manifest,
                condition,
                decoder,
                config_hash,
            )
    reused_chunks = sum(
        len(decode_record_covered_chunk_indices(record, _manifest_chunks_by_index(manifest)))
        for record in existing
    )
    if not pending:
        print(
            f"  decode {condition.label}: cached chunks present; pending=0/{len(all_tasks)} chunks",
            flush=True,
        )
    else:
        print(
            f"  decode {condition.label}: reusing {reused_chunks}/{len(all_tasks)} chunks; "
            f"decoding {len(pending)} chunks with decoder={decoder}",
            flush=True,
        )
        matrices_path = artifact_paths(sample_dir, condition)["matrices"]
        config_hash_key = "relaybp_config_hash"
        config_payload_key = "relaybp_config"

        def attach_metadata(result: dict[str, object], tasks: Sequence[DecodeTask]) -> None:
            result.update(
                {
                    "label": condition.label,
                    "decoder": decoder,
                    config_hash_key: config_hash,
                    "decode_schema_version": DECODE_SCHEMA_VERSION,
                    "shot_manifest_hash": shot_manifest_hash,
                    config_payload_key: asdict(config),
                    "mip_fallback_config": (
                        None if mip_fallback_config is None else asdict(mip_fallback_config)
                    ),
                    "relaybp_fallback_config": (
                        None if relaybp_fallback_config is None else asdict(relaybp_fallback_config)
                    ),
                    "basis": condition.basis,
                    "decoding_mode": condition.decoding_mode,
                    "p": condition.p_noise,
                    "cycles": condition.cycles,
                    "chunk_fingerprints": chunk_fingerprints_for_tasks(tasks, manifest),
                }
            )

        pending_chunk_fingerprints = chunk_fingerprints_for_tasks(pending, manifest)
        pending_record_id = decode_record_id_from_fingerprints(pending_chunk_fingerprints)
        mip_details_path: Path | None = None
        mip_detail_metadata: dict[str, object] | None = None
        relaybp_details_path: Path | None = None
        relaybp_detail_metadata: dict[str, object] | None = None
        if decoder == "relaybp":
            relaybp_details_path = relaybp_unconverged_details_path(
                decode_dir, condition, config_hash, pending_record_id
            )
            if force and relaybp_details_path.exists():
                relaybp_details_path.unlink()
            relaybp_detail_metadata = {
                "label": condition.label,
                "decoder": decoder,
                "relaybp_config_hash": config_hash,
                "relaybp_fallback_config": (
                    None if relaybp_fallback_config is None else asdict(relaybp_fallback_config)
                ),
                "record_id": pending_record_id,
                "shot_manifest_hash": shot_manifest_hash,
                "chunk_fingerprints": pending_chunk_fingerprints,
            }
        if decoder == "relaybp" and mip_fallback_config is not None and mip_fallback_config.enabled:
            mip_details_path = mip_fallback_details_path(
                decode_dir, condition, config_hash, pending_record_id
            )
            if force and mip_details_path.exists():
                mip_details_path.unlink()
            mip_detail_metadata = {
                "label": condition.label,
                "decoder": decoder,
                "relaybp_config_hash": config_hash,
                "relaybp_fallback_config": (
                    None if relaybp_fallback_config is None else asdict(relaybp_fallback_config)
                ),
                "record_id": pending_record_id,
                "shot_manifest_hash": shot_manifest_hash,
                "chunk_fingerprints": pending_chunk_fingerprints,
            }

        result = run_decode_batch(
            pending,
            matrices_path,
            config,
            decoder=decoder,
            decode_batch_size=decode_batch_size,
            progress_seconds=progress_seconds,
            mip_fallback_config=mip_fallback_config,
            relaybp_fallback_config=relaybp_fallback_config,
            mip_fallback_details_path=mip_details_path,
            mip_fallback_detail_metadata=mip_detail_metadata,
            relaybp_unconverged_details_path=relaybp_details_path,
            relaybp_unconverged_detail_metadata=relaybp_detail_metadata,
        )
        attach_metadata(result, pending)
        with jsonl_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(result, sort_keys=True) + "\n")
    all_records = read_decode_records(jsonl_path, decoder=decoder, config_hash=config_hash)
    records, pending = select_reusable_decode_records(
        all_records, all_tasks, manifest, condition, decoder, config_hash
    )
    if pending:
        raise RuntimeError(
            f"decode cache did not cover all chunks for {condition.label}: "
            f"pending={len(pending)}/{len(all_tasks)}"
        )
    return aggregate_decode_records(
        condition,
        meta,
        records,
        jsonl_path,
        config,
        config_hash,
        shot_manifest_hash,
        shots_manifest_path(sample_dir, condition),
    )


def _record_int(record: dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(record.get(key, default) or 0)
    except (TypeError, ValueError):
        return int(default)


def _record_float(record: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _add_record_int(record: dict[str, object], key: str, delta: int) -> None:
    if key in record and record.get(key) is not None:
        record[key] = _record_int(record, key) + int(delta)


def find_default_mip_retry_source_hash(
    decode_dir: Path,
    manifest: dict,
    all_tasks: Sequence[DecodeTask],
    condition: Condition,
    config: RelayBPConfig,
    mip_fallback_config: MIPFallbackConfig,
) -> str:
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    candidates: list[dict[str, object]] = []
    groups = read_all_decode_record_groups(decode_dir, "relaybp")
    for (label, _config_hash), records in groups.items():
        if label != condition.label:
            continue
        for record in records:
            if record.get("decoder", "relaybp") != "relaybp":
                continue
            if record.get("label") != condition.label:
                continue
            if relaybp_config_from_record(record) != config:
                continue
            if bool(record.get("relaybp_fallback_enabled")):
                continue
            if bool(record.get("relaybp_fallback_retry_unconverged_enabled")):
                continue
            if not bool(record.get("mip_fallback_enabled")):
                continue
            source_mip_config = record.get("mip_fallback_config")
            if not isinstance(source_mip_config, dict):
                continue
            try:
                source_gap = float(source_mip_config.get("mip_rel_gap", 0.0) or 0.0)
                source_time_limit = float(source_mip_config.get("time_limit", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isclose(
                source_gap,
                float(mip_fallback_config.mip_rel_gap),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                continue
            target_time_limit = float(mip_fallback_config.time_limit)
            if target_time_limit > 0.0:
                if source_time_limit >= target_time_limit:
                    continue
            elif source_time_limit <= 0.0:
                continue
            if _record_int(record, "mip_fallback_unsolved") <= 0:
                continue
            details_path = record.get("mip_fallback_details_path")
            if not isinstance(details_path, str) or not details_path:
                continue
            if not Path(details_path).exists():
                continue
            covered = decode_record_covered_chunk_indices(record, manifest_chunks)
            if covered != target_indices:
                continue
            candidates.append(record)
    if not candidates:
        raise RuntimeError(
            f"{condition.label}: no previous full RelayBP+MIP decode record "
            "with saved unsolved MIP details was found. Run RelayBP+MIP once "
            "with this script version, or pass --mip-retry-source-hash for a "
            "specific source hash."
        )
    source = max(candidates, key=_record_source_sort_key)
    value = source.get("relaybp_config_hash")
    if not isinstance(value, str) or not value:
        raise RuntimeError("selected MIP retry source record has no relaybp_config_hash")
    return value


def find_default_relaybp_fallback_retry_source_hash(
    decode_dir: Path,
    manifest: dict,
    all_tasks: Sequence[DecodeTask],
    condition: Condition,
    config: RelayBPConfig,
) -> str:
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    candidates: list[dict[str, object]] = []
    groups = read_all_decode_record_groups(decode_dir, "relaybp")
    for (label, _config_hash), records in groups.items():
        if label != condition.label:
            continue
        for record in records:
            if record.get("decoder", "relaybp") != "relaybp":
                continue
            if record.get("label") != condition.label:
                continue
            if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
                continue
            if relaybp_config_from_record(record) != config:
                continue
            if bool(record.get("relaybp_fallback_retry_unconverged_enabled")):
                continue
            primary_unconverged = _record_int(
                record, "primary_relaybp_unconverged", _record_int(record, "unconverged")
            )
            if primary_unconverged <= 0:
                continue
            details_path = record.get("relaybp_unconverged_details_path")
            if not isinstance(details_path, str) or not details_path:
                continue
            if not Path(details_path).exists():
                continue
            covered = decode_record_covered_chunk_indices(record, manifest_chunks)
            if not covered or not covered <= target_indices:
                continue
            candidates.append(record)
    if not candidates:
        raise RuntimeError(
            f"{condition.label}: no previous compatible RelayBP decode record "
            "with saved primary-unconverged details was found. Run a normal "
            "decode once with RELAYBP_FALLBACK_RETRY_UNCONVERGED=False, then "
            "retry fallback parameters."
        )
    source = max(candidates, key=_record_source_sort_key)
    value = source.get("relaybp_config_hash")
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            "selected RelayBP fallback retry source record has no relaybp_config_hash"
        )
    return value


def _relaybp_details_remaining_unconverged_count(record: dict[str, object]) -> int:
    details_path = record.get("relaybp_unconverged_details_path")
    if not isinstance(details_path, str) or not details_path:
        return 0
    path = Path(details_path)
    if not path.exists():
        return 0
    try:
        details = load_relaybp_unconverged_details(path)
    except Exception:
        return 0
    if int(details.get("schema_version", 0)) != 1:
        return 0
    old_converged_mask = np.asarray(details["old_converged_mask"], dtype=bool)
    return int(np.sum(~old_converged_mask))


def _bposd_retry_source_is_consistent(record: dict[str, object]) -> bool:
    remaining = _relaybp_details_remaining_unconverged_count(record)
    return remaining > 0 and remaining == _record_int(record, "unconverged")


def _bposd_retry_source_is_passthrough(record: dict[str, object]) -> bool:
    return _record_int(record, "unconverged") == 0


def _bposd_retry_source_is_usable(record: dict[str, object]) -> bool:
    return _bposd_retry_source_is_passthrough(record) or _bposd_retry_source_is_consistent(record)


def find_default_bposd_retry_source_hash(
    decode_dir: Path,
    manifest: dict,
    all_tasks: Sequence[DecodeTask],
    condition: Condition,
    config: RelayBPConfig,
) -> str:
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    shot_manifest_hash = str(manifest["manifest_hash"])
    candidates: list[dict[str, object]] = []
    groups = read_all_decode_record_groups(decode_dir, "relaybp")
    for (label, _config_hash), records in groups.items():
        if label != condition.label:
            continue
        for record in records:
            if record.get("decoder", "relaybp") != "relaybp":
                continue
            if record.get("label") != condition.label:
                continue
            if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
                continue
            if record.get("shot_manifest_hash") != shot_manifest_hash:
                continue
            if relaybp_config_from_record(record) != config:
                continue
            if bool(record.get("bposd_retry_unconverged_enabled")):
                continue
            covered = decode_record_covered_chunk_indices(record, manifest_chunks)
            if not covered or not covered <= target_indices:
                continue
            if not _bposd_retry_source_is_usable(record):
                continue
            candidates.append(record)
    if not candidates:
        raise RuntimeError(
            f"{condition.label}: no previous compatible RelayBP decode record "
            "with saved still-unconverged details was found. Run a normal "
            "RelayBP decode with detail saving first, or pass "
            "--bposd-retry-source-hash for a specific source hash."
        )
    source = max(candidates, key=_record_source_sort_key)
    value = source.get("relaybp_config_hash")
    if not isinstance(value, str) or not value:
        raise RuntimeError("selected BPOSD retry source record has no relaybp_config_hash")
    return value


def retry_relaybp_fallback_by_reconstructing_primary(
    source_record: dict[str, object],
    *,
    condition: Condition,
    source_hash: str,
    target_hash: str,
    config: RelayBPConfig,
    relaybp_fallback_config: RelayBPFallbackConfig,
    matrices_path: Path,
    manifest: dict,
    shot_manifest_hash: str,
    decode_batch_size: int,
    progress_seconds: float,
) -> dict[str, object]:
    """Re-run primary RelayBP to recover primary-unconverged rows, then fallback them.

    Older decode records do not persist per-shot primary-unconverged details. In
    that case retry-only cannot avoid the primary pass, but the strong fallback
    is still applied only to rows that primary RelayBP fails to converge.
    """
    manifest_chunks = _manifest_chunks_by_index(manifest)
    tasks = decode_tasks_for_record(source_record, manifest_chunks)
    print(
        f"    RelayBP fallback retry reconstructing primary-unconverged rows: "
        f"chunks={len(tasks)} source_hash={source_hash} target_hash={target_hash}",
        flush=True,
    )
    result = run_decode_batch(
        tasks,
        matrices_path,
        config,
        decoder="relaybp",
        decode_batch_size=decode_batch_size,
        progress_seconds=progress_seconds,
        mip_fallback_config=None,
        relaybp_fallback_config=relaybp_fallback_config,
    )
    result.update(
        {
            "label": condition.label,
            "decoder": "relaybp",
            "relaybp_config_hash": target_hash,
            "decode_schema_version": DECODE_SCHEMA_VERSION,
            "shot_manifest_hash": shot_manifest_hash,
            "relaybp_config": asdict(config),
            "mip_fallback_config": None,
            "relaybp_fallback_config": asdict(relaybp_fallback_config),
            "basis": condition.basis,
            "decoding_mode": condition.decoding_mode,
            "p": condition.p_noise,
            "cycles": condition.cycles,
            "chunk_fingerprints": chunk_fingerprints_for_tasks(tasks, manifest),
            "relaybp_fallback_retry_unconverged_enabled": True,
            "relaybp_fallback_retry_source_hash": source_hash,
            "relaybp_fallback_retry_reconstructed_primary": True,
        }
    )
    retried = _record_int(result, "primary_relaybp_unconverged")
    retry_converged = _record_int(result, "relaybp_fallback_converged")
    retry_unconverged = _record_int(result, "relaybp_fallback_unconverged")
    result["relaybp_fallback_retry_retried"] = retried
    result["relaybp_fallback_retry_converged"] = retry_converged
    result["relaybp_fallback_retry_unconverged"] = retry_unconverged
    result["relaybp_fallback_retry_failures"] = _record_int(
        result, "relaybp_fallback_failures"
    ) + _record_int(result, "u_failures")
    result["relaybp_fallback_retry_observable_mismatches"] = _record_int(
        result, "relaybp_fallback_observable_mismatches"
    ) + _record_int(result, "u_observable_mismatches")
    return result


def retry_relaybp_fallback_for_record(
    source_record: dict[str, object],
    *,
    condition: Condition,
    decode_dir: Path,
    source_hash: str,
    target_hash: str,
    config: RelayBPConfig,
    relaybp_fallback_config: RelayBPFallbackConfig,
    matrices_path: Path,
    manifest: dict,
    shot_manifest_hash: str,
    decode_batch_size: int,
    progress_seconds: float,
) -> dict[str, object]:
    record = {
        key: value for key, value in source_record.items() if not str(key).startswith("_source_")
    }
    source_relay_config = record.get("relaybp_config")
    if not isinstance(source_relay_config, dict) or stable_hash(
        {"relaybp_config": source_relay_config}
    ) != stable_hash({"relaybp_config": asdict(config)}):
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} RelayBP parameters differ "
            "from the current notebook settings. Retry-only fallback only supports "
            "changing the strong RelayBP fallback parameters."
        )

    source_converged = _record_int(record, "converged")
    source_unconverged = _record_int(record, "unconverged")
    primary_unconverged = _record_int(record, "primary_relaybp_unconverged", source_unconverged)
    if record.get("primary_relaybp_converged") is None:
        record["primary_relaybp_converged"] = source_converged
        record["primary_relaybp_unconverged"] = source_unconverged
        record["primary_relaybp_convergence_rate"] = source_converged / max(
            _record_int(record, "num_shots"), 1
        )
        record["primary_relaybp_iterations_total"] = record.get("iterations_total")
        record["primary_relaybp_avg_iterations"] = record.get("avg_iterations")
        record["primary_relaybp_max_iterations"] = record.get("max_iterations")

    def mark_retry_metadata() -> None:
        record["relaybp_config_hash"] = target_hash
        record["relaybp_config"] = asdict(config)
        record["relaybp_fallback_enabled"] = True
        record["relaybp_fallback_config"] = asdict(relaybp_fallback_config)
        record["relaybp_fallback_retry_unconverged_enabled"] = True
        record["relaybp_fallback_retry_source_hash"] = source_hash

    details_path_text = record.get("relaybp_unconverged_details_path")
    if primary_unconverged > 0 and (
        not isinstance(details_path_text, str)
        or not details_path_text
        or not Path(details_path_text).exists()
    ):
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} has "
            f"primary_relaybp_unconverged={primary_unconverged}, but this "
            "record does not contain saved primary-unconverged details. "
            "Retry-only mode will not reconstruct by rerunning primary RelayBP. "
            "Run a normal decode once with RELAYBP_FALLBACK_RETRY_UNCONVERGED=False."
        )

    if primary_unconverged <= 0:
        mark_retry_metadata()
        record["relaybp_fallback_called"] = 0
        record["relaybp_fallback_converged"] = 0
        record["relaybp_fallback_unconverged"] = 0
        record["relaybp_fallback_convergence_rate"] = None
        record["relaybp_fallback_iterations_total"] = 0
        record["relaybp_fallback_avg_iterations"] = None
        record["relaybp_fallback_max_iterations"] = 0
        record["relaybp_fallback_failures"] = 0
        record["relaybp_fallback_observable_mismatches"] = 0
        record["relaybp_fallback_seconds"] = 0.0
        record["relaybp_fallback_retry_retried"] = 0
        record["relaybp_fallback_retry_converged"] = 0
        record["relaybp_fallback_retry_unconverged"] = 0
        record["relaybp_fallback_retry_failures"] = 0
        record["relaybp_fallback_retry_observable_mismatches"] = 0
        return record
    details_path = Path(details_path_text)
    if not details_path.exists():
        raise FileNotFoundError(f"missing RelayBP unconverged details: {details_path}")
    details = load_relaybp_unconverged_details(details_path)
    if int(details["schema_version"]) != 1:
        raise RuntimeError(f"unsupported RelayBP details schema: {details_path}")

    detectors = np.asarray(details["detectors"], dtype=np.uint8)
    observables = np.asarray(details["observables"], dtype=np.uint8)
    old_converged_mask = np.asarray(details["old_converged_mask"], dtype=bool)
    old_failure_mask = np.asarray(details["old_failure_mask"], dtype=bool)
    old_mismatch_counts = np.asarray(details["old_mismatch_counts"], dtype=np.int64)
    old_iterations = np.asarray(details["old_iterations"], dtype=np.int64)
    shot_indices = np.asarray(details["shot_indices"], dtype=np.int64)
    if detectors.shape[0] != primary_unconverged:
        raise RuntimeError(
            f"RelayBP fallback details count mismatch for {condition.label}: "
            f"record={primary_unconverged} details={detectors.shape[0]}"
        )
    if old_iterations.shape[0] != detectors.shape[0]:
        raise RuntimeError(
            f"RelayBP fallback details iteration count mismatch for {condition.label}: "
            f"iterations={old_iterations.shape[0]} details={detectors.shape[0]}"
        )
    if shot_indices.shape[0] != detectors.shape[0]:
        raise RuntimeError(
            f"RelayBP fallback details shot-index count mismatch for {condition.label}: "
            f"shot_indices={shot_indices.shape[0]} details={detectors.shape[0]}"
        )

    runner = build_relaybp_runner(matrices_path, relaybp_fallback_config.relay_config())
    batch_size = int(decode_batch_size)
    if batch_size <= 0 or batch_size > detectors.shape[0]:
        batch_size = int(detectors.shape[0])
    print(
        f"    RelayBP fallback retry primary-unconverged details: shots={detectors.shape[0]} "
        f"source_hash={source_hash} target_hash={target_hash} "
        f"decode_batch_size={batch_size}",
        flush=True,
    )
    retry_t0 = time.monotonic()
    last_progress = retry_t0
    predicted = np.zeros_like(observables, dtype=np.uint8)
    converged_mask = np.zeros(detectors.shape[0], dtype=bool)
    iterations = np.zeros(detectors.shape[0], dtype=np.int64)
    decoded = 0
    for start in range(0, detectors.shape[0], batch_size):
        stop = min(start + batch_size, detectors.shape[0])
        detailed_results = runner.decode_observables_detailed_batch(
            detectors[start:stop].astype(np.uint8, copy=False),
            parallel=True,
            progress_bar=False,
        )
        predicted[start:stop] = np.vstack(
            [np.asarray(result.observables, dtype=np.uint8) for result in detailed_results]
        )
        converged_mask[start:stop] = np.asarray(
            [bool(result.converged) for result in detailed_results], dtype=bool
        )
        iterations[start:stop] = np.asarray(
            [int(result.iterations) for result in detailed_results], dtype=np.int64
        )
        decoded += stop - start
        now = time.monotonic()
        if now - last_progress >= progress_seconds or decoded == detectors.shape[0]:
            print(
                f"    RelayBP fallback retry decoded shots={decoded}/{detectors.shape[0]} "
                f"converged={int(np.sum(converged_mask))}/{decoded}",
                flush=True,
            )
            last_progress = now

    mismatch_mask = predicted != observables.astype(np.uint8, copy=False)
    mismatch_counts = np.sum(mismatch_mask, axis=1, dtype=np.int64)
    failure_mask = np.any(mismatch_mask, axis=1)

    old_failures = int(np.sum(old_failure_mask))
    old_observable_mismatches = int(np.sum(old_mismatch_counts))
    old_converged = int(np.sum(old_converged_mask))
    old_unconverged_mask = ~old_converged_mask
    old_c_iterations = int(np.sum(old_iterations[old_converged_mask]))
    old_u_iterations = int(np.sum(old_iterations[old_unconverged_mask]))
    new_failures = int(np.sum(failure_mask))
    new_observable_mismatches = int(np.sum(mismatch_counts))
    converged = int(np.sum(converged_mask))
    unconverged = int(detectors.shape[0] - converged)
    c_failures = int(np.sum(failure_mask & converged_mask))
    u_failures = int(np.sum(failure_mask & ~converged_mask))
    c_observable_mismatches = int(np.sum(mismatch_counts[converged_mask]))
    u_observable_mismatches = int(np.sum(mismatch_counts[~converged_mask]))
    iterations_total = int(np.sum(iterations))
    retry_seconds = float(time.monotonic() - retry_t0)

    _add_record_int(record, "failures", new_failures - old_failures)
    _add_record_int(
        record,
        "observable_mismatches",
        new_observable_mismatches - old_observable_mismatches,
    )
    _add_record_int(record, "converged", converged - old_converged)
    _add_record_int(record, "unconverged", -(converged - old_converged))
    _add_record_int(record, "c_failures", c_failures)
    _add_record_int(record, "u_failures", u_failures - old_failures)
    _add_record_int(record, "c_observable_mismatches", c_observable_mismatches)
    _add_record_int(
        record,
        "u_observable_mismatches",
        u_observable_mismatches - old_observable_mismatches,
    )
    _add_record_int(record, "iterations_total", iterations_total)
    record["max_iterations"] = max(_record_int(record, "max_iterations"), int(np.max(iterations)))
    if unconverged:
        _add_record_int(
            record,
            "u_iterations_total",
            int(np.sum(iterations[~converged_mask])) - old_u_iterations,
        )
        record["u_max_iterations"] = max(
            _record_int(record, "u_max_iterations"),
            int(np.max(iterations[~converged_mask])),
        )
    else:
        _add_record_int(record, "u_iterations_total", -old_u_iterations)
    if converged:
        _add_record_int(
            record,
            "c_iterations_total",
            int(np.sum(iterations[converged_mask])) - old_c_iterations,
        )
        record["c_max_iterations"] = max(
            _record_int(record, "c_max_iterations"),
            int(np.max(iterations[converged_mask])),
        )

    mark_retry_metadata()
    record["relaybp_fallback_called"] = primary_unconverged
    record["relaybp_fallback_converged"] = converged
    record["relaybp_fallback_unconverged"] = unconverged
    record["relaybp_fallback_convergence_rate"] = converged / max(primary_unconverged, 1)
    record["relaybp_fallback_iterations_total"] = iterations_total
    record["relaybp_fallback_avg_iterations"] = iterations_total / max(primary_unconverged, 1)
    record["relaybp_fallback_max_iterations"] = int(np.max(iterations)) if len(iterations) else 0
    record["relaybp_fallback_failures"] = c_failures
    record["relaybp_fallback_observable_mismatches"] = c_observable_mismatches
    record["relaybp_fallback_seconds"] = retry_seconds
    record["relaybp_fallback_retry_retried"] = primary_unconverged
    record["relaybp_fallback_retry_converged"] = converged
    record["relaybp_fallback_retry_unconverged"] = unconverged
    record["relaybp_fallback_retry_failures"] = new_failures
    record["relaybp_fallback_retry_observable_mismatches"] = new_observable_mismatches
    chunk_fingerprints = record.get("chunk_fingerprints")
    if isinstance(chunk_fingerprints, list):
        record_id = decode_record_id_from_fingerprints(chunk_fingerprints)
    else:
        record_id = stable_hash(
            {
                "source_hash": source_hash,
                "source_details_path": str(details_path),
                "num_shots": record.get("num_shots"),
            }
        )[:12]
    target_details_path = relaybp_unconverged_details_path(
        decode_dir, condition, target_hash, record_id
    )
    if target_details_path.resolve() == details_path.resolve():
        raise RuntimeError(f"Refusing to overwrite source RelayBP details: {details_path}")
    save_relaybp_unconverged_details(
        target_details_path,
        metadata={
            "detail_type": "relaybp_fallback_retry_unconverged",
            "label": condition.label,
            "source_hash": source_hash,
            "target_hash": target_hash,
            "source_details_path": str(details_path),
            "record_id": record_id,
            "shot_manifest_hash": shot_manifest_hash,
            "chunk_fingerprints": chunk_fingerprints,
            "num_primary_unconverged": primary_unconverged,
            "num_fallback_unconverged": unconverged,
        },
        relaybp_config=relaybp_fallback_config.relay_config(),
        shot_indices=shot_indices,
        detectors=detectors,
        observables=observables,
        old_converged_mask=converged_mask,
        old_failure_mask=failure_mask,
        old_mismatch_counts=mismatch_counts,
        old_iterations=iterations,
    )
    persisted = load_relaybp_unconverged_details(target_details_path)
    persisted_converged = np.asarray(persisted["old_converged_mask"], dtype=bool)
    persisted_unconverged = int(np.sum(~persisted_converged))
    if persisted_converged.shape[0] != primary_unconverged:
        raise RuntimeError(
            f"Persisted RelayBP fallback details count mismatch for {condition.label}: "
            f"record={primary_unconverged} details={persisted_converged.shape[0]}"
        )
    if persisted_unconverged != unconverged:
        raise RuntimeError(
            f"Persisted RelayBP fallback mask mismatch for {condition.label}: "
            f"record={unconverged} details={persisted_unconverged}"
        )
    record["relaybp_unconverged_details_path"] = str(target_details_path)
    record["logical_error_rate"] = _record_int(record, "failures") / max(
        _record_int(record, "num_shots"), 1
    )
    return record


def retry_unsolved_mip_for_record(
    source_record: dict[str, object],
    *,
    condition: Condition,
    decode_dir: Path,
    source_hash: str,
    target_hash: str,
    config: RelayBPConfig,
    mip_fallback_config: MIPFallbackConfig,
    matrices_path: Path,
    progress_seconds: float,
) -> dict[str, object]:
    record = {
        key: value for key, value in source_record.items() if not str(key).startswith("_source_")
    }
    if not bool(record.get("mip_fallback_enabled")):
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} was not produced with MIP fallback"
        )
    source_relay_config = record.get("relaybp_config")
    if not isinstance(source_relay_config, dict) or stable_hash(
        {"relaybp_config": source_relay_config}
    ) != stable_hash({"relaybp_config": asdict(config)}):
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} RelayBP parameters differ "
            "from the current notebook settings. Retry mode only supports changing "
            "MIP fallback parameters; rerun normal decode if RelayBP parameters changed."
        )
    old_unsolved_count = _record_int(record, "mip_fallback_unsolved")
    details_path_text = record.get("mip_fallback_details_path")
    if old_unsolved_count <= 0:
        record["relaybp_config_hash"] = target_hash
        record["relaybp_config"] = asdict(config)
        record["mip_fallback_config"] = asdict(mip_fallback_config)
        record["mip_fallback_mode"] = "deferred_parallel_retry_unsolved"
        record["mip_retry_unsolved_enabled"] = True
        record["mip_retry_source_hash"] = source_hash
        record["mip_retry_retried"] = 0
        record["mip_retry_solved"] = 0
        record["mip_retry_unsolved"] = 0
        record["mip_retry_failures"] = 0
        record["mip_retry_observable_mismatches"] = 0
        record["mip_fallback_details_path"] = None
        return record
    if not isinstance(details_path_text, str) or not details_path_text:
        raise RuntimeError(
            f"{condition.label} source record has mip_fallback_unsolved="
            f"{old_unsolved_count}, but no mip_fallback_details_path. "
            "It was likely produced before retry details were supported; rerun "
            "RelayBP+MIP once to create details."
        )
    details_path = Path(details_path_text)
    if not details_path.exists():
        raise FileNotFoundError(f"missing MIP fallback details: {details_path}")
    details = load_mip_fallback_details(details_path)
    if int(details["schema_version"]) != 1:
        raise RuntimeError(f"unsupported MIP details schema: {details_path}")

    detectors = np.asarray(details["detectors"], dtype=np.uint8)
    observables = np.asarray(details["observables"], dtype=np.uint8)
    shot_indices = np.asarray(details["shot_indices"], dtype=np.int64)
    old_failure_mask = np.asarray(details["old_failure_mask"], dtype=bool)
    old_mismatch_counts = np.asarray(details["old_mismatch_counts"], dtype=np.int64)
    if detectors.shape[0] != old_unsolved_count:
        raise RuntimeError(
            f"MIP details count mismatch for {condition.label}: "
            f"record={old_unsolved_count} details={detectors.shape[0]}"
        )

    check_matrix, observables_matrix, error_priors = load_check_matrices(matrices_path)
    predicted = np.zeros_like(observables, dtype=np.uint8)
    (
        predicted,
        actual_unsolved_mask,
        count_unsolved_mask,
        mip_stats,
        _mip_info_by_index,
    ) = run_deferred_mip_fallback(
        detectors,
        predicted,
        np.arange(detectors.shape[0], dtype=np.int64),
        check_matrix.tocsr().astype(np.uint8),
        observables_matrix.tocsr().astype(np.uint8),
        error_priors,
        mip_fallback_config,
        progress_seconds=progress_seconds,
    )

    mismatch_mask = predicted != observables.astype(np.uint8, copy=False)
    mismatch_counts = np.sum(mismatch_mask, axis=1, dtype=np.int64)
    if np.any(count_unsolved_mask):
        mismatch_counts[count_unsolved_mask] = observables.shape[1]
    failure_mask = np.any(mismatch_mask, axis=1) | count_unsolved_mask

    old_failures = int(np.sum(old_failure_mask))
    old_observable_mismatches = int(np.sum(old_mismatch_counts))
    new_failures = int(np.sum(failure_mask))
    new_observable_mismatches = int(np.sum(mismatch_counts))
    delta_failures = new_failures - old_failures
    delta_observable_mismatches = new_observable_mismatches - old_observable_mismatches

    for key in ("failures", "u_failures", "mip_fallback_failures"):
        _add_record_int(record, key, delta_failures)
    for key in (
        "observable_mismatches",
        "u_observable_mismatches",
        "mip_fallback_observable_mismatches",
    ):
        _add_record_int(record, key, delta_observable_mismatches)

    record["relaybp_config_hash"] = target_hash
    record["relaybp_config"] = asdict(config)
    record["mip_fallback_config"] = asdict(mip_fallback_config)
    record["mip_fallback_mode"] = "deferred_parallel_retry_unsolved"
    record["mip_fallback_workers"] = int(mip_stats["workers"])
    record["mip_fallback_solved"] = _record_int(record, "mip_fallback_solved") + int(
        mip_stats["solved"]
    )
    record["mip_fallback_unsolved"] = int(mip_stats["unsolved"])
    record["mip_fallback_seconds"] = _record_float(record, "mip_fallback_seconds") + float(
        mip_stats["seconds"]
    )
    record["logical_error_rate"] = _record_int(record, "failures") / max(
        _record_int(record, "num_shots"), 1
    )
    record["mip_retry_unsolved_enabled"] = True
    record["mip_retry_source_hash"] = source_hash
    record["mip_retry_retried"] = int(detectors.shape[0])
    record["mip_retry_solved"] = int(mip_stats["solved"])
    record["mip_retry_unsolved"] = int(mip_stats["unsolved"])
    record["mip_retry_failures"] = int(new_failures)
    record["mip_retry_observable_mismatches"] = int(new_observable_mismatches)

    chunk_fingerprints = record.get("chunk_fingerprints")
    if isinstance(chunk_fingerprints, list):
        record_id = decode_record_id_from_fingerprints(chunk_fingerprints)
    else:
        record_id = stable_hash({"source_hash": source_hash, "details_path": str(details_path)})[
            :12
        ]
    target_details_path = mip_fallback_details_path(decode_dir, condition, target_hash, record_id)
    if np.any(actual_unsolved_mask):
        remaining = np.flatnonzero(actual_unsolved_mask).astype(np.int64)
        save_mip_fallback_details(
            target_details_path,
            metadata={
                "detail_type": "mip_retry_unsolved",
                "label": condition.label,
                "relaybp_config_hash": target_hash,
                "mip_retry_source_hash": source_hash,
                "source_details_path": str(details_path),
                "record_id": record_id,
                "chunk_fingerprints": chunk_fingerprints,
            },
            config=mip_fallback_config,
            shot_indices=shot_indices[remaining],
            detectors=detectors[remaining],
            observables=observables[remaining],
            old_failure_mask=failure_mask[remaining],
            old_mismatch_counts=mismatch_counts[remaining],
        )
        record["mip_fallback_details_path"] = str(target_details_path)
    else:
        record["mip_fallback_details_path"] = None
    return record


def run_mip_retry_unsolved_condition(
    condition: Condition,
    sample_dir: Path,
    decode_dir: Path,
    shots: int,
    shot_chunk: int,
    config: RelayBPConfig,
    meta: dict,
    *,
    force: bool,
    progress_seconds: float,
    mip_fallback_config: MIPFallbackConfig,
    source_hash: str | None,
) -> dict[str, object]:
    _ = (shots, shot_chunk)
    if not mip_fallback_config.enabled:
        raise RuntimeError("--mip-retry-unsolved requires --mip-fallback")
    manifest = load_shots_manifest(sample_dir, condition)
    shot_manifest_hash = str(manifest["manifest_hash"])
    all_tasks = expected_decode_tasks(condition, manifest, meta)
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    resolved_source_hash = str(source_hash or "").strip()
    if not resolved_source_hash:
        resolved_source_hash = find_default_mip_retry_source_hash(
            decode_dir,
            manifest,
            all_tasks,
            condition,
            config,
            mip_fallback_config,
        )
    target_hash = relaybp_mip_retry_config_hash(config, mip_fallback_config, resolved_source_hash)
    jsonl_path = decode_jsonl_path(decode_dir, condition, "relaybp", target_hash)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if force and jsonl_path.exists():
        jsonl_path.unlink()

    source_jsonl_path = decode_jsonl_path(decode_dir, condition, "relaybp", resolved_source_hash)
    source_records_all = read_decode_records(
        source_jsonl_path, decoder="relaybp", config_hash=resolved_source_hash
    )
    if not source_records_all:
        raise RuntimeError(
            f"No source decode records found for {condition.label} "
            f"source_hash={resolved_source_hash}: {source_jsonl_path}"
        )

    source_records: list[dict[str, object]] = []
    skipped_no_unsolved = 0
    skipped_no_details = 0
    skipped_wrong_config = 0
    for record in source_records_all:
        if record.get("label") != condition.label:
            continue
        try:
            source_config = relaybp_config_from_record(record)
        except Exception:
            skipped_wrong_config += 1
            continue
        if source_config != config:
            skipped_wrong_config += 1
            continue
        if record.get("shot_manifest_hash") != shot_manifest_hash:
            skipped_wrong_config += 1
            continue
        covered = decode_record_covered_chunk_indices(record, manifest_chunks)
        if not covered or not covered <= target_indices:
            skipped_wrong_config += 1
            continue
        if bool(record.get("relaybp_fallback_enabled")):
            skipped_wrong_config += 1
            continue
        if bool(record.get("relaybp_fallback_retry_unconverged_enabled")):
            skipped_wrong_config += 1
            continue
        if not bool(record.get("mip_fallback_enabled")):
            skipped_no_details += 1
            continue
        if _record_int(record, "mip_fallback_unsolved") <= 0:
            source_records.append(record)
            skipped_no_unsolved += 1
            continue
        details_path = record.get("mip_fallback_details_path")
        if not isinstance(details_path, str) or not details_path or not Path(details_path).exists():
            skipped_no_details += 1
            continue
        source_records.append(record)

    retryable = [
        record for record in source_records if _record_int(record, "mip_fallback_unsolved") > 0
    ]
    if not retryable:
        raise RuntimeError(
            f"{condition.label} source_hash={resolved_source_hash} has no retryable "
            "MIP-unsolved records with saved mip_fallback_details_path. "
            f"skipped_no_details={skipped_no_details} "
            f"skipped_wrong_config={skipped_wrong_config}"
        )

    target_records = (
        [] if force else read_decode_records(jsonl_path, decoder="relaybp", config_hash=target_hash)
    )
    existing_record_ids: set[str] = set()
    for record in target_records:
        chunk_fingerprints = record.get("chunk_fingerprints")
        if isinstance(chunk_fingerprints, list):
            existing_record_ids.add(decode_record_id_from_fingerprints(chunk_fingerprints))

    to_process: list[dict[str, object]] = []
    for record in source_records:
        chunk_fingerprints = record.get("chunk_fingerprints")
        if isinstance(chunk_fingerprints, list):
            record_id = decode_record_id_from_fingerprints(chunk_fingerprints)
        else:
            record_id = stable_hash(
                {
                    "source_hash": resolved_source_hash,
                    "source_path": str(source_jsonl_path),
                    "num_shots": record.get("num_shots"),
                    "mip_unsolved": record.get("mip_fallback_unsolved"),
                }
            )[:12]
        if record_id not in existing_record_ids:
            to_process.append(record)

    source_chunks = sum(int(record.get("num_chunks", 0) or 0) for record in source_records)
    source_shots = sum(int(record.get("num_shots", 0) or 0) for record in source_records)
    retry_unsolved = sum(_record_int(record, "mip_fallback_unsolved") for record in to_process)
    if to_process:
        print(
            f"  MIP retry {condition.label}: source_hash={resolved_source_hash} "
            f"source_records={len(source_records)} source_chunks={source_chunks} "
            f"source_shots={source_shots} retry_records={len(to_process)} "
            f"unsolved_to_retry={retry_unsolved}",
            flush=True,
        )
        if skipped_no_unsolved:
            print(
                f"    source records with no MIP-unsolved shots are copied: {skipped_no_unsolved}",
                flush=True,
            )
        if skipped_no_details or skipped_wrong_config:
            print(
                f"    skipped records: no_details={skipped_no_details} "
                f"wrong_config={skipped_wrong_config}",
                flush=True,
            )
        matrices_path = artifact_paths(sample_dir, condition)["matrices"]
        with jsonl_path.open("a", encoding="utf-8") as fout:
            for source_record in to_process:
                retried_record = retry_unsolved_mip_for_record(
                    source_record,
                    condition=condition,
                    decode_dir=decode_dir,
                    source_hash=resolved_source_hash,
                    target_hash=target_hash,
                    config=config,
                    mip_fallback_config=mip_fallback_config,
                    matrices_path=matrices_path,
                    progress_seconds=progress_seconds,
                )
                retried_record["decode_schema_version"] = DECODE_SCHEMA_VERSION
                retried_record["shot_manifest_hash"] = shot_manifest_hash
                fout.write(json.dumps(retried_record, sort_keys=True) + "\n")
                fout.flush()
    else:
        print(
            f"  MIP retry {condition.label}: cached retry records present; "
            f"source_hash={resolved_source_hash}",
            flush=True,
        )

    all_records = read_decode_records(jsonl_path, decoder="relaybp", config_hash=target_hash)
    if not all_records:
        raise RuntimeError(
            f"MIP retry produced no records for {condition.label} target_hash={target_hash}"
        )
    return aggregate_decode_records(
        condition,
        meta,
        all_records,
        jsonl_path,
        config,
        target_hash,
        shot_manifest_hash,
        shots_manifest_path(sample_dir, condition),
    )


def run_relaybp_fallback_retry_condition(
    condition: Condition,
    sample_dir: Path,
    decode_dir: Path,
    shots: int,
    shot_chunk: int,
    config: RelayBPConfig,
    meta: dict,
    *,
    force: bool,
    progress_seconds: float,
    decode_batch_size: int,
    relaybp_fallback_config: RelayBPFallbackConfig,
    source_hash: str | None,
) -> dict[str, object]:
    _ = (shots, shot_chunk)
    if not relaybp_fallback_config.enabled:
        raise RuntimeError("--relaybp-fallback-retry-unconverged requires --relaybp-fallback")
    manifest = load_shots_manifest(sample_dir, condition)
    shot_manifest_hash = str(manifest["manifest_hash"])
    all_tasks = expected_decode_tasks(condition, manifest, meta)
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    resolved_source_hash = str(source_hash or "").strip()
    if not resolved_source_hash:
        resolved_source_hash = find_default_relaybp_fallback_retry_source_hash(
            decode_dir,
            manifest,
            all_tasks,
            condition,
            config,
        )
    target_hash = relaybp_fallback_retry_config_hash(
        config, relaybp_fallback_config, resolved_source_hash
    )
    jsonl_path = decode_jsonl_path(decode_dir, condition, "relaybp", target_hash)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if force and jsonl_path.exists():
        jsonl_path.unlink()

    source_jsonl_path = decode_jsonl_path(decode_dir, condition, "relaybp", resolved_source_hash)
    source_records_all = read_decode_records(
        source_jsonl_path,
        decoder="relaybp",
        config_hash=resolved_source_hash,
    )
    if not source_records_all:
        raise RuntimeError(
            f"No source decode records found for {condition.label} "
            f"source_hash={resolved_source_hash}: {source_jsonl_path}"
        )

    source_records: list[dict[str, object]] = []
    skipped_wrong_config = 0
    skipped_wrong_coverage = 0
    for record in source_records_all:
        if record.get("label") != condition.label:
            continue
        if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
            skipped_wrong_config += 1
            continue
        try:
            source_config = relaybp_config_from_record(record)
        except Exception:
            skipped_wrong_config += 1
            continue
        if source_config != config:
            skipped_wrong_config += 1
            continue
        covered = decode_record_covered_chunk_indices(record, manifest_chunks)
        if not covered or not covered <= target_indices:
            skipped_wrong_coverage += 1
            continue
        if bool(record.get("relaybp_fallback_retry_unconverged_enabled")):
            skipped_wrong_config += 1
            continue
        source_records.append(record)

    if not source_records:
        raise RuntimeError(
            f"{condition.label} source_hash={resolved_source_hash} has no usable "
            "source records with compatible primary RelayBP config and chunk coverage. "
            f"skipped_wrong_config={skipped_wrong_config} "
            f"skipped_wrong_coverage={skipped_wrong_coverage}"
        )

    target_records = (
        [] if force else read_decode_records(jsonl_path, decoder="relaybp", config_hash=target_hash)
    )
    existing_record_ids: set[str] = set()
    for record in target_records:
        chunk_fingerprints = record.get("chunk_fingerprints")
        if isinstance(chunk_fingerprints, list):
            existing_record_ids.add(decode_record_id_from_fingerprints(chunk_fingerprints))

    to_process: list[dict[str, object]] = []
    for record in source_records:
        chunk_fingerprints = record.get("chunk_fingerprints")
        if isinstance(chunk_fingerprints, list):
            record_id = decode_record_id_from_fingerprints(chunk_fingerprints)
        else:
            record_id = stable_hash(
                {
                    "source_hash": resolved_source_hash,
                    "source_path": str(source_jsonl_path),
                    "num_shots": record.get("num_shots"),
                    "mip_unsolved": record.get("mip_fallback_unsolved"),
                }
            )[:12]
        if record_id not in existing_record_ids:
            to_process.append(record)

    source_chunks = sum(int(record.get("num_chunks", 0) or 0) for record in source_records)
    source_shots = sum(int(record.get("num_shots", 0) or 0) for record in source_records)
    retry_unconverged = sum(
        _record_int(record, "primary_relaybp_unconverged", _record_int(record, "unconverged"))
        for record in to_process
    )
    if to_process:
        print(
            f"  RelayBP fallback retry {condition.label}: "
            f"source_hash={resolved_source_hash} target_hash={target_hash} "
            f"source_records={len(source_records)} source_chunks={source_chunks} "
            f"source_shots={source_shots} retry_records={len(to_process)} "
            f"primary_unconverged_to_retry={retry_unconverged}",
            flush=True,
        )
        if skipped_wrong_config or skipped_wrong_coverage:
            print(
                f"    skipped source records: wrong_config={skipped_wrong_config} "
                f"wrong_coverage={skipped_wrong_coverage}",
                flush=True,
            )
        matrices_path = artifact_paths(sample_dir, condition)["matrices"]
        with jsonl_path.open("a", encoding="utf-8") as fout:
            for source_record in to_process:
                retried_record = retry_relaybp_fallback_for_record(
                    source_record,
                    condition=condition,
                    decode_dir=decode_dir,
                    source_hash=resolved_source_hash,
                    target_hash=target_hash,
                    config=config,
                    relaybp_fallback_config=relaybp_fallback_config,
                    matrices_path=matrices_path,
                    manifest=manifest,
                    shot_manifest_hash=shot_manifest_hash,
                    decode_batch_size=decode_batch_size,
                    progress_seconds=progress_seconds,
                )
                retried_record["decode_schema_version"] = DECODE_SCHEMA_VERSION
                retried_record["shot_manifest_hash"] = shot_manifest_hash
                fout.write(json.dumps(retried_record, sort_keys=True) + "\n")
                fout.flush()
    else:
        print(
            f"  RelayBP fallback retry {condition.label}: cached retry records present; "
            f"source_hash={resolved_source_hash} target_hash={target_hash}",
            flush=True,
        )

    all_records = read_decode_records(jsonl_path, decoder="relaybp", config_hash=target_hash)
    if not all_records:
        raise RuntimeError(
            f"RelayBP fallback retry produced no records for "
            f"{condition.label} target_hash={target_hash}"
        )
    return aggregate_decode_records(
        condition,
        meta,
        all_records,
        jsonl_path,
        config,
        target_hash,
        shot_manifest_hash,
        shots_manifest_path(sample_dir, condition),
    )


def retry_bposd_for_record(
    source_record: dict[str, object],
    *,
    condition: Condition,
    decode_dir: Path,
    source_hash: str,
    target_hash: str,
    config: RelayBPConfig,
    bposd_config: BpOsdRetryConfig,
    matrices_path: Path,
    manifest: dict,
    shot_manifest_hash: str,
    progress_seconds: float,
    bposd_workers: int,
) -> dict[str, object]:
    record = {
        key: value for key, value in source_record.items() if not str(key).startswith("_source_")
    }
    source_relay_config = record.get("relaybp_config")
    if not isinstance(source_relay_config, dict) or stable_hash(
        {"relaybp_config": source_relay_config}
    ) != stable_hash({"relaybp_config": asdict(config)}):
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} RelayBP parameters differ "
            "from the current notebook settings. BPOSD retry-only requires the "
            "same primary RelayBP parameters as the source record."
        )

    primary_unconverged = _record_int(
        record, "primary_relaybp_unconverged", _record_int(record, "unconverged")
    )
    details_path_text = record.get("relaybp_unconverged_details_path")
    if not isinstance(details_path_text, str) or not details_path_text:
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} has no saved "
            "RelayBP unconverged details path."
        )
    details_path = Path(details_path_text)
    if not details_path.exists():
        raise FileNotFoundError(f"missing RelayBP unconverged details: {details_path}")
    details = load_relaybp_unconverged_details(details_path)
    if int(details["schema_version"]) != 1:
        raise RuntimeError(f"unsupported RelayBP details schema: {details_path}")

    shot_indices = np.asarray(details["shot_indices"], dtype=np.int64)
    detectors = np.asarray(details["detectors"], dtype=np.uint8)
    observables = np.asarray(details["observables"], dtype=np.uint8)
    old_converged_mask = np.asarray(details["old_converged_mask"], dtype=bool)
    old_failure_mask = np.asarray(details["old_failure_mask"], dtype=bool)
    old_mismatch_counts = np.asarray(details["old_mismatch_counts"], dtype=np.int64)
    old_iterations = np.asarray(details["old_iterations"], dtype=np.int64)
    if detectors.shape[0] != primary_unconverged:
        raise RuntimeError(
            f"BPOSD retry details count mismatch for {condition.label}: "
            f"record primary_unconverged={primary_unconverged} "
            f"details={detectors.shape[0]}"
        )
    if any(
        array.shape[0] != detectors.shape[0]
        for array in (
            shot_indices,
            observables,
            old_converged_mask,
            old_failure_mask,
            old_mismatch_counts,
            old_iterations,
        )
    ):
        raise RuntimeError(
            f"BPOSD retry details array shape mismatch for {condition.label}: "
            f"details={details_path}"
        )

    target_mask = ~old_converged_mask
    retry_indices = np.flatnonzero(target_mask)
    source_unconverged = _record_int(record, "unconverged")
    if int(len(retry_indices)) != source_unconverged:
        raise RuntimeError(
            f"{condition.label} source_hash={source_hash} has "
            f"record unconverged={source_unconverged}, but saved details have "
            f"old_converged_mask=false count={len(retry_indices)}. Refusing to "
            "write a BPOSD retry record with ambiguous accounting."
        )
    chunk_fingerprints = record.get("chunk_fingerprints")
    if isinstance(chunk_fingerprints, list):
        record_id = decode_record_id_from_fingerprints(chunk_fingerprints)
    else:
        record_id = stable_hash(
            {
                "source_hash": source_hash,
                "details_path": str(details_path),
                "num_shots": record.get("num_shots"),
            }
        )[:12]
    retry_details_path = bposd_retry_details_path(decode_dir, condition, target_hash, record_id)

    def mark_retry_metadata() -> None:
        record["relaybp_config_hash"] = target_hash
        record["relaybp_config"] = asdict(config)
        record["bposd_retry_unconverged_enabled"] = True
        record["bposd_retry_source_hash"] = source_hash
        record["bposd_retry_config"] = asdict(bposd_config)
        record["bposd_retry_details_path"] = str(retry_details_path)

    if not len(retry_indices):
        mark_retry_metadata()
        record["bposd_retry_retried"] = 0
        record["bposd_retry_syndrome_matched"] = 0
        record["bposd_retry_logical_correct"] = 0
        record["bposd_retry_logical_failures"] = 0
        record["bposd_retry_unresolved"] = 0
        record["bposd_retry_observable_mismatches"] = 0
        record["bposd_retry_seconds"] = 0.0
        return record

    effective_workers = max(1, min(int(bposd_workers), int(len(retry_indices))))
    print(
        f"    BPOSD retry saved-unconverged details: shots={len(retry_indices)} "
        f"source_hash={source_hash} target_hash={target_hash} "
        f"osd_method={bposd_config.osd_method} osd_order={bposd_config.osd_order} "
        f"workers={effective_workers}",
        flush=True,
    )
    retry_t0 = time.monotonic()
    last_progress = retry_t0
    syndrome_matched = np.zeros(len(retry_indices), dtype=bool)
    logical_correct = np.zeros(len(retry_indices), dtype=bool)
    mismatch_counts = np.zeros(len(retry_indices), dtype=np.int64)
    correction_weights = np.zeros(len(retry_indices), dtype=np.int64)
    decode_seconds = np.zeros(len(retry_indices), dtype=np.float64)
    error_text: list[str] = ["" for _ in range(len(retry_indices))]

    def store_bposd_result(result: dict[str, object]) -> None:
        local_index = int(result["local_index"])
        syndrome_matched[local_index] = bool(result["syndrome_matched"])
        logical_correct[local_index] = bool(result["logical_correct"])
        mismatch_counts[local_index] = int(result["mismatch_count"])
        correction_weights[local_index] = int(result["correction_weight"])
        decode_seconds[local_index] = float(result["decode_seconds"])
        error_text[local_index] = str(result["error_text"])

    def maybe_print_bposd_progress(completed: int, *, force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if force or now - last_progress >= progress_seconds:
            print(
                f"    BPOSD decoded shots={completed}/{len(retry_indices)} "
                f"syndrome_matched={int(np.sum(syndrome_matched))} "
                f"logical_correct={int(np.sum(logical_correct))}",
                flush=True,
            )
            last_progress = now

    if effective_workers == 1:
        bposd_decoder, check_matrix, observables_matrix = build_bposd_retry_decoder(
            matrices_path, bposd_config
        )
        for local_index, detail_index in enumerate(retry_indices):
            result = decode_bposd_retry_row(
                bposd_decoder,
                check_matrix,
                observables_matrix,
                local_index=int(local_index),
                detail_index=int(detail_index),
                syndrome=detectors[detail_index].astype(np.uint8, copy=False),
                observed=observables[detail_index].astype(np.uint8, copy=False),
                old_mismatch_count=int(old_mismatch_counts[detail_index]),
            )
            store_bposd_result(result)
            maybe_print_bposd_progress(
                local_index + 1,
                force=local_index + 1 == len(retry_indices),
            )
    else:
        tasks = [
            (
                int(local_index),
                int(detail_index),
                detectors[detail_index].astype(np.uint8, copy=True),
                observables[detail_index].astype(np.uint8, copy=True),
                int(old_mismatch_counts[detail_index]),
            )
            for local_index, detail_index in enumerate(retry_indices)
        ]
        ctx = mp.get_context("spawn")
        completed = 0
        with ctx.Pool(
            processes=effective_workers,
            initializer=init_bposd_retry_worker,
            initargs=(str(matrices_path), asdict(bposd_config)),
        ) as pool:
            for result in pool.imap_unordered(
                run_bposd_retry_worker_task,
                tasks,
                chunksize=1,
            ):
                store_bposd_result(result)
                completed += 1
                maybe_print_bposd_progress(
                    completed,
                    force=completed == len(retry_indices),
                )

    unresolved_mask = ~syndrome_matched
    retry_failure_mask = np.where(
        syndrome_matched,
        mismatch_counts > 0,
        old_failure_mask[retry_indices],
    )
    retry_mismatch_counts = np.where(
        syndrome_matched,
        mismatch_counts,
        old_mismatch_counts[retry_indices],
    ).astype(np.int64, copy=False)

    old_failures = int(np.sum(old_failure_mask[retry_indices]))
    old_observable_mismatches = int(np.sum(old_mismatch_counts[retry_indices]))
    new_failures = int(np.sum(retry_failure_mask))
    new_observable_mismatches = int(np.sum(retry_mismatch_counts))
    solved = int(np.sum(syndrome_matched))
    unresolved = int(np.sum(unresolved_mask))
    solved_failures = int(np.sum(retry_failure_mask[syndrome_matched]))
    solved_observable_mismatches = int(np.sum(retry_mismatch_counts[syndrome_matched]))
    unresolved_failures = int(np.sum(retry_failure_mask[unresolved_mask]))
    unresolved_observable_mismatches = int(np.sum(retry_mismatch_counts[unresolved_mask]))

    _add_record_int(record, "failures", new_failures - old_failures)
    _add_record_int(
        record,
        "observable_mismatches",
        new_observable_mismatches - old_observable_mismatches,
    )
    _add_record_int(record, "converged", solved)
    _add_record_int(record, "unconverged", -solved)
    _add_record_int(record, "c_failures", solved_failures)
    _add_record_int(record, "c_observable_mismatches", solved_observable_mismatches)
    _add_record_int(record, "u_failures", unresolved_failures - old_failures)
    _add_record_int(
        record,
        "u_observable_mismatches",
        unresolved_observable_mismatches - old_observable_mismatches,
    )

    target_iterations = old_iterations[retry_indices]
    solved_iterations = target_iterations[syndrome_matched]
    unresolved_iterations = target_iterations[unresolved_mask]
    if solved:
        _add_record_int(record, "c_iterations_total", int(np.sum(solved_iterations)))
        record["c_max_iterations"] = max(
            _record_int(record, "c_max_iterations"),
            int(np.max(solved_iterations)),
        )
    _add_record_int(
        record,
        "u_iterations_total",
        int(np.sum(unresolved_iterations)) - int(np.sum(target_iterations)),
    )
    record["u_max_iterations"] = int(np.max(unresolved_iterations)) if unresolved else 0
    record["logical_error_rate"] = _record_int(record, "failures") / max(
        _record_int(record, "num_shots"), 1
    )
    if _record_int(record, "unconverged") == 0:
        # Once all shots converge, their iteration statistics equal the totals.
        record["c_failures"] = _record_int(record, "failures")
        record["c_observable_mismatches"] = _record_int(record, "observable_mismatches")
        if record.get("iterations_total") is not None:
            record["c_iterations_total"] = _record_int(record, "iterations_total")
        if record.get("max_iterations") is not None:
            record["c_max_iterations"] = _record_int(record, "max_iterations")
        record["u_failures"] = 0
        record["u_observable_mismatches"] = 0
        record["u_iterations_total"] = 0
        record["u_max_iterations"] = 0

    retry_seconds = float(time.monotonic() - retry_t0)
    mark_retry_metadata()
    record["bposd_retry_retried"] = int(len(retry_indices))
    record["bposd_retry_syndrome_matched"] = solved
    record["bposd_retry_logical_correct"] = int(np.sum(logical_correct))
    record["bposd_retry_logical_failures"] = solved_failures
    record["bposd_retry_unresolved"] = unresolved
    record["bposd_retry_observable_mismatches"] = new_observable_mismatches
    record["bposd_retry_seconds"] = retry_seconds

    metadata = {
        "detail_type": "bposd_retry_saved_relaybp_unconverged",
        "label": condition.label,
        "source_hash": source_hash,
        "target_hash": target_hash,
        "shot_manifest_hash": shot_manifest_hash,
        "record_id": record_id,
        "num_retried": int(len(retry_indices)),
    }
    save_bposd_retry_details(
        retry_details_path,
        metadata=metadata,
        bposd_config=bposd_config,
        source_hash=source_hash,
        shot_indices=shot_indices[retry_indices],
        source_detail_indices=retry_indices.astype(np.int64),
        syndrome_matched=syndrome_matched,
        logical_correct=logical_correct,
        mismatch_counts=retry_mismatch_counts,
        correction_weights=correction_weights,
        decode_seconds=decode_seconds,
        error_text=error_text,
    )
    return record


def run_bposd_retry_condition(
    condition: Condition,
    sample_dir: Path,
    decode_dir: Path,
    shots: int,
    shot_chunk: int,
    config: RelayBPConfig,
    meta: dict,
    *,
    force: bool,
    progress_seconds: float,
    bposd_config: BpOsdRetryConfig,
    bposd_workers: int,
    source_hash: str | None,
) -> dict[str, object]:
    _ = (shots, shot_chunk)
    if not bposd_config.enabled:
        raise RuntimeError("--bposd-retry-unconverged requires enabled BPOSD config")
    manifest = load_shots_manifest(sample_dir, condition)
    shot_manifest_hash = str(manifest["manifest_hash"])
    all_tasks = expected_decode_tasks(condition, manifest, meta)
    manifest_chunks = _manifest_chunks_by_index(manifest)
    target_indices = {int(task.chunk_index) for task in all_tasks}
    resolved_source_hash = str(source_hash or "").strip()
    if not resolved_source_hash:
        resolved_source_hash = find_default_bposd_retry_source_hash(
            decode_dir,
            manifest,
            all_tasks,
            condition,
            config,
        )
    target_hash = relaybp_bposd_retry_config_hash(config, bposd_config, resolved_source_hash)
    jsonl_path = decode_jsonl_path(decode_dir, condition, "relaybp", target_hash)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if force and jsonl_path.exists():
        jsonl_path.unlink()

    source_jsonl_path = decode_jsonl_path(decode_dir, condition, "relaybp", resolved_source_hash)
    source_records_all = read_decode_records(
        source_jsonl_path,
        decoder="relaybp",
        config_hash=resolved_source_hash,
    )
    if not source_records_all:
        raise RuntimeError(
            f"No source decode records found for {condition.label} "
            f"source_hash={resolved_source_hash}: {source_jsonl_path}"
        )

    source_records: list[dict[str, object]] = []
    skipped_wrong_config = 0
    skipped_wrong_coverage = 0
    skipped_no_remaining = 0
    for record in source_records_all:
        if record.get("label") != condition.label:
            continue
        if record.get("decode_schema_version") != DECODE_SCHEMA_VERSION:
            skipped_wrong_config += 1
            continue
        try:
            source_config = relaybp_config_from_record(record)
        except Exception:
            skipped_wrong_config += 1
            continue
        if source_config != config:
            skipped_wrong_config += 1
            continue
        if bool(record.get("bposd_retry_unconverged_enabled")):
            skipped_wrong_config += 1
            continue
        covered = decode_record_covered_chunk_indices(record, manifest_chunks)
        if not covered or not covered <= target_indices:
            skipped_wrong_coverage += 1
            continue
        if not _bposd_retry_source_is_usable(record):
            skipped_no_remaining += 1
            continue
        source_records.append(record)

    if not source_records:
        raise RuntimeError(
            f"{condition.label} source_hash={resolved_source_hash} has no usable "
            "source records with saved still-unconverged RelayBP details. "
            f"skipped_wrong_config={skipped_wrong_config} "
            f"skipped_wrong_coverage={skipped_wrong_coverage} "
            f"skipped_no_remaining={skipped_no_remaining}"
        )

    target_records = (
        [] if force else read_decode_records(jsonl_path, decoder="relaybp", config_hash=target_hash)
    )
    existing_record_ids: set[str] = set()
    for record in target_records:
        chunk_fingerprints = record.get("chunk_fingerprints")
        if isinstance(chunk_fingerprints, list):
            existing_record_ids.add(decode_record_id_from_fingerprints(chunk_fingerprints))

    to_process: list[dict[str, object]] = []
    to_passthrough: list[dict[str, object]] = []
    retry_rows = 0
    for record in source_records:
        chunk_fingerprints = record.get("chunk_fingerprints")
        if isinstance(chunk_fingerprints, list):
            record_id = decode_record_id_from_fingerprints(chunk_fingerprints)
        else:
            record_id = stable_hash(
                {
                    "source_hash": resolved_source_hash,
                    "source_path": str(source_jsonl_path),
                    "num_shots": record.get("num_shots"),
                    "unconverged": record.get("unconverged"),
                }
            )[:12]
        if record_id not in existing_record_ids:
            remaining = _relaybp_details_remaining_unconverged_count(record)
            if remaining > 0:
                to_process.append(record)
                retry_rows += remaining
            else:
                to_passthrough.append(record)

    source_chunks = sum(int(record.get("num_chunks", 0) or 0) for record in source_records)
    source_shots = sum(int(record.get("num_shots", 0) or 0) for record in source_records)
    if to_process or to_passthrough:
        print(
            f"  BPOSD retry {condition.label}: "
            f"source_hash={resolved_source_hash} target_hash={target_hash} "
            f"source_records={len(source_records)} source_chunks={source_chunks} "
            f"source_shots={source_shots} passthrough_records={len(to_passthrough)} "
            f"retry_records={len(to_process)} still_unconverged_to_retry={retry_rows}",
            flush=True,
        )
        if skipped_wrong_config or skipped_wrong_coverage or skipped_no_remaining:
            print(
                f"    skipped source records: wrong_config={skipped_wrong_config} "
                f"wrong_coverage={skipped_wrong_coverage} "
                f"no_remaining={skipped_no_remaining}",
                flush=True,
            )
        matrices_path = artifact_paths(sample_dir, condition)["matrices"]
        with jsonl_path.open("a", encoding="utf-8") as fout:
            for source_record in to_passthrough:
                passthrough_record = make_bposd_passthrough_record(
                    source_record,
                    source_hash=resolved_source_hash,
                    target_hash=target_hash,
                    config=config,
                    bposd_config=bposd_config,
                    shot_manifest_hash=shot_manifest_hash,
                )
                fout.write(json.dumps(passthrough_record, sort_keys=True) + "\n")
                fout.flush()
            for source_record in to_process:
                retried_record = retry_bposd_for_record(
                    source_record,
                    condition=condition,
                    decode_dir=decode_dir,
                    source_hash=resolved_source_hash,
                    target_hash=target_hash,
                    config=config,
                    bposd_config=bposd_config,
                    matrices_path=matrices_path,
                    manifest=manifest,
                    shot_manifest_hash=shot_manifest_hash,
                    progress_seconds=progress_seconds,
                    bposd_workers=bposd_workers,
                )
                retried_record["decode_schema_version"] = DECODE_SCHEMA_VERSION
                retried_record["shot_manifest_hash"] = shot_manifest_hash
                fout.write(json.dumps(retried_record, sort_keys=True) + "\n")
                fout.flush()
    else:
        print(
            f"  BPOSD retry {condition.label}: cached retry records present; "
            f"source_hash={resolved_source_hash} target_hash={target_hash}",
            flush=True,
        )

    all_records = read_decode_records(jsonl_path, decoder="relaybp", config_hash=target_hash)
    if not all_records:
        raise RuntimeError(
            f"BPOSD retry produced no records for {condition.label} target_hash={target_hash}"
        )
    selected_records, pending = select_reusable_decode_records(
        all_records,
        all_tasks,
        manifest,
        condition,
        "relaybp",
        target_hash,
        include_retry_records=True,
    )
    if pending:
        raise RuntimeError(
            f"BPOSD retry cache does not cover all chunks for {condition.label}: "
            f"pending={len(pending)}/{len(all_tasks)}. Existing partial retry "
            "records were not written into the summary."
        )
    return aggregate_decode_records(
        condition,
        meta,
        selected_records,
        jsonl_path,
        config,
        target_hash,
        shot_manifest_hash,
        shots_manifest_path(sample_dir, condition),
    )
