#!/usr/bin/env python3
"""Parallel BP-OSD upper bounds for Cornucopia circuit-level distance.

Append a random nonzero logical constraint to the detector-check matrix
and decode the syndrome [0, ..., 0, 1] with BP-OSD. The smallest valid
fault weight found across trials is an upper bound, not a distance proof.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_construction.affine_codes import build_code
from code_construction.cornucopia_codes import SPECS
from syndrome_extraction.memory_circuit import (
    build_memory_circuit,
    logical_rows_for_basis,
)
from syndrome_extraction.syndrome_schedule import (
    StimNoiseConfig,
)

warnings.filterwarnings(
    "ignore",
    message="This is the old syntax for the `bposd_decoder`.*",
    category=UserWarning,
)


Basis = Literal["Z", "X"]
RELAYBP_PYTHON = sys.executable


@dataclass(frozen=True)
class BposdConfig:
    error_rate: float = 0.01
    bp_method: str = "ms"
    max_iter: int = 500
    osd_method: str = "osd_cs"
    osd_order: int = 7
    ms_scaling_factor: float = 0.0
    logical_combo_size: int = 0


@dataclass(frozen=True)
class ChunkTask:
    start_trial: int
    num_trials: int
    seed: int


_HDEC: sparse.csr_matrix | None = None
_OBS: sparse.csr_matrix | None = None
_BPOSD_CONFIG: BposdConfig | None = None
_SYNDROME: np.ndarray | None = None
_N_FAULTS: int | None = None
_N_DETECTORS: int | None = None
_N_OBSERVABLES: int | None = None


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("stim", "ldpc", "bposd", "relay_bp", "beliefmatching"):
        try:
            __import__(module_name)
        except Exception as exc:  # pragma: no cover - environment check
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if missing:
        raise ImportError(
            "Missing RelayBP runtime dependencies in the current Python "
            "environment. Missing: " + "; ".join(missing)
        )


def detector_matrices_from_circuit(circuit):
    from relay_bp.stim import CheckMatrices  # type: ignore

    dem = circuit.detector_error_model(decompose_errors=False)
    matrices = CheckMatrices.from_dem(
        dem,
        decomposed_hyperedges=None,
        prune_decided_errors=True,
    )
    hdec = matrices.check_matrix.tocsr().astype(np.uint8)
    obs = matrices.observables_matrix.tocsr().astype(np.uint8)
    hdec.data %= 2
    obs.data %= 2
    hdec.eliminate_zeros()
    obs.eliminate_zeros()
    return dem, hdec, obs, np.asarray(matrices.error_priors, dtype=np.float64)


def write_text_if_requested(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_problem(
    spec,
    basis: Basis,
    *,
    rounds: int,
    p_cx: float,
    write_stim_path: Path | None,
    write_dem_path: Path | None,
):
    code = build_code(spec.to_code_spec())
    logical_rows = logical_rows_for_basis(code, basis)
    noise = StimNoiseConfig(
        p_cx=float(p_cx),
        p_measure_flip=0.0,
        p_final_measure_flip=0.0,
        p_reset_flip=0.0,
    )
    circuit = build_memory_circuit(
        code,
        basis=basis,
        logical_rows=logical_rows,
        rounds=rounds,
        noise=noise,
    )
    write_text_if_requested(write_stim_path, str(circuit))
    dem, hdec, obs, error_priors = detector_matrices_from_circuit(circuit)
    write_text_if_requested(write_dem_path, str(dem))
    if hdec.shape[1] == 0:
        raise ValueError(f"{spec.name} {basis}: DEM produced no fault columns")
    if obs.shape[0] == 0 or obs.nnz == 0:
        raise ValueError(f"{spec.name} {basis}: DEM produced no observable columns")
    return {
        "code": code,
        "circuit": circuit,
        "dem": dem,
        "hdec": hdec,
        "obs": obs,
        "error_priors": error_priors,
        "logical_rows": logical_rows,
        "noise": noise,
    }


def init_worker(hdec, obs, config: BposdConfig) -> None:
    global _HDEC, _OBS, _BPOSD_CONFIG, _SYNDROME, _N_FAULTS, _N_DETECTORS, _N_OBSERVABLES
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _HDEC = hdec.tocsr().astype(np.uint8)
    _OBS = obs.tocsr().astype(np.uint8)
    _BPOSD_CONFIG = config
    _N_DETECTORS, _N_FAULTS = _HDEC.shape
    _N_OBSERVABLES = _OBS.shape[0]
    _SYNDROME = np.zeros(_N_DETECTORS + 1, dtype=np.uint8)
    _SYNDROME[-1] = 1


def xor_observable_rows(selected: np.ndarray) -> sparse.csr_matrix:
    if _OBS is None or _N_FAULTS is None:
        raise RuntimeError("worker was not initialized")
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size == 1:
        return _OBS.getrow(int(selected[0])).copy().tocsr()

    counts = np.asarray(_OBS[selected].astype(np.int32).sum(axis=0)).reshape(-1)
    indices = np.flatnonzero(counts & 1).astype(np.int32)
    data = np.ones(indices.size, dtype=np.uint8)
    return sparse.csr_matrix(
        (data, indices, np.array([0, indices.size], dtype=np.int32)),
        shape=(1, _N_FAULTS),
    )


def random_observable_constraint(
    rng: np.random.Generator,
) -> tuple[sparse.csr_matrix, list[int]]:
    if _OBS is None or _BPOSD_CONFIG is None or _N_FAULTS is None or _N_OBSERVABLES is None:
        raise RuntimeError("worker was not initialized")
    combo_size = int(_BPOSD_CONFIG.logical_combo_size)
    for _ in range(100):
        if combo_size <= 0:
            mask = rng.integers(0, 2, size=int(_N_OBSERVABLES), dtype=np.uint8)
            selected = np.flatnonzero(mask).astype(np.int64)
            if selected.size == 0:
                continue
        else:
            fixed_size = max(1, min(combo_size, int(_N_OBSERVABLES)))
            selected = rng.choice(_N_OBSERVABLES, size=fixed_size, replace=False)
            selected = np.asarray(selected, dtype=np.int64)

        row = xor_observable_rows(selected)
        row.data %= 2
        row.eliminate_zeros()
        if row.nnz:
            return row, [int(x) for x in selected.tolist()]
    raise RuntimeError("failed to sample a nonzero observable constraint row")


def run_chunk(task: ChunkTask) -> dict[str, object]:
    if _HDEC is None or _BPOSD_CONFIG is None or _SYNDROME is None or _N_FAULTS is None:
        raise RuntimeError("worker was not initialized")
    from ldpc import bposd_decoder  # type: ignore

    rng = np.random.default_rng(int(task.seed))
    best_weight: int | None = None
    best_trial: int | None = None
    best_observables: list[int] | None = None
    valid_trials = 0
    zero_weight_results = 0
    syndrome_mismatches = 0
    errors = 0
    start_time = time.monotonic()

    for offset in range(int(task.num_trials)):
        trial_index = int(task.start_trial + offset)
        try:
            logical_row, selected_observables = random_observable_constraint(rng)
            h_aug = sparse.vstack([_HDEC, logical_row], format="csr", dtype=np.uint8)
            decoder = bposd_decoder(
                h_aug,
                error_rate=float(_BPOSD_CONFIG.error_rate),
                max_iter=int(_BPOSD_CONFIG.max_iter),
                bp_method=str(_BPOSD_CONFIG.bp_method),
                ms_scaling_factor=float(_BPOSD_CONFIG.ms_scaling_factor),
                osd_method=str(_BPOSD_CONFIG.osd_method),
                osd_order=int(_BPOSD_CONFIG.osd_order),
            )
            decoder.decode(_SYNDROME)
            decoding = np.asarray(decoder.osdw_decoding, dtype=np.uint8).reshape(-1)
            decoded_syndrome = np.asarray(h_aug.dot(decoding) % 2, dtype=np.uint8).reshape(-1)
            if not np.array_equal(decoded_syndrome, _SYNDROME):
                syndrome_mismatches += 1
                continue
            weight = int(np.count_nonzero(decoding))
            if weight <= 0:
                zero_weight_results += 1
                continue
            valid_trials += 1
            if best_weight is None or weight < best_weight:
                best_weight = weight
                best_trial = trial_index
                best_observables = selected_observables
        except Exception:
            errors += 1
            if errors <= 3:
                raise

    return {
        "pid": os.getpid(),
        "start_trial": int(task.start_trial),
        "num_trials": int(task.num_trials),
        "seed": int(task.seed),
        "valid_trials": int(valid_trials),
        "zero_weight_results": int(zero_weight_results),
        "syndrome_mismatches": int(syndrome_mismatches),
        "errors": int(errors),
        "best_weight": None if best_weight is None else int(best_weight),
        "best_trial": None if best_trial is None else int(best_trial),
        "best_observables": best_observables,
        "seconds": float(time.monotonic() - start_time),
    }


def make_chunk_tasks(num_trials: int, chunk_size: int, seed: int) -> list[ChunkTask]:
    tasks: list[ChunkTask] = []
    start = 0
    while start < num_trials:
        count = min(chunk_size, num_trials - start)
        chunk_seed = int((seed + 0x9E3779B97F4A7C15 * (len(tasks) + 1)) % (2**63 - 1))
        tasks.append(ChunkTask(start_trial=start, num_trials=count, seed=chunk_seed))
        start += count
    return tasks


def run_parallel_trials(
    hdec: sparse.csr_matrix,
    obs: sparse.csr_matrix,
    *,
    config: BposdConfig,
    num_trials: int,
    workers: int,
    chunk_size: int,
    seed: int,
    jsonl_path: Path,
    progress_seconds: float,
) -> dict[str, object]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = make_chunk_tasks(num_trials, chunk_size, seed)
    best_weight: int | None = None
    best_trial: int | None = None
    best_observables: list[int] | None = None
    completed_trials = 0
    valid_trials = 0
    zero_weight_results = 0
    syndrome_mismatches = 0
    errors = 0
    t0 = time.monotonic()
    last_progress = t0

    ctx = mp.get_context("spawn")
    with jsonl_path.open("a", encoding="utf-8") as fout:
        with ctx.Pool(
            processes=int(workers),
            initializer=init_worker,
            initargs=(hdec, obs, config),
        ) as pool:
            try:
                for result in pool.imap_unordered(run_chunk, tasks, chunksize=1):
                    completed_trials += int(result["num_trials"])
                    valid_trials += int(result["valid_trials"])
                    zero_weight_results += int(result["zero_weight_results"])
                    syndrome_mismatches += int(result.get("syndrome_mismatches", 0))
                    errors += int(result["errors"])
                    chunk_best = result.get("best_weight")
                    improved = False
                    if chunk_best is not None and (
                        best_weight is None or int(chunk_best) < best_weight
                    ):
                        best_weight = int(chunk_best)
                        best_trial = (
                            None if result.get("best_trial") is None else int(result["best_trial"])
                        )
                        best_observables = result.get("best_observables")  # type: ignore[assignment]
                        improved = True
                    result["completed_trials_total"] = completed_trials
                    result["global_best_weight"] = best_weight
                    result["global_best_trial"] = best_trial
                    fout.write(json.dumps(result, sort_keys=True) + "\n")
                    fout.flush()

                    now = time.monotonic()
                    if improved or now - last_progress >= progress_seconds:
                        rate = completed_trials / max(now - t0, 1e-9)
                        print(
                            f"  progress {completed_trials}/{num_trials} "
                            f"valid={valid_trials} best={best_weight} "
                            f"rate={rate:.2f} trials/s",
                            flush=True,
                        )
                        last_progress = now
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise

    seconds = time.monotonic() - t0
    return {
        "num_trials_requested": int(num_trials),
        "num_trials_completed": int(completed_trials),
        "valid_trials": int(valid_trials),
        "zero_weight_results": int(zero_weight_results),
        "syndrome_mismatches": int(syndrome_mismatches),
        "errors": int(errors),
        "best_weight": best_weight,
        "best_trial": best_trial,
        "best_observables": best_observables,
        "seconds": float(seconds),
        "trials_per_second": float(completed_trials / max(seconds, 1e-9)),
        "jsonl_path": str(jsonl_path),
    }


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


def aggregate_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_code: dict[str, dict[str, object]] = {}
    for record in records:
        code_name = str(record.get("code_name"))
        entry = by_code.setdefault(
            code_name,
            {
                "code_name": code_name,
                "parameter_label": record.get("parameter_label"),
                "P": record.get("P"),
                "L": record.get("L"),
                "J": record.get("J"),
                "Z_basis_UB": None,
                "X_basis_UB": None,
                "min_UB": None,
            },
        )
        basis = record.get("basis")
        upper_bound = record.get("upper_bound")
        if basis == "Z":
            entry["Z_basis_UB"] = upper_bound
        elif basis == "X":
            entry["X_basis_UB"] = upper_bound
        bounds = [
            value
            for value in (entry.get("Z_basis_UB"), entry.get("X_basis_UB"))
            if value is not None
        ]
        entry["min_UB"] = None if not bounds else int(min(int(value) for value in bounds))
    return list(by_code.values())


def write_summary(output_dir: Path, records: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"
    aggregate = aggregate_records(records)
    payload = {"by_code": aggregate, "records": records}
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["Cornucopia circuit-distance BP-OSD upper-bound summary", ""]
    lines.append("By code")
    for entry in aggregate:
        lines.append(
            "{} {} Z_basis_UB={} X_basis_UB={} min_UB={}".format(
                entry.get("code_name"),
                entry.get("parameter_label"),
                entry.get("Z_basis_UB"),
                entry.get("X_basis_UB"),
                entry.get("min_UB"),
            )
        )
    lines.append("")
    lines.append("By code and basis")
    for record in records:
        seconds = float(record.get("seconds") or 0.0)
        lines.append(
            "{} {} basis={} UB={} trials={} faults={} detectors={} observables={} seconds={:.3f}".format(
                record.get("code_name"),
                record.get("parameter_label"),
                record.get("basis"),
                record.get("upper_bound"),
                record.get("num_trials_completed"),
                record.get("num_fault_columns"),
                record.get("num_detectors"),
                record.get("num_observables"),
                seconds,
            )
        )
    lines.append("")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")


def load_existing_records(output_dir: Path) -> list[dict[str, object]]:
    summary_json = output_dir / "summary.json"
    if not summary_json.exists():
        return []
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"warning: failed to load existing summary records from {summary_json}: {exc}",
            flush=True,
        )
        return []
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", default="all", help="Comma-separated SPECS names, or all.")
    parser.add_argument("--basis", choices=["Z", "X", "both"], default="both")
    parser.add_argument("--num-trials", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--p-cx", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--error-rate", type=float, default=0.01)
    parser.add_argument("--bp-method", default="ms")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--osd-method", default="osd_cs")
    parser.add_argument("--osd-order", type=int, default=7)
    parser.add_argument("--ms-scaling-factor", type=float, default=0.0)
    parser.add_argument(
        "--logical-combo-size",
        type=int,
        default=0,
        help=(
            "Observable rows XORed into each logical constraint. "
            "Default 0 samples a random nonzero GF(2) combination of all observables."
        ),
    )
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("circuit_distance/results"))
    parser.add_argument("--write-stim", action="store_true")
    parser.add_argument("--write-dem", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_runtime_dependencies()
    specs = parse_codes(args.codes)
    bases = parse_bases(args.basis)
    if args.num_trials <= 0:
        raise SystemExit("--num-trials must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    config = BposdConfig(
        error_rate=float(args.error_rate),
        bp_method=str(args.bp_method),
        max_iter=int(args.max_iter),
        osd_method=str(args.osd_method),
        osd_order=int(args.osd_order),
        ms_scaling_factor=float(args.ms_scaling_factor),
        logical_combo_size=int(args.logical_combo_size),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = load_existing_records(args.output_dir)

    print(f"Python executable: {Path(RELAYBP_PYTHON).name}", flush=True)
    if records:
        print(
            f"loaded_existing_summary_records={len(records)} from {args.output_dir / 'summary.json'}",
            flush=True,
        )
    print(
        f"codes={[spec.name for spec in specs]} bases={bases} workers={args.workers}",
        flush=True,
    )
    print(f"bposd_config={asdict(config)}", flush=True)

    for spec_index, spec in enumerate(specs):
        for basis_index, basis in enumerate(bases):
            label = f"{spec.name}_{basis.lower()}basis"
            print(f"\nBuilding {label} {spec.parameter_label}...", flush=True)
            stim_path = args.output_dir / "stim" / f"{label}.stim" if args.write_stim else None
            dem_path = args.output_dir / "dem" / f"{label}.dem" if args.write_dem else None
            problem = build_problem(
                spec,
                basis,
                rounds=int(args.rounds),
                p_cx=float(args.p_cx),
                write_stim_path=stim_path,
                write_dem_path=dem_path,
            )
            hdec = problem["hdec"]
            obs = problem["obs"]
            circuit = problem["circuit"]
            dem = problem["dem"]
            basis_seed = int(args.seed + 1000003 * spec_index + 9176 * basis_index)
            jsonl_path = args.output_dir / f"{label}_trials.jsonl"
            print(
                f"  detectors={hdec.shape[0]} observables={obs.shape[0]} "
                f"fault_columns={hdec.shape[1]} dem_errors={dem.num_errors}",
                flush=True,
            )
            result = run_parallel_trials(
                hdec,
                obs,
                config=config,
                num_trials=int(args.num_trials),
                workers=int(args.workers),
                chunk_size=int(args.chunk_size),
                seed=basis_seed,
                jsonl_path=jsonl_path,
                progress_seconds=float(args.progress_seconds),
            )
            record = {
                "code_name": spec.name,
                "parameter_label": spec.parameter_label,
                "P": int(spec.p),
                "L": int(spec.l),
                "J": int(spec.j),
                "basis": basis,
                "rounds": int(args.rounds),
                "p_cx": float(args.p_cx),
                "bposd_config": asdict(config),
                "num_detectors": int(hdec.shape[0]),
                "num_observables": int(obs.shape[0]),
                "num_fault_columns": int(hdec.shape[1]),
                "dem_num_errors": int(dem.num_errors),
                "stim_num_detectors": int(circuit.num_detectors),
                "stim_num_observables": int(circuit.num_observables),
                "upper_bound": result["best_weight"],
                **result,
            }
            records.append(record)
            write_summary(args.output_dir, records)
            print(
                "DONE {}: upper_bound={} trials={} seconds={:.3f}".format(
                    label,
                    record.get("upper_bound"),
                    record.get("num_trials_completed"),
                    float(record.get("seconds") or 0.0),
                ),
                flush=True,
            )

    write_summary(args.output_dir, records)
    print("summary_json=" + str(args.output_dir / "summary.json"), flush=True)
    print("summary_txt=" + str(args.output_dir / "summary.txt"), flush=True)


if __name__ == "__main__":
    main()
