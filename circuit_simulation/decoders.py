"""Decoder construction, retry workers, and saved fallback details."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from circuit_simulation.shot_data import load_check_matrices, stable_hash
from circuit_simulation.simulation_parameters import (
    DECODE_SCHEMA_VERSION,
    BpOsdRetryConfig,
    MIPFallbackConfig,
    RelayBPConfig,
    RelayBPFallbackConfig,
)


def decoder_config_hash(config: RelayBPConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode("utf-8")).hexdigest()[
        :12
    ]


def relaybp_config_hash(config: RelayBPConfig) -> str:
    return decoder_config_hash(config)


def relaybp_mip_config_hash(
    relay_config: RelayBPConfig,
    mip_config: MIPFallbackConfig | None,
    relaybp_fallback_config: RelayBPFallbackConfig | None = None,
) -> str:
    relaybp_fallback_enabled = (
        relaybp_fallback_config is not None and relaybp_fallback_config.enabled
    )
    if (mip_config is None or not mip_config.enabled) and not relaybp_fallback_enabled:
        return relaybp_config_hash(relay_config)
    payload: dict[str, object] = {"relaybp_config": asdict(relay_config)}
    if relaybp_fallback_enabled:
        payload["relaybp_fallback_config"] = asdict(relaybp_fallback_config)
    if mip_config is not None and mip_config.enabled:
        mip_payload = asdict(mip_config)
    # Exclude the worker count when identifying decoder settings.
        mip_payload.pop("workers", None)
        if float(mip_payload.get("wall_time_limit", 0.0) or 0.0) <= 0.0:
            mip_payload.pop("wall_time_limit", None)
        payload["mip_fallback_config"] = mip_payload
    return stable_hash(payload)[:12]


def relaybp_mip_retry_config_hash(
    relay_config: RelayBPConfig,
    mip_config: MIPFallbackConfig,
    source_hash: str,
) -> str:
    return stable_hash(
        {
            "relaybp_mip_config_hash": relaybp_mip_config_hash(relay_config, mip_config),
            "mip_retry_mode": "unsolved",
            "mip_retry_source_hash": str(source_hash),
        }
    )[:12]


def relaybp_fallback_retry_config_hash(
    relay_config: RelayBPConfig,
    relaybp_fallback_config: RelayBPFallbackConfig,
    source_hash: str,
) -> str:
    return stable_hash(
        {
            "relaybp_config": asdict(relay_config),
            "relaybp_fallback_config": asdict(relaybp_fallback_config),
            "relaybp_fallback_retry_mode": "saved_primary_unconverged_details_v1",
            "relaybp_fallback_retry_source_hash": str(source_hash),
        }
    )[:12]


def relaybp_bposd_retry_config_hash(
    relay_config: RelayBPConfig,
    bposd_config: BpOsdRetryConfig,
    source_hash: str,
) -> str:
    return stable_hash(
        {
            "relaybp_config": asdict(relay_config),
            "bposd_retry_config": asdict(bposd_config),
            "bposd_retry_mode": "saved_relaybp_unconverged_details_v1",
            "bposd_retry_source_hash": str(source_hash),
        }
    )[:12]


def build_relaybp_runner(matrices_path: Path, config: RelayBPConfig):
    import relay_bp  # type: ignore

    check_matrix, observables_matrix, error_priors = load_check_matrices(matrices_path)
    decoder = relay_bp.RelayDecoderF32(
        check_matrix,
        error_priors=np.asarray(error_priors, dtype=np.float64),
        **config.kwargs(),
    )
    return relay_bp.ObservableDecoderRunner(decoder, observables_matrix)


def build_bposd_retry_decoder(
    matrices_path: Path,
    config: BpOsdRetryConfig,
):
    from ldpc import BpOsdDecoder  # type: ignore

    check_matrix, observables_matrix, error_priors = load_check_matrices(matrices_path)
    decoder = BpOsdDecoder(
        check_matrix,
        error_channel=np.asarray(error_priors, dtype=np.float64).tolist(),
        **config.kwargs(),
    )
    return decoder, check_matrix.tocsr(), observables_matrix.tocsr()


_BPOSD_RETRY_WORKER_DECODER = None
_BPOSD_RETRY_WORKER_CHECK_MATRIX = None
_BPOSD_RETRY_WORKER_OBSERVABLES_MATRIX = None


def init_bposd_retry_worker(
    matrices_path_text: str,
    bposd_config_payload: dict[str, object],
) -> None:
    global _BPOSD_RETRY_WORKER_DECODER
    global _BPOSD_RETRY_WORKER_CHECK_MATRIX
    global _BPOSD_RETRY_WORKER_OBSERVABLES_MATRIX

    config = BpOsdRetryConfig(**bposd_config_payload)
    (
        _BPOSD_RETRY_WORKER_DECODER,
        _BPOSD_RETRY_WORKER_CHECK_MATRIX,
        _BPOSD_RETRY_WORKER_OBSERVABLES_MATRIX,
    ) = build_bposd_retry_decoder(Path(matrices_path_text), config)


def decode_bposd_retry_row(
    bposd_decoder: object,
    check_matrix: sparse.csr_matrix,
    observables_matrix: sparse.csr_matrix,
    *,
    local_index: int,
    detail_index: int,
    syndrome: np.ndarray,
    observed: np.ndarray,
    old_mismatch_count: int,
) -> dict[str, object]:
    shot_t0 = time.monotonic()
    result: dict[str, object] = {
        "local_index": int(local_index),
        "detail_index": int(detail_index),
        "syndrome_matched": False,
        "logical_correct": False,
        "mismatch_count": int(old_mismatch_count),
        "correction_weight": 0,
        "decode_seconds": 0.0,
        "error_text": "",
    }
    try:
        correction = np.asarray(bposd_decoder.decode(syndrome), dtype=np.uint8)
        correction = np.ravel(correction) % 2
        result["correction_weight"] = int(np.sum(correction, dtype=np.int64))
        syndrome_after = np.asarray(check_matrix @ correction, dtype=np.uint8) % 2
        syndrome_after = np.ravel(syndrome_after).astype(np.uint8, copy=False)
        syndrome_ok = bool(np.array_equal(syndrome_after, syndrome))
        result["syndrome_matched"] = syndrome_ok
        if syndrome_ok:
            predicted = np.asarray(observables_matrix @ correction, dtype=np.uint8) % 2
            predicted = np.ravel(predicted).astype(np.uint8, copy=False)
            mismatches = int(np.sum(predicted != observed, dtype=np.int64))
            result["mismatch_count"] = mismatches
            result["logical_correct"] = mismatches == 0
        else:
            result["error_text"] = "syndrome_mismatch"
    except Exception as exc:  # keep the shot in the unconverged bucket
        result["error_text"] = f"{type(exc).__name__}: {exc}"[:512]
    result["decode_seconds"] = float(time.monotonic() - shot_t0)
    return result


def run_bposd_retry_worker_task(
    task: tuple[int, int, np.ndarray, np.ndarray, int],
) -> dict[str, object]:
    if (
        _BPOSD_RETRY_WORKER_DECODER is None
        or _BPOSD_RETRY_WORKER_CHECK_MATRIX is None
        or _BPOSD_RETRY_WORKER_OBSERVABLES_MATRIX is None
    ):
        raise RuntimeError("BPOSD retry worker was not initialized")
    local_index, detail_index, syndrome, observed, old_mismatch_count = task
    return decode_bposd_retry_row(
        _BPOSD_RETRY_WORKER_DECODER,
        _BPOSD_RETRY_WORKER_CHECK_MATRIX,
        _BPOSD_RETRY_WORKER_OBSERVABLES_MATRIX,
        local_index=int(local_index),
        detail_index=int(detail_index),
        syndrome=np.asarray(syndrome, dtype=np.uint8),
        observed=np.asarray(observed, dtype=np.uint8),
        old_mismatch_count=int(old_mismatch_count),
    )


def make_bposd_passthrough_record(
    source_record: dict[str, object],
    *,
    source_hash: str,
    target_hash: str,
    config: RelayBPConfig,
    bposd_config: BpOsdRetryConfig,
    shot_manifest_hash: str,
) -> dict[str, object]:
    record = {
        key: value for key, value in source_record.items() if not str(key).startswith("_source_")
    }
    record["relaybp_config_hash"] = target_hash
    record["relaybp_config"] = asdict(config)
    record["decode_schema_version"] = DECODE_SCHEMA_VERSION
    record["shot_manifest_hash"] = shot_manifest_hash
    record["bposd_retry_unconverged_enabled"] = True
    record["bposd_retry_source_hash"] = source_hash
    record["bposd_retry_config"] = asdict(bposd_config)
    record["bposd_retry_details_path"] = ""
    record["bposd_retry_retried"] = 0
    record["bposd_retry_syndrome_matched"] = 0
    record["bposd_retry_logical_correct"] = 0
    record["bposd_retry_logical_failures"] = 0
    record["bposd_retry_unresolved"] = 0
    record["bposd_retry_observable_mismatches"] = 0
    record["bposd_retry_seconds"] = 0.0
    return record


def mip_decode_observables(
    check_matrix: sparse.csr_matrix,
    observables_matrix: sparse.csr_matrix,
    error_priors: np.ndarray,
    syndrome: np.ndarray,
    config: MIPFallbackConfig,
) -> tuple[np.ndarray | None, dict[str, object]]:
    from mip import (  # type: ignore
        BINARY,
        CBC,
        INTEGER,
        MINIMIZE,
        Model,
        OptimizationStatus,
        SearchEmphasis,
        xsum,
    )

    t0 = time.monotonic()
    syndrome = np.asarray(syndrome, dtype=np.int8).reshape(-1) % 2
    num_checks, num_faults = check_matrix.shape
    if syndrome.shape[0] != num_checks:
        raise ValueError(f"syndrome length {syndrome.shape[0]} != num_checks {num_checks}")
    priors = np.clip(np.asarray(error_priors, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    weights = np.maximum(np.log((1.0 - priors) / priors), 0.0)

    h = check_matrix.tocsr()
    model = Model(sense=MINIMIZE, solver_name=CBC)
    model.verbose = 0
    if config.threads > 0:
        model.threads = int(config.threads)
    if config.feasible_only:
        model.max_solutions = 1
        model.emphasis = SearchEmphasis.FEASIBILITY
    if config.mip_rel_gap > 0:
        model.max_mip_gap = float(config.mip_rel_gap)
    x = [model.add_var(var_type=BINARY) for _ in range(num_faults)]
    y = [
        model.add_var(var_type=INTEGER, lb=0, ub=max(int(h.indptr[i + 1] - h.indptr[i]), 1))
        for i in range(num_checks)
    ]
    for i in range(num_checks):
        cols = h.indices[h.indptr[i] : h.indptr[i + 1]]
        model += xsum(x[int(j)] for j in cols) - 2 * y[i] == int(syndrome[i])
    model.objective = xsum(float(weights[j]) * x[j] for j in range(num_faults))
    if config.time_limit > 0:
        status = model.optimize(max_seconds=float(config.time_limit))
    else:
        status = model.optimize()
    seconds = time.monotonic() - t0
    success = status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
    info: dict[str, object] = {
        "mip_solver": "mip",
        "mip_status": str(status),
        "mip_success": bool(success),
        "mip_message": str(status),
        "mip_feasible_only": bool(config.feasible_only),
        "mip_num_solutions": int(model.num_solutions),
        "mip_fun": None if model.objective_value is None else float(model.objective_value),
        "mip_seconds": float(seconds),
    }
    if not success:
        return None, info
    correction = np.asarray([round(var.x or 0.0) for var in x], dtype=np.uint8) % 2
    predicted_syndrome = np.asarray(check_matrix.dot(correction), dtype=np.uint8)
    predicted_syndrome = (predicted_syndrome.reshape(-1) % 2).astype(np.uint8)
    if not np.array_equal(predicted_syndrome, syndrome.astype(np.uint8)):
        info["mip_success"] = False
        info["mip_message"] = "MIP solution did not match syndrome"
        return None, info
    obs = np.asarray(observables_matrix.dot(correction), dtype=np.uint8).reshape(-1)
    return (obs % 2).astype(np.uint8, copy=False), info


_MIP_WORKER_CHECK_MATRIX: sparse.csr_matrix | None = None
_MIP_WORKER_OBSERVABLES_MATRIX: sparse.csr_matrix | None = None
_MIP_WORKER_ERROR_PRIORS: np.ndarray | None = None
_MIP_WORKER_CONFIG: MIPFallbackConfig | None = None


def _init_mip_worker(
    check_matrix: sparse.csr_matrix,
    observables_matrix: sparse.csr_matrix,
    error_priors: np.ndarray,
    config_payload: dict[str, object],
) -> None:
    global _MIP_WORKER_CHECK_MATRIX
    global _MIP_WORKER_OBSERVABLES_MATRIX
    global _MIP_WORKER_ERROR_PRIORS
    global _MIP_WORKER_CONFIG
    _MIP_WORKER_CHECK_MATRIX = check_matrix
    _MIP_WORKER_OBSERVABLES_MATRIX = observables_matrix
    _MIP_WORKER_ERROR_PRIORS = np.asarray(error_priors, dtype=np.float64)
    _MIP_WORKER_CONFIG = MIPFallbackConfig(**config_payload)


def _mip_decode_worker(
    task: tuple[int, np.ndarray],
) -> tuple[int, np.ndarray | None, dict[str, object]]:
    if (
        _MIP_WORKER_CHECK_MATRIX is None
        or _MIP_WORKER_OBSERVABLES_MATRIX is None
        or _MIP_WORKER_ERROR_PRIORS is None
        or _MIP_WORKER_CONFIG is None
    ):
        raise RuntimeError("MIP worker was not initialized")
    shot_index, syndrome = task
    prediction, info = mip_decode_observables(
        _MIP_WORKER_CHECK_MATRIX,
        _MIP_WORKER_OBSERVABLES_MATRIX,
        _MIP_WORKER_ERROR_PRIORS,
        syndrome,
        _MIP_WORKER_CONFIG,
    )
    return int(shot_index), prediction, info


def save_mip_fallback_details(
    path: Path,
    *,
    metadata: dict[str, object],
    config: MIPFallbackConfig,
    shot_indices: np.ndarray,
    detectors: np.ndarray,
    observables: np.ndarray,
    old_failure_mask: np.ndarray,
    old_mismatch_counts: np.ndarray,
) -> None:
    """Persist only unresolved MIP fallback shots for later retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int64),
        shot_indices=np.asarray(shot_indices, dtype=np.int64),
        detectors=np.asarray(detectors, dtype=np.uint8),
        observables=np.asarray(observables, dtype=np.uint8),
        old_failure_mask=np.asarray(old_failure_mask, dtype=np.uint8),
        old_mismatch_counts=np.asarray(old_mismatch_counts, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        mip_fallback_config_json=np.asarray(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        ),
    )


def load_mip_fallback_details(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:

        def scalar_text(key: str) -> str:
            value = payload[key]
            return str(value.item() if hasattr(value, "item") else value)

        return {
            "schema_version": int(payload["schema_version"][0]),
            "shot_indices": np.asarray(payload["shot_indices"], dtype=np.int64),
            "detectors": np.asarray(payload["detectors"], dtype=np.uint8),
            "observables": np.asarray(payload["observables"], dtype=np.uint8),
            "old_failure_mask": np.asarray(payload["old_failure_mask"], dtype=np.uint8).astype(
                bool
            ),
            "old_mismatch_counts": np.asarray(payload["old_mismatch_counts"], dtype=np.int64),
            "metadata": json.loads(scalar_text("metadata_json")),
            "mip_fallback_config": json.loads(scalar_text("mip_fallback_config_json")),
        }


def save_relaybp_unconverged_details(
    path: Path,
    *,
    metadata: dict[str, object],
    relaybp_config: RelayBPConfig,
    shot_indices: np.ndarray,
    detectors: np.ndarray,
    observables: np.ndarray,
    old_converged_mask: np.ndarray,
    old_failure_mask: np.ndarray,
    old_mismatch_counts: np.ndarray,
    old_iterations: np.ndarray,
) -> None:
    """Persist primary RelayBP-unconverged shots for strong fallback retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int64),
        shot_indices=np.asarray(shot_indices, dtype=np.int64),
        detectors=np.asarray(detectors, dtype=np.uint8),
        observables=np.asarray(observables, dtype=np.uint8),
        old_converged_mask=np.asarray(old_converged_mask, dtype=np.uint8),
        old_failure_mask=np.asarray(old_failure_mask, dtype=np.uint8),
        old_mismatch_counts=np.asarray(old_mismatch_counts, dtype=np.int64),
        old_iterations=np.asarray(old_iterations, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        relaybp_config_json=np.asarray(
            json.dumps(asdict(relaybp_config), sort_keys=True, separators=(",", ":"))
        ),
    )


def load_relaybp_unconverged_details(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:

        def scalar_text(key: str) -> str:
            value = payload[key]
            return str(value.item() if hasattr(value, "item") else value)

        return {
            "schema_version": int(payload["schema_version"][0]),
            "shot_indices": np.asarray(payload["shot_indices"], dtype=np.int64),
            "detectors": np.asarray(payload["detectors"], dtype=np.uint8),
            "observables": np.asarray(payload["observables"], dtype=np.uint8),
            "old_converged_mask": np.asarray(payload["old_converged_mask"], dtype=np.uint8).astype(
                bool
            ),
            "old_failure_mask": np.asarray(payload["old_failure_mask"], dtype=np.uint8).astype(
                bool
            ),
            "old_mismatch_counts": np.asarray(payload["old_mismatch_counts"], dtype=np.int64),
            "old_iterations": np.asarray(payload["old_iterations"], dtype=np.int64),
            "metadata": json.loads(scalar_text("metadata_json")),
            "relaybp_config": json.loads(scalar_text("relaybp_config_json")),
        }


def save_bposd_retry_details(
    path: Path,
    *,
    metadata: dict[str, object],
    bposd_config: BpOsdRetryConfig,
    source_hash: str,
    shot_indices: np.ndarray,
    source_detail_indices: np.ndarray,
    syndrome_matched: np.ndarray,
    logical_correct: np.ndarray,
    mismatch_counts: np.ndarray,
    correction_weights: np.ndarray,
    decode_seconds: np.ndarray,
    error_text: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray([1], dtype=np.int64),
        shot_indices=np.asarray(shot_indices, dtype=np.int64),
        source_detail_indices=np.asarray(source_detail_indices, dtype=np.int64),
        syndrome_matched=np.asarray(syndrome_matched, dtype=np.uint8),
        logical_correct=np.asarray(logical_correct, dtype=np.uint8),
        mismatch_counts=np.asarray(mismatch_counts, dtype=np.int64),
        correction_weights=np.asarray(correction_weights, dtype=np.int64),
        decode_seconds=np.asarray(decode_seconds, dtype=np.float64),
        error_text=np.asarray(list(error_text), dtype="U512"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        bposd_config_json=np.asarray(
            json.dumps(asdict(bposd_config), sort_keys=True, separators=(",", ":"))
        ),
        source_hash=np.asarray(str(source_hash)),
    )
