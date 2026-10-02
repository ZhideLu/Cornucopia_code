"""Decoder construction, retry workers, and saved unconverged-shot details."""

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
    RelayBPConfig,
    RelayBPFallbackConfig,
)


def decoder_config_hash(config: RelayBPConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode("utf-8")).hexdigest()[
        :12
    ]


def relaybp_config_hash(config: RelayBPConfig) -> str:
    return decoder_config_hash(config)


def relaybp_decode_config_hash(
    relay_config: RelayBPConfig,
    relaybp_fallback_config: RelayBPFallbackConfig | None = None,
) -> str:
    """Identify the primary RelayBP settings and the optional second pass."""
    relaybp_fallback_enabled = (
        relaybp_fallback_config is not None and relaybp_fallback_config.enabled
    )
    if not relaybp_fallback_enabled:
        return relaybp_config_hash(relay_config)
    payload: dict[str, object] = {
        "relaybp_config": asdict(relay_config),
        "relaybp_fallback_config": asdict(relaybp_fallback_config),
    }
    return stable_hash(payload)[:12]


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
