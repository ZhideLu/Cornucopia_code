"""Store circuits, sparse decoding matrices, and sampled detector records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from circuit_simulation.simulation_parameters import (
    SCRIPT_VERSION,
    Condition,
    DecoderName,
    GenerateTask,
)


def artifact_paths(output_dir: Path, condition: Condition) -> dict[str, Path]:
    base = output_dir / "artifacts" / condition.label
    return {
        "stim": base.with_suffix(".stim"),
        "dem": base.with_suffix(".dem"),
        "matrices": base.with_suffix(".matrices.npz"),
        "meta": base.with_suffix(".meta.json"),
    }


def shots_dir(output_dir: Path, condition: Condition) -> Path:
    return output_dir / "shots" / condition.label


def decode_jsonl_path(
    output_dir: Path, condition: Condition, decoder: DecoderName, config_hash: str
) -> Path:
    if decoder != "relaybp":
        raise ValueError(f"unsupported decoder: {decoder}")
    return output_dir / "decode" / f"{condition.label}_{decoder}_{config_hash}.jsonl"


def mip_fallback_details_dir(output_dir: Path) -> Path:
    return output_dir / "mip_fallback_details"


def relaybp_unconverged_details_dir(output_dir: Path) -> Path:
    return output_dir / "relaybp_unconverged_details"


def bposd_retry_details_dir(output_dir: Path) -> Path:
    return output_dir / "bposd_retry_details"


def mip_fallback_details_path(
    output_dir: Path, condition: Condition, config_hash: str, record_id: str
) -> Path:
    return (
        mip_fallback_details_dir(output_dir)
        / f"{condition.label}_relaybp_{config_hash}_{record_id}.npz"
    )


def relaybp_unconverged_details_path(
    output_dir: Path, condition: Condition, config_hash: str, record_id: str
) -> Path:
    return (
        relaybp_unconverged_details_dir(output_dir)
        / f"{condition.label}_relaybp_{config_hash}_{record_id}.npz"
    )


def bposd_retry_details_path(
    output_dir: Path, condition: Condition, config_hash: str, record_id: str
) -> Path:
    return (
        bposd_retry_details_dir(output_dir)
        / f"{condition.label}_relaybp_{config_hash}_{record_id}.npz"
    )


def shot_chunk_path(output_dir: Path, condition: Condition, chunk_index: int) -> Path:
    return shots_dir(output_dir, condition) / f"chunk_{chunk_index:06d}.npz"


def shots_manifest_path(output_dir: Path, condition: Condition) -> Path:
    return shots_dir(output_dir, condition) / "manifest.json"


def packed_width(num_bits: int) -> int:
    return (int(num_bits) + 7) // 8


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def shot_file_info(
    path: Path, expected_shots: int, num_detectors: int, num_observables: int
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            detector_shape = tuple(int(x) for x in payload["detector_shape"])
            observable_shape = tuple(int(x) for x in payload["observable_shape"])
            detector_data = payload["detector_data"]
            observable_data = payload["observable_data"]
            bitorder = str(
                payload["bitorder"].item()
                if hasattr(payload["bitorder"], "item")
                else payload["bitorder"]
            )
            ok = (
                int(payload["num_shots"]) == int(expected_shots)
                and detector_shape == (int(expected_shots), int(num_detectors))
                and observable_shape == (int(expected_shots), int(num_observables))
                and detector_data.shape == (int(expected_shots), packed_width(num_detectors))
                and observable_data.shape == (int(expected_shots), packed_width(num_observables))
                and detector_data.dtype == np.uint8
                and observable_data.dtype == np.uint8
                and bitorder == "little"
            )
            if not ok:
                return None
            return {
                **file_fingerprint(path),
                "num_shots": int(expected_shots),
                "start_shot": int(payload["start_shot"]),
                "chunk_index": int(payload["chunk_index"]),
                "seed": int(payload["seed"]),
                "detector_shape": list(detector_shape),
                "observable_shape": list(observable_shape),
                "detector_packed_shape": list(detector_data.shape),
                "observable_packed_shape": list(observable_data.shape),
                "bitorder": bitorder,
            }
    except Exception:
        return None


def shot_matches_task(task: GenerateTask) -> bool:
    """Check the stored sample dimensions, seed, and position in the run."""
    info = shot_file_info(
        Path(task.output_path), task.num_shots, task.num_detectors, task.num_observables
    )
    return info is not None and all(
        info[key] == getattr(task, key) for key in ("start_shot", "chunk_index", "seed")
    )


def build_shots_manifest(
    output_dir: Path,
    condition: Condition,
    tasks: Sequence[GenerateTask],
    meta: dict,
    shots: int,
    shot_chunk: int,
    seed: int,
) -> dict[str, object]:
    artifact = artifact_paths(output_dir, condition)
    chunks = []
    for task in tasks:
        info = shot_file_info(
            Path(task.output_path),
            task.num_shots,
            task.num_detectors,
            task.num_observables,
        )
        if info is None or any(
            info[key] != getattr(task, key) for key in ("start_shot", "chunk_index", "seed")
        ):
            raise ValueError(f"Missing or incompatible shot chunk: {task.output_path}")
        chunks.append(
            {
                "chunk_index": int(task.chunk_index),
                "start_shot": int(task.start_shot),
                "num_shots": int(task.num_shots),
                "seed": int(task.seed),
                **info,
            }
        )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "label": condition.label,
        "code_name": condition.code_name,
        "basis": condition.basis,
        "decoding_mode": condition.decoding_mode,
        "p": condition.p_noise,
        "cycles": condition.cycles,
        "shots_requested": int(shots),
        "shot_chunk": int(shot_chunk),
        "condition_seed": int(seed),
        "num_detectors": int(meta["num_detectors"]),
        "num_observables": int(meta["num_observables"]),
        "artifact_fingerprints": {
            name: file_fingerprint(path) for name, path in artifact.items() if path.exists()
        },
        "chunks": chunks,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def write_shots_manifest(
    output_dir: Path,
    condition: Condition,
    tasks: Sequence[GenerateTask],
    meta: dict,
    shots: int,
    shot_chunk: int,
    seed: int,
) -> dict[str, object]:
    manifest = build_shots_manifest(output_dir, condition, tasks, meta, shots, shot_chunk, seed)
    write_json_atomic(shots_manifest_path(output_dir, condition), manifest)
    return manifest


def load_shots_manifest(output_dir: Path, condition: Condition) -> dict[str, object]:
    path = shots_manifest_path(output_dir, condition)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing shot manifest for {condition.label}. Run --stage generate first: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    if manifest.get("manifest_hash") != expected:
        raise ValueError(f"Shot manifest hash mismatch for {condition.label}: {path}")
    artifacts = {
        name: file_fingerprint(artifact)
        for name, artifact in artifact_paths(output_dir, condition).items()
        if artifact.exists()
    }
    if manifest.get("artifact_fingerprints") != artifacts:
        raise ValueError(
            f"{condition.label}: circuit artifacts changed after sampling. "
            "Run --stage generate --force-shots before decoding."
        )
    for chunk in manifest["chunks"]:
        shot_path = Path(chunk["path"])
        if not shot_path.exists() or any(
            chunk.get(key) != value for key, value in file_fingerprint(shot_path).items()
        ):
            raise ValueError(
                f"{condition.label}: sampled shots changed: {shot_path}. "
                "Run --stage generate before decoding."
            )
    return manifest


def sparse_to_payload(prefix: str, matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    csr = matrix.tocsr().astype(np.uint8)
    csr.data %= 2
    csr.eliminate_zeros()
    return {
        f"{prefix}_data": csr.data.astype(np.uint8),
        f"{prefix}_indices": csr.indices.astype(np.int64),
        f"{prefix}_indptr": csr.indptr.astype(np.int64),
        f"{prefix}_shape": np.asarray(csr.shape, dtype=np.int64),
    }


def payload_to_sparse(payload, prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(x) for x in payload[f"{prefix}_shape"])
    return sparse.csr_matrix(
        (
            payload[f"{prefix}_data"].astype(np.uint8),
            payload[f"{prefix}_indices"].astype(np.int32),
            payload[f"{prefix}_indptr"].astype(np.int32),
        ),
        shape=shape,
        dtype=np.uint8,
    )


def save_check_matrices(path: Path, check_matrix, observables_matrix, error_priors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    payload.update(sparse_to_payload("check", check_matrix))
    payload.update(sparse_to_payload("obs", observables_matrix))
    payload["error_priors"] = np.asarray(error_priors, dtype=np.float64)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fout:
        np.savez_compressed(fout, **payload)
    tmp.replace(path)


def load_check_matrices(
    path: Path,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        check = payload_to_sparse(payload, "check")
        obs = payload_to_sparse(payload, "obs")
        priors = np.asarray(payload["error_priors"], dtype=np.float64)
    return check, obs, priors


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
