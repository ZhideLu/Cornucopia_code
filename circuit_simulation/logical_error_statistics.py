"""Logical-failure accounting and simulation summaries."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from circuit_simulation.shot_data import (
    load_shots_manifest,
    shots_manifest_path,
    write_json_atomic,
)
from circuit_simulation.simulation_parameters import (
    DECODE_SCHEMA_VERSION,
    SCRIPT_VERSION,
    Condition,
    RelayBPConfig,
)


def aggregate_decode_records(
    condition: Condition,
    meta: dict,
    records: list[dict[str, object]],
    jsonl_path: Path,
    config: RelayBPConfig,
    config_hash: str,
    shot_manifest_hash: str,
    shot_manifest_path_value: Path,
) -> dict[str, object]:
    decoded_shots = int(sum(int(record.get("num_shots", 0)) for record in records))
    failures = int(sum(int(record.get("failures", 0)) for record in records))
    has_observable_mismatches = all("observable_mismatches" in record for record in records)
    has_convergence = all("converged" in record and "unconverged" in record for record in records)
    has_convergence_breakdown = all(
        "c_failures" in record
        and "u_failures" in record
        and "c_observable_mismatches" in record
        and "u_observable_mismatches" in record
        for record in records
    )
    observable_mismatches = (
        int(sum(int(record.get("observable_mismatches", 0)) for record in records))
        if has_observable_mismatches
        else None
    )
    converged = (
        int(sum(int(record.get("converged", 0)) for record in records)) if has_convergence else None
    )
    unconverged = (
        int(sum(int(record.get("unconverged", 0)) for record in records))
        if has_convergence
        else None
    )
    iterations_total = (
        int(sum(int(record.get("iterations_total", 0)) for record in records))
        if has_convergence
        else None
    )
    max_iterations = (
        int(max((int(record.get("max_iterations", 0)) for record in records), default=0))
        if has_convergence
        else None
    )
    c_failures = (
        int(sum(int(record.get("c_failures", 0)) for record in records))
        if has_convergence_breakdown
        else None
    )
    c_observable_mismatches = (
        int(sum(int(record.get("c_observable_mismatches", 0)) for record in records))
        if has_convergence_breakdown
        else None
    )
    c_iterations_total = (
        int(sum(int(record.get("c_iterations_total", 0)) for record in records))
        if has_convergence_breakdown
        else None
    )
    c_max_iterations = (
        int(max((int(record.get("c_max_iterations", 0)) for record in records), default=0))
        if has_convergence_breakdown
        else None
    )
    u_failures = (
        int(sum(int(record.get("u_failures", 0)) for record in records))
        if has_convergence_breakdown
        else None
    )
    u_observable_mismatches = (
        int(sum(int(record.get("u_observable_mismatches", 0)) for record in records))
        if has_convergence_breakdown
        else None
    )
    u_iterations_total = (
        int(sum(int(record.get("u_iterations_total", 0)) for record in records))
        if has_convergence_breakdown
        else None
    )
    u_max_iterations = (
        int(max((int(record.get("u_max_iterations", 0)) for record in records), default=0))
        if has_convergence_breakdown
        else None
    )
    has_relaybp_primary = isinstance(config, RelayBPConfig) and all(
        "primary_relaybp_converged" in record and "primary_relaybp_unconverged" in record
        for record in records
    )
    primary_relaybp_converged = (
        int(sum(int(record.get("primary_relaybp_converged", 0)) for record in records))
        if has_relaybp_primary
        else None
    )
    primary_relaybp_unconverged = (
        int(sum(int(record.get("primary_relaybp_unconverged", 0)) for record in records))
        if has_relaybp_primary
        else None
    )
    primary_relaybp_iterations_total = (
        int(sum(int(record.get("primary_relaybp_iterations_total", 0)) for record in records))
        if has_relaybp_primary
        else None
    )
    primary_relaybp_max_iterations = (
        int(
            max(
                (int(record.get("primary_relaybp_max_iterations", 0)) for record in records),
                default=0,
            )
        )
        if has_relaybp_primary
        else None
    )
    primary_relaybp_avg_iterations = (
        None
        if primary_relaybp_iterations_total is None or decoded_shots <= 0
        else primary_relaybp_iterations_total / decoded_shots
    )
    primary_relaybp_convergence_rate = (
        None
        if primary_relaybp_converged is None or decoded_shots <= 0
        else primary_relaybp_converged / decoded_shots
    )
    has_relaybp_fallback = any(bool(record.get("relaybp_fallback_enabled")) for record in records)
    relaybp_fallback_config = None
    if has_relaybp_fallback:
        for record in records:
            payload = record.get("relaybp_fallback_config")
            if isinstance(payload, dict):
                relaybp_fallback_config = payload
                break
    relaybp_fallback_called = (
        int(sum(int(record.get("relaybp_fallback_called", 0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_converged = (
        int(sum(int(record.get("relaybp_fallback_converged", 0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_unconverged = (
        int(sum(int(record.get("relaybp_fallback_unconverged", 0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_failures = (
        int(sum(int(record.get("relaybp_fallback_failures", 0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_observable_mismatches = (
        int(sum(int(record.get("relaybp_fallback_observable_mismatches", 0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_iterations_total = (
        int(sum(int(record.get("relaybp_fallback_iterations_total", 0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_avg_iterations = (
        None
        if relaybp_fallback_iterations_total is None or not relaybp_fallback_called
        else relaybp_fallback_iterations_total / relaybp_fallback_called
    )
    relaybp_fallback_max_iterations = (
        int(
            max(
                (int(record.get("relaybp_fallback_max_iterations", 0)) for record in records),
                default=0,
            )
        )
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_seconds = (
        float(sum(float(record.get("relaybp_fallback_seconds", 0.0)) for record in records))
        if has_relaybp_fallback
        else None
    )
    relaybp_fallback_convergence_rate = (
        None
        if relaybp_fallback_called is None
        or relaybp_fallback_called <= 0
        or relaybp_fallback_converged is None
        else relaybp_fallback_converged / relaybp_fallback_called
    )
    has_relaybp_fallback_retry = any(
        bool(record.get("relaybp_fallback_retry_unconverged_enabled")) for record in records
    )
    relaybp_fallback_retry_source_hashes = (
        sorted(
            {
                str(record.get("relaybp_fallback_retry_source_hash"))
                for record in records
                if record.get("relaybp_fallback_retry_source_hash")
            }
        )
        if has_relaybp_fallback_retry
        else []
    )
    relaybp_fallback_retry_retried = (
        int(sum(int(record.get("relaybp_fallback_retry_retried", 0)) for record in records))
        if has_relaybp_fallback_retry
        else None
    )
    relaybp_fallback_retry_converged = (
        int(sum(int(record.get("relaybp_fallback_retry_converged", 0)) for record in records))
        if has_relaybp_fallback_retry
        else None
    )
    relaybp_fallback_retry_unconverged = (
        int(sum(int(record.get("relaybp_fallback_retry_unconverged", 0)) for record in records))
        if has_relaybp_fallback_retry
        else None
    )
    relaybp_fallback_retry_failures = (
        int(sum(int(record.get("relaybp_fallback_retry_failures", 0)) for record in records))
        if has_relaybp_fallback_retry
        else None
    )
    relaybp_fallback_retry_observable_mismatches = (
        int(
            sum(
                int(record.get("relaybp_fallback_retry_observable_mismatches", 0))
                for record in records
            )
        )
        if has_relaybp_fallback_retry
        else None
    )
    has_bposd_retry = any(bool(record.get("bposd_retry_unconverged_enabled")) for record in records)
    bposd_retry_config = None
    if has_bposd_retry:
        for record in records:
            payload = record.get("bposd_retry_config")
            if isinstance(payload, dict):
                bposd_retry_config = payload
                break
    bposd_retry_source_hashes = (
        sorted(
            {
                str(record.get("bposd_retry_source_hash"))
                for record in records
                if record.get("bposd_retry_source_hash")
            }
        )
        if has_bposd_retry
        else []
    )
    bposd_retry_retried = (
        int(sum(int(record.get("bposd_retry_retried", 0)) for record in records))
        if has_bposd_retry
        else None
    )
    bposd_retry_syndrome_matched = (
        int(sum(int(record.get("bposd_retry_syndrome_matched", 0)) for record in records))
        if has_bposd_retry
        else None
    )
    bposd_retry_logical_correct = (
        int(sum(int(record.get("bposd_retry_logical_correct", 0)) for record in records))
        if has_bposd_retry
        else None
    )
    bposd_retry_logical_failures = (
        int(sum(int(record.get("bposd_retry_logical_failures", 0)) for record in records))
        if has_bposd_retry
        else None
    )
    bposd_retry_unresolved = (
        int(sum(int(record.get("bposd_retry_unresolved", 0)) for record in records))
        if has_bposd_retry
        else None
    )
    bposd_retry_observable_mismatches = (
        int(sum(int(record.get("bposd_retry_observable_mismatches", 0)) for record in records))
        if has_bposd_retry
        else None
    )
    bposd_retry_seconds = (
        float(sum(float(record.get("bposd_retry_seconds", 0.0)) for record in records))
        if has_bposd_retry
        else None
    )
    has_mip_fallback = any(bool(record.get("mip_fallback_enabled")) for record in records)
    mip_fallback_config = None
    if has_mip_fallback:
        for record in records:
            payload = record.get("mip_fallback_config")
            if isinstance(payload, dict):
                mip_fallback_config = payload
                break
    mip_fallback_called = (
        int(sum(int(record.get("mip_fallback_called", 0)) for record in records))
        if has_mip_fallback
        else None
    )
    mip_fallback_solved = (
        int(sum(int(record.get("mip_fallback_solved", 0)) for record in records))
        if has_mip_fallback
        else None
    )
    mip_fallback_unsolved = (
        int(sum(int(record.get("mip_fallback_unsolved", 0)) for record in records))
        if has_mip_fallback
        else None
    )
    mip_fallback_failures = (
        int(sum(int(record.get("mip_fallback_failures", 0)) for record in records))
        if has_mip_fallback
        else None
    )
    mip_fallback_observable_mismatches = (
        int(sum(int(record.get("mip_fallback_observable_mismatches", 0)) for record in records))
        if has_mip_fallback
        else None
    )
    mip_fallback_seconds = (
        float(sum(float(record.get("mip_fallback_seconds", 0.0)) for record in records))
        if has_mip_fallback
        else None
    )
    mip_fallback_mode = None
    if has_mip_fallback:
        for record in records:
            value = record.get("mip_fallback_mode")
            if value is not None:
                mip_fallback_mode = value
                break
    mip_fallback_workers = (
        int(max((int(record.get("mip_fallback_workers", 0)) for record in records), default=0))
        if has_mip_fallback
        else None
    )
    has_mip_retry_unsolved = any(
        bool(record.get("mip_retry_unsolved_enabled")) for record in records
    )
    mip_retry_source_hashes = (
        sorted(
            {
                str(record.get("mip_retry_source_hash"))
                for record in records
                if record.get("mip_retry_source_hash")
            }
        )
        if has_mip_retry_unsolved
        else []
    )
    mip_retry_retried = (
        int(sum(int(record.get("mip_retry_retried", 0)) for record in records))
        if has_mip_retry_unsolved
        else None
    )
    mip_retry_solved = (
        int(sum(int(record.get("mip_retry_solved", 0)) for record in records))
        if has_mip_retry_unsolved
        else None
    )
    mip_retry_unsolved = (
        int(sum(int(record.get("mip_retry_unsolved", 0)) for record in records))
        if has_mip_retry_unsolved
        else None
    )
    mip_retry_failures = (
        int(sum(int(record.get("mip_retry_failures", 0)) for record in records))
        if has_mip_retry_unsolved
        else None
    )
    mip_retry_observable_mismatches = (
        int(sum(int(record.get("mip_retry_observable_mismatches", 0)) for record in records))
        if has_mip_retry_unsolved
        else None
    )
    ler = None if decoded_shots == 0 else failures / decoded_shots
    per_cycle = None if ler is None else 1.0 - (1.0 - ler) ** (1.0 / max(condition.cycles, 1))
    num_logicals = int(meta.get("num_observables", 0))
    per_logical = (
        None
        if per_cycle is None or num_logicals <= 0
        else 1.0 - (1.0 - per_cycle) ** (1.0 / num_logicals)
    )
    direct_per_logical = (
        None
        if observable_mismatches is None
        or decoded_shots <= 0
        or num_logicals <= 0
        or condition.cycles <= 0
        else observable_mismatches / (decoded_shots * num_logicals * int(condition.cycles))
    )
    convergence_rate = (
        None if converged is None or decoded_shots <= 0 else converged / decoded_shots
    )
    avg_iterations = (
        None if iterations_total is None or decoded_shots <= 0 else iterations_total / decoded_shots
    )

    def group_rates(
        group_shots: int | None,
        group_failures: int | None,
        group_observable_mismatches: int | None,
        group_iterations_total: int | None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        if group_shots is None or group_failures is None or group_shots <= 0:
            return None, None, None, None
        group_ler = group_failures / group_shots
        group_per_cycle = 1.0 - (1.0 - group_ler) ** (1.0 / max(condition.cycles, 1))
        group_per_logical = (
            None if num_logicals <= 0 else 1.0 - (1.0 - group_per_cycle) ** (1.0 / num_logicals)
        )
        group_direct = (
            None
            if group_observable_mismatches is None or num_logicals <= 0 or condition.cycles <= 0
            else group_observable_mismatches / (group_shots * num_logicals * int(condition.cycles))
        )
        group_avg_iterations = (
            None if group_iterations_total is None else group_iterations_total / group_shots
        )
        return group_ler, group_per_logical, group_direct, group_avg_iterations

    c_ler, c_per_logical, c_direct_per_logical, c_avg_iterations = group_rates(
        converged, c_failures, c_observable_mismatches, c_iterations_total
    )
    u_ler, u_per_logical, u_direct_per_logical, u_avg_iterations = group_rates(
        unconverged, u_failures, u_observable_mismatches, u_iterations_total
    )
    stderr = None if ler is None else math.sqrt(max(ler * (1.0 - ler), 0.0) / max(decoded_shots, 1))
    decoder = "relaybp"
    record = {
        "label": condition.label,
        "decoder": decoder,
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
        "shots_decoded": decoded_shots,
        "failures": failures,
        "observable_mismatches": observable_mismatches,
        "converged": converged,
        "unconverged": unconverged,
        "convergence_rate": convergence_rate,
        "iterations_total": iterations_total,
        "avg_iterations": avg_iterations,
        "max_iterations": max_iterations,
        "c_failures": c_failures,
        "c_LER": c_ler,
        "c_LER_per_cycle_per_logical": c_per_logical,
        "c_observable_mismatches": c_observable_mismatches,
        "c_direct_per_logical_LER": c_direct_per_logical,
        "c_iterations_total": c_iterations_total,
        "c_avg_iterations": c_avg_iterations,
        "c_max_iterations": c_max_iterations,
        "u_failures": u_failures,
        "u_LER": u_ler,
        "u_LER_per_cycle_per_logical": u_per_logical,
        "u_observable_mismatches": u_observable_mismatches,
        "u_direct_per_logical_LER": u_direct_per_logical,
        "u_iterations_total": u_iterations_total,
        "u_avg_iterations": u_avg_iterations,
        "u_max_iterations": u_max_iterations,
        "primary_relaybp_converged": primary_relaybp_converged,
        "primary_relaybp_unconverged": primary_relaybp_unconverged,
        "primary_relaybp_convergence_rate": primary_relaybp_convergence_rate,
        "primary_relaybp_iterations_total": primary_relaybp_iterations_total,
        "primary_relaybp_avg_iterations": primary_relaybp_avg_iterations,
        "primary_relaybp_max_iterations": primary_relaybp_max_iterations,
        "relaybp_fallback_enabled": bool(has_relaybp_fallback),
        "relaybp_fallback_config": relaybp_fallback_config,
        "relaybp_fallback_called": relaybp_fallback_called,
        "relaybp_fallback_converged": relaybp_fallback_converged,
        "relaybp_fallback_unconverged": relaybp_fallback_unconverged,
        "relaybp_fallback_convergence_rate": relaybp_fallback_convergence_rate,
        "relaybp_fallback_failures": relaybp_fallback_failures,
        "relaybp_fallback_observable_mismatches": (relaybp_fallback_observable_mismatches),
        "relaybp_fallback_iterations_total": relaybp_fallback_iterations_total,
        "relaybp_fallback_avg_iterations": relaybp_fallback_avg_iterations,
        "relaybp_fallback_max_iterations": relaybp_fallback_max_iterations,
        "relaybp_fallback_seconds": relaybp_fallback_seconds,
        "relaybp_fallback_retry_unconverged_enabled": bool(has_relaybp_fallback_retry),
        "relaybp_fallback_retry_source_hashes": (relaybp_fallback_retry_source_hashes),
        "relaybp_fallback_retry_retried": relaybp_fallback_retry_retried,
        "relaybp_fallback_retry_converged": relaybp_fallback_retry_converged,
        "relaybp_fallback_retry_unconverged": relaybp_fallback_retry_unconverged,
        "relaybp_fallback_retry_failures": relaybp_fallback_retry_failures,
        "relaybp_fallback_retry_observable_mismatches": (
            relaybp_fallback_retry_observable_mismatches
        ),
        "bposd_retry_unconverged_enabled": bool(has_bposd_retry),
        "bposd_retry_config": bposd_retry_config,
        "bposd_retry_source_hashes": bposd_retry_source_hashes,
        "bposd_retry_retried": bposd_retry_retried,
        "bposd_retry_syndrome_matched": bposd_retry_syndrome_matched,
        "bposd_retry_logical_correct": bposd_retry_logical_correct,
        "bposd_retry_logical_failures": bposd_retry_logical_failures,
        "bposd_retry_unresolved": bposd_retry_unresolved,
        "bposd_retry_observable_mismatches": bposd_retry_observable_mismatches,
        "bposd_retry_seconds": bposd_retry_seconds,
        "mip_fallback_enabled": bool(has_mip_fallback),
        "mip_fallback_config": mip_fallback_config,
        "mip_fallback_called": mip_fallback_called,
        "mip_fallback_solved": mip_fallback_solved,
        "mip_fallback_unsolved": mip_fallback_unsolved,
        "mip_fallback_failures": mip_fallback_failures,
        "mip_fallback_observable_mismatches": mip_fallback_observable_mismatches,
        "mip_fallback_mode": mip_fallback_mode,
        "mip_fallback_workers": mip_fallback_workers,
        "mip_fallback_seconds": mip_fallback_seconds,
        "mip_retry_unsolved_enabled": bool(has_mip_retry_unsolved),
        "mip_retry_source_hashes": mip_retry_source_hashes,
        "mip_retry_retried": mip_retry_retried,
        "mip_retry_solved": mip_retry_solved,
        "mip_retry_unsolved": mip_retry_unsolved,
        "mip_retry_failures": mip_retry_failures,
        "mip_retry_observable_mismatches": mip_retry_observable_mismatches,
        "logical_error_rate": ler,
        "logical_error_rate_stderr": stderr,
        "logical_error_rate_per_cycle": per_cycle,
        "logical_error_rate_per_cycle_per_logical": per_logical,
        "direct_per_logical_LER": direct_per_logical,
        "num_detectors": int(meta.get("num_detectors", 0)),
        "num_observables": int(meta.get("num_observables", 0)),
        "num_error_columns": int(meta.get("num_error_columns", 0)),
        "num_dem_errors": int(meta.get("num_dem_errors", 0)),
        "detectors_removed_for_xz": int(meta.get("detectors_removed_for_xz", 0)),
        "decode_jsonl_path": str(jsonl_path),
        "shot_manifest_path": str(shot_manifest_path_value),
        "shot_manifest_hash": shot_manifest_hash,
        "decode_schema_version": DECODE_SCHEMA_VERSION,
    }
    record["relaybp_config_hash"] = config_hash
    record["relaybp_config"] = asdict(config)
    return record


def condition_generation_summary(
    condition: Condition, output_dir: Path, shots: int, shot_chunk: int, meta: dict
) -> dict[str, object]:
    manifest = (
        load_shots_manifest(output_dir, condition)
        if shots_manifest_path(output_dir, condition).exists()
        else None
    )
    chunks = [] if manifest is None else list(manifest.get("chunks", []))
    available = int(sum(int(chunk.get("num_shots", 0)) for chunk in chunks))
    chunks_available = len(chunks)
    chunks_expected = math.ceil(shots / shot_chunk)
    return {
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
        "shots_requested": shots,
        "shots_available": available,
        "chunks_available": chunks_available,
        "chunks_expected": chunks_expected,
        "shot_manifest_path": (
            None if manifest is None else str(shots_manifest_path(output_dir, condition))
        ),
        "shot_manifest_hash": (None if manifest is None else manifest.get("manifest_hash")),
        "logical_error_rate": None,
        "observable_mismatches": None,
        "direct_per_logical_LER": None,
        "converged": None,
        "unconverged": None,
        "convergence_rate": None,
        "iterations_total": None,
        "avg_iterations": None,
        "max_iterations": None,
        "c_failures": None,
        "c_LER": None,
        "c_LER_per_cycle_per_logical": None,
        "c_observable_mismatches": None,
        "c_direct_per_logical_LER": None,
        "c_iterations_total": None,
        "c_avg_iterations": None,
        "c_max_iterations": None,
        "u_failures": None,
        "u_LER": None,
        "u_LER_per_cycle_per_logical": None,
        "u_observable_mismatches": None,
        "u_direct_per_logical_LER": None,
        "u_iterations_total": None,
        "u_avg_iterations": None,
        "u_max_iterations": None,
        "num_detectors": int(meta.get("num_detectors", 0)),
        "num_observables": int(meta.get("num_observables", 0)),
        "num_error_columns": int(meta.get("num_error_columns", 0)),
        "num_dem_errors": int(meta.get("num_dem_errors", 0)),
        "detectors_removed_for_xz": int(meta.get("detectors_removed_for_xz", 0)),
    }


def summary_record_key(record: dict[str, object]) -> tuple[object]:
    if "shots_decoded" not in record:
        return (record.get("label"),)
    decoder = record.get("decoder")
    if decoder is None:
        decoder = "relaybp"
    try:
        p_value: object = f"{float(record.get('p')):.12g}"
    except (TypeError, ValueError):
        p_value = record.get("p")
    return (
        decoder,
        record.get("code_name"),
        record.get("parameter_label"),
        record.get("P"),
        record.get("expected_d"),
        record.get("basis"),
        record.get("decoding_mode"),
        record.get("cycles"),
        p_value,
    )


def load_existing_summary_records(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("records", [])
    merged: list[dict[str, object]] = []
    for record in records:
        if isinstance(record, dict):
            upsert_summary_record(merged, record)
    return merged


def upsert_summary_record(records: list[dict[str, object]], record: dict[str, object]) -> None:
    key = summary_record_key(record)
    for index, existing in enumerate(records):
        if summary_record_key(existing) == key:
            records[index] = record
            return
    records.append(record)


def _summary_int(record: dict[str, object], key: str, default: int = 0) -> int:
    value = record.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary_float(record: dict[str, object], key: str, default: float = 0.0) -> float:
    value = record.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary_text(value: object) -> str:
    return "None" if value is None else str(value)


def _format_sig3(value: object) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def _format_float4(value: object) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _format_count(value: object) -> str:
    if value is None:
        return "None"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_p_value(value: object) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_aligned_table(headers: Sequence[str], values: Sequence[str]) -> list[str]:
    widths = [
        max(len(str(header)), len(str(value))) + 2
        for header, value in zip(headers, values, strict=True)
    ]
    return [
        "".join(
            str(header).ljust(width) for header, width in zip(headers, widths, strict=True)
        ).rstrip(),
        "".join(
            str(value).rjust(width) for value, width in zip(values, widths, strict=True)
        ).rstrip(),
    ]


def _format_kv_block(title: str, rows: Sequence[tuple[str, str]]) -> list[str]:
    key_width = max((len(key) for key, _value in rows), default=0)
    lines = [title]
    for key, value in rows:
        lines.append(f"  {key:<{key_width}}  {value}")
    return lines


def _summary_sort_key(record: dict[str, object]) -> tuple[object, ...]:
    mode = str(record.get("decoding_mode") or "unknown")
    mode_order = {"xyz": 0, "xz": 1}.get(mode, 99)
    basis = str(record.get("basis") or "")
    basis_order = {"Z": 0, "X": 1}.get(basis, 99)
    return (
        mode_order,
        mode,
        _summary_float(record, "p"),
        _summary_int(record, "P"),
        str(record.get("code_name") or ""),
        basis_order,
        basis,
        _summary_int(record, "cycles"),
        str(record.get("label") or ""),
    )


def _format_summary_record(record: dict[str, object]) -> str:
    shots = record.get("shots_decoded", record.get("shots_available"))
    lines = [str(record.get("label"))]
    lines.append(
        "meta: "
        f"basis={_summary_text(record.get('basis'))}  "
        f"p={_format_p_value(record.get('p'))}  "
        f"cycles={_format_count(record.get('cycles'))}  "
        f"detectors={_format_count(record.get('num_detectors'))}  "
        f"observables={_format_count(record.get('num_observables'))}  "
        f"convergence_rate={_format_sig3(record.get('convergence_rate'))}"
    )
    decoder_name = str(record.get("decoder", "relaybp"))
    config = record.get("relaybp_config")
    if decoder_name == "relaybp" and isinstance(config, dict):
        lines.append(
            "decoder: relaybp  "
            f"gamma0={_summary_text(config.get('gamma0'))}  "
            f"pre_iter={_summary_text(config.get('pre_iter'))}  "
            f"num_sets={_summary_text(config.get('num_sets'))}  "
            f"set_max_iter={_summary_text(config.get('set_max_iter'))}  "
            f"gamma_dist=({_summary_text(config.get('gamma_dist_min'))},{_summary_text(config.get('gamma_dist_max'))})  "
            f"stop_nconv={_summary_text(config.get('stop_nconv'))}"
        )
    relaybp_fallback_config = record.get("relaybp_fallback_config")
    if (
        decoder_name == "relaybp"
        and isinstance(relaybp_fallback_config, dict)
        and relaybp_fallback_config.get("enabled")
    ):
        fallback_trigger = (
            "saved_unconverged_details"
            if record.get("relaybp_fallback_retry_unconverged_enabled")
            else "primary_relaybp_nonconverged"
        )
        lines.append(
            "fallback: relaybp_strong  "
            f"trigger={fallback_trigger}  "
            f"gamma0={_summary_text(relaybp_fallback_config.get('gamma0'))}  "
            f"pre_iter={_summary_text(relaybp_fallback_config.get('pre_iter'))}  "
            f"num_sets={_summary_text(relaybp_fallback_config.get('num_sets'))}  "
            f"set_max_iter={_summary_text(relaybp_fallback_config.get('set_max_iter'))}  "
            f"gamma_dist=({_summary_text(relaybp_fallback_config.get('gamma_dist_min'))},{_summary_text(relaybp_fallback_config.get('gamma_dist_max'))})  "
            f"stop_nconv={_summary_text(relaybp_fallback_config.get('stop_nconv'))}"
        )
    mip_config = record.get("mip_fallback_config")
    if decoder_name == "relaybp" and isinstance(mip_config, dict) and mip_config.get("enabled"):
        mip_trigger = (
            "relaybp_fallback_unconverged"
            if isinstance(relaybp_fallback_config, dict) and relaybp_fallback_config.get("enabled")
            else "relaybp_nonconverged"
        )
        lines.append(
            "fallback: mip  "
            f"time_limit={_summary_text(mip_config.get('time_limit'))}  "
            f"mip_rel_gap={_summary_text(mip_config.get('mip_rel_gap'))}  "
            f"feasible_only={_summary_text(mip_config.get('feasible_only'))}  "
            f"workers={_summary_text(mip_config.get('workers'))}  "
            f"threads={_summary_text(mip_config.get('threads'))}  "
            f"trigger={mip_trigger}  "
            f"count_unsolved_as_failure={_summary_text(mip_config.get('count_unsolved_as_failure'))}"
        )
    if decoder_name == "relaybp" and record.get("primary_relaybp_converged") is not None:
        lines.extend(
            _format_kv_block(
                "primary relaybp:",
                [
                    (
                        "primary_converged",
                        _format_count(record.get("primary_relaybp_converged")),
                    ),
                    (
                        "primary_unconverged",
                        _format_count(record.get("primary_relaybp_unconverged")),
                    ),
                    (
                        "primary_convergence_rate",
                        _format_sig3(record.get("primary_relaybp_convergence_rate")),
                    ),
                    (
                        "primary_avg_iterations",
                        _format_float4(record.get("primary_relaybp_avg_iterations")),
                    ),
                    (
                        "primary_max_iterations",
                        _format_count(record.get("primary_relaybp_max_iterations")),
                    ),
                ],
            )
        )
    if decoder_name == "relaybp" and bool(record.get("relaybp_fallback_enabled")):
        lines.extend(
            _format_kv_block(
                "relaybp fallback:",
                [
                    ("called", _format_count(record.get("relaybp_fallback_called"))),
                    (
                        "converged",
                        _format_count(record.get("relaybp_fallback_converged")),
                    ),
                    (
                        "unconverged",
                        _format_count(record.get("relaybp_fallback_unconverged")),
                    ),
                    (
                        "convergence_rate",
                        _format_sig3(record.get("relaybp_fallback_convergence_rate")),
                    ),
                    (
                        "fallback_failures",
                        _format_count(record.get("relaybp_fallback_failures")),
                    ),
                    (
                        "fallback_observable_mismatches",
                        _format_count(record.get("relaybp_fallback_observable_mismatches")),
                    ),
                    (
                        "fallback_avg_iterations",
                        _format_float4(record.get("relaybp_fallback_avg_iterations")),
                    ),
                    (
                        "fallback_max_iterations",
                        _format_count(record.get("relaybp_fallback_max_iterations")),
                    ),
                    (
                        "fallback_seconds",
                        _format_float4(record.get("relaybp_fallback_seconds")),
                    ),
                ],
            )
        )
    if decoder_name == "relaybp" and bool(record.get("relaybp_fallback_retry_unconverged_enabled")):
        lines.extend(
            _format_kv_block(
                "relaybp fallback retry-only:",
                [
                    (
                        "source_hashes",
                        ",".join(record.get("relaybp_fallback_retry_source_hashes") or []),
                    ),
                    (
                        "retried",
                        _format_count(record.get("relaybp_fallback_retry_retried")),
                    ),
                    (
                        "converged",
                        _format_count(record.get("relaybp_fallback_retry_converged")),
                    ),
                    (
                        "unconverged",
                        _format_count(record.get("relaybp_fallback_retry_unconverged")),
                    ),
                    (
                        "retry_failures",
                        _format_count(record.get("relaybp_fallback_retry_failures")),
                    ),
                    (
                        "retry_observable_mismatches",
                        _format_count(record.get("relaybp_fallback_retry_observable_mismatches")),
                    ),
                ],
            )
        )
    if decoder_name == "relaybp" and bool(record.get("bposd_retry_unconverged_enabled")):
        bposd_config = record.get("bposd_retry_config")
        if isinstance(bposd_config, dict):
            lines.append(
                "fallback: bposd_retry  "
                f"trigger=saved_relaybp_unconverged_details  "
                f"max_iter={_summary_text(bposd_config.get('max_iter'))}  "
                f"bp_method={_summary_text(bposd_config.get('bp_method'))}  "
                f"ms_scaling_factor={_summary_text(bposd_config.get('ms_scaling_factor'))}  "
                f"schedule={_summary_text(bposd_config.get('schedule'))}  "
                f"osd_method={_summary_text(bposd_config.get('osd_method'))}  "
                f"osd_order={_summary_text(bposd_config.get('osd_order'))}  "
                f"threads={_summary_text(bposd_config.get('threads'))}"
            )
        lines.extend(
            _format_kv_block(
                "bposd retry-only:",
                [
                    (
                        "source_hashes",
                        ",".join(record.get("bposd_retry_source_hashes") or []),
                    ),
                    ("retried", _format_count(record.get("bposd_retry_retried"))),
                    (
                        "syndrome_matched",
                        _format_count(record.get("bposd_retry_syndrome_matched")),
                    ),
                    (
                        "logical_correct",
                        _format_count(record.get("bposd_retry_logical_correct")),
                    ),
                    (
                        "logical_failures",
                        _format_count(record.get("bposd_retry_logical_failures")),
                    ),
                    (
                        "unresolved",
                        _format_count(record.get("bposd_retry_unresolved")),
                    ),
                    (
                        "retry_observable_mismatches",
                        _format_count(record.get("bposd_retry_observable_mismatches")),
                    ),
                    (
                        "retry_seconds",
                        _format_float4(record.get("bposd_retry_seconds")),
                    ),
                ],
            )
        )
    lines.extend(
        _format_kv_block(
            "all shots:",
            [
                ("shots", _format_count(shots)),
                ("failures", _format_count(record.get("failures"))),
                ("LER", _format_sig3(record.get("logical_error_rate"))),
                (
                    "LER_per_cycle_per_logical",
                    _format_sig3(record.get("logical_error_rate_per_cycle_per_logical")),
                ),
                (
                    "observable_mismatches",
                    _format_count(record.get("observable_mismatches")),
                ),
                (
                    "direct_per_logical_LER",
                    _format_sig3(record.get("direct_per_logical_LER")),
                ),
                ("avg_iterations", _format_float4(record.get("avg_iterations"))),
                ("max_iterations", _format_count(record.get("max_iterations"))),
            ],
        )
    )
    c_title = (
        "relaybp-converged after fallback shots:"
        if bool(record.get("relaybp_fallback_enabled"))
        else "converged shots:"
    )
    lines.extend(
        _format_kv_block(
            c_title,
            [
                ("converged", _format_count(record.get("converged"))),
                ("c_failures", _format_count(record.get("c_failures"))),
                ("c_LER", _format_sig3(record.get("c_LER"))),
                (
                    "c_LER_per_cycle_per_logical",
                    _format_sig3(record.get("c_LER_per_cycle_per_logical")),
                ),
                (
                    "c_observable_mismatches",
                    _format_count(record.get("c_observable_mismatches")),
                ),
                (
                    "c_direct_per_logical_LER",
                    _format_sig3(record.get("c_direct_per_logical_LER")),
                ),
                ("c_avg_iterations", _format_float4(record.get("c_avg_iterations"))),
                ("c_max_iterations", _format_count(record.get("c_max_iterations"))),
            ],
        )
    )
    u_title = (
        "still relaybp-unconverged before MIP shots:"
        if bool(record.get("relaybp_fallback_enabled"))
        else "unconverged shots:"
    )
    lines.extend(
        _format_kv_block(
            u_title,
            [
                ("unconverged", _format_count(record.get("unconverged"))),
                ("u_failures", _format_count(record.get("u_failures"))),
                ("u_LER", _format_sig3(record.get("u_LER"))),
                (
                    "u_LER_per_cycle_per_logical",
                    _format_sig3(record.get("u_LER_per_cycle_per_logical")),
                ),
                (
                    "u_observable_mismatches",
                    _format_count(record.get("u_observable_mismatches")),
                ),
                (
                    "u_direct_per_logical_LER",
                    _format_sig3(record.get("u_direct_per_logical_LER")),
                ),
                ("u_avg_iterations", _format_float4(record.get("u_avg_iterations"))),
                ("u_max_iterations", _format_count(record.get("u_max_iterations"))),
            ],
        )
    )
    if bool(record.get("mip_fallback_enabled")):
        lines.extend(
            _format_kv_block(
                "mip fallback:",
                [
                    ("called", _format_count(record.get("mip_fallback_called"))),
                    ("solved", _format_count(record.get("mip_fallback_solved"))),
                    ("unsolved", _format_count(record.get("mip_fallback_unsolved"))),
                    ("mip_failures", _format_count(record.get("mip_fallback_failures"))),
                    (
                        "mip_observable_mismatches",
                        _format_count(record.get("mip_fallback_observable_mismatches")),
                    ),
                    ("mode", _summary_text(record.get("mip_fallback_mode"))),
                    ("workers", _format_count(record.get("mip_fallback_workers"))),
                    (
                        "mip_seconds",
                        _format_float4(record.get("mip_fallback_seconds")),
                    ),
                    (
                        "retry_source_hashes",
                        ",".join(record.get("mip_retry_source_hashes") or []),
                    ),
                    ("retry_retried", _format_count(record.get("mip_retry_retried"))),
                    ("retry_solved", _format_count(record.get("mip_retry_solved"))),
                    ("retry_unsolved", _format_count(record.get("mip_retry_unsolved"))),
                    ("retry_failures", _format_count(record.get("mip_retry_failures"))),
                    (
                        "retry_observable_mismatches",
                        _format_count(record.get("mip_retry_observable_mismatches")),
                    ),
                ],
            )
        )
    return "\n".join(lines)


def write_summary(
    output_dir: Path, records: list[dict[str, object]], summary_stem: str = "summary"
) -> None:
    write_json_atomic(
        output_dir / f"{summary_stem}.json",
        {
            "script_version": SCRIPT_VERSION,
            "generated_at_unix": time.time(),
            "records": records,
        },
    )

    lines = [
        "Cornucopia circuit-level noise simulation summary",
        "Grouped by decoding_mode, then physical error rate p, then code.",
        "",
    ]
    sorted_records = sorted(records, key=_summary_sort_key)
    if not sorted_records:
        lines.extend(["(no records)", ""])
        (output_dir / f"{summary_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    sentinel = object()
    current_mode: object = sentinel
    current_p: object = sentinel
    current_code: object = sentinel
    for record in sorted_records:
        mode = record.get("decoding_mode") or "unknown"
        p_value = _format_p_value(record.get("p"))
        code_key = (
            record.get("code_name") or "unknown",
            record.get("parameter_label"),
            record.get("P"),
            record.get("expected_d"),
        )
        if mode != current_mode:
            if current_mode is not sentinel:
                lines.append("")
            lines.append(f"## mode={mode}")
            current_mode = mode
            current_p = sentinel
            current_code = sentinel
        if p_value != current_p:
            if current_p is not sentinel:
                lines.append("")
            lines.append(f"### p={p_value}")
            current_p = p_value
            current_code = sentinel
        if code_key != current_code:
            code_name, parameter_label, p_code, expected_d = code_key
            lines.append("")
            lines.append(f"code={code_name} {parameter_label} P={p_code} expected_d={expected_d}")
            current_code = code_key
        lines.append(_format_summary_record(record))
        lines.append("")
    lines.append("")
    (output_dir / f"{summary_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
