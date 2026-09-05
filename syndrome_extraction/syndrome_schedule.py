#!/usr/bin/env python3
"""Stim syndrome circuits for Cornucopia codes.

The parallel syndrome-extraction order is:

  t=00: Z=G3  X=F0
  t=01: Z=G2  X=F1
  t=02: Z=G1  X=F2
  t=03: Z=G0  X=F3
  t=04: Z=G5  X=F4
  t=05: Z=G4  X=F5
  t=06: Z=F5  X=G4
  t=07: Z=F4  X=G5
  t=08: Z=F3  X=G0
  t=09: Z=F2  X=G1
  t=10: Z=F1  X=G2
  t=11: Z=F0  X=G3
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
from scipy import sparse

MeasurementBasis = Literal["Z", "X"]
AffineMap = Tuple[int, int]


PARALLEL_Z_ORDER: tuple[tuple[str, int], ...] = (
    ("G", 3),
    ("G", 2),
    ("G", 1),
    ("G", 0),
    ("G", 5),
    ("G", 4),
    ("F", 5),
    ("F", 4),
    ("F", 3),
    ("F", 2),
    ("F", 1),
    ("F", 0),
)

PARALLEL_X_ORDER: tuple[tuple[str, int], ...] = (
    ("F", 0),
    ("F", 1),
    ("F", 2),
    ("F", 3),
    ("F", 4),
    ("F", 5),
    ("G", 4),
    ("G", 5),
    ("G", 0),
    ("G", 1),
    ("G", 2),
    ("G", 3),
)


@dataclass(frozen=True)
class StimNoiseConfig:
    """Circuit-level noise used by the constructed syndrome circuit."""

    p_cx: float = 0.0
    p_measure_flip: float = 0.0
    p_final_measure_flip: float = 0.0
    p_reset_flip: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "p_cx": float(self.p_cx),
            "p_measure_flip": float(self.p_measure_flip),
            "p_final_measure_flip": float(self.p_final_measure_flip),
            "p_reset_flip": float(self.p_reset_flip),
        }


@dataclass(frozen=True)
class APMLayer:
    basis: MeasurementBasis
    family: str
    index: int
    edges: tuple[tuple[int, int], ...]

    @property
    def label(self) -> str:
        return f"{self.family}{self.index}"

    def __len__(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class CircuitBuildResult:
    circuit: object
    z_layers: tuple[APMLayer, ...]
    x_layers: tuple[APMLayer, ...]
    data_qubits: tuple[int, ...]
    z_ancillas: tuple[int, ...]
    x_ancillas: tuple[int, ...]
    noise: StimNoiseConfig


def require_stim():
    try:
        import stim  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("stim is required to build and validate these circuits.") from exc
    return stim


def order_labels(order: Sequence[tuple[str, int]]) -> list[str]:
    return [f"{family}{index}" for family, index in order]


def _mod_inverse(a: int, modulus: int) -> int:
    a %= modulus
    if gcd(a, modulus) != 1:
        raise ValueError(f"{a} is not invertible modulo {modulus}")
    return pow(a, -1, modulus)


def _local_images(apm: AffineMap, p: int, *, transpose: bool) -> np.ndarray:
    a, b = int(apm[0]) % p, int(apm[1]) % p
    xs = np.arange(p, dtype=np.int64)
    if transpose:
        inv_a = _mod_inverse(a, p)
        return ((inv_a * (xs - b)) % p).astype(np.int64)
    return ((a * xs + b) % p).astype(np.int64)


def build_apm_layers(
    code,
    basis: MeasurementBasis,
    order: Sequence[tuple[str, int]],
) -> tuple[APMLayer, ...]:
    """Build one edge layer per APM block for H_X or H_Z."""
    basis = basis.upper()  # type: ignore[assignment]
    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    p = int(code.spec.p)
    half = int(code.spec.half)
    active_rows = tuple(int(row) for row in code.spec.active_set)
    f_params = tuple(code.spec.f_params)
    g_params = tuple(code.spec.g_params)

    layers: list[APMLayer] = []
    for family, index in order:
        if family not in ("F", "G"):
            raise ValueError(f"bad APM family {family!r}")
        apm = f_params[index] if family == "F" else g_params[index]
        local_data = _local_images(apm, p, transpose=basis == "Z")
        edges: list[tuple[int, int]] = []
        for active_index, active_row in enumerate(active_rows):
            check_base = active_index * p
            if basis == "X":
                data_block = (index + active_row) % half
                if family == "G":
                    data_block += half
            else:
                data_block = (active_row - index) % half
                if family == "F":
                    data_block += half
            data_base = data_block * p
            for check_local in range(p):
                edges.append((check_base + check_local, data_base + int(local_data[check_local])))
        layers.append(APMLayer(basis=basis, family=family, index=index, edges=tuple(edges)))
    return tuple(layers)


def layers_to_sparse(layers: Sequence[APMLayer], shape: tuple[int, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for layer in layers:
        for check, data in layer.edges:
            rows.append(check)
            cols.append(data)
    values = np.ones(len(rows), dtype=np.uint8)
    out = sparse.coo_matrix((values, (rows, cols)), shape=shape, dtype=np.uint8).tocsr()
    out.data %= 2
    out.eliminate_zeros()
    return out


def matching_ok(layers: Sequence[APMLayer]) -> bool:
    for layer in layers:
        checks = [check for check, _data in layer.edges]
        data = [data for _check, data in layer.edges]
        if len(checks) != len(set(checks)):
            return False
        if len(data) != len(set(data)):
            return False
    return True


def validate_layer_reconstruction(code, z_layers, x_layers) -> dict[str, object]:
    z_reconstructed = layers_to_sparse(z_layers, code.HZ.shape)
    x_reconstructed = layers_to_sparse(x_layers, code.HX.shape)
    return {
        "z_matrix_matches": (z_reconstructed != code.HZ).nnz == 0,
        "x_matrix_matches": (x_reconstructed != code.HX).nnz == 0,
        "z_matching_ok": matching_ok(z_layers),
        "x_matching_ok": matching_ok(x_layers),
        "z_layer_sizes": [len(layer) for layer in z_layers],
        "x_layer_sizes": [len(layer) for layer in x_layers],
    }


def validate_parallel_overlaps(z_layers, x_layers) -> tuple[bool, list[dict[str, object]]]:
    if len(z_layers) != len(x_layers):
        raise ValueError("Z and X layer counts differ")
    rows: list[dict[str, object]] = []
    ok = True
    for tick, (z_layer, x_layer) in enumerate(zip(z_layers, x_layers, strict=True)):
        z_data = {data for _check, data in z_layer.edges}
        x_data = {data for _check, data in x_layer.edges}
        overlap = sorted(z_data & x_data)
        ok = ok and not overlap
        rows.append(
            {
                "tick": int(tick),
                "z_layer": z_layer.label,
                "x_layer": x_layer.label,
                "overlap_count": len(overlap),
                "overlap_sample": overlap[:10],
            }
        )
    return ok, rows


def _append_tick(circuit, tick: bool) -> None:
    if tick:
        circuit.append("TICK")


def _append_reset_flip(circuit, qubits: Sequence[int], basis: MeasurementBasis, p: float) -> None:
    if p <= 0 or not qubits:
        return
    circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", list(qubits), float(p))


def _append_measurement_noise(
    circuit,
    qubits: Sequence[int],
    basis: MeasurementBasis,
    p_flip: float,
) -> None:
    if p_flip <= 0 or not qubits:
        return
    circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", list(qubits), float(p_flip))


def _record_targets(stim, record_indices: Sequence[int], current_record_count: int):
    return [stim.target_rec(int(index) - int(current_record_count)) for index in record_indices]


def _append_z_layer_cx(
    circuit, layer: APMLayer, z_ancillas, data_qubits, noise: StimNoiseConfig
) -> None:
    for check, data in layer.edges:
        qd = data_qubits[data]
        qa = z_ancillas[check]
        circuit.append("CX", [qd, qa])
        if noise.p_cx > 0:
            circuit.append("DEPOLARIZE2", [qd, qa], float(noise.p_cx))


def _append_x_layer_cx(
    circuit, layer: APMLayer, x_ancillas, data_qubits, noise: StimNoiseConfig
) -> None:
    for check, data in layer.edges:
        qd = data_qubits[data]
        qa = x_ancillas[check]
        circuit.append("CX", [qa, qd])
        if noise.p_cx > 0:
            circuit.append("DEPOLARIZE2", [qa, qd], float(noise.p_cx))


def _measure_parallel_round(
    stim,
    circuit,
    z_layers,
    x_layers,
    z_ancillas,
    x_ancillas,
    data_qubits,
    noise: StimNoiseConfig,
    previous_z_measurements: list[int] | None,
    previous_x_measurements: list[int] | None,
    measurement_records: list[int],
    tick_between_layers: bool,
) -> tuple[list[int], list[int]]:
    circuit.append("R", z_ancillas)
    _append_reset_flip(circuit, z_ancillas, "Z", noise.p_reset_flip)
    circuit.append("RX", x_ancillas)
    _append_reset_flip(circuit, x_ancillas, "X", noise.p_reset_flip)
    _append_tick(circuit, tick_between_layers)

    for z_layer, x_layer in zip(z_layers, x_layers, strict=True):
        z_data = {data for _check, data in z_layer.edges}
        x_data = {data for _check, data in x_layer.edges}
        if z_data & x_data:
            raise ValueError(f"data overlap in tick Z={z_layer.label}, X={x_layer.label}")
        _append_z_layer_cx(circuit, z_layer, z_ancillas, data_qubits, noise)
        _append_x_layer_cx(circuit, x_layer, x_ancillas, data_qubits, noise)
        _append_tick(circuit, tick_between_layers)

    z_current: list[int] = []
    x_current: list[int] = []
    _append_measurement_noise(circuit, z_ancillas, "Z", noise.p_measure_flip)
    circuit.append("M", z_ancillas)
    for ancilla in z_ancillas:
        z_current.append(len(measurement_records))
        measurement_records.append(ancilla)

    _append_measurement_noise(circuit, x_ancillas, "X", noise.p_measure_flip)
    circuit.append("MX", x_ancillas)
    for ancilla in x_ancillas:
        x_current.append(len(measurement_records))
        measurement_records.append(ancilla)

    current_count = len(measurement_records)
    for check, rec_idx in enumerate(z_current):
        record_indices = [rec_idx]
        if previous_z_measurements is not None:
            record_indices.append(previous_z_measurements[check])
        circuit.append(
            "DETECTOR",
            _record_targets(stim, record_indices, current_count),
            [0, check, current_count],
        )

    if previous_x_measurements is not None:
        for check, rec_idx in enumerate(x_current):
            record_indices = [rec_idx, previous_x_measurements[check]]
            circuit.append(
                "DETECTOR",
                _record_targets(stim, record_indices, current_count),
                [1, check, current_count],
            )
    return z_current, x_current


def _append_final_data_z_detectors(
    stim,
    circuit,
    code,
    data_meas_indices: Sequence[int],
    previous_z_measurements: Sequence[int] | None,
    current_record_count: int,
) -> None:
    hz = code.HZ.tocsr()
    for check in range(code.HZ.shape[0]):
        record_indices = [data_meas_indices[int(q)] for q in hz.getrow(check).indices]
        if previous_z_measurements is not None:
            record_indices.append(previous_z_measurements[check])
        circuit.append(
            "DETECTOR",
            _record_targets(stim, record_indices, current_record_count),
            [2, check, current_record_count],
        )


def build_syndrome_circuit(
    code,
    *,
    rounds: int = 2,
    noise: StimNoiseConfig | None = None,
    tick_between_layers: bool = True,
) -> CircuitBuildResult:
    """Build a Z-memory syndrome circuit with parallel X/Z check extraction."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    stim = require_stim()
    noise_cfg = StimNoiseConfig() if noise is None else noise
    z_layers = build_apm_layers(code, "Z", PARALLEL_Z_ORDER)
    x_layers = build_apm_layers(code, "X", PARALLEL_X_ORDER)
    reconstruction = validate_layer_reconstruction(code, z_layers, x_layers)
    if not all(
        reconstruction[key]
        for key in ("z_matrix_matches", "x_matrix_matches", "z_matching_ok", "x_matching_ok")
    ):
        raise ValueError(f"invalid APM layer reconstruction: {reconstruction}")
    overlap_ok, overlap_rows = validate_parallel_overlaps(z_layers, x_layers)
    if not overlap_ok:
        raise ValueError(f"parallel schedule has overlaps: {overlap_rows}")

    circuit = stim.Circuit()
    data_qubits = tuple(range(code.n))
    m_z = code.HZ.shape[0]
    m_x = code.HX.shape[0]
    z_ancillas = tuple(range(code.n, code.n + m_z))
    x_ancillas = tuple(range(code.n + m_z, code.n + m_z + m_x))
    measurement_records: list[int] = []

    circuit.append("R", data_qubits)
    _append_reset_flip(circuit, data_qubits, "Z", noise_cfg.p_reset_flip)
    _append_tick(circuit, tick_between_layers)

    prev_z = None
    prev_x = None
    for _round in range(int(rounds)):
        prev_z, prev_x = _measure_parallel_round(
            stim,
            circuit,
            z_layers,
            x_layers,
            z_ancillas,
            x_ancillas,
            data_qubits,
            noise_cfg,
            prev_z,
            prev_x,
            measurement_records,
            tick_between_layers,
        )
        _append_tick(circuit, tick_between_layers)

    _append_measurement_noise(circuit, data_qubits, "Z", noise_cfg.p_final_measure_flip)
    circuit.append("M", data_qubits)
    data_meas_indices = list(range(len(measurement_records), len(measurement_records) + code.n))
    measurement_records.extend(data_qubits)
    _append_final_data_z_detectors(
        stim,
        circuit,
        code,
        data_meas_indices,
        prev_z,
        len(measurement_records),
    )

    return CircuitBuildResult(
        circuit=circuit,
        z_layers=tuple(z_layers),
        x_layers=tuple(x_layers),
        data_qubits=data_qubits,
        z_ancillas=z_ancillas,
        x_ancillas=x_ancillas,
        noise=noise_cfg,
    )


def validate_syndrome_schedule(
    code,
    *,
    rounds: int = 2,
    noise: StimNoiseConfig | None = None,
    write_stim_path: Path | None = None,
) -> dict[str, object]:
    """Validate layer order and Stim deterministic detectors for one code."""
    z_layers = build_apm_layers(code, "Z", PARALLEL_Z_ORDER)
    x_layers = build_apm_layers(code, "X", PARALLEL_X_ORDER)
    reconstruction = validate_layer_reconstruction(code, z_layers, x_layers)
    overlap_ok, overlap_rows = validate_parallel_overlaps(z_layers, x_layers)
    result = build_syndrome_circuit(code, rounds=rounds, noise=noise)
    if write_stim_path is not None:
        write_stim_path.parent.mkdir(parents=True, exist_ok=True)
        write_stim_path.write_text(str(result.circuit), encoding="utf-8")

    payload: dict[str, object] = {
        "name": code.spec.name,
        "parameter_label": code.spec.parameter_label,
        "P": int(code.spec.p),
        "L": int(code.spec.l),
        "J": int(code.spec.j),
        "n": int(code.n),
        "mX": int(code.HX.shape[0]),
        "mZ": int(code.HZ.shape[0]),
        "rounds": int(rounds),
        "x_order": order_labels(PARALLEL_X_ORDER),
        "z_order": order_labels(PARALLEL_Z_ORDER),
        "data_disjoint": bool(overlap_ok),
        "layer_overlaps": overlap_rows,
        "layer_reconstruction": reconstruction,
        "num_detectors": int(result.circuit.num_detectors),
        "num_observables": int(result.circuit.num_observables),
        "noise": result.noise.to_dict(),
        "stim_path": None if write_stim_path is None else str(write_stim_path),
    }
    try:
        dem = result.circuit.detector_error_model()
    except Exception as exc:
        payload["deterministic_detectors"] = False
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc).splitlines()[0] if str(exc) else repr(exc)
        return payload

    payload["deterministic_detectors"] = True
    payload["dem_num_detectors"] = int(dem.num_detectors)
    payload["dem_num_observables"] = int(dem.num_observables)
    payload["dem_num_errors"] = int(dem.num_errors)
    return payload
