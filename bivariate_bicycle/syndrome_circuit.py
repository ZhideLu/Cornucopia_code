#!/usr/bin/env python3
"""Eight-layer syndrome-extraction schedule for bivariate-bicycle codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import sparse

from bivariate_bicycle.code_construction import BBCode, build_bb_code, term_image

Basis = Literal["Z", "X"]

BB_Z_ORDER: tuple[str, ...] = ("A1", "A3", "B1", "B2", "B3", "A2", "IDLE", "IDLE")
BB_X_ORDER: tuple[str, ...] = ("IDLE", "A2", "B2", "B1", "B3", "A1", "A3", "IDLE")


@dataclass(frozen=True)
class StimNoiseConfig:
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
class BBLayer:
    basis: Basis
    label: str
    edges: tuple[tuple[int, int], ...]

    def __len__(self) -> int:
        return len(self.edges)


def build_bb_layers(code: BBCode, basis: Basis, order: Sequence[str]) -> tuple[BBLayer, ...]:
    spec = code.spec
    n0 = spec.block_size
    layers: list[BBLayer] = []
    for label in order:
        if label == "IDLE":
            layers.append(BBLayer(basis=basis, label=label, edges=()))
            continue
        if label not in spec.terms:
            raise ValueError(f"unknown BB layer label {label!r}")
        edges: list[tuple[int, int]] = []
        for check in range(n0):
            if basis == "X":
                data = term_image(spec, label, check, transpose=False)
                if label.startswith("B"):
                    data += n0
            else:
                data = term_image(spec, label, check, transpose=True)
                if label.startswith("A"):
                    data += n0
            edges.append((check, data))
        layers.append(BBLayer(basis=basis, label=label, edges=tuple(edges)))
    return tuple(layers)


def layers_to_sparse(layers: Sequence[BBLayer], shape: tuple[int, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for layer in layers:
        for check, data in layer.edges:
            rows.append(int(check))
            cols.append(int(data))
    values = np.ones(len(rows), dtype=np.uint8)
    out = sparse.coo_matrix((values, (rows, cols)), shape=shape, dtype=np.uint8).tocsr()
    out.data %= 2
    out.eliminate_zeros()
    return out


def matching_ok(layers: Sequence[BBLayer]) -> bool:
    for layer in layers:
        checks = [check for check, _data in layer.edges]
        data = [data for _check, data in layer.edges]
        if len(checks) != len(set(checks)):
            return False
        if len(data) != len(set(data)):
            return False
    return True


def validate_layer_reconstruction(code: BBCode, z_layers, x_layers) -> dict[str, object]:
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


def _record_targets(stim, record_indices: Sequence[int], current_record_count: int):
    return [stim.target_rec(int(index) - int(current_record_count)) for index in record_indices]


def _append_tick(circuit, enabled: bool) -> None:
    if enabled:
        circuit.append("TICK")


def _append_reset_flip(circuit, qubits: Sequence[int], basis: Basis, p: float) -> None:
    if p > 0 and qubits:
        circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", list(qubits), float(p))


def _append_measurement_flip(circuit, qubits: Sequence[int], basis: Basis, p: float) -> None:
    if p > 0 and qubits:
        circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", list(qubits), float(p))


def _append_z_layer_cx(circuit, layer: BBLayer, z_ancillas, data_qubits, noise) -> None:
    for check, data in layer.edges:
        qd = data_qubits[int(data)]
        qa = z_ancillas[int(check)]
        circuit.append("CX", [qd, qa])
        if noise.p_cx > 0:
            circuit.append("DEPOLARIZE2", [qd, qa], float(noise.p_cx))


def _append_x_layer_cx(circuit, layer: BBLayer, x_ancillas, data_qubits, noise) -> None:
    for check, data in layer.edges:
        qd = data_qubits[int(data)]
        qa = x_ancillas[int(check)]
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
    memory_basis: Basis,
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
    _append_measurement_flip(circuit, z_ancillas, "Z", noise.p_measure_flip)
    circuit.append("M", z_ancillas)
    for ancilla in z_ancillas:
        z_current.append(len(measurement_records))
        measurement_records.append(ancilla)

    _append_measurement_flip(circuit, x_ancillas, "X", noise.p_measure_flip)
    circuit.append("MX", x_ancillas)
    for ancilla in x_ancillas:
        x_current.append(len(measurement_records))
        measurement_records.append(ancilla)

    current_count = len(measurement_records)
    if previous_z_measurements is not None or memory_basis == "Z":
        for check, rec_idx in enumerate(z_current):
            record_indices = [rec_idx]
            if previous_z_measurements is not None:
                record_indices.append(previous_z_measurements[check])
            circuit.append(
                "DETECTOR",
                _record_targets(stim, record_indices, current_count),
                [0, check, current_count],
            )

    if previous_x_measurements is not None or memory_basis == "X":
        for check, rec_idx in enumerate(x_current):
            record_indices = [rec_idx]
            if previous_x_measurements is not None:
                record_indices.append(previous_x_measurements[check])
            circuit.append(
                "DETECTOR",
                _record_targets(stim, record_indices, current_count),
                [1, check, current_count],
            )

    return z_current, x_current


def _append_final_data_detectors_and_observables(
    stim,
    circuit,
    code: BBCode,
    basis: Basis,
    logical_rows: sparse.csr_matrix,
    data_meas_indices: Sequence[int],
    previous_measurements: Sequence[int] | None,
    current_record_count: int,
) -> None:
    checks = code.HZ.tocsr() if basis == "Z" else code.HX.tocsr()
    detector_coord0 = 2 if basis == "Z" else 3
    for check in range(checks.shape[0]):
        record_indices = [data_meas_indices[int(q)] for q in checks.getrow(check).indices]
        if previous_measurements is not None:
            record_indices.append(previous_measurements[check])
        circuit.append(
            "DETECTOR",
            _record_targets(stim, record_indices, current_record_count),
            [detector_coord0, check, current_record_count],
        )

    for obs_idx in range(logical_rows.shape[0]):
        support = logical_rows.getrow(obs_idx).indices
        if len(support) == 0:
            continue
        record_indices = [data_meas_indices[int(q)] for q in support]
        circuit.append(
            "OBSERVABLE_INCLUDE",
            _record_targets(stim, record_indices, current_record_count),
            int(obs_idx),
        )


def to_dense_binary(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.uint8) % 2


def logical_rows_for_basis(code: BBCode, basis: Basis) -> sparse.csr_matrix:
    from bposd.css import css_code  # type: ignore

    qcode = css_code(hx=to_dense_binary(code.HX), hz=to_dense_binary(code.HZ))
    qcode.test(show_tests=False)
    rows = qcode.lz if basis == "Z" else qcode.lx
    return sparse.csr_matrix(to_dense_binary(rows), dtype=np.uint8)


def build_memory_circuit(
    code: BBCode,
    *,
    basis: Basis,
    logical_rows: sparse.csr_matrix,
    rounds: int,
    noise: StimNoiseConfig,
    tick_between_layers: bool = True,
):
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    import stim  # type: ignore

    z_layers = build_bb_layers(code, "Z", BB_Z_ORDER)
    x_layers = build_bb_layers(code, "X", BB_X_ORDER)
    reconstruction = validate_layer_reconstruction(code, z_layers, x_layers)
    if not all(
        reconstruction[key]
        for key in ("z_matrix_matches", "x_matrix_matches", "z_matching_ok", "x_matching_ok")
    ):
        raise ValueError(f"invalid BB layer reconstruction: {reconstruction}")
    overlap_ok, overlap_rows = validate_parallel_overlaps(z_layers, x_layers)
    if not overlap_ok:
        raise ValueError(f"parallel BB schedule has overlaps: {overlap_rows}")

    circuit = stim.Circuit()
    data_qubits = tuple(range(code.n))
    m_z = code.HZ.shape[0]
    m_x = code.HX.shape[0]
    z_ancillas = tuple(range(code.n, code.n + m_z))
    x_ancillas = tuple(range(code.n + m_z, code.n + m_z + m_x))
    measurement_records: list[int] = []

    circuit.append("R" if basis == "Z" else "RX", data_qubits)
    _append_reset_flip(circuit, data_qubits, basis, noise.p_reset_flip)
    _append_tick(circuit, tick_between_layers)

    prev_z = None
    prev_x = None
    for _round in range(rounds):
        prev_z, prev_x = _measure_parallel_round(
            stim,
            circuit,
            z_layers,
            x_layers,
            z_ancillas,
            x_ancillas,
            data_qubits,
            noise,
            basis,
            prev_z,
            prev_x,
            measurement_records,
            tick_between_layers,
        )
        _append_tick(circuit, tick_between_layers)

    _append_measurement_flip(circuit, data_qubits, basis, noise.p_final_measure_flip)
    circuit.append("M" if basis == "Z" else "MX", data_qubits)
    data_meas_indices = list(range(len(measurement_records), len(measurement_records) + code.n))
    measurement_records.extend(data_qubits)
    _append_final_data_detectors_and_observables(
        stim,
        circuit,
        code,
        basis,
        logical_rows,
        data_meas_indices,
        prev_z if basis == "Z" else prev_x,
        len(measurement_records),
    )
    return circuit


def build_code_and_validate(spec):
    code = build_bb_code(spec)
    z_layers = build_bb_layers(code, "Z", BB_Z_ORDER)
    x_layers = build_bb_layers(code, "X", BB_X_ORDER)
    reconstruction = validate_layer_reconstruction(code, z_layers, x_layers)
    overlap_ok, overlap_rows = validate_parallel_overlaps(z_layers, x_layers)
    return code, reconstruction, overlap_ok, overlap_rows
