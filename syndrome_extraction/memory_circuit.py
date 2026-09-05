"""Logical-memory circuits for the Cornucopia syndrome-extraction schedule."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import sparse

from syndrome_extraction.syndrome_schedule import (
    PARALLEL_X_ORDER,
    PARALLEL_Z_ORDER,
    StimNoiseConfig,
    build_apm_layers,
    validate_layer_reconstruction,
    validate_parallel_overlaps,
)

Basis = Literal["Z", "X"]


def to_dense_binary(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.uint8) % 2


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


def _append_z_layer_cx(circuit, layer, z_ancillas, data_qubits, noise: StimNoiseConfig) -> None:
    for check, data in layer.edges:
        qd = data_qubits[int(data)]
        qa = z_ancillas[int(check)]
        circuit.append("CX", [qd, qa])
        if noise.p_cx > 0:
            circuit.append("DEPOLARIZE2", [qd, qa], float(noise.p_cx))


def _append_x_layer_cx(circuit, layer, x_ancillas, data_qubits, noise: StimNoiseConfig) -> None:
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
    code,
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


def build_memory_circuit(
    code,
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

    z_layers = build_apm_layers(code, "Z", PARALLEL_Z_ORDER)
    x_layers = build_apm_layers(code, "X", PARALLEL_X_ORDER)
    reconstruction = validate_layer_reconstruction(code, z_layers, x_layers)
    if not all(
        reconstruction[key]
        for key in (
            "z_matrix_matches",
            "x_matrix_matches",
            "z_matching_ok",
            "x_matching_ok",
        )
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

    circuit.append("R" if basis == "Z" else "RX", data_qubits)
    _append_reset_flip(circuit, data_qubits, basis, noise.p_reset_flip)
    _append_tick(circuit, tick_between_layers)

    prev_z = None
    prev_x = None
    for _ in range(rounds):
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


def logical_rows_for_basis(code, basis: Basis) -> sparse.csr_matrix:
    from bposd.css import css_code  # type: ignore

    hx = to_dense_binary(code.HX)
    hz = to_dense_binary(code.HZ)
    qcode = css_code(hx=hx, hz=hz)
    qcode.test()
    rows = qcode.lz if basis == "Z" else qcode.lx
    return sparse.csr_matrix(to_dense_binary(rows), dtype=np.uint8)
