#!/usr/bin/env python3
"""Stim-native rotated surface-code memory circuits with matched noise."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Basis = Literal["Z", "X"]


@dataclass(frozen=True)
class SurfaceNoiseConfig:
    p_cx: float = 0.0
    p_measure_flip: float = 0.0
    p_final_measure_flip: float = 0.0
    p_reset_flip: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


def _require_stim():
    try:
        import stim  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime environment check
        raise ImportError("surface-code circuit construction requires stim") from exc
    return stim


def _target_qubit(target) -> int:
    if not target.is_qubit_target:
        raise ValueError(f"expected a qubit target, got {target!r}")
    return int(target.value)


def _collect_h_targets(circuit) -> set[int]:
    stim = _require_stim()
    targets: set[int] = set()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            targets.update(_collect_h_targets(item.body_copy()))
        elif item.name == "H":
            targets.update(_target_qubit(target) for target in item.targets_copy())
    return targets


def _append_reset_in_original_order(output, targets, x_ancillas: set[int]) -> None:
    for target in targets:
        qubit = _target_qubit(target)
        output.append("RX" if qubit in x_ancillas else "R", [qubit])


def _append_measure_reset_in_original_order(output, targets, x_ancillas: set[int]) -> None:
    # One target per instruction preserves Stim's measurement-record ordering.
    for target in targets:
        qubit = _target_qubit(target)
        output.append("MRX" if qubit in x_ancillas else "MR", [qubit])


def _normalise_body_to_basis_operations(circuit, x_ancillas: set[int]):
    stim = _require_stim()
    output = stim.Circuit()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            body = _normalise_body_to_basis_operations(item.body_copy(), x_ancillas)
            output.append(stim.CircuitRepeatBlock(item.repeat_count, body))
            continue
        if item.name == "H":
            continue
        if item.name == "R":
            _append_reset_in_original_order(output, item.targets_copy(), x_ancillas)
            continue
        if item.name == "MR":
            _append_measure_reset_in_original_order(output, item.targets_copy(), x_ancillas)
            continue
        output.append(item)
    return output


def normalise_to_basis_operations(circuit):
    """Replace X-check ancilla H gates by RX/MRX without changing records."""
    x_ancillas = _collect_h_targets(circuit)
    if not x_ancillas:
        raise ValueError("Stim surface-code circuit contains no X-check ancillas")
    output = _normalise_body_to_basis_operations(circuit, x_ancillas)
    if any(item.name == "H" for item in output.flattened()):
        raise AssertionError("H-free surface-code normalisation failed")
    return output


def build_ideal_surface_memory_circuit(distance: int, basis: Basis, rounds: int):
    stim = _require_stim()
    if isinstance(distance, bool) or not isinstance(distance, int):
        raise TypeError("distance must be an integer")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if distance < 2:
        raise ValueError("distance must be at least 2")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if basis not in ("Z", "X"):
        raise ValueError("basis must be Z or X")
    task = "surface_code:rotated_memory_z" if basis == "Z" else "surface_code:rotated_memory_x"
    generated = stim.Circuit.generated(
        task,
        distance=distance,
        rounds=rounds,
    )
    return normalise_to_basis_operations(generated)


def _append_pauli_error(output, name: str, targets, probability: float) -> None:
    if probability > 0:
        output.append(name, targets, float(probability))


def _inject_noise_body(circuit, noise: SurfaceNoiseConfig):
    stim = _require_stim()
    output = stim.Circuit()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            body = _inject_noise_body(item.body_copy(), noise)
            output.append(stim.CircuitRepeatBlock(item.repeat_count, body))
            continue

        name = item.name
        targets = item.targets_copy()
        if name == "CX":
            output.append(item)
            _append_pauli_error(output, "DEPOLARIZE2", targets, noise.p_cx)
        elif name == "R":
            output.append(item)
            _append_pauli_error(output, "X_ERROR", targets, noise.p_reset_flip)
        elif name == "RX":
            output.append(item)
            _append_pauli_error(output, "Z_ERROR", targets, noise.p_reset_flip)
        elif name == "MR":
            _append_pauli_error(output, "X_ERROR", targets, noise.p_measure_flip)
            output.append(item)
            _append_pauli_error(output, "X_ERROR", targets, noise.p_reset_flip)
        elif name == "MRX":
            _append_pauli_error(output, "Z_ERROR", targets, noise.p_measure_flip)
            output.append(item)
            _append_pauli_error(output, "Z_ERROR", targets, noise.p_reset_flip)
        elif name == "M":
            _append_pauli_error(output, "X_ERROR", targets, noise.p_final_measure_flip)
            output.append(item)
        elif name == "MX":
            _append_pauli_error(output, "Z_ERROR", targets, noise.p_final_measure_flip)
            output.append(item)
        else:
            output.append(item)
    return output


def build_surface_memory_circuit(
    distance: int,
    *,
    basis: Basis,
    rounds: int,
    noise: SurfaceNoiseConfig,
):
    ideal = build_ideal_surface_memory_circuit(distance, basis, rounds)
    return _inject_noise_body(ideal, noise)


def _detector_matches_basis(coords: list[float], basis: Basis) -> bool:
    if len(coords) < 2:
        return True
    parity = int(round(float(coords[0]) + float(coords[1]))) % 4
    return parity == (0 if basis == "Z" else 2)


def _filter_detector_body(circuit, basis: Basis):
    stim = _require_stim()
    output = stim.Circuit()
    for item in circuit:
        if isinstance(item, stim.CircuitRepeatBlock):
            body = _filter_detector_body(item.body_copy(), basis)
            output.append(stim.CircuitRepeatBlock(item.repeat_count, body))
            continue
        if item.name == "DETECTOR":
            coords = [float(value) for value in item.gate_args_copy()]
            if not _detector_matches_basis(coords, basis):
                continue
        output.append(item)
    return output


def filter_surface_detectors_for_xz(circuit, basis: Basis):
    """Keep only checks of the CSS type relevant to the memory basis."""
    if basis not in ("Z", "X"):
        raise ValueError("basis must be Z or X")
    filtered = _filter_detector_body(circuit, basis)
    kept = int(filtered.num_detectors)
    removed = int(circuit.num_detectors) - kept
    return filtered, kept, removed


def cx_layer_count(circuit) -> int:
    return sum(item.name == "CX" for item in circuit.flattened())


def validate_four_cx_layers_per_round(circuit, rounds: int) -> dict[str, object]:
    flattened = circuit.flattened()
    cx_layers = [item for item in flattened if item.name == "CX"]
    cx_layers_per_round: list[int] = []
    current_round_layers = 0
    inside_measure_reset_group = False
    overlaps: list[dict[str, object]] = []
    layer_index = 0
    for item in flattened:
        if item.name == "CX":
            inside_measure_reset_group = False
            current_round_layers += 1
            qubits = [_target_qubit(target) for target in item.targets_copy()]
            if len(qubits) != len(set(qubits)):
                overlaps.append({"layer": layer_index, "targets": qubits})
            layer_index += 1
        elif item.name in ("MR", "MRX") and not inside_measure_reset_group:
            cx_layers_per_round.append(current_round_layers)
            current_round_layers = 0
            inside_measure_reset_group = True
    expected_layers = [4] * int(rounds)
    return {
        "rounds": int(rounds),
        "cx_layers": len(cx_layers),
        "expected_cx_layers": 4 * int(rounds),
        "cx_layers_per_round": cx_layers_per_round,
        "four_layers_per_round": cx_layers_per_round == expected_layers,
        "cx_layer_disjoint": not overlaps,
        "overlaps": overlaps,
    }
