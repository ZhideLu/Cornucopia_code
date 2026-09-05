"""Compare noiseless schedules with the reference Stim circuits."""

from pathlib import Path

import pytest

from code_construction.affine_codes import build_code
from code_construction.cornucopia_codes import SPECS
from syndrome_extraction.syndrome_schedule import (
    build_syndrome_circuit,
    validate_syndrome_schedule,
)

stim = pytest.importorskip("stim")

DATA = Path(__file__).resolve().parents[1] / "syndrome_extraction" / "stim_circuits"


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_schedule_matches_reference_and_has_deterministic_detectors(spec):
    code = build_code(spec.to_code_spec())
    result = build_syndrome_circuit(code, rounds=2)
    reference = stim.Circuit((DATA / f"{spec.name}_r2.stim").read_text())
    assert result.circuit == reference
    report = validate_syndrome_schedule(code, rounds=2)
    assert report["data_disjoint"]
    assert report["deterministic_detectors"]
    assert all(report["layer_reconstruction"].values())
