"""Compare reconstructed codes with the supplied parity-check matrices."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import io, sparse

from bivariate_bicycle.code_construction import SPECS as BB_SPECS
from bivariate_bicycle.code_construction import build_bb_code, code_summary
from code_construction.affine_codes import build_code, gf2_rank, permutation_matrix
from code_construction.cornucopia_codes import SPECS, construct_one

DATA = Path(__file__).resolve().parents[1] / "code_construction"


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_cornucopia_matches_reference(spec, tmp_path):
    code = build_code(spec.to_code_spec())
    for basis, matrix in (("X", code.HX), ("Z", code.HZ)):
        reference = io.mmread(DATA / "matrices" / f"{spec.name}_H{basis}.mtx").tocsr()
        assert matrix.shape == reference.shape
        assert (matrix != reference).nnz == 0
    reference = next(
        r
        for r in json.loads((DATA / "constructed_codes_summary.json").read_text())
        if r["name"] == spec.name
    )
    actual = construct_one(spec, write_matrices=False, matrix_dir=tmp_path)
    assert actual == reference
    assert actual["checks"]["all_checks_ok"]


@pytest.mark.parametrize("spec", BB_SPECS, ids=lambda spec: spec.name)
def test_bivariate_bicycle_css_and_dimension(spec):
    code = build_bb_code(spec)
    assert np.all((code.HX @ code.HZ.T).data % 2 == 0)
    assert code_summary(spec)["k"] == 12


def test_binary_rank_uses_gf2_not_real_arithmetic():
    matrix = sparse.csr_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert np.linalg.matrix_rank(matrix.toarray()) == 3
    assert gf2_rank(matrix) == 2


def test_affine_maps_are_permutations_only_for_unit_multipliers():
    matrix = permutation_matrix(8, 5, 21)
    assert np.all(np.asarray(matrix.sum(axis=0)) == 1)
    assert np.all(np.asarray(matrix.sum(axis=1)) == 1)
    with pytest.raises(ValueError, match="not a unit"):
        permutation_matrix(7, 5, 21)
