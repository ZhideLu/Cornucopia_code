#!/usr/bin/env python3
"""Bivariate-bicycle code construction for the two BB examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse

Axis = Literal["x", "y"]


@dataclass(frozen=True)
class BBCodeSpec:
    name: str
    parameter_label: str
    ell: int
    m: int
    a: tuple[int, int, int]
    b: tuple[int, int, int]
    expected_d: int

    @property
    def block_size(self) -> int:
        return int(self.ell) * int(self.m)

    @property
    def n(self) -> int:
        return 2 * self.block_size

    @property
    def terms(self) -> dict[str, tuple[Axis, int]]:
        a1, a2, a3 = self.a
        b1, b2, b3 = self.b
        return {
            "A1": ("x", int(a1)),
            "A2": ("y", int(a2)),
            "A3": ("y", int(a3)),
            "B1": ("y", int(b1)),
            "B2": ("x", int(b2)),
            "B3": ("x", int(b3)),
        }


SPECS: tuple[BBCodeSpec, ...] = (
    BBCodeSpec(
        name="bb_144_12_12",
        parameter_label="[[144,12,12]]",
        ell=12,
        m=6,
        a=(3, 1, 2),
        b=(3, 1, 2),
        expected_d=12,
    ),
    BBCodeSpec(
        name="bb_288_12_18",
        parameter_label="[[288,12,18]]",
        ell=12,
        m=12,
        a=(3, 2, 7),
        b=(3, 1, 2),
        expected_d=18,
    ),
)


@dataclass(frozen=True)
class BBCode:
    spec: BBCodeSpec
    HX: sparse.csr_matrix
    HZ: sparse.csr_matrix
    A: sparse.csr_matrix
    B: sparse.csr_matrix
    term_matrices: dict[str, sparse.csr_matrix]

    @property
    def n(self) -> int:
        return int(self.HX.shape[1])

    def num_logicals(self) -> int:
        return int(self.n - gf2_rank(self.HX) - gf2_rank(self.HZ))


def _coord_to_index(row: int, col: int, m: int) -> int:
    return int(row) * int(m) + int(col)


def term_image(spec: BBCodeSpec, label: str, row: int, *, transpose: bool = False) -> int:
    """Column touched by a monomial term row in row-major coordinates."""
    axis, power = spec.terms[label]
    ell = int(spec.ell)
    m = int(spec.m)
    i, j = divmod(int(row), m)
    sign = -1 if transpose else 1
    if axis == "x":
        i = (i + sign * int(power)) % ell
    elif axis == "y":
        j = (j + sign * int(power)) % m
    else:  # pragma: no cover
        raise ValueError(f"unknown axis {axis!r}")
    return _coord_to_index(i, j, m)


def term_matrix(spec: BBCodeSpec, label: str) -> sparse.csr_matrix:
    rows = np.arange(spec.block_size, dtype=np.int64)
    cols = np.fromiter(
        (term_image(spec, label, int(row), transpose=False) for row in rows),
        dtype=np.int64,
        count=spec.block_size,
    )
    data = np.ones(spec.block_size, dtype=np.uint8)
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(spec.block_size, spec.block_size), dtype=np.uint8
    ).tocsr()


def _binary_sum(matrices: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    out = sum(matrices[1:], matrices[0].copy()).tocsr().astype(np.uint8)
    out.data %= 2
    out.eliminate_zeros()
    return out


def build_bb_code(spec: BBCodeSpec) -> BBCode:
    terms = {label: term_matrix(spec, label) for label in spec.terms}
    A = _binary_sum([terms["A1"], terms["A2"], terms["A3"]])
    B = _binary_sum([terms["B1"], terms["B2"], terms["B3"]])
    HX = sparse.hstack([A, B], format="csr", dtype=np.uint8)
    HZ = sparse.hstack([B.T, A.T], format="csr", dtype=np.uint8)
    HX.data %= 2
    HZ.data %= 2
    HX.eliminate_zeros()
    HZ.eliminate_zeros()
    if (HX @ HZ.T).nnz:
        product = (HX @ HZ.T).tocsr()
        product.data %= 2
        product.eliminate_zeros()
        if product.nnz:
            raise ValueError(f"{spec.name}: H_X and H_Z are not orthogonal")
    return BBCode(spec=spec, HX=HX, HZ=HZ, A=A, B=B, term_matrices=terms)


def gf2_rank(matrix: sparse.spmatrix | np.ndarray) -> int:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    rows = np.asarray(dense, dtype=np.uint8) % 2
    if rows.size == 0:
        return 0
    rank = 0
    n_rows, n_cols = rows.shape
    for col in range(n_cols):
        pivot = None
        for row in range(rank, n_rows):
            if rows[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            rows[[rank, pivot]] = rows[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and rows[row, col]:
                rows[row] ^= rows[rank]
        rank += 1
        if rank == n_rows:
            break
    return int(rank)


def code_summary(spec: BBCodeSpec) -> dict[str, int | str]:
    code = build_bb_code(spec)
    return {
        "name": spec.name,
        "parameter_label": spec.parameter_label,
        "ell": int(spec.ell),
        "m": int(spec.m),
        "n": int(code.n),
        "rank_hx": gf2_rank(code.HX),
        "rank_hz": gf2_rank(code.HZ),
        "k": code.num_logicals(),
        "expected_d": int(spec.expected_d),
    }


def parse_code_names(value: str | None) -> list[BBCodeSpec]:
    if value is None or value.strip().lower() in ("", "all"):
        return list(SPECS)
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    specs = [spec for spec in SPECS if spec.name in wanted]
    missing = sorted(wanted - {spec.name for spec in specs})
    if missing:
        raise SystemExit(
            f"Unknown BB code names: {missing}. Available: {[spec.name for spec in SPECS]}"
        )
    return specs
