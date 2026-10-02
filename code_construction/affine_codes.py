#!/usr/bin/env python3
"""Affine-permutation CSS parity-check construction over GF(2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Optional, Tuple

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class CodeSpec:
    """Parameters of an affine-permutation CSS code.

    p is the permutation-block size P; l and j are the block-column and
    active block-row counts L and J. Each (a, b) pair denotes ax+b modulo P.
    """

    name: str
    parameter_label: str
    p: int
    l: int
    j: int
    f_params: Tuple[Tuple[int, int], ...]
    g_params: Tuple[Tuple[int, int], ...]
    active_rows: Optional[Tuple[int, ...]] = None

    @property
    def n(self) -> int:
        return self.p * self.l

    @property
    def half(self) -> int:
        return self.l // 2

    @property
    def active_set(self) -> Tuple[int, ...]:
        if self.active_rows is None:
            return tuple(range(self.j))
        return tuple(int(row) for row in self.active_rows)


@dataclass(frozen=True)
class AffineCSSCode:
    """Constructed CSS parity-check matrices."""

    spec: CodeSpec
    HX: sparse.csr_matrix
    HZ: sparse.csr_matrix

    @property
    def n(self) -> int:
        return self.spec.n

    def ranks(self) -> Tuple[int, int]:
        return gf2_rank(self.HX), gf2_rank(self.HZ)

    def num_logicals(self) -> int:
        rank_x, rank_z = self.ranks()
        return int(self.n - rank_x - rank_z)

    def orthogonality_ok(self) -> bool:
        product = (self.HX @ self.HZ.T).tocoo()
        if product.nnz == 0:
            return True
        return bool(np.all((product.data.astype(np.int64) & 1) == 0))


def permutation_matrix(a: int, b: int, p: int, transpose: bool = False) -> sparse.csr_matrix:
    """Return the permutation matrix for x -> a*x + b mod p."""
    a = int(a) % int(p)
    b = int(b) % int(p)
    p = int(p)
    if gcd(a, p) != 1:
        raise ValueError(f"Multiplier {a} is not a unit modulo {p}.")

    rows = np.arange(p, dtype=np.int64)
    cols = ((a * rows + b) % p).astype(np.int64)
    if transpose:
        rows, cols = cols, rows
    data = np.ones(p, dtype=np.uint8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(p, p), dtype=np.uint8)


def build_code(spec: CodeSpec) -> AffineCSSCode:
    """Assemble sparse CSS checks from cyclically indexed affine blocks.

    Row i contains [F_(j-i), G_(j-i)] in H_X and
    [G_(i-j)^T, F_(i-j)^T] in H_Z, with indices modulo L/2.
    Commutation is a property of the supplied maps and must be checked.
    """
    if spec.l % 2:
        raise ValueError("L must be even so it can split into F and G halves.")
    if len(spec.f_params) != spec.half or len(spec.g_params) != spec.half:
        raise ValueError("Expected exactly L/2 F maps and L/2 G maps.")
    if spec.j < 1 or spec.j > spec.half:
        raise ValueError("J must satisfy 1 <= J <= L/2.")
    active_rows = spec.active_set
    if len(active_rows) != spec.j:
        raise ValueError("The active row set must contain exactly J rows.")
    if len(set(active_rows)) != len(active_rows):
        raise ValueError("The active row set cannot contain duplicates.")
    if any(row < 0 or row >= spec.half for row in active_rows):
        raise ValueError("Active rows must lie in 0..L/2-1.")

    f_mats = [permutation_matrix(a, b, spec.p) for a, b in spec.f_params]
    g_mats = [permutation_matrix(a, b, spec.p) for a, b in spec.g_params]
    f_t = [matrix.T.tocsr() for matrix in f_mats]
    g_t = [matrix.T.tocsr() for matrix in g_mats]

    hx_rows = []
    hz_rows = []
    for row in active_rows:
        hx_blocks = [f_mats[(col - row) % spec.half] for col in range(spec.half)]
        hx_blocks += [g_mats[(col - row) % spec.half] for col in range(spec.half)]
        hz_blocks = [g_t[(row - col) % spec.half] for col in range(spec.half)]
        hz_blocks += [f_t[(row - col) % spec.half] for col in range(spec.half)]
        hx_rows.append(sparse.hstack(hx_blocks, format="csr", dtype=np.uint8))
        hz_rows.append(sparse.hstack(hz_blocks, format="csr", dtype=np.uint8))

    hx = sparse.vstack(hx_rows, format="csr", dtype=np.uint8)
    hz = sparse.vstack(hz_rows, format="csr", dtype=np.uint8)
    hx.data %= 2
    hz.data %= 2
    return AffineCSSCode(spec=spec, HX=hx, HZ=hz)


def _rows_to_packed_uint64(matrix: sparse.spmatrix) -> Tuple[np.ndarray, int]:
    mat = matrix.tocsr().astype(np.uint8)
    row_indices, col_indices = mat.nonzero()
    nrows, ncols = mat.shape
    nwords = (ncols + 63) // 64
    packed = np.zeros((nrows, nwords), dtype=np.uint64)
    one = np.uint64(1)
    for row, col in zip(row_indices, col_indices):
        packed[int(row), int(col) >> 6] ^= one << np.uint64(int(col) & 63)
    return packed, ncols


def gf2_rank(matrix: sparse.spmatrix) -> int:
    """Compute the rank of a sparse binary matrix over GF(2)."""
    packed, ncols = _rows_to_packed_uint64(matrix)
    if packed.size == 0:
        return 0

    rows = packed.copy()
    nrows = rows.shape[0]
    rank = 0
    for col in range(ncols):
        word = col >> 6
        bit = np.uint64(1) << np.uint64(col & 63)
        pivot = None
        for row in range(rank, nrows):
            if rows[row, word] & bit:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            rows[[rank, pivot]] = rows[[pivot, rank]]
        for row in range(nrows):
            if row != rank and (rows[row, word] & bit):
                rows[row] ^= rows[rank]
        rank += 1
        if rank == nrows:
            break
    return int(rank)
