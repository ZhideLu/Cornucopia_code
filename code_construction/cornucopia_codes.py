#!/usr/bin/env python3
"""Parameters and algebraic checks for the eight Cornucopia code instances.

Seven instances are reported in the paper. ``cornucopia_p267_d18`` is an
additional distance-18 instance that is not reported there.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from scipy import io as scipy_io

from code_construction.affine_codes import CodeSpec, build_code

AffineMap = Tuple[int, int]


@dataclass(frozen=True)
class CornucopiaSpec:
    """One supplied code instance; expected_d is a nominal distance label."""

    name: str
    parameter_label: str
    p: int
    l: int
    j: int
    expected_k: int
    expected_d: int
    a_params: Tuple[AffineMap, ...]
    b_params: Tuple[AffineMap, ...]
    expected_noncommuting_pairs: Tuple[Tuple[int, int, int], ...]
    expected_reduced_group: str

    @property
    def n(self) -> int:
        return self.p * self.l

    def to_code_spec(self) -> CodeSpec:
        return CodeSpec(
            name=self.name,
            parameter_label=self.parameter_label,
            p=self.p,
            l=self.l,
            j=self.j,
            a_params=self.a_params,
            b_params=self.b_params,
            active_rows=(0, 1, 2),
        )


SPECS: Tuple[CornucopiaSpec, ...] = (
    CornucopiaSpec(
        name="cornucopia_p21_d6",
        parameter_label="[[252,130,6]]",
        p=21,
        l=12,
        j=3,
        expected_k=130,
        expected_d=6,
        expected_reduced_group="cyclic C7, abelian",
        a_params=((1, 16), (8, 8), (1, 15), (1, 15), (1, 18), (1, 12)),
        b_params=((1, 12), (1, 3), (1, 7), (8, 5), (1, 9), (1, 3)),
        expected_noncommuting_pairs=((0, 3, 14), (1, 2, 7)),
    ),
    CornucopiaSpec(
        name="cornucopia_p48_d8",
        parameter_label="[[576,292,8]]",
        p=48,
        l=12,
        j=3,
        expected_k=292,
        expected_d=8,
        expected_reduced_group="cyclic C16, abelian",
        a_params=((1, 1), (17, 5), (1, 42), (1, 12), (1, 12), (1, 24)),
        b_params=((1, 6), (1, 12), (1, 10), (17, 11), (1, 6), (1, 3)),
        expected_noncommuting_pairs=((0, 3, 32), (1, 2, 16)),
    ),
    CornucopiaSpec(
        name="cornucopia_p75_d10",
        parameter_label="[[900,454,10]]",
        p=75,
        l=12,
        j=3,
        expected_k=454,
        expected_d=10,
        expected_reduced_group="cyclic C25, abelian",
        a_params=((1, 73), (26, 32), (1, 57), (1, 12), (1, 69), (1, 6)),
        b_params=((1, 9), (1, 57), (1, 61), (26, 17), (1, 60), (1, 57)),
        expected_noncommuting_pairs=((0, 3, 50), (1, 2, 25)),
    ),
    CornucopiaSpec(
        name="cornucopia_p87_d12",
        parameter_label="[[1044,526,12]]",
        p=87,
        l=12,
        j=3,
        expected_k=526,
        expected_d=12,
        expected_reduced_group="cyclic C29, abelian",
        a_params=((1, 31), (59, 80), (1, 78), (1, 51), (1, 18), (1, 6)),
        b_params=((1, 27), (1, 69), (1, 70), (59, 47), (1, 21), (1, 84)),
        expected_noncommuting_pairs=((0, 3, 29), (1, 2, 58)),
    ),
    CornucopiaSpec(
        name="cornucopia_p147_d14",
        parameter_label="[[1764,886,14]]",
        p=147,
        l=12,
        j=3,
        expected_k=886,
        expected_d=14,
        expected_reduced_group="cyclic C49, abelian",
        a_params=((1, 142), (50, 5), (1, 33), (1, 123), (1, 120), (1, 105)),
        b_params=((1, 42), (1, 87), (1, 130), (50, 95), (1, 45), (1, 78)),
        expected_noncommuting_pairs=((0, 3, 98), (1, 2, 49)),
    ),
    CornucopiaSpec(
        name="cornucopia_p192_d16",
        parameter_label="[[2304,1156,16]]",
        p=192,
        l=12,
        j=3,
        expected_k=1156,
        expected_d=16,
        expected_reduced_group="cyclic C64, abelian",
        a_params=((1, 19), (65, 182), (1, 153), (1, 132), (1, 135), (1, 51)),
        b_params=((1, 84), (1, 21), (1, 112), (65, 14), (1, 120), (1, 183)),
        expected_noncommuting_pairs=((0, 3, 128), (1, 2, 64)),
    ),
    CornucopiaSpec(
        name="cornucopia_p237_d18",
        parameter_label="[[2844,1426,18]]",
        p=237,
        l=12,
        j=3,
        expected_k=1426,
        expected_d=18,
        expected_reduced_group="cyclic C79, abelian",
        a_params=((1, 85), (80, 128), (1, 213), (1, 18), (1, 198), (1, 165)),
        b_params=((1, 24), (1, 120), (1, 157), (80, 53), (1, 147), (1, 21)),
        expected_noncommuting_pairs=((0, 3, 158), (1, 2, 79)),
    ),
    # Additional distance-18 instance; it is not reported in the paper.
    CornucopiaSpec(
        name="cornucopia_p267_d18",
        parameter_label="[[3204,1606,18]]",
        p=267,
        l=12,
        j=3,
        expected_k=1606,
        expected_d=18,
        expected_reduced_group="cyclic C89, abelian",
        a_params=((1, 193), (179, 14), (1, 147), (1, 258), (1, 12), (1, 243)),
        b_params=((1, 120), (1, 60), (1, 232), (179, 50), (1, 96), (1, 15)),
        expected_noncommuting_pairs=((0, 3, 89), (1, 2, 178)),
    ),
)


def commutation_residue(left: AffineMap, right: AffineMap, modulus: int) -> int:
    a, b = left
    c, d = right
    return (d * (a - 1) - b * (c - 1)) % modulus


def noncommuting_ab_pairs(
    a_params: Sequence[AffineMap],
    b_params: Sequence[AffineMap],
    modulus: int,
) -> Tuple[Tuple[int, int, int], ...]:
    out = []
    for i, a_map in enumerate(a_params):
        for j, b_map in enumerate(b_params):
            residue = commutation_residue(a_map, b_map, modulus)
            if residue:
                out.append((i, j, residue))
    return tuple(out)


def all_maps_commute_mod(
    a_params: Sequence[AffineMap],
    b_params: Sequence[AffineMap],
    modulus: int,
) -> bool:
    maps = tuple(a_params) + tuple(b_params)
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            if commutation_residue(maps[i], maps[j], modulus):
                return False
    return True


def translation_count(a_params: Sequence[AffineMap], b_params: Sequence[AffineMap], p: int) -> int:
    return sum(1 for a, _b in (*a_params, *b_params) if a % p == 1)


def assert_valid_affine_maps(spec: CornucopiaSpec) -> None:
    for label, params in (("A", spec.a_params), ("B", spec.b_params)):
        if len(params) != 6:
            raise ValueError(f"{spec.name}: expected 6 {label} maps")
        for index, (a, _b) in enumerate(params):
            if gcd(int(a), spec.p) != 1:
                raise ValueError(
                    f"{spec.name}: {label}{index} multiplier {a} is not a unit mod {spec.p}"
                )


def matrix_market_write(matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = matrix.astype("int8").tocoo()
    mat.data = mat.data % 2
    scipy_io.mmwrite(str(path), mat)


def construct_one(spec: CornucopiaSpec, *, write_matrices: bool, matrix_dir: Path) -> dict:
    assert_valid_affine_maps(spec)
    code = build_code(spec.to_code_spec())
    rank_hx, rank_hz = code.ranks()
    k = code.num_logicals()
    q = spec.p // 3
    ab_noncommuting = noncommuting_ab_pairs(spec.a_params, spec.b_params, spec.p)
    reduced_abelian = all_maps_commute_mod(spec.a_params, spec.b_params, q)
    row_wx = np.asarray(code.HX.sum(axis=1)).ravel()
    row_wz = np.asarray(code.HZ.sum(axis=1)).ravel()
    col_wx = np.asarray(code.HX.sum(axis=0)).ravel()
    col_wz = np.asarray(code.HZ.sum(axis=0)).ravel()

    if write_matrices:
        matrix_market_write(code.HX, matrix_dir / f"{spec.name}_HX.mtx")
        matrix_market_write(code.HZ, matrix_dir / f"{spec.name}_HZ.mtx")

    checks = {
        "n_matches": code.n == spec.n,
        "k_matches": k == spec.expected_k,
        "orthogonality_ok": code.orthogonality_ok(),
        "noncommuting_pairs_match": ab_noncommuting == spec.expected_noncommuting_pairs,
        "reduced_mod_p_over_3_abelian": reduced_abelian,
    }
    checks["all_checks_ok"] = all(checks.values())

    return {
        "name": spec.name,
        "parameter_label": spec.parameter_label,
        "P": spec.p,
        "P_over_3": q,
        "L": spec.l,
        "J": spec.j,
        "active_rows": [0, 1, 2],
        "expected_d": spec.expected_d,
        "n": code.n,
        "expected_k": spec.expected_k,
        "k": k,
        "rank_HX": rank_hx,
        "rank_HZ": rank_hz,
        "orthogonality_ok": checks["orthogonality_ok"],
        "translation_count": translation_count(spec.a_params, spec.b_params, spec.p),
        "noncommuting_pairs": [list(item) for item in ab_noncommuting],
        "expected_noncommuting_pairs": [list(item) for item in spec.expected_noncommuting_pairs],
        "expected_reduced_group": spec.expected_reduced_group,
        "reduced_mod_p_over_3_abelian": reduced_abelian,
        "row_weight_HX_min": int(row_wx.min()),
        "row_weight_HX_max": int(row_wx.max()),
        "row_weight_HZ_min": int(row_wz.min()),
        "row_weight_HZ_max": int(row_wz.max()),
        "col_weight_HX_min": int(col_wx.min()),
        "col_weight_HX_max": int(col_wx.max()),
        "col_weight_HZ_min": int(col_wz.min()),
        "col_weight_HZ_max": int(col_wz.max()),
        "A": [list(item) for item in spec.a_params],
        "B": [list(item) for item in spec.b_params],
        "checks": checks,
    }


def write_text_summary(records: Sequence[dict], path: Path) -> None:
    lines = ["Cornucopia code construction summary", ""]
    for record in records:
        checks = record["checks"]
        lines.extend(
            [
                f"{record['name']} {record['parameter_label']}",
                f"  P={record['P']} L={record['L']} J={record['J']}",
                f"  [[n,k,d]]=[[{record['n']},{record['k']},{record['expected_d']}]] ranks=({record['rank_HX']},{record['rank_HZ']})",
                f"  orthogonality_ok={record['orthogonality_ok']} reduced_mod_P_over_3_abelian={record['reduced_mod_p_over_3_abelian']}",
                f"  noncommuting_pairs={record['noncommuting_pairs']}",
                f"  A={record['A']}",
                f"  B={record['B']}",
                f"  checks={checks}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
