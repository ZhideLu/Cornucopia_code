#!/usr/bin/env python3
"""Rotated planar surface-code specifications used by circuit simulation."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_DISTANCES = (6, 8, 10, 12, 14, 16, 18)


@dataclass(frozen=True)
class SurfaceCodeSpec:
    distance: int

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, int):
            raise TypeError("surface-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("surface-code distance must be at least 2")

    @property
    def name(self) -> str:
        return f"surface_d{self.distance}"

    @property
    def n(self) -> int:
        return int(self.distance) ** 2

    @property
    def k(self) -> int:
        return 1

    @property
    def expected_d(self) -> int:
        return int(self.distance)

    @property
    def parameter_label(self) -> str:
        return f"[[{self.n},{self.k},{self.expected_d}]]"


SPECS = tuple(SurfaceCodeSpec(distance) for distance in DEFAULT_DISTANCES)


def parse_code_names(value: str | None) -> list[SurfaceCodeSpec]:
    if value is None or value.strip().lower() in ("", "all"):
        return list(SPECS)

    requested = [item.strip() for item in value.split(",") if item.strip()]
    specs_by_name = {spec.name: spec for spec in SPECS}
    selected: list[SurfaceCodeSpec] = []
    invalid: list[str] = []
    for name in requested:
        spec = specs_by_name.get(name)
        if spec is None:
            match = re.fullmatch(r"surface_d([0-9]+)", name)
            if match is None or int(match.group(1)) < 2:
                invalid.append(name)
                continue
            spec = SurfaceCodeSpec(int(match.group(1)))
        selected.append(spec)
    if invalid:
        raise SystemExit(
            f"Invalid surface-code names: {invalid}. Use surface_dN with an integer N >= 2."
        )
    return selected
