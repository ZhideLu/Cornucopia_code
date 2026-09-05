#!/usr/bin/env python3
"""Numeric and smoke tests for Fig. 3."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from figures.overhead.plot_overhead import (
    PlotConfig,
    bb_points,
    cornucopia_cycle_times_ms,
    cornucopia_points,
    make_figure,
    surface_points,
    validate_plot_config,
)


def test_cornucopia_overhead_includes_data_and_both_check_types() -> None:
    first = cornucopia_points()[0]
    assert first.data_qubits == 252
    assert first.check_qubits == 126
    assert first.total_physical_qubits == 378
    assert math.isclose(first.overhead_per_logical, 378 / 130)


def test_bb_overhead_uses_n_checks() -> None:
    points = bb_points()
    assert [point.total_physical_qubits for point in points] == [288, 576]
    assert [point.overhead_per_logical for point in points] == [24.0, 48.0]


def test_surface_overhead_matches_rotated_memory_qubit_counts() -> None:
    points = surface_points()
    assert [point.total_physical_qubits for point in points] == [
        97,
        161,
        241,
        337,
        721,
    ]


def test_cycle_total_row_is_transcribed_exactly() -> None:
    assert cornucopia_cycle_times_ms() == (
        (6, 10.41),
        (8, 11.92),
        (10, 12.87),
        (12, 13.22),
        (14, 14.68),
        (16, 15.48),
        (18, 16.18),
    )


def test_config_rejects_missing_family_style() -> None:
    config = PlotConfig(marker_choice={"Cornucopia": "o"})
    with pytest.raises(ValueError, match="missing families"):
        validate_plot_config(config)


def test_default_endpoint_annotations_target_reference_distances() -> None:
    config = PlotConfig()
    assert config.annotation_distance == {
        "Cornucopia": 18,
        "Surface": 19,
        "BB": 18,
    }
    assert config.annotation_font_size == {
        "Cornucopia": 10.0,
        "Surface": 10.0,
        "BB": 10.0,
    }
    assert config.y_axis_title == (
        "Physical overhead per logical qubit",
        "Atom-rearrangement\ntime per cycle (ms)",
    )


def test_figure_smoke_writes_only_pdf_and_csv(tmp_path: Path) -> None:
    pdf_path, data_path = make_figure(tmp_path, output_stem="smoke")
    assert pdf_path.is_file() and pdf_path.stat().st_size > 1000
    assert data_path.is_file() and data_path.stat().st_size > 100
    assert not (tmp_path / "smoke.png").exists()
