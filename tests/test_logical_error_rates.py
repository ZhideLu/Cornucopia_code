"""Regression checks for the retained final two-panel logical-error figure.

The oracle is the final figure's published-data selection and mathematical
model, checked independently from CSVs shipped in this repository. No original
server checkout, raw decode summaries, or notebook execution is needed.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import numpy as np
import pytest

from figures.logical_error_rates import plot_error_rates as plotting

DATA = Path(__file__).resolve().parents[1] / "figures/logical_error_rates/data"
RATE = "LER_per_cycle_per_logical"
CORN_LABELS = (
    "[[252,130,6]]",
    "[[576,292,8]]",
    "[[900,454,10]]",
    "[[1044,526,12]]",
    "[[1764,886,14]]",
    "[[2304,1156,16]]",
    "[[2844,1426,18]]",
)
COMPARISON_LABELS = (
    "Cornucopia [[1044,526,12]]",
    "Cornucopia [[2844,1426,18]]",
    "BB [[144,12,12]]",
    "BB [[288,12,18]]",
    "Surface [[169,1,13]]",
    "Surface [[361,1,19]]",
)


def csv_rows(filename):
    with (DATA / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def data_artists(ax):
    return [item for item in ax.containers if isinstance(item, ErrorbarContainer)]


def fit_artists(ax):
    return [
        line
        for line in ax.lines
        if line.get_linestyle() not in ("None", "none", "")
        and line.get_label() != "break-even"
    ]


@pytest.fixture(scope="module")
def stacked_figure():
    figure = plotting.build_stacked_figure()
    yield figure
    plt.close(figure)


def test_final_panel_selection_scales_and_break_even(stacked_figure):
    assert len(stacked_figure.axes) == 2
    top, bottom = stacked_figure.axes
    assert [item.get_label() for item in data_artists(top)] == list(CORN_LABELS)
    assert [item.get_label() for item in data_artists(bottom)] == list(
        COMPARISON_LABELS
    )
    assert [len(fit_artists(ax)) for ax in (top, bottom)] == [7, 6]
    assert [len(item.lines[0].get_xdata()) for item in data_artists(top)] == [
        6,
        6,
        6,
        6,
        4,
        6,
        4,
    ]
    assert [len(item.lines[0].get_xdata()) for item in data_artists(bottom)] == [4] * 6
    for index, ax in enumerate((top, bottom)):
        assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
        np.testing.assert_allclose(ax.get_xlim(), (0.001 / 1.12, 0.0045 * 1.12))
        np.testing.assert_allclose(ax.get_ylim(), (1e-15, (1e-2, 1e-4)[index]), atol=0)
        assert [text.get_text() for text in ax.texts] == [("a", "b")[index]]
    assert top.get_shared_x_axes().joined(top, bottom)
    guides = [line for line in top.lines if line.get_label() == "break-even"]
    assert len(guides) == 1
    guide = guides[0]
    np.testing.assert_allclose(guide.get_xdata(), np.geomspace(0.001, 0.004, 300))
    np.testing.assert_array_equal(guide.get_ydata(), guide.get_xdata())
    assert (guide.get_linestyle(), guide.get_linewidth(), guide.get_color()) == (
        "--",
        1.4,
        "0.35",
    )
    assert not any(line.get_label() == "break-even" for line in bottom.lines)


def assert_measurements_and_curve(
    container, curve, rows, coefficient, family, curve_max
):
    """Check plotted values against CSVs, not against plotting helper results."""
    xs = np.array([float(row["p"]) for row in rows])
    ys = np.array([float(row["y"]) for row in rows])
    errors = np.array([float(row["yerr"]) for row in rows])
    marker = container.lines[0]
    np.testing.assert_array_equal(marker.get_xdata(), xs)
    np.testing.assert_array_equal(marker.get_ydata(), ys)
    segments = np.asarray(container.lines[2][0].get_segments())
    np.testing.assert_array_equal(segments[:, :, 0], np.column_stack((xs, xs)))
    np.testing.assert_allclose(
        segments[:, :, 1], np.column_stack((ys - errors, ys + errors)), atol=0
    )

    fit_x = np.asarray(curve.get_xdata())
    np.testing.assert_allclose(fit_x, np.geomspace(0.001, curve_max, 300))
    distance = int(coefficient["expected_d"])
    polynomial = np.polyval(
        [float(coefficient[key]) for key in ("c2", "c1", "c0")], fit_x
    )
    expected_fit = fit_x ** (distance / 2) * np.exp(polynomial)
    np.testing.assert_allclose(curve.get_ydata(), expected_fit, rtol=2e-10, atol=0)
    styles = {"Cornucopia": ("d", 10, "-", 1.4), "BB": ("*", 12, "--", 1.2)}
    expected_marker, size, line_style, line_width = styles[family]
    assert (marker.get_marker(), marker.get_markersize()) == (expected_marker, size)
    assert (curve.get_linestyle(), curve.get_linewidth()) == (line_style, line_width)
    assert marker.get_color() == curve.get_color()


def test_cornucopia_and_bb_measurements_errors_and_fits_match_tables(stacked_figure):
    top, bottom = stacked_figure.axes
    for panel, family, prefix in (
        (top, "Cornucopia", "cornucopia"),
        (bottom, "Cornucopia", "cornucopia"),
        (bottom, "BB", "bb"),
    ):
        points = csv_rows(f"{prefix}_xz_average_fit_data.csv")
        coefficients = csv_rows(f"{prefix}_xz_average_fit_coefficients.csv")
        artists = {
            item.get_label(): (item, curve)
            for item, curve in zip(data_artists(panel), fit_artists(panel), strict=True)
        }
        for coefficient in coefficients:
            label = coefficient["parameter_label"]
            if panel is bottom:
                label = f"{family} {label}"
            if label not in artists:
                continue
            rows = sorted(
                (row for row in points if row["code_name"] == coefficient["code_name"]),
                key=lambda row: float(row["p"]),
            )
            if panel is bottom and family == "Cornucopia":
                rows = [row for row in rows if float(row["p"]) <= 0.0025]
            curve_max = (
                max(float(row["p"]) for row in rows) if family == "BB" else 0.0025
            )
            assert_measurements_and_curve(
                *artists[label], rows, coefficient, family, curve_max
            )


def test_surface_fit_uses_unweighted_log_model_and_counting_error(stacked_figure):
    bottom = stacked_figure.axes[1]
    rows = csv_rows("surface_xz_average_rates.csv")
    for index, distance in enumerate((13, 19), start=4):
        selected = sorted(
            (row for row in rows if int(row["expected_d"]) == distance),
            key=lambda row: float(row["p"]),
        )
        xs = np.array([float(row["p"]) for row in selected])
        ys = np.array([float(row[RATE]) for row in selected])
        failures = np.array([int(row["failures"]) for row in selected])
        points, curve = data_artists(bottom)[index], fit_artists(bottom)[index]
        np.testing.assert_array_equal(points.lines[0].get_xdata(), xs)
        np.testing.assert_array_equal(points.lines[0].get_ydata(), ys)
        segments = np.asarray(points.lines[2][0].get_segments())
        errors = ys / np.sqrt(failures)
        np.testing.assert_allclose(
            segments[:, :, 1], np.column_stack((ys - errors, ys + errors)), atol=0
        )
        # Scaled polynomial fitting is an independent oracle for the same
        # unweighted quadratic log model; no production fitting helper is used.
        polynomial = np.polynomial.Polynomial.fit(
            xs, np.log(ys / xs ** (distance / 2)), 2
        )
        fit_x = np.asarray(curve.get_xdata())
        np.testing.assert_allclose(fit_x, np.geomspace(0.001, xs.max(), 300))
        np.testing.assert_allclose(
            curve.get_ydata(),
            fit_x ** (distance / 2) * np.exp(polynomial(fit_x)),
            rtol=2e-8,
            atol=0,
        )
        assert (points.lines[0].get_marker(), points.lines[0].get_markersize()) == (
            "s",
            8,
        )
        assert (curve.get_linestyle(), curve.get_linewidth()) == ("-.", 1.2)


def test_only_final_pdf_is_written_from_five_standalone_tables(tmp_path):
    names = {
        "cornucopia_xz_average_fit_data.csv",
        "cornucopia_xz_average_fit_coefficients.csv",
        "bb_xz_average_fit_data.csv",
        "bb_xz_average_fit_coefficients.csv",
        "surface_xz_average_rates.csv",
    }
    assert set(plotting.TABLES.values()) == names
    inputs, output = tmp_path / "tables", tmp_path / "output"
    inputs.mkdir()
    for name in names:
        shutil.copyfile(DATA / name, inputs / name)
    result = plotting.plot_error_rates(inputs, output)
    assert result == output / "stacked_error_rate_fits.pdf"
    assert plotting.OUTPUT_FILENAME == result.name
    assert list(output.iterdir()) == [result]
    assert result.read_bytes().startswith(b"%PDF-")
    assert {path.name for path in inputs.iterdir()} == names


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown_setting": 1},
        {"height": [6]},
        {"width": [7, 0]},
        {"axis_label_size": [-1, 14]},
        {"subplot_label_x": ["bad", 0]},
        {"grid_line_width": [0, 0.5]},
        {"grid_alpha": [0.3, 1.1]},
        {"y_limits": [(0, 1e-2), (1e-15, 1e-4)]},
        {"y_limits": [(1e-2, 1e-15), (1e-15, 1e-4)]},
        {"figure_margins": {"left": 0.9, "right": 0.1, "bottom": 0.1, "top": 0.9}},
        {"pdf_margin_inches": {"left": -1, "right": 0, "bottom": 0, "top": 0}},
        {"break_even_style": {"color": "black", "linestyle": "--", "linewidth": 0}},
        {"break_even_x_limits": (0.004, 0.001)},
        {"marker_choice": {"Cornucopia": "o"}},
    ],
)
def test_invalid_display_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        plotting.build_stacked_figure(display_settings=settings)


def test_display_overrides_do_not_mutate_defaults():
    original = deepcopy(plotting.DEFAULT_DISPLAY_SETTINGS)
    override = {
        "height": [4, 3],
        "marker_choice": {"Cornucopia": "o", "BB": "*", "Surface": "s"},
    }
    figure = plotting.build_stacked_figure(display_settings=override)
    try:
        np.testing.assert_array_equal(figure.get_size_inches(), [7, 7])
        assert data_artists(figure.axes[0])[0].lines[0].get_marker() == "o"
        assert plotting.DEFAULT_DISPLAY_SETTINGS == original
        assert override["height"] == [4, 3]
    finally:
        plt.close(figure)


def test_surface_shots_are_combined_and_zero_failure_points_are_excluded(tmp_path):
    template = {
        "expected_d": 13,
        "code_name": "surface_d13",
        "parameter_label": "[[169,1,13]]",
        "p": 0.002,
        RATE: 1e-7,
        "failures": 2,
        "x_shots": 3,
        "z_shots": 7,
    }
    records = [
        {**template, "p": 0.003},
        {**template, "p": 0.002, "x_shots": 0, "z_shots": 1},
        {**template, "p": 0.001, "failures": 0},
        {**template, "p": 0.0015, RATE: 0},
        {**template, "p": 0.004, "x_shots": 0, "z_shots": 0},
        {**template, "p": 0.006},
        {**template, "p": 0},
    ]
    path = tmp_path / "surface.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(template))
        writer.writeheader()
        writer.writerows(records)
    groups = plotting.surface_groups_from_table(path)
    assert set(groups) == {(13, "surface_d13", "[[169,1,13]]")}
    selected = next(iter(groups.values()))
    assert [record["p"] for record in selected] == [0.002, 0.003]
    assert [record["shots"] for record in selected] == [1, 10]
    assert all(record["failures"] == 2 for record in selected)
