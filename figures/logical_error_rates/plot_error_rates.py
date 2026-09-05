"""Plot the retained two-panel fitted comparison from the supplied X/Z tables.

Panel a contains the Cornucopia codes; panel b compares Cornucopia, BB, and
surface codes. The raw decode summaries and decoder packages are not needed.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.transforms import Bbox

RATE_COLUMN = FIT_Y_COLUMN = "LER_per_cycle_per_logical"
FIT_WEIGHT_MODE = "unweighted_log"
MIN_SHOTS = 1
SURFACE_FIT_MAX_P = 0.005
COMMON_AXIS_CURVE_MIN_P = 0.001
COMMON_AXIS_Y_MIN = 1e-15
COMMON_AXIS_Y_MAJOR_TICKS = [10.0 ** (-exp) for exp in range(1, 16, 2)]
COMPARISON_DISTANCES = {12, 18}
STACKED_EXCLUDED_SURFACE_DISTANCES = {11}
AVERAGE_SAVE_PREFIX = "cornucopia_xz_average_fit"
BB_FIT_SAVE_PREFIX = "bb_xz_average_fit"
OUTPUT_FILENAME = "stacked_error_rate_fits.pdf"
TABLES = {
    "Cornucopia points": "cornucopia_xz_average_fit_data.csv",
    "Cornucopia coefficients": "cornucopia_xz_average_fit_coefficients.csv",
    "BB points": "bb_xz_average_fit_data.csv",
    "BB coefficients": "bb_xz_average_fit_coefficients.csv",
    "Surface X/Z averages": "surface_xz_average_rates.csv",
}
FONT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

DEFAULT_DISPLAY_SETTINGS = {
    "height": [6, 5.4],
    "width": [7, 7],
    "axis_label_size": [14.5, 14.5],
    "tick_label_size": [12, 12],
    "legend_font_size": [11, 11],
    "subplot_label_size": [15, 15],
    "subplot_label_x": [-0.125, -0.125],
    "subplot_label_y": [0.98, 0.98],
    "grid_line_width": [0.5, 0.5],
    "grid_alpha": [0.3, 0.3],
    "y_limits": [(1e-15, 0.01), (1e-15, 0.0001)],
    "figure_margins": {"left": 0.14, "right": 0.98, "bottom": 0.07, "top": 0.99},
    "pdf_margin_inches": {"left": 0.01, "right": 0.05, "bottom": 0.01, "top": 0.01},
    "marker_choice": {"Cornucopia": "d", "BB": "*", "Surface": "s"},
    "line_style": {"Cornucopia": "-", "BB": "--", "Surface": "-."},
    "marker_size": {"Cornucopia": 10, "BB": 12, "Surface": 8},
    "line_width": {"Cornucopia": 1.4, "BB": 1.2, "Surface": 1.2},
    "break_even_style": {"color": "0.35", "linestyle": "--", "linewidth": 1.4},
    "break_even_x_limits": (0.001, 0.004),
    "panel_gap": 0.18,
}


def read_table(path: Path) -> list[dict[str, str]]:
    """Read CSV rows without changing the supplied numeric precision."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


read_plot_csv = read_table


def distance_model(p, distance: int, c0: float, c1: float, c2: float) -> np.ndarray:
    """Evaluate the distance model for positive physical error probabilities."""
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("Physical error rates must be positive")
    return model_curve(p, distance, np.asarray([c0, c1, c2], dtype=float))


def surface_groups_from_table(path: Path) -> dict[tuple[int, str, str], list[dict]]:
    """Restore the numeric average records used by the original surface fit.

    The exported table stores X and Z shot counts separately. Their sum is
    the average record's shot count; the stored total failures determines
    the original counting-error approximation.
    """
    groups = defaultdict(list)
    for row in read_table(path):
        record = dict(row)
        for key in ("expected_d", "failures", "x_shots", "z_shots"):
            record[key] = int(row[key])
        record["p"] = float(row["p"])
        record[FIT_Y_COLUMN] = float(row[FIT_Y_COLUMN])
        record["shots"] = record["x_shots"] + record["z_shots"]
        if not (
            0 < record["p"] <= SURFACE_FIT_MAX_P
            and record[FIT_Y_COLUMN] > 0
            and record["shots"] >= MIN_SHOTS
            and record["failures"] > 0
        ):
            continue
        key = (record["expected_d"], record["code_name"], record["parameter_label"])
        groups[key].append(record)
    for rows in groups.values():
        rows.sort(key=lambda record: record["p"])
    if not any(key[0] not in STACKED_EXCLUDED_SURFACE_DISTANCES for key in groups):
        raise ValueError("No eligible surface-code X/Z-average points")
    return dict(groups)


def setup_plot_style() -> tuple[list[str], list[str]]:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.dpi": 160,
        }
    )
    colors = [
        "#0072B2",
        "#3DBB8D",
        "#F0A21A",
        "#111111",
        "#F26B8A",
        "#6F4CC3",
        "#8C8C8C",
    ]
    markers = ["d", "o", "s", "^", "v", "P", "X"]
    return colors, markers


def apply_common_log_axis_style(
    ax,
    x_values,
    y_values_for_limits=None,
) -> None:
    """Use the same log-axis settings for raw-rate and fitted-model plots."""
    clean_x = [float(x) for x in x_values if x is not None and float(x) > 0]
    clean_x.append(COMMON_AXIS_CURVE_MIN_P)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if clean_x:
        ax.set_xlim(min(clean_x) / 1.12, max(clean_x) * 1.12)

    if y_values_for_limits:
        clean_y = [
            float(y) for y in y_values_for_limits if y is not None and float(y) > 0
        ]
    else:
        clean_y = []
    if clean_y:
        ymax = max(max(clean_y) * 1.4, COMMON_AXIS_Y_MAJOR_TICKS[0])
    else:
        ymax = COMMON_AXIS_Y_MAJOR_TICKS[0]
    ax.set_ylim(COMMON_AXIS_Y_MIN, ymax)
    ax.set_yticks(COMMON_AXIS_Y_MAJOR_TICKS)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.65, alpha=0.45)


def failure_column_for_y(y_column: str) -> str:
    if y_column.startswith("c_"):
        return "c_failures"
    if y_column.startswith("u_"):
        return "u_failures"
    return "failures"


def yerr_from_failures(record: dict[str, object], y_column: str) -> float:
    y_value = record.get(y_column)
    failures = record.get(failure_column_for_y(y_column))
    if not (isinstance(y_value, float) and y_value > 0):
        return 0.0
    if not (isinstance(failures, int) and failures > 0):
        return 0.0
    return float(y_value) / math.sqrt(failures)


def distance_fit(
    rows: list[dict[str, object]], fit_d: int
) -> tuple[np.ndarray, dict[str, float]]:
    if len(rows) < 2:
        raise ValueError(f"Need at least 2 data points for fit, got {len(rows)}")
    p = np.asarray([float(r["p"]) for r in rows], dtype=float)
    y = np.asarray([float(r[FIT_Y_COLUMN]) for r in rows], dtype=float)
    failures = np.asarray([max(int(r["failures"]), 1) for r in rows], dtype=float)
    target = np.log(y) - 0.5 * float(fit_d) * np.log(p)
    fit_degree = min(2, len(rows) - 1)
    if fit_degree == 2:
        design = np.column_stack([np.ones_like(p), p, p * p])
    else:
        design = np.column_stack([np.ones_like(p), p])
    if FIT_WEIGHT_MODE == "unweighted_log":
        sqrt_w = np.ones_like(p)
    elif FIT_WEIGHT_MODE == "failure_weighted_log":
        sqrt_w = np.sqrt(failures)
    else:
        raise ValueError(f"unknown FIT_WEIGHT_MODE={FIT_WEIGHT_MODE!r}")
    coeff_raw, *_ = np.linalg.lstsq(
        design * sqrt_w[:, None], target * sqrt_w, rcond=None
    )
    pred_target = design @ coeff_raw
    if fit_degree == 2:
        coeff = coeff_raw
    else:
        coeff = np.asarray([float(coeff_raw[0]), float(coeff_raw[1]), 0.0], dtype=float)
    pred_log_y = 0.5 * float(fit_d) * np.log(p) + pred_target
    residual = np.log(y) - pred_log_y
    if FIT_WEIGHT_MODE == "failure_weighted_log":
        rmse_log = float(np.sqrt(np.average(residual * residual, weights=failures)))
    else:
        rmse_log = float(np.sqrt(np.mean(residual * residual)))
    stats = {
        "weight_mode": FIT_WEIGHT_MODE,
        "rmse_log": rmse_log,
        "max_abs_residual_log": float(np.max(np.abs(residual))),
        "num_points": int(len(rows)),
        "fit_degree": int(fit_degree),
        "failure_weight_sum": int(np.sum(failures)),
    }
    return coeff, stats


def model_curve(p: np.ndarray, fit_d: int, coeff: np.ndarray) -> np.ndarray:
    c0, c1, c2 = [float(x) for x in coeff]
    return np.exp(0.5 * float(fit_d) * np.log(p) + c0 + c1 * p + c2 * p * p)


def build_stacked_figure(
    data_dir: Path | None = None, display_settings: dict | None = None
):
    """Build the final two-panel figure; the caller owns and closes it.

    Display settings override the retained notebook's defaults. They do not
    alter code selection, fitted coefficients, or X/Z averaging conventions.
    """
    data_dir = (
        Path(__file__).resolve().parent / "data" if data_dir is None else Path(data_dir)
    )
    missing = [
        data_dir / filename
        for filename in TABLES.values()
        if not (data_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing final-figure tables: " + ", ".join(map(str, missing))
        )
    settings = deepcopy(DEFAULT_DISPLAY_SETTINGS)
    if display_settings is not None:
        unknown = set(display_settings) - set(settings)
        if unknown:
            raise ValueError(f"Unknown display settings: {sorted(unknown)}")
        settings.update(deepcopy(display_settings))
    OUT_DIR = data_dir
    surface_groups = surface_groups_from_table(
        data_dir / "surface_xz_average_rates.csv"
    )
    height = settings["height"]
    width = settings["width"]
    axis_label_size = settings["axis_label_size"]
    tick_label_size = settings["tick_label_size"]
    legend_font_size = settings["legend_font_size"]
    subplot_label_size = settings["subplot_label_size"]
    subplot_label_x = settings["subplot_label_x"]
    subplot_label_y = settings["subplot_label_y"]
    grid_line_width = settings["grid_line_width"]
    grid_alpha = settings["grid_alpha"]
    y_limits = settings["y_limits"]
    figure_margins = settings["figure_margins"]
    pdf_margin_inches = settings["pdf_margin_inches"]
    marker_choice = settings["marker_choice"]
    line_style = settings["line_style"]
    marker_size = settings["marker_size"]
    line_width = settings["line_width"]
    break_even_style = settings["break_even_style"]
    break_even_x_limits = settings["break_even_x_limits"]
    panel_gap = settings["panel_gap"]
    with plt.rc_context(FONT_STYLE):
        setup_plot_style()

        def _validate_stacked_display_settings() -> None:
            if len(height) != 2 or len(width) != 2:
                raise ValueError("height and width must each contain [top, bottom]")
            if any(float(value) <= 0 for value in [*height, *width]):
                raise ValueError("height and width values must be positive")
            for name, values in [
                ("axis_label_size", axis_label_size),
                ("tick_label_size", tick_label_size),
                ("legend_font_size", legend_font_size),
                ("subplot_label_size", subplot_label_size),
            ]:
                if len(values) != 2:
                    raise ValueError(f"{name} must contain [top, bottom]")
                if any(float(value) <= 0 for value in values):
                    raise ValueError(f"{name} values must be positive")
            for name, values in [
                ("subplot_label_x", subplot_label_x),
                ("subplot_label_y", subplot_label_y),
            ]:
                if len(values) != 2:
                    raise ValueError(f"{name} must contain [panel a, panel b]")
                try:
                    [float(value) for value in values]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} values must be numeric") from exc
            for name, values in [
                ("grid_line_width", grid_line_width),
                ("grid_alpha", grid_alpha),
            ]:
                if len(values) != 2:
                    raise ValueError(f"{name} must contain [panel a, panel b]")
                try:
                    numeric_values = [float(value) for value in values]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} values must be numeric") from exc
                if name == "grid_line_width" and any(
                    value <= 0 for value in numeric_values
                ):
                    raise ValueError("grid_line_width values must be positive")
                if name == "grid_alpha" and any(
                    not 0 <= value <= 1 for value in numeric_values
                ):
                    raise ValueError("grid_alpha values must be between 0 and 1")
            if len(y_limits) != 2:
                raise ValueError(
                    "y_limits must contain [(a_min, a_max), (b_min, b_max)]"
                )
            for panel_index, limits in enumerate(y_limits):
                if len(limits) != 2:
                    raise ValueError(f"y_limits[{panel_index}] must contain (min, max)")
                try:
                    lower, upper = [float(value) for value in limits]
                except (TypeError, ValueError) as exc:
                    raise ValueError("y_limits values must be numeric") from exc
                if lower <= 0 or upper <= lower:
                    raise ValueError("Each y_limits pair must satisfy 0 < min < max")
            required_margin_keys = {
                "left",
                "right",
                "bottom",
                "top",
            }
            for name, margins in [
                ("figure_margins", figure_margins),
                ("pdf_margin_inches", pdf_margin_inches),
            ]:
                if set(margins) != required_margin_keys:
                    raise ValueError(
                        f"{name} must contain exactly {sorted(required_margin_keys)}"
                    )
                try:
                    numeric_margins = {
                        key: float(value) for key, value in margins.items()
                    }
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} values must be numeric") from exc
                if name == "pdf_margin_inches" and any(
                    value < 0 for value in numeric_margins.values()
                ):
                    raise ValueError("pdf_margin_inches values must be nonnegative")
            figure_left = float(figure_margins["left"])
            figure_right = float(figure_margins["right"])
            figure_bottom = float(figure_margins["bottom"])
            figure_top = float(figure_margins["top"])
            if not 0 <= figure_left < figure_right <= 1:
                raise ValueError("figure_margins must satisfy 0 <= left < right <= 1")
            if not 0 <= figure_bottom < figure_top <= 1:
                raise ValueError("figure_margins must satisfy 0 <= bottom < top <= 1")
            required_break_even_keys = {
                "color",
                "linestyle",
                "linewidth",
            }
            if set(break_even_style) != required_break_even_keys:
                raise ValueError(
                    "break_even_style must contain exactly "
                    f"{sorted(required_break_even_keys)}"
                )
            if float(break_even_style["linewidth"]) <= 0:
                raise ValueError("break_even_style linewidth must be positive")
            if len(break_even_x_limits) != 2:
                raise ValueError("break_even_x_limits must contain (min, max)")
            try:
                break_even_x_min, break_even_x_max = [
                    float(value) for value in break_even_x_limits
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError("break_even_x_limits values must be numeric") from exc
            if break_even_x_min <= 0 or break_even_x_max <= break_even_x_min:
                raise ValueError("break_even_x_limits must satisfy 0 < min < max")
            families = {"Cornucopia", "BB", "Surface"}
            for name, values in [
                ("marker_choice", marker_choice),
                ("line_style", line_style),
                ("marker_size", marker_size),
                ("line_width", line_width),
            ]:
                missing = families - set(values)
                if missing:
                    raise ValueError(f"{name} is missing families: {sorted(missing)}")

        def _stacked_cornucopia_panel(ax) -> tuple[list[float], list[float]]:
            data_rows = read_plot_csv(OUT_DIR / f"{AVERAGE_SAVE_PREFIX}_data.csv")
            coefficient_rows = read_plot_csv(
                OUT_DIR / f"{AVERAGE_SAVE_PREFIX}_coefficients.csv"
            )
            colors, _ = setup_plot_style()
            all_x = [COMMON_AXIS_CURVE_MIN_P]
            all_y: list[float] = []

            for index, coefficient_row in enumerate(coefficient_rows):
                expected_d = int(coefficient_row["expected_d"])
                code_name = coefficient_row["code_name"]
                rows = sorted(
                    [
                        row
                        for row in data_rows
                        if int(row["expected_d"]) == expected_d
                        and row["code_name"] == code_name
                    ],
                    key=lambda row: float(row["p"]),
                )
                if not rows:
                    continue

                xs = np.asarray([float(row["p"]) for row in rows], dtype=float)
                ys = np.asarray([float(row["y"]) for row in rows], dtype=float)
                yerrs = np.asarray([float(row["yerr"]) for row in rows], dtype=float)
                fit_xs = np.asarray(
                    [
                        float(row["p"])
                        for row in rows
                        if row["used_for_fit"].lower() == "true"
                    ],
                    dtype=float,
                )
                if fit_xs.size < 2:
                    raise RuntimeError(
                        f"Not enough Cornucopia fit points for {code_name}"
                    )

                fit_d = int(coefficient_row.get("fit_d") or expected_d)
                coeff = np.asarray(
                    [
                        float(coefficient_row["c0"]),
                        float(coefficient_row["c1"]),
                        float(coefficient_row["c2"]),
                    ],
                    dtype=float,
                )
                curve_min = min(
                    float(coefficient_row["curve_min_p"]),
                    float(fit_xs.min()),
                )
                curve_x = np.geomspace(curve_min, float(fit_xs.max()), 300)
                curve_y = model_curve(curve_x, fit_d, coeff)
                color = colors[index % len(colors)]
                marker = marker_choice["Cornucopia"]

                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    fmt=marker,
                    markersize=marker_size["Cornucopia"],
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    ecolor=color,
                    elinewidth=1.1,
                    capsize=3.2,
                    capthick=1.1,
                    linestyle="none",
                    label=coefficient_row["parameter_label"],
                    zorder=3,
                )
                ax.plot(
                    curve_x,
                    curve_y,
                    linestyle=line_style["Cornucopia"],
                    color=color,
                    linewidth=line_width["Cornucopia"],
                    alpha=0.9,
                    zorder=2,
                )
                all_x.extend(xs.tolist())
                all_x.extend(curve_x.tolist())
                all_y.extend((ys + yerrs).tolist())
                all_y.extend(np.maximum(ys - yerrs, ys * 0.5).tolist())
                all_y.extend(curve_y.tolist())

            ax.set_xlabel(
                "Physical error rate, $p$",
                fontsize=axis_label_size[0],
            )
            ax.set_ylabel(
                r"LER per logical per cycle, $p_L$",
                fontsize=axis_label_size[0],
            )
            apply_common_log_axis_style(ax, all_x, all_y)
            ax.set_ylim(*[float(value) for value in y_limits[0]])
            ax.grid(
                True,
                which="both",
                linestyle=":",
                linewidth=float(grid_line_width[0]),
                alpha=float(grid_alpha[0]),
            )
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=tick_label_size[0],
            )
            ax.legend(
                frameon=False,
                fontsize=legend_font_size[0],
                loc="lower right",
            )
            return all_x, all_y

        def _stacked_family_comparison_panel(
            ax,
        ) -> tuple[list[float], list[float]]:
            sources = [
                {
                    "family": "Cornucopia",
                    "basis": "X/Z avg",
                    "data_path": OUT_DIR / f"{AVERAGE_SAVE_PREFIX}_data.csv",
                    "coefficients_path": (
                        OUT_DIR / f"{AVERAGE_SAVE_PREFIX}_coefficients.csv"
                    ),
                    "curve_to_all_points": False,
                    "markersize": marker_size["Cornucopia"],
                },
                {
                    "family": "BB",
                    "basis": "X/Z avg",
                    "data_path": OUT_DIR / f"{BB_FIT_SAVE_PREFIX}_data.csv",
                    "coefficients_path": (
                        OUT_DIR / f"{BB_FIT_SAVE_PREFIX}_coefficients.csv"
                    ),
                    "curve_to_all_points": True,
                    "markersize": marker_size["BB"],
                },
            ]
            colors, _ = setup_plot_style()
            all_x = [COMMON_AXIS_CURVE_MIN_P]
            all_y: list[float] = []

            for source in sources:
                data_rows = read_plot_csv(source["data_path"])
                coefficient_rows = read_plot_csv(source["coefficients_path"])
                style_index = {
                    (int(row["expected_d"]), row["code_name"]): index
                    for index, row in enumerate(coefficient_rows)
                }

                for coefficient_row in coefficient_rows:
                    expected_d = int(coefficient_row["expected_d"])
                    if expected_d not in COMPARISON_DISTANCES:
                        continue

                    code_name = coefficient_row["code_name"]
                    rows = sorted(
                        [
                            row
                            for row in data_rows
                            if int(row["expected_d"]) == expected_d
                            and row["code_name"] == code_name
                            and (
                                source["family"] != "Cornucopia"
                                or float(row["p"])
                                <= float(coefficient_row["fit_max_p"])
                            )
                        ],
                        key=lambda row: float(row["p"]),
                    )
                    if not rows:
                        raise RuntimeError(
                            f"No stacked-plot data for {source['family']} {code_name}"
                        )

                    xs = np.asarray(
                        [float(row["p"]) for row in rows],
                        dtype=float,
                    )
                    ys = np.asarray(
                        [float(row["y"]) for row in rows],
                        dtype=float,
                    )
                    yerrs = np.asarray(
                        [float(row["yerr"]) for row in rows],
                        dtype=float,
                    )
                    fit_xs = np.asarray(
                        [
                            float(row["p"])
                            for row in rows
                            if row["used_for_fit"].lower() == "true"
                        ],
                        dtype=float,
                    )
                    if fit_xs.size < 2:
                        raise RuntimeError(
                            f"Not enough stacked fit points for "
                            f"{source['family']} {code_name}"
                        )

                    fit_d = int(coefficient_row.get("fit_d") or expected_d)
                    coeff = np.asarray(
                        [
                            float(coefficient_row["c0"]),
                            float(coefficient_row["c1"]),
                            float(coefficient_row["c2"]),
                        ],
                        dtype=float,
                    )
                    curve_min = min(
                        float(coefficient_row["curve_min_p"]),
                        float(fit_xs.min()),
                    )
                    curve_max = (
                        float(xs.max())
                        if source["curve_to_all_points"]
                        else float(fit_xs.max())
                    )
                    curve_x = np.geomspace(curve_min, curve_max, 300)
                    curve_y = model_curve(curve_x, fit_d, coeff)
                    index = style_index[(expected_d, code_name)]
                    color = colors[index % len(colors)]
                    marker = marker_choice[source["family"]]

                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerrs,
                        fmt=marker,
                        markersize=source["markersize"],
                        color=color,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        ecolor=color,
                        elinewidth=1.1,
                        capsize=3.2,
                        capthick=1.1,
                        linestyle="none",
                        label=(
                            f"{source['family']} {coefficient_row['parameter_label']}"
                        ),
                        zorder=3,
                    )
                    ax.plot(
                        curve_x,
                        curve_y,
                        linestyle=line_style[source["family"]],
                        color=color,
                        linewidth=line_width[source["family"]],
                        alpha=0.9,
                        zorder=2,
                    )
                    all_x.extend(xs.tolist())
                    all_x.extend(curve_x.tolist())
                    all_y.extend((ys + yerrs).tolist())
                    all_y.extend(np.maximum(ys - yerrs, ys * 0.5).tolist())
                    all_y.extend(curve_y.tolist())

            surface_colors = {
                13: colors[4 % len(colors)],
                19: "#8c564b",
            }
            for (
                expected_d,
                code_name,
                parameter_label,
            ), rows in sorted(surface_groups.items()):
                if expected_d in STACKED_EXCLUDED_SURFACE_DISTANCES:
                    continue
                if len(rows) < 2:
                    raise RuntimeError(
                        f"Not enough surface-code points for {code_name}: {len(rows)}"
                    )

                coeff, _ = distance_fit(rows, expected_d)
                xs = np.asarray(
                    [float(row["p"]) for row in rows],
                    dtype=float,
                )
                ys = np.asarray(
                    [float(row[FIT_Y_COLUMN]) for row in rows],
                    dtype=float,
                )
                yerrs = np.asarray(
                    [yerr_from_failures(row, FIT_Y_COLUMN) for row in rows],
                    dtype=float,
                )
                curve_x = np.geomspace(
                    COMMON_AXIS_CURVE_MIN_P,
                    float(xs.max()),
                    300,
                )
                curve_y = model_curve(curve_x, expected_d, coeff)
                color = surface_colors.get(expected_d, f"C{expected_d % 10}")
                marker = marker_choice["Surface"]

                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    fmt=marker,
                    markersize=marker_size["Surface"],
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    ecolor=color,
                    elinewidth=1.1,
                    capsize=3.2,
                    capthick=1.1,
                    linestyle="none",
                    label=f"Surface {parameter_label}",
                    zorder=3,
                )
                ax.plot(
                    curve_x,
                    curve_y,
                    linestyle=line_style["Surface"],
                    color=color,
                    linewidth=line_width["Surface"],
                    alpha=0.9,
                    zorder=2,
                )
                all_x.extend(xs.tolist())
                all_x.extend(curve_x.tolist())
                all_y.extend((ys + yerrs).tolist())
                all_y.extend(np.maximum(ys - yerrs, ys * 0.5).tolist())
                all_y.extend(curve_y.tolist())

            ax.set_xlabel(
                "Physical error rate, $p$",
                fontsize=axis_label_size[1],
            )
            ax.set_ylabel(
                r"LER per logical per cycle, $p_L$",
                fontsize=axis_label_size[1],
            )
            apply_common_log_axis_style(ax, all_x, all_y)
            ax.set_ylim(*[float(value) for value in y_limits[1]])
            ax.grid(
                True,
                which="both",
                linestyle=":",
                linewidth=float(grid_line_width[1]),
                alpha=float(grid_alpha[1]),
            )
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=tick_label_size[1],
            )
            ax.legend(
                frameon=False,
                fontsize=legend_font_size[1],
                loc="lower right",
            )
            return all_x, all_y

        def _add_centered_stacked_axis(
            fig,
            outer_spec,
            panel_width: float,
            maximum_width: float,
            sharex=None,
        ):
            side_width = max((maximum_width - panel_width) / 2.0, 0.0)
            if side_width <= 1e-12:
                return fig.add_subplot(outer_spec, sharex=sharex)
            inner_spec = outer_spec.subgridspec(
                1,
                3,
                width_ratios=[side_width, panel_width, side_width],
                wspace=0.0,
            )
            return fig.add_subplot(inner_spec[0, 1], sharex=sharex)

        _validate_stacked_display_settings()
        maximum_width = max(float(value) for value in width)
        fig = plt.figure(
            figsize=(
                maximum_width,
                sum(float(value) for value in height),
            )
        )
        outer_grid = fig.add_gridspec(
            2,
            1,
            height_ratios=height,
            hspace=panel_gap,
        )
        stacked_top_ax = _add_centered_stacked_axis(
            fig,
            outer_grid[0],
            float(width[0]),
            maximum_width,
        )
        stacked_bottom_ax = _add_centered_stacked_axis(
            fig,
            outer_grid[1],
            float(width[1]),
            maximum_width,
            sharex=stacked_top_ax,
        )
        top_x, _ = _stacked_cornucopia_panel(stacked_top_ax)
        bottom_x, _ = _stacked_family_comparison_panel(stacked_bottom_ax)

        shared_x = [*top_x, *bottom_x]
        clean_shared_x = [
            float(value) for value in shared_x if value is not None and float(value) > 0
        ]
        if clean_shared_x:
            shared_xlim = (
                min(clean_shared_x) / 1.12,
                max(clean_shared_x) * 1.12,
            )
            stacked_top_ax.set_xlim(*shared_xlim)
            stacked_bottom_ax.set_xlim(*shared_xlim)

            break_even_x = np.geomspace(
                float(break_even_x_limits[0]),
                float(break_even_x_limits[1]),
                300,
            )
            stacked_top_ax.plot(
                break_even_x,
                break_even_x,
                color=break_even_style["color"],
                linestyle=break_even_style["linestyle"],
                linewidth=float(break_even_style["linewidth"]),
                label="break-even",
                zorder=1,
            )
            stacked_top_ax.legend(
                frameon=False,
                fontsize=legend_font_size[0],
                loc="lower right",
            )

        for panel_index, (panel_ax, panel_label) in enumerate(
            [
                (stacked_top_ax, "a"),
                (stacked_bottom_ax, "b"),
            ]
        ):
            panel_ax.text(
                float(subplot_label_x[panel_index]),
                float(subplot_label_y[panel_index]),
                panel_label,
                transform=panel_ax.transAxes,
                fontsize=subplot_label_size[panel_index],
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
                zorder=10,
            )

        stacked_top_ax.tick_params(
            axis="x",
            which="both",
            labelbottom=True,
        )
        fig.align_ylabels([stacked_top_ax, stacked_bottom_ax])
        fig.subplots_adjust(
            left=float(figure_margins["left"]),
            right=float(figure_margins["right"]),
            bottom=float(figure_margins["bottom"]),
            top=float(figure_margins["top"]),
            hspace=panel_gap,
        )
        return fig


def plot_error_rates(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    display_settings: dict | None = None,
) -> Path:
    """Write only stacked_error_rate_fits.pdf from the final-figure tables."""
    output_dir = (
        Path("figures/logical_error_rates/results")
        if output_dir is None
        else Path(output_dir)
    )
    settings = deepcopy(DEFAULT_DISPLAY_SETTINGS)
    if display_settings is not None:
        settings.update(deepcopy(display_settings))
    fig = build_stacked_figure(data_dir, display_settings)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        margins = settings["pdf_margin_inches"]
        with plt.rc_context(FONT_STYLE):
            fig.canvas.draw()
            bbox = fig.get_tightbbox(fig.canvas.get_renderer())
            pdf_bbox = Bbox.from_extents(
                bbox.x0 - float(margins["left"]),
                bbox.y0 - float(margins["bottom"]),
                bbox.x1 + float(margins["right"]),
                bbox.y1 + float(margins["top"]),
            )
            path = output_dir / OUTPUT_FILENAME
            fig.savefig(path, bbox_inches=pdf_bbox, pad_inches=0)
    finally:
        plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/logical_error_rates/results")
    )
    args = parser.parse_args()
    print(plot_error_rates(args.data_dir, args.output_dir))


if __name__ == "__main__":
    main()
