#!/usr/bin/env python3
"""Create Fig. 3: physical overhead and Cornucopia rearrangement time."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.transforms import Bbox

DEFAULT_OUTPUT_DIR = Path("figures/overhead/results")
FAMILIES = ("Cornucopia", "Surface", "BB")


@dataclass(frozen=True)
class CodePoint:
    family: str
    distance: int
    data_qubits: int
    check_qubits: int
    logical_qubits: int

    @property
    def total_physical_qubits(self) -> int:
        return self.data_qubits + self.check_qubits

    @property
    def overhead_per_logical(self) -> float:
        return self.total_physical_qubits / self.logical_qubits


@dataclass(frozen=True)
class PlotConfig:
    # Physical size of [panel a, panel b] in inches.
    height: tuple[float, float] = (4.5, 4.5)
    width: tuple[float, float] = (5.4, 5.4)

    axis_label_size: tuple[float, float] = (14.0, 14.0)
    y_axis_title: tuple[str, str] = (
        "Physical overhead per logical qubit",
        "Atom-rearrangement\ntime per cycle (ms)",
    )
    tick_label_size: tuple[float, float] = (11.0, 11.0)
    legend_font_size: tuple[float, float] = (11.0, 11.0)
    subplot_label_size: tuple[float, float] = (16.0, 16.0)
    subplot_label_x: tuple[float, float] = (-0.14, -0.20)
    subplot_label_y: tuple[float, float] = (1.02, 1.02)

    grid_line_width: tuple[float, float] = (0.65, 0.65)
    grid_alpha: tuple[float, float] = (0.28, 0.28)
    x_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (5.5, 19.5),
        (5.5, 18.5),
    )
    y_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (2.5, 1000.0),
        (9.5, 17.0),
    )

    figure_margins: dict[str, float] = field(
        default_factory=lambda: {
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.16,
            "top": 0.96,
        }
    )
    pdf_margin_inches: dict[str, float] = field(
        default_factory=lambda: {
            "left": 0.02,
            "right": 0.02,
            "bottom": 0.02,
            "top": 0.02,
        }
    )
    panel_gap: float = 0.34

    marker_choice: dict[str, str] = field(
        default_factory=lambda: {
            "Cornucopia": "o",
            "Surface": "^",
            "BB": "s",
        }
    )
    panel_b_marker: str = "D"
    line_style: dict[str, str] = field(
        default_factory=lambda: {
            "Cornucopia": "-",
            "Surface": "--",
            "BB": "-.",
        }
    )
    marker_size: dict[str, float] = field(
        default_factory=lambda: {
            "Cornucopia": 7.0,
            "Surface": 7.0,
            "BB": 7.0,
        }
    )
    line_width: dict[str, float] = field(
        default_factory=lambda: {
            "Cornucopia": 2.0,
            "Surface": 2.0,
            "BB": 2.0,
        }
    )
    family_color: dict[str, str] = field(
        default_factory=lambda: {
            "Cornucopia": "#0072B2",
            "Surface": "#009E73",
            "BB": "#D55E00",
        }
    )
    legend_location: tuple[str | None, str | None] = ("upper left", None)
    marker_edge_color: str = "white"
    marker_edge_width: float = 0.8
    annotation_font_size: dict[str, float] = field(
        default_factory=lambda: {
            "Cornucopia": 10.0,
            "Surface": 10.0,
            "BB": 10.0,
        }
    )
    annotation_text_color: str = "black"
    annotation_background_color: str = "white"
    annotation_background_alpha: float = 0.9
    annotation_background_pad: float = 0.6
    annotation_distance: dict[str, int] = field(
        default_factory=lambda: {
            "Cornucopia": 18,
            "Surface": 19,
            "BB": 18,
        }
    )
    annotation_offset_points: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "Cornucopia": (0.0, 9.0),
            "Surface": (-7.0, -25.0),
            "BB": (0.0, 9.0),
        }
    )
    annotation_horizontal_alignment: dict[str, str] = field(
        default_factory=lambda: {
            "Cornucopia": "center",
            "Surface": "right",
            "BB": "center",
        }
    )
    annotation_value_format: dict[str, str] = field(
        default_factory=lambda: {
            "Cornucopia": ".3g",
            "Surface": ".3g",
            "BB": ".3g",
        }
    )


def cornucopia_points() -> tuple[CodePoint, ...]:
    # Each L=12, J=3 code has 12P data qubits and 2JP=6P checks.
    parameters = (
        (21, 130, 6),
        (48, 292, 8),
        (75, 454, 10),
        (87, 526, 12),
        (147, 886, 14),
        (192, 1156, 16),
        (237, 1426, 18),
    )
    return tuple(
        CodePoint(
            family="Cornucopia",
            distance=distance,
            data_qubits=12 * p,
            check_qubits=6 * p,
            logical_qubits=k,
        )
        for p, k, distance in parameters
    )


def bb_points() -> tuple[CodePoint, ...]:
    # BB H_X and H_Z each contain n/2 measured check rows.
    return (
        CodePoint("BB", 12, data_qubits=144, check_qubits=144, logical_qubits=12),
        CodePoint("BB", 18, data_qubits=288, check_qubits=288, logical_qubits=12),
    )


def surface_points() -> tuple[CodePoint, ...]:
    # A rotated distance-d memory uses d^2 data and d^2-1 check qubits.
    return tuple(
        CodePoint(
            family="Surface",
            distance=distance,
            data_qubits=distance * distance,
            check_qubits=distance * distance - 1,
            logical_qubits=1,
        )
        for distance in (7, 9, 11, 13, 19)
    )


def cornucopia_cycle_times_ms() -> tuple[tuple[int, float], ...]:
    # Extended Table 2, final "Cycle Total (ms)" row.
    return (
        (6, 10.41),
        (8, 11.92),
        (10, 12.87),
        (12, 13.22),
        (14, 14.68),
        (16, 15.48),
        (18, 16.18),
    )


def configure_fonts(font_dir: Path | None = None) -> str:
    """Use a bundled sans-serif font, or register explicitly supplied TTF files."""
    family = "DejaVu Sans"
    if font_dir is not None:
        font_paths = sorted(Path(font_dir).glob("*.ttf"))
        if not font_paths:
            raise FileNotFoundError("The supplied font directory contains no TTF files")
        for path in font_paths:
            font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(font_paths[0])).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.cal": family,
            "mathtext.sf": family,
            "mathtext.tt": family,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    return family


def _validate_pair(name: str, values, *, positive: bool = False) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain [panel a, panel b]")
    numeric = [float(value) for value in values]
    if positive and any(value <= 0 for value in numeric):
        raise ValueError(f"{name} values must be positive")


def validate_plot_config(config: PlotConfig) -> None:
    for name in (
        "height",
        "width",
        "axis_label_size",
        "tick_label_size",
        "legend_font_size",
        "subplot_label_size",
        "grid_line_width",
    ):
        _validate_pair(name, getattr(config, name), positive=True)
    for name in ("subplot_label_x", "subplot_label_y"):
        _validate_pair(name, getattr(config, name))
    if len(config.y_axis_title) != 2:
        raise ValueError("y_axis_title must contain [panel a, panel b]")
    if any(not str(value).strip() for value in config.y_axis_title):
        raise ValueError("y_axis_title values cannot be empty")

    _validate_pair("grid_alpha", config.grid_alpha)
    if any(not 0 <= float(value) <= 1 for value in config.grid_alpha):
        raise ValueError("grid_alpha values must be between 0 and 1")

    for name, limits in (
        ("x_limits", config.x_limits),
        ("y_limits", config.y_limits),
    ):
        if len(limits) != 2:
            raise ValueError(f"{name} must contain [panel a, panel b]")
        for panel_index, pair in enumerate(limits):
            if len(pair) != 2:
                raise ValueError(f"{name}[{panel_index}] must contain (min, max)")
            lower, upper = [float(value) for value in pair]
            if upper <= lower:
                raise ValueError(f"{name}[{panel_index}] must satisfy min < max")
            if name == "y_limits" and panel_index == 0 and lower <= 0:
                raise ValueError("panel a log-scale y minimum must be positive")

    required_margin_keys = {"left", "right", "bottom", "top"}
    for name, margins in (
        ("figure_margins", config.figure_margins),
        ("pdf_margin_inches", config.pdf_margin_inches),
    ):
        if set(margins) != required_margin_keys:
            raise ValueError(f"{name} must contain exactly {sorted(required_margin_keys)}")
        numeric = {key: float(value) for key, value in margins.items()}
        if name == "pdf_margin_inches" and any(value < 0 for value in numeric.values()):
            raise ValueError("pdf_margin_inches values must be nonnegative")

    figure_left = float(config.figure_margins["left"])
    figure_right = float(config.figure_margins["right"])
    figure_bottom = float(config.figure_margins["bottom"])
    figure_top = float(config.figure_margins["top"])
    if not 0 <= figure_left < figure_right <= 1:
        raise ValueError("figure_margins must satisfy 0 <= left < right <= 1")
    if not 0 <= figure_bottom < figure_top <= 1:
        raise ValueError("figure_margins must satisfy 0 <= bottom < top <= 1")
    if float(config.panel_gap) < 0:
        raise ValueError("panel_gap must be nonnegative")
    if float(config.marker_edge_width) < 0:
        raise ValueError("marker_edge_width must be nonnegative")
    if not str(config.panel_b_marker).strip():
        raise ValueError("panel_b_marker cannot be empty")
    if not 0 <= float(config.annotation_background_alpha) <= 1:
        raise ValueError("annotation_background_alpha must be between 0 and 1")
    if float(config.annotation_background_pad) < 0:
        raise ValueError("annotation_background_pad must be nonnegative")

    for name in (
        "marker_choice",
        "line_style",
        "marker_size",
        "line_width",
        "family_color",
        "annotation_font_size",
        "annotation_distance",
        "annotation_offset_points",
        "annotation_horizontal_alignment",
        "annotation_value_format",
    ):
        values = getattr(config, name)
        missing = set(FAMILIES) - set(values)
        if missing:
            raise ValueError(f"{name} is missing families: {sorted(missing)}")
    if any(float(config.marker_size[name]) <= 0 for name in FAMILIES):
        raise ValueError("marker_size values must be positive")
    if any(float(config.line_width[name]) <= 0 for name in FAMILIES):
        raise ValueError("line_width values must be positive")
    if any(float(config.annotation_font_size[name]) <= 0 for name in FAMILIES):
        raise ValueError("annotation_font_size values must be positive")
    if len(config.legend_location) != 2:
        raise ValueError("legend_location must contain [panel a, panel b]")
    for family in FAMILIES:
        offset = config.annotation_offset_points[family]
        if len(offset) != 2:
            raise ValueError(f"annotation_offset_points[{family!r}] must contain (x, y)")
        float(offset[0])
        float(offset[1])
        format(
            1.0,
            str(config.annotation_value_format[family]),
        )


def _plot_family(
    ax,
    points: Iterable[CodePoint],
    *,
    family: str,
    config: PlotConfig,
) -> None:
    ordered = sorted(points, key=lambda point: point.distance)
    ax.plot(
        [point.distance for point in ordered],
        [point.overhead_per_logical for point in ordered],
        label=family,
        color=config.family_color[family],
        marker=config.marker_choice[family],
        linestyle=config.line_style[family],
        linewidth=float(config.line_width[family]),
        markersize=float(config.marker_size[family]),
        markeredgecolor=config.marker_edge_color,
        markeredgewidth=float(config.marker_edge_width),
        zorder=3,
    )


def _annotate_overhead_endpoints(
    ax,
    config: PlotConfig,
) -> None:
    points_by_family = {
        "Cornucopia": cornucopia_points(),
        "Surface": surface_points(),
        "BB": bb_points(),
    }
    for family, points in points_by_family.items():
        target_distance = int(config.annotation_distance[family])
        point = next(
            (candidate for candidate in points if candidate.distance == target_distance),
            None,
        )
        if point is None:
            raise ValueError(f"No {family} point at annotation distance d={target_distance}")
        offset_x, offset_y = config.annotation_offset_points[family]
        label = format(
            point.overhead_per_logical,
            str(config.annotation_value_format[family]),
        )
        ax.annotate(
            label,
            xy=(point.distance, point.overhead_per_logical),
            xytext=(float(offset_x), float(offset_y)),
            textcoords="offset points",
            color=config.annotation_text_color,
            fontsize=float(config.annotation_font_size[family]),
            ha=config.annotation_horizontal_alignment[family],
            va="bottom",
            annotation_clip=False,
            bbox={
                "boxstyle": (f"square,pad={float(config.annotation_background_pad)}"),
                "facecolor": config.annotation_background_color,
                "edgecolor": "none",
                "alpha": float(config.annotation_background_alpha),
            },
            zorder=4,
        )


def _style_axis(ax, config: PlotConfig, panel_index: int) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=float(config.tick_label_size[panel_index]),
        width=1.0,
        length=5,
    )
    ax.tick_params(axis="both", which="minor", width=0.8, length=3)
    ax.grid(
        True,
        which="both",
        linewidth=float(config.grid_line_width[panel_index]),
        alpha=float(config.grid_alpha[panel_index]),
        color="#8C8C8C",
        linestyle=":",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _add_centered_axis(
    figure,
    outer_spec,
    panel_height: float,
    maximum_height: float,
):
    side_height = max((maximum_height - panel_height) / 2.0, 0.0)
    if side_height <= 1e-12:
        return figure.add_subplot(outer_spec)
    inner_spec = outer_spec.subgridspec(
        3,
        1,
        height_ratios=[side_height, panel_height, side_height],
        hspace=0.0,
    )
    return figure.add_subplot(inner_spec[1, 0])


def write_data_csv(path: Path) -> None:
    times = dict(cornucopia_cycle_times_ms())
    rows = list(cornucopia_points()) + list(bb_points()) + list(surface_points())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "distance",
                "data_qubits",
                "check_qubits",
                "total_physical_qubits",
                "logical_qubits",
                "physical_overhead_per_logical",
                "cycle_time_ms",
            ]
        )
        for point in rows:
            writer.writerow(
                [
                    point.family,
                    point.distance,
                    point.data_qubits,
                    point.check_qubits,
                    point.total_physical_qubits,
                    point.logical_qubits,
                    f"{point.overhead_per_logical:.12g}",
                    times.get(point.distance, "") if point.family == "Cornucopia" else "",
                ]
            )


def make_figure(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: PlotConfig | None = None,
    *,
    output_stem: str = "fig3",
) -> tuple[Path, Path]:
    config = config or PlotConfig()
    validate_plot_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts()

    maximum_height = max(float(value) for value in config.height)
    figure = plt.figure(
        figsize=(
            sum(float(value) for value in config.width),
            maximum_height,
        )
    )
    outer_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=config.width,
        wspace=float(config.panel_gap),
    )
    ax_a = _add_centered_axis(
        figure,
        outer_grid[0],
        float(config.height[0]),
        maximum_height,
    )
    ax_b = _add_centered_axis(
        figure,
        outer_grid[1],
        float(config.height[1]),
        maximum_height,
    )

    _plot_family(
        ax_a,
        cornucopia_points(),
        family="Cornucopia",
        config=config,
    )
    _plot_family(
        ax_a,
        surface_points(),
        family="Surface",
        config=config,
    )
    _plot_family(
        ax_a,
        bb_points(),
        family="BB",
        config=config,
    )
    ax_a.set_yscale("log")
    ax_a.set_xlim(*[float(value) for value in config.x_limits[0]])
    ax_a.set_ylim(*[float(value) for value in config.y_limits[0]])
    ax_a.set_xticks([6, 8, 10, 12, 14, 16, 18])
    ax_a.yaxis.set_major_locator(LogLocator(base=10))
    ax_a.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    ax_a.set_xlabel(
        r"Distance, $d$",
        fontsize=float(config.axis_label_size[0]),
    )
    ax_a.set_ylabel(
        config.y_axis_title[0],
        fontsize=float(config.axis_label_size[0]),
    )
    _annotate_overhead_endpoints(ax_a, config)
    if config.legend_location[0] is not None:
        ax_a.legend(
            loc=config.legend_location[0],
            fontsize=float(config.legend_font_size[0]),
            frameon=False,
            handlelength=2.4,
        )
    _style_axis(ax_a, config, 0)

    time_points = cornucopia_cycle_times_ms()
    family = "Cornucopia"
    ax_b.plot(
        [distance for distance, _time in time_points],
        [time for _distance, time in time_points],
        label=family,
        color=config.family_color[family],
        marker=config.panel_b_marker,
        linestyle=config.line_style[family],
        linewidth=float(config.line_width[family]),
        markersize=float(config.marker_size[family]),
        markeredgecolor=config.marker_edge_color,
        markeredgewidth=float(config.marker_edge_width),
        zorder=3,
    )
    ax_b.set_xlim(*[float(value) for value in config.x_limits[1]])
    ax_b.set_ylim(*[float(value) for value in config.y_limits[1]])
    ax_b.set_xticks([6, 8, 10, 12, 14, 16, 18])
    ax_b.set_xlabel(
        r"Distance, $d$",
        fontsize=float(config.axis_label_size[1]),
    )
    ax_b.set_ylabel(
        config.y_axis_title[1],
        fontsize=float(config.axis_label_size[1]),
    )
    if config.legend_location[1] is not None:
        ax_b.legend(
            loc=config.legend_location[1],
            fontsize=float(config.legend_font_size[1]),
            frameon=False,
            handlelength=2.4,
        )
    _style_axis(ax_b, config, 1)

    for panel_index, (label, axis) in enumerate((("a", ax_a), ("b", ax_b))):
        axis.text(
            float(config.subplot_label_x[panel_index]),
            float(config.subplot_label_y[panel_index]),
            label,
            transform=axis.transAxes,
            fontsize=float(config.subplot_label_size[panel_index]),
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
            zorder=10,
        )

    figure.subplots_adjust(
        left=float(config.figure_margins["left"]),
        right=float(config.figure_margins["right"]),
        bottom=float(config.figure_margins["bottom"]),
        top=float(config.figure_margins["top"]),
        wspace=float(config.panel_gap),
    )

    figure.canvas.draw()
    tight_bbox_inches = figure.get_tightbbox(figure.canvas.get_renderer())
    pdf_bbox_inches = Bbox.from_extents(
        tight_bbox_inches.x0 - float(config.pdf_margin_inches["left"]),
        tight_bbox_inches.y0 - float(config.pdf_margin_inches["bottom"]),
        tight_bbox_inches.x1 + float(config.pdf_margin_inches["right"]),
        tight_bbox_inches.y1 + float(config.pdf_margin_inches["top"]),
    )

    pdf_path = output_dir / f"{output_stem}.pdf"
    data_path = output_dir / f"{output_stem}_data.csv"
    figure.savefig(
        pdf_path,
        bbox_inches=pdf_bbox_inches,
        pad_inches=0,
    )
    plt.close(figure)
    write_data_csv(data_path)
    return pdf_path, data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fig3.pdf and fig3_data.csv.",
    )
    parser.add_argument("--output-stem", default="fig3")
    args = parser.parse_args()
    pdf_path, data_path = make_figure(
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    print(f"pdf={pdf_path}")
    print(f"data={data_path}")


if __name__ == "__main__":
    main()
