from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import typer

from wahlumfragen.model import SimulationResult, load_simulation_result

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION_RESULT_PATH = PROJECT_ROOT / "data" / "simulation_result.pkl"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

BACKGROUND = "#F8FAFC"
PANEL = "#FFFFFF"
INK = "#111827"
MUTED = "#64748B"
GRID = "#E2E8F0"
REFERENCE = "#94A3B8"

PARTY_LABELS: Mapping[str, str] = {
    "cdu_csu": "CDU/CSU",
    "spd": "SPD",
    "gruene": "Gruene",
    "fdp": "FDP",
    "linke": "Linke",
    "afd": "AfD",
    "bsw": "BSW",
    "sonstige": "Sonstige",
}

PARTY_COLORS: Mapping[str, str] = {
    "cdu_csu": "#111827",
    "spd": "#E11D48",
    "gruene": "#16A34A",
    "fdp": "#EAB308",
    "linke": "#BE185D",
    "afd": "#2563EB",
    "bsw": "#7C3AED",
    "sonstige": "#6B7280",
}

FIGURE_BUILDERS: Mapping[str, Callable[[SimulationResult], Figure]]


def create_all_figures(result: SimulationResult) -> dict[str, Figure]:
    """Create all presentation figures from an already loaded simulation result."""
    return {name: builder(result) for name, builder in FIGURE_BUILDERS.items()}


def save_all_figures(
    result: SimulationResult,
    output_dir: Path | str = DEFAULT_FIGURE_DIR,
    formats: Sequence[str] = ("png",),
    dpi: int = 180,
) -> list[Path]:
    """Save all presentation figures and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    figures = create_all_figures(result)
    for stem, figure in figures.items():
        for file_format in formats:
            normalized_format = file_format.removeprefix(".").lower()
            path = output_path / f"{stem}.{normalized_format}"
            figure.savefig(
                path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
                format=normalized_format,
            )
            saved_paths.append(path)
        plt.close(figure)
    return saved_paths


def plot_vote_share_forecast(result: SimulationResult) -> Figure:
    """Plot weighted vote shares with 95 percent simulation intervals."""
    _set_theme()
    table = result.uncertainty_intervals.copy()
    table["weighted_average"] = result.weighted_average
    table = table.sort_values("weighted_average")

    figure, axis = _figure(figsize=(10, 6.2))
    _add_header(
        figure,
        "Vote-share forecast",
        "Weighted polling average with 95 percent Monte Carlo uncertainty intervals",
    )

    y_positions = np.arange(len(table))
    x_max = max(40.0, float(table["upper_95"].max() * 100) + 6.0)

    for y_position, (party, row) in zip(y_positions, table.iterrows(), strict=True):
        color = PARTY_COLORS.get(party, MUTED)
        lower = float(row["lower_95"] * 100)
        upper = float(row["upper_95"] * 100)
        average = float(row["weighted_average"] * 100)

        axis.plot(
            [lower, upper],
            [y_position, y_position],
            color=color,
            alpha=0.22,
            linewidth=11,
            solid_capstyle="round",
        )
        axis.plot(
            [lower, upper],
            [y_position, y_position],
            color=color,
            alpha=0.78,
            linewidth=2.4,
            solid_capstyle="round",
        )
        axis.scatter(average, y_position, s=88, color=color, edgecolor=PANEL, linewidth=1.8, zorder=3)
        axis.text(x_max - 0.4, y_position, f"{average:.1f}%", ha="right", va="center", fontsize=10, color=INK)

    axis.axvline(5, color=REFERENCE, linewidth=1.2, linestyle=(0, (4, 4)))
    axis.text(5.25, len(table) - 0.35, "5 percent threshold", color=MUTED, fontsize=9, ha="left", va="center")

    axis.set_xlim(0, x_max)
    axis.set_yticks(y_positions, [_party_label(party) for party in table.index])
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_xlabel("Vote share", color=MUTED, labelpad=10)
    _style_axis(axis)

    legend = [
        Line2D([0], [0], color=MUTED, linewidth=6, alpha=0.25, label="95 percent interval"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=INK,
            markeredgecolor=PANEL,
            markersize=9,
            label="Weighted average",
        ),
    ]
    axis.legend(handles=legend, loc="lower right", frameon=False, labelcolor=MUTED)
    figure.subplots_adjust(left=0.18, right=0.96, top=0.78, bottom=0.14)
    return figure


def plot_threshold_probabilities(result: SimulationResult) -> Figure:
    """Plot the probability that each party clears the electoral threshold."""
    _set_theme()
    probabilities = result.threshold_probabilities.sort_values()

    figure, axis = _figure(figsize=(9.5, 5.4))
    _add_header(
        figure,
        "Threshold risk",
        "Probability that each party clears the 5 percent threshold",
    )

    y_positions = np.arange(len(probabilities))
    values = probabilities.to_numpy(dtype=float) * 100
    colors = [PARTY_COLORS.get(party, MUTED) for party in probabilities.index]
    axis.barh(y_positions, values, height=0.58, color=colors, alpha=0.9)

    for y_position, value in zip(y_positions, values, strict=True):
        if value >= 88:
            axis.text(
                value - 2.0,
                y_position,
                f"{value:.0f}%",
                ha="right",
                va="center",
                color=PANEL,
                fontsize=10,
                fontweight="bold",
            )
        else:
            axis.text(
                value + 1.4,
                y_position,
                f"{value:.0f}%",
                ha="left",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold",
            )

    axis.axvline(50, color=REFERENCE, linewidth=1.1, linestyle=(0, (4, 4)))
    axis.text(51, len(probabilities) - 0.35, "even odds (50% probability)", color=MUTED, fontsize=9, ha="left", va="center")

    axis.set_xlim(0, 100)
    axis.set_yticks(y_positions, [_party_label(party) for party in probabilities.index])
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_xlabel("Probability", color=MUTED, labelpad=10)
    _style_axis(axis)
    figure.subplots_adjust(left=0.18, right=0.94, top=0.76, bottom=0.15)
    return figure


def plot_coalition_probabilities(result: SimulationResult) -> Figure:
    """Plot majority probabilities for the configured coalition scenarios."""
    _set_theme()
    probabilities = result.coalition_probabilities.sort_values()

    figure, axis = _figure(figsize=(10.5, 5.8))
    _add_header(
        figure,
        "Coalition pathways",
        "Probability of a parliamentary majority in the simulated seat shares",
    )

    y_positions = np.arange(len(probabilities))
    values = probabilities.to_numpy(dtype=float) * 100
    colors = [_coalition_color(value) for value in values]
    axis.barh(y_positions, values, height=0.58, color=colors, alpha=0.92)

    for y_position, value in zip(y_positions, values, strict=True):
        if value >= 84:
            axis.text(
                value - 2.0,
                y_position,
                f"{value:.0f}%",
                ha="right",
                va="center",
                color=PANEL,
                fontsize=10,
                fontweight="bold",
            )
        else:
            axis.text(
                value + 1.3,
                y_position,
                f"{value:.0f}%",
                ha="left",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold",
            )

    axis.axvline(50, color=REFERENCE, linewidth=1.2, linestyle=(0, (4, 4)))
    axis.text(51, len(probabilities) - 0.35, "majority threshold", color=MUTED, fontsize=9, ha="left", va="center")

    axis.set_xlim(0, 100)
    axis.set_yticks(y_positions, probabilities.index)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_xlabel("even odds (50% probability)", color=MUTED, labelpad=10)
    _style_axis(axis)
    figure.subplots_adjust(left=0.30, right=0.95, top=0.77, bottom=0.15)
    return figure


def plot_eligible_vote_share_distribution(result: SimulationResult) -> Figure:
    """Plot simulated eligible vote-share distributions after threshold handling."""
    _set_theme()
    seat_shares = result.simulated_seats.copy()
    visible_parties = [
        party
        for party in seat_shares.columns
        if float(seat_shares[party].quantile(0.975)) > 0
    ]
    medians = seat_shares[visible_parties].median().sort_values()
    ordered_parties = list(medians.index)
    samples = [seat_shares[party].to_numpy(dtype=float) * 100 for party in ordered_parties]

    figure, axis = _figure(figsize=(10, 6.2))
    _add_header(
        figure,
        "Eligible vote-share distribution",
        "Simulation draws after applying the simplified threshold rule",
    )

    y_positions = np.arange(len(ordered_parties))
    violin_parts = axis.violinplot(
        samples,
        positions=y_positions,
        orientation="horizontal",
        widths=0.82,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body, party in zip(violin_parts["bodies"], ordered_parties, strict=True):
        color = PARTY_COLORS.get(party, MUTED)
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.18)

    for y_position, party, party_samples in zip(y_positions, ordered_parties, samples, strict=True):
        color = PARTY_COLORS.get(party, MUTED)
        lower, median, upper = np.quantile(party_samples, [0.025, 0.5, 0.975])
        axis.plot(
            [lower, upper],
            [y_position, y_position],
            color=color,
            linewidth=2.4,
            alpha=0.72,
            solid_capstyle="round",
        )
        axis.scatter(median, y_position, s=74, color=color, edgecolor=PANEL, linewidth=1.6, zorder=3)
        axis.text(median + 0.9, y_position + 0.18, f"{median:.1f}%", fontsize=9, color=INK, ha="left", va="bottom")

    x_max = max(55.0, max(np.quantile(party_samples, 0.995) for party_samples in samples) + 5.0)
    axis.axvline(50, color=REFERENCE, linewidth=1.15, linestyle=(0, (4, 4)))
    axis.text(50.7, len(ordered_parties) - 0.35, "single-party majority", color=MUTED, fontsize=9, ha="left", va="center")

    axis.set_xlim(0, x_max)
    axis.set_yticks(y_positions, [_party_label(party) for party in ordered_parties])
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.set_xlabel("Eligible vote share", color=MUTED, labelpad=10)
    _style_axis(axis)
    figure.subplots_adjust(left=0.18, right=0.96, top=0.78, bottom=0.14)
    return figure


# HELPER FUNCTIONS
def _set_theme() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "figure.facecolor": BACKGROUND,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "savefig.facecolor": BACKGROUND,
            "svg.fonttype": "none",
            "xtick.color": MUTED,
            "ytick.color": INK,
        }
    )


def _figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(PANEL)
    return figure, axis


def _add_header(figure: Figure, title: str, subtitle: str) -> None:
    figure.text(0.08, 0.95, title, fontsize=20, fontweight="bold", color=INK, ha="left", va="top")
    figure.text(0.08, 0.905, subtitle, fontsize=10.5, color=MUTED, ha="left", va="top")


def _style_axis(axis: Axes) -> None:
    axis.grid(axis="x", color=GRID, linewidth=0.9)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def _party_label(party: str) -> str:
    return PARTY_LABELS.get(party, party.replace("_", " ").title())


def _coalition_color(value: float) -> str:
    if value >= 50:
        return "#15803D"
    if value >= 20:
        return "#D97706"
    return "#64748B"


FIGURE_BUILDERS = {
    "vote_share_forecast": plot_vote_share_forecast,
    "threshold_probabilities": plot_threshold_probabilities,
    "coalition_probabilities": plot_coalition_probabilities,
    "eligible_vote_share_distribution": plot_eligible_vote_share_distribution,
}


def main(
    result_path: Path = DEFAULT_SIMULATION_RESULT_PATH,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    dpi: int = 180,
    save_svg: bool = False,
) -> None:
    """Load a saved simulation result and write presentation figures."""
    result = load_simulation_result(result_path)
    formats = ("png", "svg") if save_svg else ("png",)
    saved_paths = save_all_figures(result, output_dir=output_dir, formats=formats, dpi=dpi)

    print(f"Saved {len(saved_paths)} figure file(s):")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    typer.run(main)
