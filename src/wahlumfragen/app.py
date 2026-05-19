from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from wahlumfragen.data import load_poll_csv
from wahlumfragen.model import SimulationResult, load_simulation_result, simulate_election
from wahlumfragen.visualize import PARTY_LABELS, create_all_figures

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLL_PATH = PROJECT_ROOT / "data" / "sample.csv"
DEFAULT_RESULT_PATH = PROJECT_ROOT / "data" / "simulation_result.pkl"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

DEFAULT_DRAWS = 50_000
MIN_DRAWS = 500
MAX_DRAWS = 200_000
DRAW_STEP = 500


@dataclass(frozen=True)
class FigureSpec:
    """Display metadata for one dashboard figure."""

    name: str
    tab_label: str
    caption: str


@dataclass(frozen=True)
class DashboardMetrics:
    """Compact headline metrics for the dashboard."""

    leading_party: str
    leading_vote_share: float
    threshold_party: str
    threshold_probability: float
    top_coalition: str
    top_coalition_probability: float
    poll_count: int
    latest_poll_date: str
    total_sample_size: int


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        name="vote_share_forecast",
        tab_label="Forecast",
        caption="Weighted polling average with central Monte Carlo intervals.",
    ),
    FigureSpec(
        name="eligible_vote_share_distribution",
        tab_label="Distribution",
        caption="Simulated eligible vote shares after applying the simplified threshold rule.",
    ),
    FigureSpec(
        name="threshold_probabilities",
        tab_label="Thresholds",
        caption="Estimated chance that each party clears the 5 percent electoral threshold.",
    ),
    FigureSpec(
        name="coalition_probabilities",
        tab_label="Coalitions",
        caption="Majority probabilities for selected parliamentary coalition scenarios.",
    ),
)


def load_poll_rows(data_path: Path | str = DEFAULT_POLL_PATH) -> list[dict]:
    """Load the local poll CSV rows used by the dashboard."""
    rows = load_poll_csv(data_path)
    if not rows:
        msg = f"No poll rows found at {Path(data_path)}"
        raise FileNotFoundError(msg)
    return rows


def load_baseline_result(result_path: Path | str = DEFAULT_RESULT_PATH) -> SimulationResult:
    """Load the saved baseline simulation result."""
    return load_simulation_result(result_path)


def get_static_figure_paths(figure_dir: Path | str = DEFAULT_FIGURE_DIR) -> dict[str, Path]:
    """Return expected static PNG paths keyed by dashboard figure name."""
    output_dir = Path(figure_dir)
    return {spec.name: output_dir / f"{spec.name}.png" for spec in FIGURE_SPECS}


def missing_static_figures(paths: Mapping[str, Path] | None = None) -> list[Path]:
    """Return static figure paths that are not available on disk."""
    figure_paths = get_static_figure_paths() if paths is None else paths
    return [path for path in figure_paths.values() if not path.exists()]


def validate_draw_count(n_draws: int) -> int:
    """Validate and normalize a Streamlit-provided draw count."""
    draw_count = int(n_draws)
    if draw_count < MIN_DRAWS or draw_count > MAX_DRAWS:
        msg = f"n_draws must be between {MIN_DRAWS} and {MAX_DRAWS}"
        raise ValueError(msg)
    return draw_count


def run_simulation(
    n_draws: int,
    data_path: Path | str = DEFAULT_POLL_PATH,
    seed: int = 42,
) -> SimulationResult:
    """Run a session-only simulation without overwriting saved artifacts."""
    draw_count = validate_draw_count(n_draws)
    rows = load_poll_rows(data_path)
    return simulate_election(rows, n_draws=draw_count, seed=seed, save_result_path=None)


def create_dashboard_figures(result: SimulationResult) -> dict[str, Figure]:
    """Create Matplotlib figures in dashboard order."""
    figures = create_all_figures(result)
    return {spec.name: figures[spec.name] for spec in FIGURE_SPECS}


def build_dashboard_metrics(
    result: SimulationResult,
    poll_rows: Sequence[Mapping[str, object]],
) -> DashboardMetrics:
    """Build headline metrics from a simulation result and its poll rows."""
    polls = pd.DataFrame(poll_rows)
    latest_poll_date = pd.to_datetime(polls["published_at"], errors="raise").max().strftime("%d %b %Y")
    total_sample_size = int(pd.to_numeric(polls["sample_size"], errors="raise").sum())

    leading_party = str(result.weighted_average.idxmax())
    threshold_party = str((result.threshold_probabilities - 0.5).abs().idxmin())
    top_coalition = str(result.coalition_probabilities.idxmax())

    return DashboardMetrics(
        leading_party=party_label(leading_party),
        leading_vote_share=float(result.weighted_average[leading_party]),
        threshold_party=party_label(threshold_party),
        threshold_probability=float(result.threshold_probabilities[threshold_party]),
        top_coalition=top_coalition,
        top_coalition_probability=float(result.coalition_probabilities[top_coalition]),
        poll_count=len(polls),
        latest_poll_date=latest_poll_date,
        total_sample_size=total_sample_size,
    )


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a proportion as a human-readable percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_integer(value: int) -> str:
    """Format an integer with thousands separators."""
    return f"{value:,}"


def party_label(party: str) -> str:
    """Return the display label for one party key."""
    return PARTY_LABELS.get(party, party.replace("_", " ").title())
