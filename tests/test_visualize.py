from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from wahlumfragen.data import generate_sample_polls
from wahlumfragen.model import simulate_election
from wahlumfragen.visualize import create_all_figures, save_all_figures


EXPECTED_FIGURES = {
    "vote_share_forecast",
    "threshold_probabilities",
    "coalition_probabilities",
    "eligible_vote_share_distribution",
}


def _sample_rows():
    """Return synthetic poll rows as dictionaries."""
    return [poll.as_csv_row() for poll in generate_sample_polls()]


def _sample_result():
    """Return a compact simulation result for visualization smoke tests."""
    return simulate_election(_sample_rows(), n_draws=300, seed=123, save_result_path=None)


def test_create_all_figures_returns_expected_figure_names():
    """Test that all Streamlit-ready figure builders produce Matplotlib figures."""
    figures = create_all_figures(_sample_result())

    assert set(figures) == EXPECTED_FIGURES
    assert all(isinstance(figure, Figure) for figure in figures.values())

    for figure in figures.values():
        plt.close(figure)


def test_save_all_figures_writes_non_empty_pngs(tmp_path):
    """Test that the visualization export writes the expected figure files."""
    saved_paths = save_all_figures(_sample_result(), output_dir=tmp_path, formats=("png",), dpi=80)

    assert {path.stem for path in saved_paths} == EXPECTED_FIGURES
    assert all(path.suffix == ".png" for path in saved_paths)
    assert all(path.exists() and path.stat().st_size > 0 for path in saved_paths)
