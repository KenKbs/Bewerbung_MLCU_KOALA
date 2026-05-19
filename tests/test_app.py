import sys

from wahlumfragen.app import (
    FIGURE_SPECS,
    build_dashboard_metrics,
    get_static_figure_paths,
    load_poll_rows,
    run_simulation,
)


def test_app_helpers_import_without_streamlit():
    """Test that app helpers do not require importing Streamlit."""
    assert "streamlit" not in sys.modules


def test_static_figure_paths_match_dashboard_specs():
    """Test that every dashboard figure points at an existing static PNG."""
    paths = get_static_figure_paths()

    assert set(paths) == {spec.name for spec in FIGURE_SPECS}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def test_session_simulation_builds_dashboard_metrics():
    """Test that a small session-only simulation supports dashboard metrics."""
    poll_rows = load_poll_rows()
    result = run_simulation(500)
    metrics = build_dashboard_metrics(result, poll_rows)

    assert metrics.poll_count == len(poll_rows)
    assert metrics.leading_party
    assert metrics.leading_vote_share > 0
    assert 0 <= metrics.threshold_probability <= 1
    assert 0 <= metrics.top_coalition_probability <= 1
