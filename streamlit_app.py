from __future__ import annotations

import html

import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from wahlumfragen.app import (
    DEFAULT_DRAWS,
    DRAW_STEP,
    FIGURE_SPECS,
    MAX_DRAWS,
    MIN_DRAWS,
    DashboardMetrics,
    build_dashboard_metrics,
    create_dashboard_figures,
    format_integer,
    format_percent,
    get_static_figure_paths,
    load_baseline_result,
    load_poll_rows,
    missing_static_figures,
    run_simulation,
)
from wahlumfragen.model import SimulationResult


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="Wahlumfragen Forecast", layout="wide")
    _inject_css()

    poll_rows = load_poll_rows()
    baseline_result = load_baseline_result()

    _render_sidebar()

    active_result = st.session_state.get("simulation_result", baseline_result)
    active_draws = int(st.session_state.get("simulation_draws", DEFAULT_DRAWS))
    source_label = "Session simulation" if "simulation_result" in st.session_state else "Baseline simulation"
    metrics = build_dashboard_metrics(active_result, poll_rows)

    _render_header(source_label=source_label, active_draws=active_draws)
    _render_metrics(metrics)
    _render_chart_tabs(active_result=active_result, baseline_result=baseline_result)


def _render_sidebar() -> None:
    """Render sidebar controls for session-only simulation runs."""
    with st.sidebar:
        st.markdown("## Simulation")
        st.caption("Run a fresh Monte Carlo forecast while keeping the saved figures unchanged.")
        n_draws = st.number_input(
            "Monte Carlo samples",
            min_value=MIN_DRAWS,
            max_value=MAX_DRAWS,
            value=DEFAULT_DRAWS,
            step=DRAW_STEP,
        )

        if st.button("Run new simulation", type="primary", width="stretch"):
            with st.spinner(f"Running {format_integer(int(n_draws))} draws"):
                st.session_state["simulation_result"] = run_simulation(int(n_draws))
                st.session_state["simulation_draws"] = int(n_draws)
            st.success("Simulation complete.")

        if "simulation_result" in st.session_state:
            if st.button("Show baseline again", width="stretch"):
                st.session_state.pop("simulation_result", None)
                st.session_state.pop("simulation_draws", None)
                st.rerun()

        st.markdown("---")
        st.markdown("### Model")
        st.caption("Recency-weighted polls, latent normal uncertainty, softmax vote shares, 5 percent threshold.")


def _render_header(source_label: str, active_draws: int) -> None:
    """Render the dashboard header."""
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">Bundestag polling prototype</div>
            <h1>Monte Carlo election forecast</h1>
            <p>
                A compact reviewer dashboard for the synthetic Wahlumfragen prototype:
                weighted polling averages, threshold risk, eligible seat-share distributions,
                and coalition majority paths.
            </p>
            <div class="status-row">
                <span class="status-pill">{html.escape(source_label)}</span>
                <span class="status-pill">{html.escape(format_integer(active_draws))} draws</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(metrics: DashboardMetrics) -> None:
    """Render compact headline metric tiles."""
    columns = st.columns(4)
    _metric_card(
        columns[0],
        label="Leading party",
        value=metrics.leading_party,
        caption=f"{format_percent(metrics.leading_vote_share)} weighted average",
        accent="#111827",
    )
    _metric_card(
        columns[1],
        label="Threshold watch",
        value=metrics.threshold_party,
        caption=f"{format_percent(metrics.threshold_probability, decimals=0)} chance above 5 percent",
        accent="#7C3AED",
    )
    _metric_card(
        columns[2],
        label="Top coalition path",
        value=metrics.top_coalition,
        caption=f"{format_percent(metrics.top_coalition_probability, decimals=0)} simulated majority chance",
        accent="#15803D",
    )
    _metric_card(
        columns[3],
        label="Poll base",
        value=f"{metrics.poll_count} polls",
        caption=f"{format_integer(metrics.total_sample_size)} respondents, latest {metrics.latest_poll_date}",
        accent="#2563EB",
    )


def _metric_card(column, label: str, value: str, caption: str, accent: str) -> None:
    """Render one HTML metric card in a Streamlit column."""
    column.markdown(
        f"""
        <article class="metric-card" style="--accent: {html.escape(accent)};">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _render_chart_tabs(active_result: SimulationResult, baseline_result: SimulationResult) -> None:
    """Render the four forecast charts and the future raw-polls placeholder."""
    static_paths = get_static_figure_paths()
    missing_paths = missing_static_figures(static_paths)
    use_generated_figures = "simulation_result" in st.session_state or bool(missing_paths)
    figures = create_dashboard_figures(active_result if use_generated_figures else baseline_result) if use_generated_figures else {}

    if missing_paths and "simulation_result" not in st.session_state:
        missing_names = ", ".join(path.name for path in missing_paths)
        st.warning(f"Static PNGs missing ({missing_names}); rendering from the saved simulation result instead.")

    tabs = st.tabs([spec.tab_label for spec in FIGURE_SPECS] + ["Raw Polls Soon"])
    for tab, spec in zip(tabs[: len(FIGURE_SPECS)], FIGURE_SPECS, strict=True):
        with tab:
            st.markdown(f"<p class='chart-caption'>{html.escape(spec.caption)}</p>", unsafe_allow_html=True)
            if figures:
                _render_matplotlib_figure(figures[spec.name])
            else:
                st.image(str(static_paths[spec.name]), width="stretch")

    with tabs[-1]:
        _render_raw_polls_placeholder()

    for figure in figures.values():
        plt.close(figure)


def _render_matplotlib_figure(figure: Figure) -> None:
    """Render one Matplotlib figure in Streamlit."""
    st.pyplot(figure, width="stretch", clear_figure=False)


def _render_raw_polls_placeholder() -> None:
    """Render the planned raw-polls view placeholder."""
    st.markdown(
        """
        <section class="placeholder">
            <div class="eyebrow">Coming next</div>
            <h2>Pollster-level raw poll explorer</h2>
            <p>
                This space is reserved for institute filters, latest-poll snapshots,
                sample-size and method summaries, and compact party bar charts.
            </p>
            <div class="placeholder-tags">
                <span>Institute filter</span>
                <span>Recent polls</span>
                <span>Sample size</span>
                <span>Mode summary</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _inject_css() -> None:
    """Inject the dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748B;
            --line: #E2E8F0;
            --panel: #FFFFFF;
            --soft: #F8FAFC;
            --accent: #0F766E;
        }

        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 46%, #F8FAFC 100%);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #E5E7EB;
        }

        [data-testid="stSidebar"] input {
            color: #111827;
            background: #FFFFFF;
        }

        div.stButton > button {
            border-radius: 6px;
            border: 1px solid #0F766E;
            font-weight: 700;
        }

        .hero {
            padding: 1.15rem 0 0.95rem;
            margin-bottom: 1.1rem;
            border-bottom: 1px solid rgba(100, 116, 139, 0.22);
        }

        .eyebrow {
            color: #0F766E;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .hero h1 {
            color: var(--ink);
            font-size: 3rem;
            line-height: 1.04;
            letter-spacing: 0;
            margin: 0.22rem 0 0.45rem;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.62;
            max-width: 860px;
            margin: 0;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.95rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.35rem 0.7rem;
            border-radius: 6px;
            border: 1px solid #BAE6FD;
            background: #F0F9FF;
            color: #0369A1;
            font-size: 0.86rem;
            font-weight: 800;
        }

        .metric-card {
            min-height: 8.1rem;
            padding: 0.95rem 1rem;
            border: 1px solid var(--line);
            border-top: 4px solid var(--accent);
            border-radius: 8px;
            background: var(--panel);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 850;
            letter-spacing: 0;
            line-height: 1.15;
            margin-top: 0.52rem;
            overflow-wrap: anywhere;
        }

        .metric-caption {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.38;
            margin-top: 0.45rem;
        }

        .chart-caption {
            color: var(--muted);
            font-size: 0.98rem;
            margin: 0.35rem 0 0.75rem;
        }

        .placeholder {
            border: 1px dashed #CBD5E1;
            border-radius: 8px;
            background: #FFFFFF;
            padding: 2rem;
            margin-top: 0.6rem;
        }

        .placeholder h2 {
            color: var(--ink);
            font-size: 1.65rem;
            letter-spacing: 0;
            margin: 0.25rem 0 0.55rem;
        }

        .placeholder p {
            color: var(--muted);
            max-width: 760px;
            line-height: 1.58;
        }

        .placeholder-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .placeholder-tags span {
            border: 1px solid #D9E2EC;
            border-radius: 6px;
            background: #F8FAFC;
            color: #334155;
            font-weight: 750;
            padding: 0.4rem 0.65rem;
        }

        button[data-baseweb="tab"] {
            font-weight: 800;
            letter-spacing: 0;
        }

        @media (max-width: 760px) {
            .hero h1 {
                font-size: 2.2rem;
            }

            .metric-card {
                min-height: 7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
