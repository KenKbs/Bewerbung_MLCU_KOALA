# Wahlumfragen Prototype

This repository is a compact election-polling prototype built as supporting material for the MCML/KOALA HiWi
application. It is intentionally small: the goal is to demonstrate a working end-to-end workflow for polling data,
uncertainty modelling, Monte Carlo simulation, threshold handling, coalition probabilities, and a reviewer-friendly
Streamlit dashboard.

The project does not try to be a production election forecast. The included polls are synthetic, the model is
simplified, and the dashboard is meant to make the modelling choices inspectable rather than to publish real political
predictions.

## What This Repo Contains

- Synthetic German Bundestag polling data in `data/sample.csv`.
- A recency- and sample-size-weighted polling average.
- A simple latent-normal uncertainty model with a fixed covariance matrix and log-softmax vote-share draws.
- A Monte Carlo simulation with 50,000 saved baseline draws in `data/simulation_result.pkl`.
- A simplified 5 percent threshold step and selected coalition majority probabilities.
- Static Matplotlib figures in `reports/figures/`.
- A Streamlit dashboard in `streamlit_app.py` that can show the saved baseline result or run a fresh session-only
  simulation.
- Tests for the data loader, model, visualization exports, and app helper layer.

## Quick Start With uv

Prerequisites:

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

From a fresh clone:

```bash
cd Bewerbung_Wahlumfragen
uv sync --locked
uv run invoke app
```

Open the dashboard at [http://localhost:8501](http://localhost:8501).

The repository already includes the sample CSV, the saved baseline simulation, and the exported PNG figures, so the
dashboard should work immediately after `uv sync --locked`.

## Quick Start With Docker

Prerequisites:

- Docker

Build and run the Streamlit app image:

```bash
cd Bewerbung_Wahlumfragen
uv run invoke docker-build
docker run --rm -p 8501:8501 wahlumfragen-app
```
If uv run does not work:

```bash
docker build -f docker/app.Dockerfile -t wahlumfragen-app .
docker run --rm -p 8501:8501 wahlumfragen-app
```

Open the dashboard at [http://localhost:8501](http://localhost:8501).
The Docker image installs only runtime dependencies and copies the app, source package, sample data, baseline result,
and static figures into the container.

## Recreate The Artifacts

The committed artifacts are enough to run the app. If you want to rebuild them locally, use the invoke tasks:

```bash
uv run invoke generate-data
uv run invoke simulate-polls --draws=50000
uv run invoke plot
```

Equivalent direct commands:

```bash
uv run src/wahlumfragen/data.py data/sample.csv
uv run src/wahlumfragen/model.py --n-draws=50000
uv run src/wahlumfragen/visualize.py
```

The Streamlit sidebar can also run a fresh simulation interactively. Those dashboard runs stay in the current
Streamlit session and do not overwrite `data/simulation_result.pkl` or the PNG files.

## Development Commands

```bash
uv run pytest tests/
uv run ruff check .
uv run ruff format .
uv run invoke --list
```

The project uses `uv.lock` for reproducible installs. For deployment-style installs without dev tools, use:

```bash
uv sync --locked --no-dev
```

## Project Structure

```txt
.
├── data/
│   ├── sample.csv                 # Synthetic poll input data
│   └── simulation_result.pkl      # Saved baseline Monte Carlo result
├── docker/
│   └── app.Dockerfile             # Streamlit runtime image
├── reports/
│   ├── Ausschreibung-HiWi(ne)-KOALA.docx.pdf
│   └── figures/                   # Exported dashboard PNGs
├── src/wahlumfragen/
│   ├── app.py                     # Dashboard helper functions
│   ├── data.py                    # Synthetic data generation and CSV loading
│   ├── model.py                   # Poll weighting, uncertainty model, simulation
│   └── visualize.py               # Matplotlib figure builders
├── tests/                         # Pytest suite
├── streamlit_app.py               # Streamlit UI entry point
├── tasks.py                       # Invoke task wrappers
├── pyproject.toml                 # Package metadata and dependencies
└── uv.lock                        # Locked dependency versions
```

## Model Summary

1. Load wide-format poll rows from `data/sample.csv`.
2. Weight polls by recency and sample size:
   - recency uses a 14-day half-life;
   - sample size uses `sqrt(n)`;
   - weights are normalized to sum to one.
3. Compute a weighted average over the party vote-share columns.
4. Convert the weighted average to latent log scores.
5. Draw Monte Carlo samples from a multivariate normal distribution with a fixed covariance matrix.
6. Convert each draw back to vote shares with softmax.
7. Apply a simplified 5 percent threshold and renormalize eligible parties as seat-share proxies.
8. Estimate threshold probabilities and selected coalition majority probabilities from the simulated draws.

## Limitations

- The polling data are synthetic and hand-written for the prototype. They are not scraped, live, or real polling data.
- The uncertainty model is deliberately simple. It uses a fixed covariance matrix instead of fitted pollster, mode,
  time, or party-specific error structures.
- The threshold and seat-share logic is a proxy. It does not model constituencies, direct mandates, overhang seats,
  leveling seats, turnout, tactical voting, or official Bundestag seat allocation rules.
- Coalition probabilities are computed only for a small configured set of coalition scenarios.
- The dashboard includes a placeholder for a future raw-poll explorer, but that view is not implemented yet.
- `train.py`, `evaluate.py`, and the API scaffold are not part of the working prototype path.

## Purpose

This is best read as a small, inspectable technical sample: it shows data handling, reproducible Python packaging,
simulation code, visual output, tests, Docker packaging, and a simple Streamlit presentation layer. The modelling choices
are intentionally transparent so reviewers can see what was implemented within the limited scope.
