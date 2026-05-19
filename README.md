# Wahlumfragen Prototype

Small Python/Streamlit prototype for election-poll aggregation, Monte Carlo simulation, threshold handling, and coalition probability visualization.

The project demonstrates an end-to-end workflow:

- load synthetic Bundestag polling data
- compute a recency- and sample-size-weighted polling average
- simulate plausible vote-share outcomes 
- apply a simplified 5% threshold
- estimate threshold and coalition probabilities
- visualize the results in a Streamlit dashboard

This is a technical prototype, not a production election forecast.

## Quick Start With uv

Prerequisites:

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

```bash
cd Bewerbung_Wahlumfragen
uv sync --locked
uv run invoke app
```

Open the dashboard at:

```txt
http://localhost:8501
```

The repository includes the sample data, a saved baseline simulation, and exported PNG figures.

## Quick Start With Docker

Prerequisite:

- Docker

```bash
cd Bewerbung_Wahlumfragen
uv run invoke docker-build
docker run --rm -p 8501:8501 wahlumfragen-app
```

Without `uv`:

```bash
docker build -f docker/app.Dockerfile -t wahlumfragen-app .
docker run --rm -p 8501:8501 wahlumfragen-app
```

Open:

```txt
http://localhost:8501
```

## Recreate Artifacts
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

## Model
The polls are weighted by recency and sample size using a simple weighting scheme. The resulting weighted polling average is transformed into latent log-support scores and used as the center of a latent multivariate normal distribution. A fixed covariance matrix is defined in [`src/wahlumfragen/model.py`](src/wahlumfragen/model.py). Simulated latent scores are then transformed back with softmax to obtain valid vote-share probabilities that are non-negative and sum to 100%.

Coalition probabilities are estimated by applying a simplified 5% threshold, renormalizing eligible parties, and checking whether configured coalitions exceed 50%.

## Limitations
- Endless limitations... As it's only a small prototype.
- Polling data are synthetic and included only for demonstration.
- The uncertainty model uses a fixed covariance structure rather than fitted historical polling errors.
- Coalition majorities use a simplified eligible-vote-share proxy, not official Bundestag seat allocation.
- Direct mandates, overhang/leveling seats, turnout, tactical voting, and district-level effects are not modeled.
- No database is setup for better handling a lot of poll data etc. etc.
