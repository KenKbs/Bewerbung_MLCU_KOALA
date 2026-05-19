from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import typer

from wahlumfragen.data import DEFAULT_SAMPLE_PATH, PARTY_COLUMNS, load_poll_csv

# Party order follows PARTY_COLUMNS:
# cdu_csu, spd, gruene, fdp, linke, afd, bsw, sonstige
DEFAULT_LATENT_COVARIANCE = np.array(
    [ # cdu,    spd,     grüne, fdp     linke    afd     bsw     sonstige
        [0.040, -0.002, -0.002, -0.002, -0.002, -0.002, -0.002, -0.002],  # cdu_csu
        [-0.002, 0.040, -0.002, -0.002, -0.002, -0.002, -0.002, -0.002],  # spd
        [-0.002, -0.002, 0.040, -0.002, -0.002, -0.002, -0.002, -0.002],  # gruene
        [-0.002, -0.002, -0.002, 0.040, -0.002, -0.002, -0.002, -0.002],  # fdp
        [-0.002, -0.002, -0.002, -0.002, 0.040, -0.002, -0.002, -0.002],  # linke
        [-0.002, -0.002, -0.002, -0.002, -0.002, 0.040, -0.002, -0.002],  # afd
        [-0.002, -0.002, -0.002, -0.002, -0.002, -0.002, 0.040, -0.002],  # bsw
        [-0.002, -0.002, -0.002, -0.002, -0.002, -0.002, -0.002, 0.040],  # sonstige
    ],
    dtype=float,
)

DEFAULT_COALITIONS: Mapping[str, tuple[str, ...]] = {
    "CDU/CSU + SPD": ("cdu_csu", "spd"),
    "CDU/CSU + Gruene": ("cdu_csu", "gruene"),
    "CDU/CSU + SPD + Gruene": ("cdu_csu", "spd", "gruene"),
    "SPD + Gruene + FDP": ("spd", "gruene", "fdp"),
    "SPD + Gruene + Linke": ("spd", "gruene", "linke"),
    "CDU/CSU + Gruene + FDP": ("cdu_csu", "gruene", "fdp"),
}


@dataclass(frozen=True)
class SimulationResult:
    """Container for the polling aggregation and Monte Carlo output."""

    weighted_average: pd.Series
    poll_weights: pd.Series
    latent_mean: pd.Series
    latent_covariance: pd.DataFrame
    simulated_votes: pd.DataFrame
    simulated_seats: pd.DataFrame
    uncertainty_intervals: pd.DataFrame
    threshold_probabilities: pd.Series
    coalition_probabilities: pd.Series


def compute_poll_weights(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    half_life_days: float = 14.0,
    sample_size_power: float = 0.5,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Compute normalized poll weights from recency and sample size. 
       We AGGREGATE the polls.

    Args:
        rows: Poll rows with at least ``published_at`` and ``sample_size``.
        half_life_days: Number of days after which recency weight halves.
        sample_size_power: Exponent applied to sample size. ``0.5`` uses sqrt(n).
        as_of_date: Date from which recency is measured. Defaults to latest poll date.

    Returns:
        Normalized weights aligned to the input row order.
    """
    if half_life_days <= 0:
        msg = "half_life_days must be positive"
        raise ValueError(msg)
    if sample_size_power < 0:
        msg = "sample_size_power must be non-negative"
        raise ValueError(msg)

    polls = _polls_to_dataframe(rows)
    reference_date = pd.Timestamp(as_of_date) if as_of_date is not None else polls["published_at"].max()
    age_days = (reference_date - polls["published_at"]).dt.days.clip(lower=0)
    recency_weights = np.power(0.5, age_days.to_numpy(dtype=float) / half_life_days)
    sample_weights = np.power(polls["sample_size"].to_numpy(dtype=float), sample_size_power)
    raw_weights = recency_weights * sample_weights

    if np.isclose(raw_weights.sum(), 0.0):
        msg = "poll weights sum to zero"
        raise ValueError(msg)

    return pd.Series(raw_weights / raw_weights.sum(), index=polls.index, name="weight")


def compute_weighted_average(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    party_columns: Sequence[str] = PARTY_COLUMNS,
    half_life_days: float = 14.0,
    sample_size_power: float = 0.5,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Compute a recency-weighted polling average as party vote-share proportions.
       Function to compute the weights explicitly. 

    Args:
        rows: Poll rows in wide format with party percentages.
        party_columns: Columns to aggregate.
        half_life_days: Number of days after which recency weight halves.
        sample_size_power: Exponent applied to sample size.
        as_of_date: Date from which recency is measured. Defaults to latest poll date.

    Returns:
        Weighted average vote shares as proportions summing to one.
    """
    polls = _polls_to_dataframe(rows, party_columns=party_columns)

    weights = compute_poll_weights(
        polls,
        half_life_days=half_life_days,
        sample_size_power=sample_size_power,
        as_of_date=as_of_date,
    )
    party_shares = polls[list(party_columns)] / 100.0
    weighted_average = party_shares.mul(weights, axis=0).sum(axis=0)
    return _normalize_series(weighted_average)


def sample_vote_shares(
    mean_share: pd.Series,
    latent_covariance: np.ndarray = DEFAULT_LATENT_COVARIANCE,
    n_draws: int = 10_000,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Sample vote shares via latent normal scores followed by softmax.
       We are MODELING the Uncertatinty
       Draw from multivariate normal --> put into (logged) softmax
       To get Voting Probabilites

    Args:
        mean_share: Aggregated party vote shares as proportions.
        latent_covariance: Constant covariance matrix for the latent normal scores.
        n_draws: Number of Monte Carlo draws.
        seed: Random seed for reproducible simulations.

    Returns:
        Latent mean, labelled latent covariance, and simulated vote-share probabilities.
    """
    if n_draws <= 0:
        msg = "n_draws must be positive"
        raise ValueError(msg)

    mean = _mean_share_to_series(mean_share)
    covariance = _validate_latent_covariance(latent_covariance, n_parties=len(mean))

    latent_mean = pd.Series(np.log(mean.to_numpy()), index=mean.index, name="latent_mean")
    latent_covariance_df = pd.DataFrame(covariance, index=mean.index, columns=mean.index)

    rng = np.random.default_rng(seed)

    # Draw scores from multivariate normal distribution
    latent_draws = rng.multivariate_normal(
        mean=latent_mean.to_numpy(),
        cov=covariance,
        size=n_draws,
        check_valid="raise",
    )
    
    # Get Simulated votes via Softmax
    simulated_votes = pd.DataFrame(_softmax(latent_draws), columns=mean.index)
    return latent_mean, latent_covariance_df, simulated_votes


def simulate_election(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    party_columns: Sequence[str] = PARTY_COLUMNS,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    n_draws: int = 10_000,
    threshold: float = 0.05,
    seed: int = 42,
    half_life_days: float = 14.0,
    sample_size_power: float = 0.5,
    latent_covariance: np.ndarray = DEFAULT_LATENT_COVARIANCE,
    threshold_exemptions: Sequence[str] = ("sonstige",),
    as_of_date: str | pd.Timestamp | None = None,
) -> SimulationResult:
    """Simulate election outcomes from latent normal scores and softmax probabilities.

    Args:
        rows: Poll rows in wide format with party percentages.
        party_columns: Party columns to model.
        coalitions: Mapping from coalition display names to party columns.
        n_draws: Number of Monte Carlo draws.
        threshold: Electoral threshold as a proportion.
        seed: Random seed for reproducible simulations.
        half_life_days: Number of days after which recency weight halves.
        sample_size_power: Exponent applied to sample size.
        latent_covariance: Constant covariance matrix for latent party scores.
        threshold_exemptions: Parties exempt from threshold probability reporting and seat eligibility.
        as_of_date: Date from which recency is measured. Defaults to latest poll date.

    Returns:
        SimulationResult with averages, latent parameters, simulated draws, and summaries.
    """
    if not 0 <= threshold <= 1:
        msg = "threshold must be between 0 and 1"
        raise ValueError(msg)

    # Get polls
    polls = _polls_to_dataframe(rows, party_columns=party_columns)
    _validate_coalitions(coalitions, party_columns)

    # Calculate poll weights
    poll_weights = compute_poll_weights(
        polls,
        half_life_days=half_life_days,
        sample_size_power=sample_size_power,
        as_of_date=as_of_date,
    )
    # Aggregate polls with weights
    weighted_average = compute_weighted_average(
        polls,
        party_columns=party_columns,
        half_life_days=half_life_days,
        sample_size_power=sample_size_power,
        as_of_date=as_of_date,
    )
    # Model uncertainty
    latent_mean, latent_covariance_df, simulated_votes = sample_vote_shares(
        weighted_average,
        latent_covariance=latent_covariance,
        n_draws=n_draws,
        seed=seed,
    )

    # Calculate "5% Hürde" etc. uncertainty intervalls etc. 
    simulated_seats = _apply_threshold(simulated_votes, threshold=threshold, threshold_exemptions=threshold_exemptions)
    threshold_probabilities = _threshold_probabilities(
        simulated_votes,
        threshold=threshold,
        threshold_exemptions=threshold_exemptions,
    )
    coalition_probabilities = _coalition_probabilities(simulated_seats, coalitions=coalitions)
    uncertainty_intervals = _uncertainty_intervals(simulated_votes)

    return SimulationResult(
        weighted_average=weighted_average,
        poll_weights=poll_weights,
        latent_mean=latent_mean,
        latent_covariance=latent_covariance_df,
        simulated_votes=simulated_votes,
        simulated_seats=simulated_seats,
        uncertainty_intervals=uncertainty_intervals,
        threshold_probabilities=threshold_probabilities,
        coalition_probabilities=coalition_probabilities,
    )


def summarize_results(result: SimulationResult) -> dict[str, pd.DataFrame | pd.Series]:
    """Return display-ready summaries from a simulation result."""
    average = result.weighted_average.rename("weighted_average")
    return {
        "weighted_average": average,
        "uncertainty_intervals": result.uncertainty_intervals,
        "threshold_probabilities": result.threshold_probabilities,
        "coalition_probabilities": result.coalition_probabilities.sort_values(ascending=False),
    }


# HELPER FUNCTIONS 
def _validate_latent_covariance(covariance: np.ndarray, n_parties: int) -> np.ndarray:
    """Validate the fixed latent covariance matrix used by the simulator."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (n_parties, n_parties):
        msg = f"latent_covariance must have shape {(n_parties, n_parties)}, got {covariance.shape}"
        raise ValueError(msg)
    if not np.allclose(covariance, covariance.T):
        msg = "latent_covariance must be symmetric"
        raise ValueError(msg)

    eigvals = np.linalg.eigvalsh(covariance)
    assert eigvals.min() > -1e-8
    return covariance


def _softmax(values: np.ndarray) -> np.ndarray:
    """Convert latent scores into vote-share probabilities."""
    shifted_values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted_values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _polls_to_dataframe(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    party_columns: Sequence[str] = PARTY_COLUMNS,
) -> pd.DataFrame:
    """Convert poll rows into a validated dataframe."""
    polls = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if polls.empty:
        msg = "no polls available"
        raise ValueError(msg)

    required_columns = {"published_at", "sample_size", *party_columns}
    missing_columns = required_columns.difference(polls.columns)
    if missing_columns:
        msg = f"missing required poll columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    polls = polls.copy()
    polls["published_at"] = pd.to_datetime(polls["published_at"], errors="raise")
    polls["sample_size"] = pd.to_numeric(polls["sample_size"], errors="raise")
    for party in party_columns:
        polls[party] = pd.to_numeric(polls[party], errors="raise")

    if (polls["sample_size"] <= 0).any():
        msg = "sample_size must be positive for every poll"
        raise ValueError(msg)

    return polls


def _mean_share_to_series(mean_share: pd.Series | Sequence[float]) -> pd.Series:
    """Convert a mean-share object to a labelled series."""
    if isinstance(mean_share, pd.Series):
        mean = mean_share.astype(float).copy()
    else:
        mean = pd.Series(mean_share, dtype=float)

    if mean.empty:
        msg = "mean_share must not be empty"
        raise ValueError(msg)
    if (mean <= 0).any():
        msg = "mean_share must contain positive values before taking log scores"
        raise ValueError(msg)

    return _normalize_series(mean)


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a non-negative series to sum to one."""
    total = float(series.sum())
    if total <= 0:
        msg = "cannot normalize shares with non-positive total"
        raise ValueError(msg)
    return series / total


def _apply_threshold(
    simulated_votes: pd.DataFrame,
    threshold: float,
    threshold_exemptions: Sequence[str],
) -> pd.DataFrame:
    """Apply a simplified 5 percent threshold and return normalized seat-share proxies."""
    exempt = set(threshold_exemptions)
    seat_shares = simulated_votes.copy()
    threshold_parties = [party for party in simulated_votes.columns if party not in exempt]
    seat_shares[threshold_parties] = seat_shares[threshold_parties].where(
        seat_shares[threshold_parties] >= threshold,
        0.0,
    )
    for party in exempt.intersection(seat_shares.columns):
        seat_shares[party] = 0.0

    eligible_total = seat_shares.sum(axis=1)
    seat_shares = seat_shares.div(eligible_total.replace(0.0, np.nan), axis=0)
    return seat_shares.fillna(0.0)


def _threshold_probabilities(
    simulated_votes: pd.DataFrame,
    threshold: float,
    threshold_exemptions: Sequence[str],
) -> pd.Series:
    """Compute the probability that each non-exempt party clears the threshold."""
    exempt = set(threshold_exemptions)
    probabilities = {
        party: float((simulated_votes[party] >= threshold).mean())
        for party in simulated_votes.columns
        if party not in exempt
    }
    return pd.Series(probabilities, name="threshold_probability")


def _coalition_probabilities(
    simulated_seats: pd.DataFrame,
    coalitions: Mapping[str, Sequence[str]],
) -> pd.Series:
    """Compute majority probabilities for configured coalitions."""
    probabilities = {
        coalition_name: float((simulated_seats[list(parties)].sum(axis=1) > 0.5).mean())
        for coalition_name, parties in coalitions.items()
    }
    return pd.Series(probabilities, name="majority_probability")


def _uncertainty_intervals(simulated_votes: pd.DataFrame) -> pd.DataFrame:
    """Compute central 95 percent intervals for simulated party vote shares."""
    intervals = simulated_votes.quantile([0.025, 0.5, 0.975]).T
    intervals.columns = ["lower_95", "median", "upper_95"]
    return intervals


def _validate_coalitions(coalitions: Mapping[str, Sequence[str]], party_columns: Sequence[str]) -> None:
    """Validate that configured coalitions only reference modeled parties."""
    party_set = set(party_columns)
    unknown_parties = sorted({party for parties in coalitions.values() for party in parties if party not in party_set})
    if unknown_parties:
        msg = f"coalitions reference unknown parties: {unknown_parties}"
        raise ValueError(msg)


def main(data_path: Path = DEFAULT_SAMPLE_PATH,
         n_draws: int = 50000) -> None:
    """Run the prototype model on the sample data and print compact summaries."""
    rows = load_poll_csv(data_path)
    result = simulate_election(rows, n_draws=n_draws)
    summaries = summarize_results(result)

    print("Weighted polling average")
    print((summaries["weighted_average"] * 100).round(2))
    print("\nThreshold probabilities")
    print((summaries["threshold_probabilities"] * 100).round(1))
    print("\nCoalition majority probabilities")
    print((summaries["coalition_probabilities"] * 100).round(1))


if __name__ == "__main__":
    typer.run(main)
