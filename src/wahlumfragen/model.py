from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wahlumfragen.data import DEFAULT_SAMPLE_PATH, PARTY_COLUMNS, load_poll_csv

# Define default coalations as a mapping, maybe add afd + CDU...
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
    correlation: pd.DataFrame
    covariance: pd.DataFrame
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

    # AGGREGATION --> RECENCY + SAMPLE SIZE
    weights = compute_poll_weights(
        polls,
        half_life_days=half_life_days,
        sample_size_power=sample_size_power,
        as_of_date=as_of_date,
    )
    # Get parties from that poll and convert to float e.g. 27.5% --> 0.275
    party_shares = polls[list(party_columns)] / 100.0
    weighted_average = party_shares.mul(weights, axis=0).sum(axis=0) # Multiply by weight
    return _normalize_series(weighted_average) # Normalize_series, makes sure everything sumes to 100% (individual value / sum of all values)


def build_correlation_matrix(n_parties: int, off_diagonal_correlation: float = -0.10) -> np.ndarray:
    """Build a PSD equicorrelation matrix for competing party vote shares.

    Args:
        n_parties: Number of parties included in the model.
        off_diagonal_correlation: Shared off-diagonal correlation coefficient.

    Returns:
        Equicorrelation matrix with ones on the diagonal.
    """
    if n_parties < 2:
        msg = "n_parties must be at least 2"
        raise ValueError(msg)

    lower_bound = -1 / (n_parties - 1)
    if off_diagonal_correlation < lower_bound:
        msg = (
            f"off_diagonal_correlation={off_diagonal_correlation} is below the PSD lower bound "
            f"{lower_bound:.6f} for {n_parties} parties"
        )
        raise ValueError(msg)
    if off_diagonal_correlation > 1:
        msg = "off_diagonal_correlation must be <= 1"
        raise ValueError(msg)

    corr = np.full((n_parties, n_parties), off_diagonal_correlation, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


def build_covariance(
    mean_share: pd.Series | Sequence[float],
    model_error: float = 0.015,
    off_diagonal_correlation: float = -0.10,
    effective_sample_size: float = 1_500.0,
) -> pd.DataFrame:
    """Build a PSD covariance matrix around the weighted polling average.

    Args:
        mean_share: Mean party vote shares as proportions.
        model_error: Additional polling/model uncertainty floor in share points.
        off_diagonal_correlation: Shared negative correlation between different parties.
        effective_sample_size: Approximate sample size used for binomial polling variance.

    Returns:
        Covariance matrix as a labelled dataframe.
    """
    if model_error < 0:
        msg = "model_error must be non-negative"
        raise ValueError(msg)
    if effective_sample_size <= 0:
        msg = "effective_sample_size must be positive"
        raise ValueError(msg)

    mean = _mean_share_to_series(mean_share)
    corr = build_correlation_matrix(len(mean), off_diagonal_correlation=off_diagonal_correlation)
    sampling_variance = np.maximum(mean.to_numpy() * (1.0 - mean.to_numpy()), 0.0) / effective_sample_size
    std = np.sqrt(sampling_variance + model_error**2)
    cov = np.outer(std, std) * corr

    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > -1e-8

    return pd.DataFrame(cov, index=mean.index, columns=mean.index)


def simulate_election(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    party_columns: Sequence[str] = PARTY_COLUMNS,
    coalitions: Mapping[str, Sequence[str]] = DEFAULT_COALITIONS,
    n_draws: int = 10_000,
    threshold: float = 0.05,
    seed: int = 42,
    half_life_days: float = 14.0,
    sample_size_power: float = 0.5,
    model_error: float = 0.015,
    off_diagonal_correlation: float = -0.10,
    effective_sample_size: float = 1_500.0,
    threshold_exemptions: Sequence[str] = ("sonstige",),
    as_of_date: str | pd.Timestamp | None = None,
) -> SimulationResult:
    """Simulate election outcomes from a correlated polling-average distribution.

    Args:
        rows: Poll rows in wide format with party percentages.
        party_columns: Party columns to model.
        coalitions: Mapping from coalition display names to party columns.
        n_draws: Number of Monte Carlo draws.
        threshold: Electoral threshold as a proportion.
        seed: Random seed for reproducible simulations.
        half_life_days: Number of days after which recency weight halves.
        sample_size_power: Exponent applied to sample size.
        model_error: Additional polling/model uncertainty floor in share points.
        off_diagonal_correlation: Shared negative correlation between different parties.
        effective_sample_size: Approximate sample size used for binomial polling variance.
        threshold_exemptions: Parties exempt from threshold probability reporting and seat eligibility.
        as_of_date: Date from which recency is measured. Defaults to latest poll date.

    Returns:
        SimulationResult with averages, matrices, simulated draws, and probability summaries.
    """
    if n_draws <= 0:
        msg = "n_draws must be positive"
        raise ValueError(msg)
    if not 0 <= threshold <= 1:
        msg = "threshold must be between 0 and 1"
        raise ValueError(msg)

    polls = _polls_to_dataframe(rows, party_columns=party_columns)
    _validate_coalitions(coalitions, party_columns)

    poll_weights = compute_poll_weights(
        polls,
        half_life_days=half_life_days,
        sample_size_power=sample_size_power,
        as_of_date=as_of_date,
    )
    weighted_average = compute_weighted_average(
        polls,
        party_columns=party_columns,
        half_life_days=half_life_days,
        sample_size_power=sample_size_power,
        as_of_date=as_of_date,
    )
    covariance = build_covariance(
        weighted_average,
        model_error=model_error,
        off_diagonal_correlation=off_diagonal_correlation,
        effective_sample_size=effective_sample_size,
    )
    correlation = pd.DataFrame(
        build_correlation_matrix(len(party_columns), off_diagonal_correlation=off_diagonal_correlation),
        index=party_columns,
        columns=party_columns,
    )

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        mean=weighted_average.to_numpy(),
        cov=covariance.to_numpy(),
        size=n_draws,
        check_valid="raise",
    )
    simulated_votes = _clip_and_normalize_draws(draws, columns=party_columns)
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
        correlation=correlation,
        covariance=covariance,
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
    if (mean < 0).any():
        msg = "mean_share cannot contain negative values"
        raise ValueError(msg)

    return _normalize_series(mean)


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a non-negative series to sum to one, by dividing
    each individual percentage by the sum of percantages."""
    total = float(series.sum())
    if total <= 0:
        msg = "cannot normalize shares with non-positive total"
        raise ValueError(msg)
    return series / total


def _clip_and_normalize_draws(draws: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Clip negative simulated vote shares and normalize each draw to one."""
    clipped = np.clip(draws, a_min=0.0, a_max=None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    zero_rows = np.isclose(row_sums[:, 0], 0.0)
    if zero_rows.any():
        clipped[zero_rows, :] = 1.0 / clipped.shape[1]
        row_sums = clipped.sum(axis=1, keepdims=True)
    normalized = clipped / row_sums
    return pd.DataFrame(normalized, columns=columns)


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


def main(data_path: Path = DEFAULT_SAMPLE_PATH) -> None:
    """Run the prototype model on the sample data and print compact summaries."""
    rows = load_poll_csv(data_path)
    result = simulate_election(rows)
    summaries = summarize_results(result)

    print("Weighted polling average")
    print((summaries["weighted_average"] * 100).round(2))
    print("\nThreshold probabilities")
    print((summaries["threshold_probabilities"] * 100).round(1))
    print("\nCoalition majority probabilities")
    print((summaries["coalition_probabilities"] * 100).round(1))


if __name__ == "__main__":
    main()
