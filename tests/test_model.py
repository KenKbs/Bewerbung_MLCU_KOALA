import numpy as np
import pytest

from wahlumfragen.data import PARTY_COLUMNS, generate_sample_polls
from wahlumfragen.model import (
    build_correlation_matrix,
    build_covariance,
    compute_poll_weights,
    compute_weighted_average,
    simulate_election,
)


def _sample_rows():
    """Return synthetic poll rows as dictionaries."""
    return [poll.as_csv_row() for poll in generate_sample_polls()]


def test_poll_weights_sum_to_one_and_favor_recent_polls():
    """Test recency and sample-size based poll weighting."""
    weights = compute_poll_weights(_sample_rows())

    assert np.isclose(weights.sum(), 1.0)
    assert weights.iloc[-1] > weights.iloc[0]


def test_weighted_average_sums_to_one():
    """Test that the weighted average is returned as valid vote-share proportions."""
    weighted_average = compute_weighted_average(_sample_rows())

    assert list(weighted_average.index) == list(PARTY_COLUMNS)
    assert np.isclose(weighted_average.sum(), 1.0)


def test_correlation_matrix_has_negative_off_diagonal_entries():
    """Test the explicit competition correlation structure."""
    corr = build_correlation_matrix(len(PARTY_COLUMNS), off_diagonal_correlation=-0.10)
    off_diagonal = corr[~np.eye(len(PARTY_COLUMNS), dtype=bool)]

    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(off_diagonal, -0.10)


def test_correlation_matrix_rejects_non_psd_rho():
    """Test that invalid equicorrelation parameters are rejected."""
    lower_bound = -1 / (len(PARTY_COLUMNS) - 1)

    with pytest.raises(ValueError, match="PSD lower bound"):
        build_correlation_matrix(len(PARTY_COLUMNS), off_diagonal_correlation=lower_bound - 0.01)


def test_covariance_matrix_is_symmetric_and_positive_semidefinite():
    """Test that the covariance matrix passes the numerical PSD eigenvalue check."""
    weighted_average = compute_weighted_average(_sample_rows())
    cov = build_covariance(weighted_average, off_diagonal_correlation=-0.10)
    eigvals = np.linalg.eigvalsh(cov)

    assert np.allclose(cov, cov.T)
    assert eigvals.min() > -1e-8


def test_simulated_vote_shares_are_valid_probabilities():
    """Test Monte Carlo draws after clipping and normalization."""
    result = simulate_election(_sample_rows(), n_draws=1_000, seed=123)

    assert (result.simulated_votes >= 0).all().all()
    assert np.allclose(result.simulated_votes.sum(axis=1), 1.0)
    assert result.threshold_probabilities.between(0, 1).all()
    assert result.coalition_probabilities.between(0, 1).all()


def test_simulation_is_deterministic_with_fixed_seed():
    """Test that a fixed seed produces repeatable simulations."""
    first = simulate_election(_sample_rows(), n_draws=500, seed=123)
    second = simulate_election(_sample_rows(), n_draws=500, seed=123)

    assert np.allclose(first.simulated_votes.to_numpy(), second.simulated_votes.to_numpy())
    assert np.allclose(first.coalition_probabilities.to_numpy(), second.coalition_probabilities.to_numpy())
