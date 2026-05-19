import numpy as np
import pytest

from wahlumfragen.data import PARTY_COLUMNS, generate_sample_polls
from wahlumfragen.model import (
    DEFAULT_LATENT_COVARIANCE,
    compute_poll_weights,
    compute_weighted_average,
    sample_vote_shares,
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


def test_default_latent_covariance_has_negative_cross_party_covariances():
    """Test the explicit latent covariance structure."""
    off_diagonal = DEFAULT_LATENT_COVARIANCE[~np.eye(len(PARTY_COLUMNS), dtype=bool)]

    assert np.all(np.diag(DEFAULT_LATENT_COVARIANCE) > 0)
    assert np.all(off_diagonal < 0)


def test_default_latent_covariance_is_symmetric_and_positive_semidefinite():
    """Test that the constant latent covariance matrix passes the PSD eigenvalue check."""
    eigvals = np.linalg.eigvalsh(DEFAULT_LATENT_COVARIANCE)

    assert np.allclose(DEFAULT_LATENT_COVARIANCE, DEFAULT_LATENT_COVARIANCE.T)
    assert eigvals.min() > -1e-8


def test_sample_vote_shares_uses_log_mean_and_softmax():
    """Test latent score sampling followed by softmax probability conversion."""
    weighted_average = compute_weighted_average(_sample_rows())
    latent_mean, latent_covariance, simulated_votes = sample_vote_shares(weighted_average, n_draws=1_000, seed=123)
    softmax_center = np.exp(latent_mean) / np.exp(latent_mean).sum()

    assert np.allclose(softmax_center.to_numpy(), weighted_average.to_numpy())
    assert np.allclose(latent_covariance.to_numpy(), DEFAULT_LATENT_COVARIANCE)
    assert (simulated_votes > 0).all().all()
    assert np.allclose(simulated_votes.sum(axis=1), 1.0)


def test_sample_vote_shares_rejects_wrong_covariance_shape():
    """Test that the latent covariance shape must match the number of parties."""
    weighted_average = compute_weighted_average(_sample_rows())

    with pytest.raises(ValueError, match="latent_covariance"):
        sample_vote_shares(weighted_average, latent_covariance=np.eye(2))


def test_simulated_vote_shares_are_valid_probabilities():
    """Test Monte Carlo draws after softmax conversion."""
    result = simulate_election(_sample_rows(), n_draws=1_000, seed=123)

    assert (result.simulated_votes > 0).all().all()
    assert np.allclose(result.simulated_votes.sum(axis=1), 1.0)
    assert result.threshold_probabilities.between(0, 1).all()
    assert result.coalition_probabilities.between(0, 1).all()


def test_simulation_is_deterministic_with_fixed_seed():
    """Test that a fixed seed produces repeatable simulations."""
    first = simulate_election(_sample_rows(), n_draws=500, seed=123)
    second = simulate_election(_sample_rows(), n_draws=500, seed=123)

    assert np.allclose(first.simulated_votes.to_numpy(), second.simulated_votes.to_numpy())
    assert np.allclose(first.coalition_probabilities.to_numpy(), second.coalition_probabilities.to_numpy())
