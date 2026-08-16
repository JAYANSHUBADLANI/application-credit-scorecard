from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_scorecard.decisioning import (
    account_economics,
    cutoff_curve,
    optimal_cutoff,
    swap_set_analysis,
)
from credit_scorecard.metrics import (
    characteristic_stability_index,
    gini_coefficient,
    hosmer_lemeshow,
    ks_statistic,
    population_stability_index,
)
from credit_scorecard.reject_inference import SyntheticAcceptancePolicy, parcel_rejects


def _policy_frame(rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "EXT_SOURCE_2": rng.uniform(size=rows),
            "EXT_SOURCE_3": rng.uniform(size=rows),
            "APP_CREDIT_INCOME_RATIO": rng.lognormal(size=rows),
        }
    )


def test_synthetic_acceptance_is_reproducible_and_hits_target() -> None:
    frame = _policy_frame()
    policy = SyntheticAcceptancePolicy(0.65, 0.12, 42).fit(frame)
    first = policy.sample(frame)
    second = policy.sample(frame)
    assert np.array_equal(first, second)
    assert abs(policy.acceptance_probability(frame).mean() - 0.65) < 1e-6
    assert abs(first.mean() - 0.65) < 0.03
    assert policy.metadata()["target_used"] is False


def test_parcelling_hides_rejected_truth_and_multiplier_increases_risk() -> None:
    scores = np.linspace(400, 800, 100)
    accepted = np.arange(100) % 3 != 0
    truth = (scores < 560).astype(float)
    observed = np.where(accepted, truth, np.nan)
    low = parcel_rejects(scores, observed, accepted, 1.0, 5)
    high = parcel_rejects(scores, observed, accepted, 2.0, 5)
    assert np.all(high.expected_target[~accepted] >= low.expected_target[~accepted])
    leaked = truth.copy()
    with pytest.raises(ValueError, match="must be hidden"):
        parcel_rejects(scores, leaked, accepted, 1.5, 5)


def test_discrimination_and_stability_reference_cases() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    assert gini_coefficient(target, probability) == 1.0
    assert ks_statistic(target, probability)[0] == 1.0
    statistic, p_value, table = hosmer_lemeshow(target, probability, groups=2)
    assert np.isfinite(statistic)
    assert 0 <= p_value <= 1
    assert len(table) == 2
    expected = np.arange(100, dtype=float)
    psi, _ = population_stability_index(expected, expected, bins=10)
    assert abs(psi) < 1e-12
    bins = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    summary, _ = characteristic_stability_index(bins, bins)
    assert summary.loc[0, "csi"] == 0


def _economic_settings() -> dict[str, float | int]:
    return {
        "ead_factor": 1.0,
        "loss_given_default": 0.5,
        "net_margin_rate": 0.1,
        "pre_default_bad_margin_rate": 0.02,
        "acquisition_cost": 100.0,
        "operating_cost": 50.0,
        "cutoff_grid_size": 11,
    }


def test_economics_reconcile_and_cutoff_curve_is_complete() -> None:
    probability = np.array([0.01, 0.10, 0.50])
    exposure = np.array([1000.0, 2000.0, 3000.0])
    economics = account_economics(probability, exposure, _economic_settings())
    assert np.allclose(
        economics["expected_profit"],
        economics["expected_revenue"] - economics["expected_loss"] - economics["expected_cost"],
    )
    curve = cutoff_curve(
        [700, 600, 500], probability, exposure, _economic_settings(), [0, 0, 1]
    )
    assert {"approval_rate", "observed_bad_rate", "expected_loss", "expected_profit"}.issubset(curve)


def test_cutoff_can_reject_all_negative_value_accounts() -> None:
    settings = _economic_settings()
    settings["loss_given_default"] = 1.0
    settings["net_margin_rate"] = 0.0
    curve = cutoff_curve(
        [700, 600, 500],
        [0.9, 0.8, 0.7],
        [1000, 1000, 1000],
        settings,
        [1, 1, 1],
    )
    optimum = optimal_cutoff(curve)
    assert optimum["approved_count"] == 0
    assert optimum["expected_profit"] == 0


def test_swap_set_has_equal_swap_in_and_swap_out() -> None:
    result = swap_set_analysis(
        applicant_ids=[1, 2, 3, 4, 5, 6],
        baseline_scores=[6, 5, 4, 3, 2, 1],
        challenger_scores=[1, 6, 5, 4, 3, 2],
        target=[0, 0, 1, 0, 1, 1],
        exposure=[1000] * 6,
        approval_rate=0.5,
        economics_settings=_economic_settings(),
    )
    counts = result.applicants["segment"].value_counts()
    assert counts["swap_in"] == counts["swap_out"]
    assert result.summary["applicants"].sum() == 6


def test_swap_set_allows_zero_approval_policy() -> None:
    result = swap_set_analysis(
        applicant_ids=[1, 2, 3],
        baseline_scores=[3, 2, 1],
        challenger_scores=[1, 2, 3],
        target=[0, 1, 0],
        exposure=[1000, 1000, 1000],
        approval_rate=0.0,
        economics_settings=_economic_settings(),
    )
    assert set(result.applicants["segment"]) == {"both_declined"}
