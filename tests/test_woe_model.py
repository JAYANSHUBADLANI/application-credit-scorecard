from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scorecard.model import ScoreScaling, fit_scorecard, verify_scaling
from credit_scorecard.woe import OTHER_BIN, WOETransformer


def test_numeric_woe_is_monotonic_and_finite() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=2000)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    frame = pd.DataFrame({"risk": x})
    transformer = WOETransformer(
        maximum_prebins=10,
        minimum_bin_fraction=0.05,
        smoothing=0.5,
    ).fit(frame, y)
    table = transformer.features_["risk"].table
    regular = table[table["bin"].str.startswith("N")]
    differences = regular["bad_rate"].diff().dropna()
    assert (differences.ge(-1e-12).all()) or (differences.le(1e-12).all())
    assert np.isfinite(transformer.transform(frame).to_numpy()).all()
    assert transformer.features_["risk"].iv > 0


def test_categorical_missing_and_unseen_are_stable() -> None:
    frame = pd.DataFrame({"category": ["a"] * 50 + ["b"] * 50 + [None] * 20})
    y = np.array([0] * 45 + [1] * 5 + [0] * 25 + [1] * 25 + [0] * 15 + [1] * 5)
    transformer = WOETransformer(
        maximum_categorical_bins=4,
        minimum_bin_fraction=0.01,
    ).fit(frame, y)
    assigned = transformer.assign_bins(pd.DataFrame({"category": ["new", None]}))
    transformed = transformer.transform(pd.DataFrame({"category": ["new", None]}))
    assert assigned.iloc[0, 0] == OTHER_BIN
    assert np.isfinite(transformed.to_numpy()).all()


def test_fractional_woe_reestimation_keeps_edges() -> None:
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    target = np.array([0] * 80 + [1] * 20)
    transformer = WOETransformer(maximum_prebins=5, minimum_bin_fraction=0.1).fit(frame, target)
    edges = list(transformer.features_["x"].edges)
    expected = np.linspace(0.01, 0.5, len(frame))
    updated = transformer.reestimate_woe(frame, expected)
    assert updated.features_["x"].edges == edges
    assert updated.features_["x"].woe_by_bin != transformer.features_["x"].woe_by_bin


def test_score_scaling_base_odds_pdo_and_pd_reconstruction() -> None:
    scaling = ScoreScaling(600, 50, 20)
    base_pd = 1 / 51
    base_log_odds = np.log(base_pd / (1 - base_pd))
    base_score = scaling.score_from_log_odds_bad(base_log_odds)
    doubled_pd = 1 / 101
    doubled_score = scaling.score_from_log_odds_bad(np.log(doubled_pd / (1 - doubled_pd)))
    assert np.isclose(base_score, 600)
    assert np.isclose(doubled_score - base_score, 20)
    assert np.isclose(scaling.probability_from_score(base_score), base_pd)


def test_scorecard_predictions_reconstruct_from_exact_scores() -> None:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"x": rng.normal(size=800)})
    target = rng.binomial(1, 1 / (1 + np.exp(-frame["x"])))
    transformer = WOETransformer(maximum_prebins=8, minimum_bin_fraction=0.05).fit(frame, target)
    settings = {
        "regularization_c": 1.0,
        "maximum_iterations": 1000,
        "class_weight": None,
        "base_score": 600,
        "base_odds_good_to_bad": 50,
        "points_to_double_odds": 20,
    }
    model = fit_scorecard(frame, target, transformer, ["x"], settings, "test")
    verify_scaling(model)
    coefficient = model.coefficient_table().set_index("feature")
    assert coefficient.loc["x", "sign_consistent"]
    assert "__BASE_POINTS__" in set(model.points_table()["feature"])
    assert np.allclose(model.predict_proba(frame), model.scaling.probability_from_score(model.score(frame)))
    assert np.corrcoef(model.score(frame), model.predict_proba(frame))[0, 1] < -0.99
