"""Logistic scorecard fitting, points scaling, and reason codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from credit_scorecard.woe import WOETransformer


_USE_CONFIGURED_CLASS_WEIGHT = object()


@dataclass(frozen=True)
class ScoreScaling:
    base_score: float
    base_odds_good_to_bad: float
    points_to_double_odds: float

    @property
    def factor(self) -> float:
        return self.points_to_double_odds / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds_good_to_bad)

    def score_from_log_odds_bad(self, log_odds_bad: np.ndarray | float) -> np.ndarray:
        return self.offset - self.factor * np.asarray(log_odds_bad)

    def probability_from_score(self, score: np.ndarray | float) -> np.ndarray:
        log_odds_bad = (self.offset - np.asarray(score)) / self.factor
        return expit(log_odds_bad)


@dataclass
class ScorecardModel:
    transformer: WOETransformer
    selected_features: list[str]
    estimator: LogisticRegression
    scaling: ScoreScaling
    name: str

    def transformed(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.transformer.transform(X, self.selected_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.transformed(X))[:, 1]

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.decision_function(self.transformed(X))

    def score(self, X: pd.DataFrame, rounded: bool = False) -> np.ndarray:
        result = self.scaling.score_from_log_odds_bad(self.decision_function(X))
        return np.rint(result).astype(int) if rounded else result

    @property
    def base_points(self) -> float:
        return self.scaling.offset - self.scaling.factor * float(self.estimator.intercept_[0])

    def coefficient_table(self) -> pd.DataFrame:
        characteristics = pd.DataFrame(
            {
                "feature": self.selected_features,
                "coefficient": self.estimator.coef_[0],
                "odds_ratio_per_woe_unit": np.exp(self.estimator.coef_[0]),
            }
        )
        characteristics["expected_sign"] = "nonpositive"
        characteristics["sign_consistent"] = characteristics["coefficient"].le(1e-8)
        characteristics["review_flag"] = np.where(
            characteristics["sign_consistent"], "", "positive_coefficient_review"
        )
        intercept = pd.DataFrame(
            [
                {
                    "feature": "__INTERCEPT__",
                    "coefficient": float(self.estimator.intercept_[0]),
                    "odds_ratio_per_woe_unit": float(np.exp(self.estimator.intercept_[0])),
                    "expected_sign": "not_applicable",
                    "sign_consistent": True,
                    "review_flag": "",
                }
            ]
        )
        return pd.concat([intercept, characteristics], ignore_index=True)

    def points_table(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = [
            {
                "feature": "__BASE_POINTS__",
                "bin": "BASE",
                "woe": np.nan,
                "coefficient": float(self.estimator.intercept_[0]),
                "exact_points": self.base_points,
                "display_points": int(np.rint(self.base_points)),
            }
        ]
        coefficients = dict(zip(self.selected_features, self.estimator.coef_[0], strict=True))
        for feature in self.selected_features:
            definition = self.transformer.features_[feature]
            for bin_id, woe in definition.woe_by_bin.items():
                exact_points = -self.scaling.factor * coefficients[feature] * woe
                rows.append(
                    {
                        "feature": feature,
                        "bin": bin_id,
                        "woe": woe,
                        "coefficient": coefficients[feature],
                        "exact_points": exact_points,
                        "display_points": int(np.rint(exact_points)),
                    }
                )
        return pd.DataFrame(rows)

    def reason_codes(self, X: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
        """Return largest point shortfalls from each characteristic's best bin."""
        bins = self.transformer.assign_bins(X, self.selected_features)
        points = self.points_table()
        best = points.groupby("feature")["exact_points"].max().to_dict()
        lookup = points.set_index(["feature", "bin"])["exact_points"].to_dict()
        rows: list[dict[str, Any]] = []
        for position, index in enumerate(X.index):
            shortfalls = []
            for feature in self.selected_features:
                bin_id = bins.iloc[position][feature]
                current = lookup.get((feature, bin_id), 0.0)
                shortfalls.append((feature, bin_id, float(best[feature] - current)))
            shortfalls.sort(key=lambda item: item[2], reverse=True)
            row: dict[str, Any] = {"index": index}
            for rank, (feature, bin_id, shortfall) in enumerate(shortfalls[:top_n], start=1):
                row[f"reason_{rank}"] = feature
                row[f"reason_{rank}_bin"] = bin_id
                row[f"reason_{rank}_point_shortfall"] = shortfall
            rows.append(row)
        return pd.DataFrame(rows).set_index("index")


def _new_estimator(
    settings: dict[str, Any],
    class_weight: str | dict | None | object = _USE_CONFIGURED_CLASS_WEIGHT,
) -> LogisticRegression:
    configured_weight = (
        settings.get("class_weight")
        if class_weight is _USE_CONFIGURED_CLASS_WEIGHT
        else class_weight
    )
    return LogisticRegression(
        C=float(settings["regularization_c"]),
        max_iter=int(settings["maximum_iterations"]),
        class_weight=configured_weight,
        solver="lbfgs",
        random_state=42,
    )


def make_scaling(settings: dict[str, Any]) -> ScoreScaling:
    return ScoreScaling(
        float(settings["base_score"]),
        float(settings["base_odds_good_to_bad"]),
        float(settings["points_to_double_odds"]),
    )


def fit_scorecard(
    X: pd.DataFrame,
    y: Iterable[int],
    transformer: WOETransformer,
    selected_features: list[str],
    model_settings: dict[str, Any],
    name: str,
    sample_weight: Iterable[float] | None = None,
) -> ScorecardModel:
    transformed = transformer.transform(X, selected_features)
    estimator = _new_estimator(model_settings)
    estimator.fit(transformed, np.asarray(list(y), dtype=int), sample_weight=sample_weight)
    return ScorecardModel(
        transformer,
        selected_features,
        estimator,
        make_scaling(model_settings),
        name,
    )


def fit_fractional_scorecard(
    X: pd.DataFrame,
    expected_y: Iterable[float],
    transformer: WOETransformer,
    selected_features: list[str],
    model_settings: dict[str, Any],
    name: str,
    sample_weight: Iterable[float] | None = None,
) -> ScorecardModel:
    """Fit expected binomial likelihood by weighted good and bad row copies."""
    transformed = transformer.transform(X, selected_features)
    target = np.asarray(list(expected_y), dtype=float)
    base_weight = (
        np.ones(len(target), dtype=float)
        if sample_weight is None
        else np.asarray(list(sample_weight), dtype=float)
    )
    if np.any((target < 0) | (target > 1)):
        raise ValueError("Fractional targets must be between zero and one")
    duplicated = pd.concat([transformed, transformed], ignore_index=True)
    labels = np.concatenate([np.ones(len(target), dtype=int), np.zeros(len(target), dtype=int)])
    weights = np.concatenate([target * base_weight, (1.0 - target) * base_weight])
    positive = weights > 0
    estimator = _new_estimator(model_settings, class_weight=None)
    estimator.fit(duplicated.loc[positive], labels[positive], sample_weight=weights[positive])
    return ScorecardModel(
        transformer,
        selected_features,
        estimator,
        make_scaling(model_settings),
        name,
    )


def verify_scaling(model: ScorecardModel, tolerance: float = 1e-10) -> None:
    """Verify the score equation and stated base-odds contract."""
    base_probability = 1.0 / (1.0 + model.scaling.base_odds_good_to_bad)
    base_log_odds = logit(base_probability)
    score = float(model.scaling.score_from_log_odds_bad(base_log_odds))
    if not np.isclose(score, model.scaling.base_score, atol=tolerance):
        raise AssertionError("Base score and base odds do not reconcile")
    doubled_good_odds_probability = 1.0 / (1.0 + 2.0 * model.scaling.base_odds_good_to_bad)
    doubled_score = float(
        model.scaling.score_from_log_odds_bad(logit(doubled_good_odds_probability))
    )
    if not np.isclose(
        doubled_score - score,
        model.scaling.points_to_double_odds,
        atol=tolerance,
    ):
        raise AssertionError("PDO scaling does not add the configured points for doubled odds")
