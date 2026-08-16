"""Credit risk discrimination, calibration, and stability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def gini_coefficient(y_true: Iterable[int], probability_bad: Iterable[float]) -> float:
    return float(2.0 * roc_auc_score(y_true, probability_bad) - 1.0)


def ks_statistic(y_true: Iterable[int], probability_bad: Iterable[float]) -> tuple[float, float]:
    false_positive, true_positive, thresholds = roc_curve(y_true, probability_bad)
    differences = true_positive - false_positive
    index = int(np.argmax(differences))
    return float(differences[index]), float(thresholds[index])


def hosmer_lemeshow(
    y_true: Iterable[int],
    probability_bad: Iterable[float],
    groups: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    target = np.asarray(list(y_true), dtype=int)
    probability = np.clip(np.asarray(list(probability_bad), dtype=float), 1e-9, 1 - 1e-9)
    group = pd.qcut(probability, q=groups, labels=False, duplicates="drop")
    frame = pd.DataFrame({"target": target, "probability": probability, "group": group})
    table = frame.groupby("group", observed=True).agg(
        count=("target", "size"),
        observed_bad=("target", "sum"),
        expected_bad=("probability", "sum"),
        average_probability=("probability", "mean"),
    )
    table["observed_good"] = table["count"] - table["observed_bad"]
    table["expected_good"] = table["count"] - table["expected_bad"]
    bad_term = np.square(table["observed_bad"] - table["expected_bad"]) / table[
        "expected_bad"
    ].clip(lower=1e-9)
    good_term = np.square(table["observed_good"] - table["expected_good"]) / table[
        "expected_good"
    ].clip(lower=1e-9)
    statistic = float((bad_term + good_term).sum())
    degrees = max(len(table) - 2, 1)
    return statistic, float(chi2.sf(statistic, degrees)), table.reset_index()


def expected_calibration_error(
    y_true: Iterable[int],
    probability_bad: Iterable[float],
    bins: int = 10,
) -> float:
    target = np.asarray(list(y_true), dtype=float)
    probability = np.asarray(list(probability_bad), dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    membership = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = len(target)
    error = 0.0
    for bin_id in range(bins):
        mask = membership == bin_id
        if mask.any():
            error += mask.mean() * abs(target[mask].mean() - probability[mask].mean())
    return float(error)


def calibration_intercept_slope(
    y_true: Iterable[int], probability_bad: Iterable[float]
) -> tuple[float, float]:
    target = np.asarray(list(y_true), dtype=int)
    probability = np.clip(np.asarray(list(probability_bad), dtype=float), 1e-8, 1 - 1e-8)
    log_odds = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    estimator.fit(log_odds, target)
    return float(estimator.intercept_[0]), float(estimator.coef_[0, 0])


def validation_metrics(
    y_true: Iterable[int],
    probability_bad: Iterable[float],
    calibration_bins: int = 10,
) -> tuple[dict[str, float], pd.DataFrame]:
    target = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(probability_bad), dtype=float)
    ks, threshold = ks_statistic(target, probability)
    hl, hl_p, calibration = hosmer_lemeshow(target, probability, calibration_bins)
    intercept, slope = calibration_intercept_slope(target, probability)
    metrics = {
        "auc": float(roc_auc_score(target, probability)),
        "gini": gini_coefficient(target, probability),
        "ks": ks,
        "ks_probability_threshold": threshold,
        "brier_score": float(brier_score_loss(target, probability)),
        "expected_calibration_error": expected_calibration_error(
            target, probability, calibration_bins
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "hosmer_lemeshow_statistic": hl,
        "hosmer_lemeshow_p_value": hl_p,
    }
    return metrics, calibration


def bootstrap_interval(
    y_true: Iterable[int],
    probability_bad: Iterable[float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    samples: int = 200,
    random_seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    target = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(probability_bad), dtype=float)
    generator = np.random.default_rng(random_seed)
    values = []
    for _ in range(samples):
        indices = generator.integers(0, len(target), len(target))
        if np.unique(target[indices]).size < 2:
            continue
        values.append(metric(target[indices], probability[indices]))
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(values, [alpha, 1.0 - alpha]))


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    expected_array = np.asarray(list(expected), dtype=float)
    actual_array = np.asarray(list(actual), dtype=float)
    expected_observed = expected_array[np.isfinite(expected_array)]
    if not len(expected_observed):
        raise ValueError("PSI expected population has no finite values")
    inner = np.unique(np.quantile(expected_observed, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    expected_bins = np.digitize(expected_array, edges[1:-1])
    actual_bins = np.digitize(actual_array, edges[1:-1])
    rows = []
    total = 0.0
    for bin_id in range(len(edges) - 1):
        expected_share = max(float(np.mean(expected_bins == bin_id)), epsilon)
        actual_share = max(float(np.mean(actual_bins == bin_id)), epsilon)
        contribution = (actual_share - expected_share) * np.log(actual_share / expected_share)
        total += contribution
        rows.append(
            {
                "bin": bin_id,
                "lower_bound": edges[bin_id],
                "upper_bound": edges[bin_id + 1],
                "expected_share": expected_share,
                "actual_share": actual_share,
                "psi_contribution": contribution,
            }
        )
    return float(total), pd.DataFrame(rows)


def characteristic_stability_index(
    expected_bins: pd.DataFrame,
    actual_bins: pd.DataFrame,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    summary = []
    for feature in expected_bins.columns:
        expected_share = expected_bins[feature].astype(str).value_counts(normalize=True)
        actual_share = actual_bins[feature].astype(str).value_counts(normalize=True)
        all_bins = sorted(set(expected_share.index) | set(actual_share.index))
        total = 0.0
        for bin_id in all_bins:
            expected_value = max(float(expected_share.get(bin_id, 0.0)), epsilon)
            actual_value = max(float(actual_share.get(bin_id, 0.0)), epsilon)
            contribution = (actual_value - expected_value) * np.log(actual_value / expected_value)
            total += contribution
            details.append(
                {
                    "feature": feature,
                    "bin": bin_id,
                    "expected_share": expected_value,
                    "actual_share": actual_value,
                    "csi_contribution": contribution,
                }
            )
        summary.append({"feature": feature, "csi": total})
    return pd.DataFrame(summary).sort_values("csi", ascending=False), pd.DataFrame(details)

