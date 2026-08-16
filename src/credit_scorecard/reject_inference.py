"""Synthetic biased acceptance and assumption-dependent parcelling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass
class SyntheticAcceptancePolicy:
    target_acceptance_rate: float = 0.65
    temperature: float = 0.12
    random_seed: int = 42
    sorted_reference_: dict[str, np.ndarray] = field(default_factory=dict)
    signs_: dict[str, float] = field(default_factory=dict)
    intercept_: float | None = None

    def _component_contract(self, columns: Iterable[str]) -> dict[str, float]:
        available = set(columns)
        candidates = {
            "EXT_SOURCE_1": -1.0,
            "EXT_SOURCE_2": -1.0,
            "EXT_SOURCE_3": -1.0,
            "APP_CREDIT_INCOME_RATIO": 1.0,
            "APP_ANNUITY_INCOME_RATIO": 1.0,
            "BUREAU_BALANCE_DELINQUENCY_RATE_MAX": 1.0,
            "BUREAU_DAYS_OVERDUE_MAX": 1.0,
            "PREV_REFUSED_RATE": 1.0,
            "INST_LATE_RATE": 1.0,
        }
        selected = {feature: sign for feature, sign in candidates.items() if feature in available}
        if len(selected) < 2:
            numeric = [column for column in columns if pd.api.types.is_numeric_dtype(column)]
            raise ValueError(
                "Synthetic acceptance needs at least two configured risk components; "
                f"available matches: {list(selected)}, numeric fallback is intentionally disabled: {numeric[:3]}"
            )
        return selected

    @staticmethod
    def _percentile(values: pd.Series, reference: np.ndarray) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        percentiles = np.full(len(numeric), 0.5, dtype=float)
        observed = np.isfinite(numeric)
        if len(reference):
            percentiles[observed] = np.searchsorted(
                reference, numeric[observed], side="right"
            ) / len(reference)
        return percentiles

    def fit(self, X: pd.DataFrame) -> "SyntheticAcceptancePolicy":
        self.signs_ = self._component_contract(X.columns)
        self.sorted_reference_ = {}
        for feature in self.signs_:
            values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
            self.sorted_reference_[feature] = np.sort(values[np.isfinite(values)])
        risk = self.risk_index(X)
        low, high = -20.0, 20.0
        for _ in range(100):
            midpoint = (low + high) / 2.0
            rate = expit(midpoint - risk / self.temperature).mean()
            if rate < self.target_acceptance_rate:
                low = midpoint
            else:
                high = midpoint
        self.intercept_ = (low + high) / 2.0
        return self

    def risk_index(self, X: pd.DataFrame) -> np.ndarray:
        if not self.signs_:
            raise RuntimeError("Acceptance policy is not fitted")
        components = []
        for feature, sign in self.signs_.items():
            percentile = self._percentile(X[feature], self.sorted_reference_[feature])
            components.append(sign * (percentile - 0.5))
        risk = np.mean(np.vstack(components), axis=0)
        return risk - np.median(risk)

    def acceptance_probability(self, X: pd.DataFrame) -> np.ndarray:
        if self.intercept_ is None:
            raise RuntimeError("Acceptance policy is not fitted")
        return expit(self.intercept_ - self.risk_index(X) / self.temperature)

    def sample(self, X: pd.DataFrame, seed_offset: int = 0) -> np.ndarray:
        probability = self.acceptance_probability(X)
        generator = np.random.default_rng(self.random_seed + seed_offset)
        return generator.random(len(X)) < probability

    def metadata(self) -> dict[str, object]:
        return {
            "target_acceptance_rate": self.target_acceptance_rate,
            "temperature": self.temperature,
            "random_seed": self.random_seed,
            "features": self.signs_,
            "intercept": self.intercept_,
            "target_used": False,
        }


@dataclass(frozen=True)
class ParcelingResult:
    expected_target: np.ndarray
    score_band: np.ndarray
    table: pd.DataFrame


def parcel_rejects(
    scores: Iterable[float],
    observed_target: Iterable[float],
    accepted: Iterable[bool],
    bad_odds_multiplier: float,
    bands: int = 10,
) -> ParcelingResult:
    """Assign rejected bad mass from accepted bad odds within frozen score bands."""
    score = np.asarray(list(scores), dtype=float)
    target = np.asarray(list(observed_target), dtype=float)
    accepted_mask = np.asarray(list(accepted), dtype=bool)
    if np.isnan(target[accepted_mask]).any():
        raise ValueError("Accepted applicants must have observed outcomes")
    if np.isfinite(target[~accepted_mask]).any():
        raise ValueError("Rejected outcomes must be hidden before parcelling")
    if bad_odds_multiplier <= 0:
        raise ValueError("Bad-odds multiplier must be positive")
    quantiles = np.linspace(0, 1, bands + 1)
    inner = np.unique(np.quantile(score[accepted_mask], quantiles[1:-1]))
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    band = pd.cut(score, bins=edges, labels=False, include_lowest=True).astype(int)
    rows: list[dict[str, float | int]] = []
    inferred_rate: dict[int, float] = {}
    global_rate = float(np.nanmean(target[accepted_mask]))
    for band_id in range(len(edges) - 1):
        accepted_in_band = accepted_mask & (band == band_id)
        accepted_bad_rate = (
            float(np.nanmean(target[accepted_in_band])) if accepted_in_band.any() else global_rate
        )
        numerator = bad_odds_multiplier * accepted_bad_rate
        rejected_bad_rate = numerator / (
            1.0 - accepted_bad_rate + numerator
        ) if accepted_bad_rate < 1 else 1.0
        inferred_rate[band_id] = float(np.clip(rejected_bad_rate, 0.0, 1.0))
        rows.append(
            {
                "score_band": band_id,
                "lower_score": float(edges[band_id]),
                "upper_score": float(edges[band_id + 1]),
                "accepted_count": int(accepted_in_band.sum()),
                "rejected_count": int(((~accepted_mask) & (band == band_id)).sum()),
                "accepted_bad_rate": accepted_bad_rate,
                "bad_odds_multiplier": bad_odds_multiplier,
                "inferred_rejected_bad_rate": inferred_rate[band_id],
            }
        )
    expected = target.copy()
    expected[~accepted_mask] = np.asarray([inferred_rate[int(value)] for value in band[~accepted_mask]])
    return ParcelingResult(expected, band, pd.DataFrame(rows))

