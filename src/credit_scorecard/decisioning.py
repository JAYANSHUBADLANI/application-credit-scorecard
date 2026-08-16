"""Profit cutoffs, score bands, and equal-volume swap-set analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def account_economics(
    probability_bad: Iterable[float],
    exposure: Iterable[float],
    settings: dict[str, Any],
) -> pd.DataFrame:
    probability = np.asarray(list(probability_bad), dtype=float)
    exposure_array = np.asarray(list(exposure), dtype=float)
    if len(probability) != len(exposure_array):
        raise ValueError("Probability and exposure lengths differ")
    ead = exposure_array * float(settings.get("ead_factor", 1.0))
    lgd = float(settings["loss_given_default"])
    good_revenue = ead * float(settings["net_margin_rate"])
    bad_revenue = ead * float(settings["pre_default_bad_margin_rate"])
    fixed_cost = float(settings["acquisition_cost"]) + float(settings["operating_cost"])
    profit_good = good_revenue - fixed_cost
    profit_bad = bad_revenue - fixed_cost - ead * lgd
    expected_revenue = (1.0 - probability) * good_revenue + probability * bad_revenue
    expected_loss = probability * ead * lgd
    expected_cost = np.full(len(probability), fixed_cost)
    expected_profit = expected_revenue - expected_loss - expected_cost
    return pd.DataFrame(
        {
            "probability_bad": probability,
            "exposure": ead,
            "profit_good": profit_good,
            "profit_bad": profit_bad,
            "expected_revenue": expected_revenue,
            "expected_loss": expected_loss,
            "expected_cost": expected_cost,
            "expected_profit": expected_profit,
        }
    )


def cutoff_curve(
    scores: Iterable[float],
    probability_bad: Iterable[float],
    exposure: Iterable[float],
    settings: dict[str, Any],
    observed_target: Iterable[int] | None = None,
) -> pd.DataFrame:
    score = np.asarray(list(scores), dtype=float)
    probability = np.asarray(list(probability_bad), dtype=float)
    economics = account_economics(probability, exposure, settings)
    target = None if observed_target is None else np.asarray(list(observed_target), dtype=int)
    grid_size = int(settings["cutoff_grid_size"])
    reject_all_cutoff = np.nextafter(float(np.max(score)), np.inf)
    cutoffs = np.unique(
        np.append(np.quantile(score, np.linspace(0, 1, grid_size)), reject_all_cutoff)
    )
    rows = []
    for cutoff in cutoffs:
        approved = score >= cutoff
        approved_count = int(approved.sum())
        if not approved_count:
            row = {
                "score_cutoff": float(cutoff),
                "approved_count": 0,
                "approval_rate": 0.0,
                "average_predicted_bad_rate": np.nan,
                "expected_defaults": 0.0,
                "expected_revenue": 0.0,
                "expected_loss": 0.0,
                "expected_cost": 0.0,
                "expected_profit": 0.0,
            }
            if target is not None:
                row["observed_bad_rate"] = np.nan
                row["realized_profit"] = 0.0
            rows.append(row)
            continue
        row = {
            "score_cutoff": float(cutoff),
            "approved_count": approved_count,
            "approval_rate": float(approved.mean()),
            "average_predicted_bad_rate": float(probability[approved].mean()),
            "expected_defaults": float(probability[approved].sum()),
            "expected_revenue": float(economics.loc[approved, "expected_revenue"].sum()),
            "expected_loss": float(economics.loc[approved, "expected_loss"].sum()),
            "expected_cost": float(economics.loc[approved, "expected_cost"].sum()),
            "expected_profit": float(economics.loc[approved, "expected_profit"].sum()),
        }
        if target is not None:
            realized_profit = np.where(
                target[approved].eq(1) if isinstance(target, pd.Series) else target[approved] == 1,
                economics.loc[approved, "profit_bad"].to_numpy(),
                economics.loc[approved, "profit_good"].to_numpy(),
            )
            row["observed_bad_rate"] = float(target[approved].mean())
            row["realized_profit"] = float(realized_profit.sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_cutoff", ignore_index=True)


def optimal_cutoff(curve: pd.DataFrame) -> pd.Series:
    if curve.empty:
        raise ValueError("Cannot choose a cutoff from an empty curve")
    return curve.loc[curve["expected_profit"].idxmax()]


def risk_appetite_options(
    curve: pd.DataFrame,
    maximum_bad_rates: Iterable[float],
) -> pd.DataFrame:
    """Return the highest validation approval rate satisfying each risk limit."""
    if "observed_bad_rate" not in curve:
        raise ValueError("Risk appetite options require observed validation outcomes")
    rows = []
    for limit in maximum_bad_rates:
        eligible = curve[curve["observed_bad_rate"] <= float(limit)]
        if eligible.empty:
            choice = curve.loc[curve["approved_count"].idxmin()]
        else:
            choice = eligible.loc[eligible["approval_rate"].idxmax()]
        rows.append(
            {
                "maximum_observed_bad_rate": float(limit),
                "feasible": True,
                "score_cutoff": float(choice["score_cutoff"]),
                "approval_rate": float(choice["approval_rate"]),
                "observed_bad_rate": float(choice["observed_bad_rate"]),
                "expected_profit": float(choice["expected_profit"]),
            }
        )
    return pd.DataFrame(rows)


def apply_locked_cutoff(
    scores: Iterable[float],
    cutoff: float,
    manual_review_width_points: float,
) -> np.ndarray:
    score = np.asarray(list(scores), dtype=float)
    lower = cutoff - manual_review_width_points / 2.0
    upper = cutoff + manual_review_width_points / 2.0
    return np.where(score >= upper, "auto_approve", np.where(score >= lower, "manual_review", "auto_decline"))


@dataclass(frozen=True)
class SwapSetResult:
    summary: pd.DataFrame
    applicants: pd.DataFrame


def swap_set_analysis(
    applicant_ids: Iterable[int],
    baseline_scores: Iterable[float],
    challenger_scores: Iterable[float],
    target: Iterable[int],
    exposure: Iterable[float],
    approval_rate: float,
    economics_settings: dict[str, Any],
) -> SwapSetResult:
    """Compare scorecards at exactly the same deterministic approval count."""
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.asarray(list(applicant_ids)),
            "baseline_score": np.asarray(list(baseline_scores), dtype=float),
            "challenger_score": np.asarray(list(challenger_scores), dtype=float),
            "target": np.asarray(list(target), dtype=int),
            "exposure": np.asarray(list(exposure), dtype=float),
        }
    )
    approved_count = int(np.floor(len(frame) * approval_rate))
    approved_count = min(max(approved_count, 0), len(frame))
    baseline_order = frame.sort_values(
        ["baseline_score", "SK_ID_CURR"], ascending=[False, True]
    ).index[:approved_count]
    challenger_order = frame.sort_values(
        ["challenger_score", "SK_ID_CURR"], ascending=[False, True]
    ).index[:approved_count]
    frame["baseline_approved"] = frame.index.isin(baseline_order)
    frame["challenger_approved"] = frame.index.isin(challenger_order)
    frame["segment"] = np.select(
        [
            frame["baseline_approved"] & frame["challenger_approved"],
            ~frame["baseline_approved"] & frame["challenger_approved"],
            frame["baseline_approved"] & ~frame["challenger_approved"],
        ],
        ["both_approved", "swap_in", "swap_out"],
        default="both_declined",
    )
    economics = account_economics(
        np.zeros(len(frame)), frame["exposure"], economics_settings
    )
    frame["realized_value"] = np.where(
        frame["target"].eq(1), economics["profit_bad"], economics["profit_good"]
    )
    summary = frame.groupby("segment", observed=True).agg(
        applicants=("SK_ID_CURR", "size"),
        bads=("target", "sum"),
        bad_rate=("target", "mean"),
        exposure=("exposure", "sum"),
        realized_value=("realized_value", "sum"),
    ).reset_index()
    swap_in_count = int((frame["segment"] == "swap_in").sum())
    swap_out_count = int((frame["segment"] == "swap_out").sum())
    if swap_in_count != swap_out_count:
        raise AssertionError("Equal-volume swap set has unequal swap-in and swap-out counts")
    return SwapSetResult(summary, frame)
