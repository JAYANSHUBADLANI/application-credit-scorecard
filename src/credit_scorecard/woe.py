"""Monotonic coarse classing, WOE transformation, and IV selection."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


MISSING_BIN = "MISSING"
OTHER_BIN = "OTHER"
RARE_LEVEL = "__RARE__"


@dataclass
class FeatureBins:
    feature: str
    kind: str
    edges: list[float] = field(default_factory=list)
    category_to_bin: dict[str, str] = field(default_factory=dict)
    woe_by_bin: dict[str, float] = field(default_factory=dict)
    iv: float = 0.0
    table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _normalize_category(values: pd.Series) -> pd.Series:
    result = values.astype("string")
    return result.where(result.notna(), "__MISSING__")


def _weighted_bin_table(
    bins: pd.Series,
    y: np.ndarray,
    weights: np.ndarray,
    smoothing: float,
    ordered_bins: list[str],
) -> tuple[pd.DataFrame, dict[str, float], float]:
    raw = pd.DataFrame(
        {
            "bin": bins.astype(str).to_numpy(),
            "bad": y * weights,
            "good": (1.0 - y) * weights,
            "weight": weights,
        }
    )
    grouped = raw.groupby("bin", observed=True).agg(
        count=("weight", "sum"),
        bad=("bad", "sum"),
        good=("good", "sum"),
    )
    grouped = grouped.reindex(ordered_bins, fill_value=0.0)
    total_bad = float(grouped["bad"].sum())
    total_good = float(grouped["good"].sum())
    active = grouped["count"].gt(0)
    count_bins = int(active.sum())
    if total_bad <= 0 or total_good <= 0:
        raise ValueError("WOE requires positive weighted good and bad counts")
    grouped["bad_rate"] = grouped["bad"] / grouped["count"].replace(0, np.nan)
    grouped["bad_distribution"] = 0.0
    grouped["good_distribution"] = 0.0
    grouped.loc[active, "bad_distribution"] = (
        grouped.loc[active, "bad"] + smoothing
    ) / (total_bad + smoothing * count_bins)
    grouped.loc[active, "good_distribution"] = (
        grouped.loc[active, "good"] + smoothing
    ) / (total_good + smoothing * count_bins)
    grouped["woe"] = 0.0
    grouped.loc[active, "woe"] = np.log(
        grouped.loc[active, "good_distribution"]
        / grouped.loc[active, "bad_distribution"]
    )
    grouped["iv_contribution"] = (
        grouped["good_distribution"] - grouped["bad_distribution"]
    ) * grouped["woe"]
    iv = float(grouped["iv_contribution"].sum())
    grouped.index.name = "bin"
    table = grouped.reset_index()
    woe = dict(zip(table["bin"], table["woe"], strict=True))
    return table, woe, iv


def _summarize_numeric(
    values: pd.Series,
    y: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> pd.DataFrame:
    nonmissing = values.notna().to_numpy()
    if not nonmissing.any():
        return pd.DataFrame(columns=["code", "count", "bad", "bad_rate"])
    codes = pd.cut(values[nonmissing].astype(float), bins=edges, labels=False, include_lowest=True)
    frame = pd.DataFrame(
        {
            "code": codes.astype(int).to_numpy(),
            "weight": weights[nonmissing],
            "bad": y[nonmissing] * weights[nonmissing],
        }
    )
    summary = frame.groupby("code", observed=True).agg(count=("weight", "sum"), bad=("bad", "sum"))
    summary = summary.reindex(range(len(edges) - 1), fill_value=0.0).reset_index()
    summary["bad_rate"] = summary["bad"] / summary["count"].replace(0, np.nan)
    return summary


def _merge_small_bins(
    values: pd.Series,
    y: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    minimum_count: float,
) -> np.ndarray:
    edges = edges.copy()
    while len(edges) > 3:
        summary = _summarize_numeric(values, y, weights, edges)
        small = summary.index[summary["count"].lt(minimum_count)].tolist()
        if not small:
            break
        index = small[0]
        rates = summary["bad_rate"].fillna(summary["bad_rate"].mean()).to_numpy()
        if index == 0:
            remove_at = 1
        elif index == len(summary) - 1:
            remove_at = index
        else:
            left_gap = abs(rates[index] - rates[index - 1])
            right_gap = abs(rates[index] - rates[index + 1])
            remove_at = index if left_gap <= right_gap else index + 1
        edges = np.delete(edges, remove_at)
    return edges


def _pava_blocks(rates: np.ndarray, weights: np.ndarray, increasing: bool) -> list[tuple[int, int]]:
    blocks: list[dict[str, float | int]] = []
    direction = 1.0 if increasing else -1.0
    for index, (rate, weight) in enumerate(zip(rates, weights, strict=True)):
        blocks.append(
            {
                "start": index,
                "end": index,
                "weight": float(max(weight, 1e-12)),
                "value": float(rate) * direction,
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            total = float(left["weight"]) + float(right["weight"])
            value = (
                float(left["value"]) * float(left["weight"])
                + float(right["value"]) * float(right["weight"])
            ) / total
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": total,
                    "value": value,
                }
            )
    return [(int(block["start"]), int(block["end"])) for block in blocks]


def _block_sse(rates: np.ndarray, weights: np.ndarray, blocks: list[tuple[int, int]]) -> float:
    total = 0.0
    for start, end in blocks:
        local_weights = weights[start : end + 1]
        local_rates = rates[start : end + 1]
        average = np.average(local_rates, weights=np.maximum(local_weights, 1e-12))
        total += float(np.sum(local_weights * np.square(local_rates - average)))
    return total


def _enforce_monotonic_edges(
    values: pd.Series,
    y: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    summary = _summarize_numeric(values, y, weights, edges)
    if len(summary) <= 2:
        return edges
    rates = summary["bad_rate"].interpolate(limit_direction="both").fillna(0.0).to_numpy()
    counts = summary["count"].to_numpy()
    increasing = _pava_blocks(rates, counts, True)
    decreasing = _pava_blocks(rates, counts, False)
    blocks = increasing if _block_sse(rates, counts, increasing) <= _block_sse(rates, counts, decreasing) else decreasing
    new_edges = [float(edges[0])]
    for _, end in blocks:
        new_edges.append(float(edges[end + 1]))
    return np.asarray(new_edges, dtype=float)


class WOETransformer:
    """Fit training-only WOE bins with an explicit event convention."""

    def __init__(
        self,
        maximum_prebins: int = 20,
        maximum_categorical_bins: int = 8,
        minimum_bin_fraction: float = 0.03,
        smoothing: float = 0.5,
    ) -> None:
        self.maximum_prebins = maximum_prebins
        self.maximum_categorical_bins = maximum_categorical_bins
        self.minimum_bin_fraction = minimum_bin_fraction
        self.smoothing = smoothing
        self.features_: dict[str, FeatureBins] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Iterable[float],
        sample_weight: Iterable[float] | None = None,
    ) -> "WOETransformer":
        target = np.asarray(list(y), dtype=float)
        weights = (
            np.ones(len(target), dtype=float)
            if sample_weight is None
            else np.asarray(list(sample_weight), dtype=float)
        )
        if len(X) != len(target) or len(weights) != len(target):
            raise ValueError("X, y, and sample_weight must have equal lengths")
        if np.any((target < 0) | (target > 1)):
            raise ValueError("WOE event values must be between zero and one")
        self.features_ = {}
        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]) and not pd.api.types.is_bool_dtype(X[feature]):
                definition = self._fit_numeric(feature, X[feature], target, weights)
            else:
                definition = self._fit_categorical(feature, X[feature], target, weights)
            self.features_[feature] = definition
        return self

    def _fit_numeric(
        self,
        feature: str,
        values: pd.Series,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> FeatureBins:
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonmissing = numeric.dropna()
        if nonmissing.nunique() <= 1:
            edges = np.asarray([-np.inf, np.inf])
        else:
            quantiles = np.linspace(0, 1, self.maximum_prebins + 1)
            inner = np.unique(nonmissing.quantile(quantiles[1:-1]).to_numpy(dtype=float))
            edges = np.concatenate(([-np.inf], inner, [np.inf]))
        minimum_count = max(float(weights.sum()) * self.minimum_bin_fraction, 1.0)
        edges = _merge_small_bins(numeric, y, weights, edges, minimum_count)
        edges = _enforce_monotonic_edges(numeric, y, weights, edges)
        codes = pd.cut(numeric, bins=edges, labels=False, include_lowest=True)
        bin_ids = codes.map(lambda value: MISSING_BIN if pd.isna(value) else f"N{int(value)}")
        ordered = [f"N{index}" for index in range(len(edges) - 1)] + [MISSING_BIN]
        table, woe, iv = _weighted_bin_table(bin_ids, y, weights, self.smoothing, ordered)
        table["lower_bound"] = table["bin"].map(
            {f"N{i}": edges[i] for i in range(len(edges) - 1)}
        )
        table["upper_bound"] = table["bin"].map(
            {f"N{i}": edges[i + 1] for i in range(len(edges) - 1)}
        )
        return FeatureBins(feature, "numeric", edges.tolist(), {}, woe, iv, table)

    def _fit_categorical(
        self,
        feature: str,
        values: pd.Series,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> FeatureBins:
        normalized = _normalize_category(values)
        missing_mask = normalized.eq("__MISSING__")
        frame = pd.DataFrame(
            {
                "category": normalized[~missing_mask].to_numpy(),
                "weight": weights[~missing_mask.to_numpy()],
                "bad": (y * weights)[~missing_mask.to_numpy()],
            }
        )
        summary = frame.groupby("category", observed=True).agg(count=("weight", "sum"), bad=("bad", "sum"))
        minimum_count = max(float(weights.sum()) * self.minimum_bin_fraction, 1.0)
        rare = set(summary.index[summary["count"].lt(minimum_count)].astype(str))
        mapped_nonmissing = normalized[~missing_mask].map(
            lambda value: RARE_LEVEL if str(value) in rare else str(value)
        )
        collapsed = pd.DataFrame(
            {
                "category": mapped_nonmissing.to_numpy(),
                "weight": weights[~missing_mask.to_numpy()],
                "bad": (y * weights)[~missing_mask.to_numpy()],
            }
        )
        summary = collapsed.groupby("category", observed=True).agg(count=("weight", "sum"), bad=("bad", "sum"))
        summary["bad_rate"] = summary["bad"] / summary["count"].replace(0, np.nan)
        summary = summary.sort_values(["bad_rate", "count"], ascending=[True, False])

        groups: list[list[str]] = []
        target_count = max(float(summary["count"].sum()) / self.maximum_categorical_bins, 1.0)
        current: list[str] = []
        current_count = 0.0
        for category, row in summary.iterrows():
            current.append(str(category))
            current_count += float(row["count"])
            remaining_categories = len(summary) - sum(len(group) for group in groups) - len(current)
            if (
                current_count >= target_count
                and len(groups) < self.maximum_categorical_bins - 1
                and remaining_categories > 0
            ):
                groups.append(current)
                current = []
                current_count = 0.0
        if current:
            groups.append(current)

        category_to_bin: dict[str, str] = {}
        for index, group in enumerate(groups):
            for category in group:
                category_to_bin[category] = f"C{index}"
        for category in normalized[~missing_mask].astype(str).unique():
            collapsed_category = RARE_LEVEL if category in rare else category
            category_to_bin[category] = category_to_bin[collapsed_category]
        category_to_bin["__MISSING__"] = MISSING_BIN
        bin_ids = normalized.map(lambda value: category_to_bin.get(str(value), OTHER_BIN))
        ordered = [f"C{index}" for index in range(len(groups))]
        for special in (MISSING_BIN, OTHER_BIN):
            if special not in ordered:
                ordered.append(special)
        table, woe, iv = _weighted_bin_table(bin_ids, y, weights, self.smoothing, ordered)
        members: dict[str, str] = {}
        for category, bin_id in category_to_bin.items():
            members[bin_id] = f"{members.get(bin_id, '')}|{category}".strip("|")
        table["categories"] = table["bin"].map(members).fillna("")
        return FeatureBins(feature, "categorical", [], category_to_bin, woe, iv, table)

    def assign_bins(self, X: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        selected = features or list(self.features_)
        output: dict[str, pd.Series] = {}
        for feature in selected:
            if feature not in X:
                raise KeyError(f"Feature {feature!r} is absent from input data")
            definition = self.features_[feature]
            if definition.kind == "numeric":
                numeric = pd.to_numeric(X[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
                codes = pd.cut(
                    numeric,
                    bins=np.asarray(definition.edges),
                    labels=False,
                    include_lowest=True,
                )
                output[feature] = codes.map(
                    lambda value: MISSING_BIN if pd.isna(value) else f"N{int(value)}"
                )
            else:
                normalized = _normalize_category(X[feature])
                output[feature] = normalized.map(
                    lambda value: definition.category_to_bin.get(str(value), OTHER_BIN)
                )
        return pd.DataFrame(output, index=X.index)

    def transform(self, X: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        selected = features or list(self.features_)
        assigned = self.assign_bins(X, selected)
        transformed = {
            feature: assigned[feature].map(self.features_[feature].woe_by_bin).fillna(0.0).astype(float)
            for feature in selected
        }
        return pd.DataFrame(transformed, index=X.index)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Iterable[float],
        sample_weight: Iterable[float] | None = None,
    ) -> pd.DataFrame:
        return self.fit(X, y, sample_weight).transform(X)

    def iv_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"feature": feature, "kind": definition.kind, "iv": definition.iv}
                for feature, definition in self.features_.items()
            ]
        ).sort_values("iv", ascending=False, ignore_index=True)

    def bin_table(self) -> pd.DataFrame:
        tables = []
        for feature, definition in self.features_.items():
            table = definition.table.copy()
            table.insert(0, "feature", feature)
            table.insert(1, "kind", definition.kind)
            tables.append(table)
        return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()

    def reestimate_woe(
        self,
        X: pd.DataFrame,
        expected_y: Iterable[float],
        sample_weight: Iterable[float] | None = None,
        features: list[str] | None = None,
    ) -> "WOETransformer":
        """Recalculate WOE values on frozen cutpoints using fractional event mass."""
        result = copy.deepcopy(self)
        selected = features or list(result.features_)
        target = np.asarray(list(expected_y), dtype=float)
        weights = (
            np.ones(len(target), dtype=float)
            if sample_weight is None
            else np.asarray(list(sample_weight), dtype=float)
        )
        assigned = result.assign_bins(X, selected)
        for feature in selected:
            definition = result.features_[feature]
            old_table = definition.table.copy()
            ordered = list(definition.woe_by_bin)
            table, woe, iv = _weighted_bin_table(
                assigned[feature], target, weights, result.smoothing, ordered
            )
            descriptor_columns = [
                column
                for column in ("lower_bound", "upper_bound", "categories")
                if column in old_table
            ]
            if descriptor_columns:
                table = table.merge(
                    old_table[["bin", *descriptor_columns]],
                    on="bin",
                    how="left",
                )
            definition.woe_by_bin = woe
            definition.iv = iv
            definition.table = table
        return result


def eligible_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    relational: bool = True,
) -> list[str]:
    """Apply explicit data-quality, sensitivity, and cardinality gates."""
    settings = config["features"]
    excluded = set(settings["exclude_columns"])
    patterns = [re.compile(pattern) for pattern in settings.get("exclude_name_patterns", [])]
    relational_prefixes = ("BUREAU_", "PREV_", "INST_", "CC_", "POS_")
    candidates: list[str] = []
    for column in frame.columns:
        if column in excluded or any(pattern.search(column) for pattern in patterns):
            continue
        if not relational and column.startswith(relational_prefixes):
            continue
        series = frame[column]
        if float(series.notna().mean()) < float(settings["minimum_non_null_fraction"]):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            if int(series.nunique(dropna=True)) > int(settings["maximum_unique_categories"]):
                continue
        candidates.append(column)
    return candidates


def select_features(
    transformer: WOETransformer,
    transformed_development: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """Select by IV, then remove highly correlated lower-IV characteristics."""
    settings = config["binning"]
    iv_table = transformer.iv_table()
    iv_table["status"] = "outside_iv_range"
    allowed = iv_table[
        iv_table["iv"].between(
            float(settings["iv_minimum"]),
            float(settings["iv_maximum"]),
            inclusive="both",
        )
    ]
    selected: list[str] = []
    maximum_correlation = float(settings["correlation_maximum"])
    maximum_features = int(settings["maximum_features"])
    for feature in allowed["feature"]:
        if not selected:
            selected.append(feature)
        else:
            correlations = transformed_development[selected].corrwith(
                transformed_development[feature]
            ).abs()
            if not correlations.gt(maximum_correlation).any():
                selected.append(feature)
        if len(selected) >= maximum_features:
            break
    iv_table.loc[iv_table["feature"].isin(allowed["feature"]), "status"] = "iv_eligible"
    iv_table.loc[iv_table["feature"].isin(selected), "status"] = "selected"
    return selected, iv_table
