"""Machine-generated exploratory analysis for the verified applicant matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_scorecard.config import project_path


def _binned_bad_rates(frame: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[feature], errors="coerce")
    observed = values.notna()
    rows = []
    if observed.sum() and values[observed].nunique() > 1:
        membership = pd.qcut(values[observed], q=10, duplicates="drop")
        local = pd.DataFrame(
            {"bin": membership.astype(str), "target": frame.loc[observed, target].to_numpy()}
        )
        grouped = local.groupby("bin", observed=True, sort=False)["target"].agg(["size", "mean"])
        for bin_id, row in grouped.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "bin": str(bin_id),
                    "applicants": int(row["size"]),
                    "bad_rate": float(row["mean"]),
                }
            )
    missing = ~observed
    if missing.any():
        rows.append(
            {
                "feature": feature,
                "bin": "MISSING",
                "applicants": int(missing.sum()),
                "bad_rate": float(frame.loc[missing, target].mean()),
            }
        )
    return pd.DataFrame(rows)


def create_eda(config: dict[str, Any]) -> dict[str, Any]:
    """Create target, missingness, and key-driver evidence from real processed data."""
    processed = project_path(config, "processed_data") / "applicants_labeled.parquet"
    if not processed.is_file():
        raise FileNotFoundError("Labeled applicant matrix is absent. Run build-features first.")
    frame = pd.read_parquet(processed)
    target = str(config["data"]["target"])
    target_table = frame[target].value_counts(dropna=False).sort_index().rename_axis("target").reset_index(name="applicants")
    target_table["share"] = target_table["applicants"] / len(frame)

    missingness = pd.DataFrame(
        {
            "feature": frame.columns,
            "null_count": [int(frame[column].isna().sum()) for column in frame.columns],
            "null_fraction": [float(frame[column].isna().mean()) for column in frame.columns],
        }
    ).sort_values(["null_fraction", "feature"], ascending=[False, True], ignore_index=True)

    priority = [
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "APP_CREDIT_INCOME_RATIO",
        "BUREAU_BALANCE_DELINQUENCY_RATE_MAX",
        "PREV_REFUSED_RATE",
        "INST_LATE_RATE",
        "CC_UTILIZATION_MEAN",
        "POS_DELINQUENCY_RATE",
    ]
    key_features = [feature for feature in priority if feature in frame][:6]
    driver_tables = [_binned_bad_rates(frame, feature, target) for feature in key_features]
    driver_table = (
        pd.concat(driver_tables, ignore_index=True) if driver_tables else pd.DataFrame()
    )

    tables = project_path(config, "artifacts") / "tables"
    figures = project_path(config, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    target_table.to_csv(tables / "eda_target_balance.csv", index=False)
    missingness.to_csv(tables / "eda_missingness.csv", index=False)
    driver_table.to_csv(tables / "eda_key_driver_bad_rates.csv", index=False)

    os.environ.setdefault("MPLCONFIGDIR", str(figures / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(figures / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=target_table, x="target", y="share", color="#3267a8", ax=ax)
    ax.set_title("Application target balance")
    ax.set_ylabel("Applicant share")
    fig.tight_layout()
    fig.savefig(figures / "eda_target_balance.png", dpi=180)
    plt.close(fig)

    top_missing = missingness[~missingness["feature"].isin([target, "SK_ID_CURR"])].head(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(data=top_missing, y="feature", x="null_fraction", color="#7b9e4b", ax=ax)
    ax.set_title("Highest applicant feature missingness")
    ax.set_xlabel("Null fraction")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures / "eda_missingness.png", dpi=180)
    plt.close(fig)

    if not driver_table.empty:
        plot_features = key_features[:4]
        fig, axes = plt.subplots(len(plot_features), 1, figsize=(10, 3.2 * len(plot_features)))
        axes_array = np.atleast_1d(axes)
        for ax, feature in zip(axes_array, plot_features, strict=True):
            local = driver_table[driver_table["feature"] == feature]
            sns.barplot(data=local, x="bin", y="bad_rate", color="#b55d4c", ax=ax)
            ax.set_title(feature)
            ax.tick_params(axis="x", rotation=35)
            ax.set_xlabel("")
        fig.suptitle("Observed bad rates across selected application-time drivers", y=1.01)
        fig.tight_layout()
        fig.savefig(figures / "eda_key_drivers.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "applicants": int(len(frame)),
        "bad_rate": float(frame[target].mean()),
        "features": int(len(frame.columns) - 2),
        "key_driver_tables": key_features,
    }
    with (project_path(config, "artifacts") / "eda_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary

