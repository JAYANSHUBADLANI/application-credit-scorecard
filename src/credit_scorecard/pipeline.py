"""End-to-end scorecard development and validation pipeline."""

from __future__ import annotations

import json
import os
import platform
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_scorecard.config import ensure_output_directories, project_path
from credit_scorecard.data import DatasetSplit, make_split, validate_split
from credit_scorecard.decisioning import (
    account_economics,
    apply_locked_cutoff,
    cutoff_curve,
    optimal_cutoff,
    risk_appetite_options,
    swap_set_analysis,
)
from credit_scorecard.features import save_feature_matrix
from credit_scorecard.metrics import (
    bootstrap_interval,
    characteristic_stability_index,
    gini_coefficient,
    ks_statistic,
    population_stability_index,
    validation_metrics,
)
from credit_scorecard.model import (
    ScorecardModel,
    fit_fractional_scorecard,
    fit_scorecard,
    verify_scaling,
)
from credit_scorecard.reject_inference import SyntheticAcceptancePolicy, parcel_rejects
from credit_scorecard.woe import WOETransformer, eligible_features, select_features


@dataclass
class ModelBundle:
    split: DatasetSplit
    acceptance_policy: SyntheticAcceptancePolicy
    application_only: ScorecardModel
    accepted_only: ScorecardModel
    parcelled: ScorecardModel
    oracle: ScorecardModel
    parcel_sensitivity: dict[str, ScorecardModel]
    accepted_development_ids: np.ndarray
    selected_features: list[str]
    config_snapshot: dict[str, Any]


def _load_feature_matrices(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = project_path(config, "processed_data")
    labeled_path = processed / "applicants_labeled.parquet"
    unlabeled_path = processed / "applicants_unlabeled.parquet"
    if not labeled_path.is_file() or not unlabeled_path.is_file():
        raise FileNotFoundError("Feature matrices are absent. Run build-features first.")
    return pd.read_parquet(labeled_path), pd.read_parquet(unlabeled_path)


def _load_labeled_matrix(config: dict[str, Any]) -> pd.DataFrame:
    path = project_path(config, "processed_data") / "applicants_labeled.parquet"
    if not path.is_file():
        raise FileNotFoundError("Labeled feature matrix is absent. Run build-features first.")
    return pd.read_parquet(path)


def _subset_by_ids(frame: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    indexed = frame.set_index("SK_ID_CURR", drop=False)
    return indexed.loc[ids].reset_index(drop=True)


def _new_transformer(config: dict[str, Any]) -> WOETransformer:
    settings = config["binning"]
    return WOETransformer(
        maximum_prebins=int(settings["maximum_prebins"]),
        maximum_categorical_bins=int(settings["maximum_categorical_bins"]),
        minimum_bin_fraction=float(settings["minimum_bin_fraction"]),
        smoothing=float(settings["smoothing"]),
    )


def _fit_model_specification(
    development: pd.DataFrame,
    accepted: np.ndarray,
    config: dict[str, Any],
    relational: bool,
    name: str,
) -> tuple[ScorecardModel, pd.DataFrame, pd.DataFrame]:
    target_name = config["data"]["target"]
    accepted_frame = development.loc[accepted].reset_index(drop=True)
    candidates = eligible_features(accepted_frame, config, relational=relational)
    if not candidates:
        raise ValueError(f"No eligible predictors for model {name}")
    target = accepted_frame[target_name].astype(int)
    transformer = _new_transformer(config).fit(accepted_frame[candidates], target)
    transformed = transformer.transform(accepted_frame[candidates])
    selected, iv_table = select_features(transformer, transformed, config)
    if not selected:
        raise ValueError(f"WOE and IV selection retained no predictors for model {name}")
    complete_bin_table = transformer.bin_table()
    removed_for_sign: list[str] = []
    while True:
        model = fit_scorecard(
            accepted_frame[selected],
            target,
            transformer,
            selected,
            config["model"],
            name,
        )
        tolerance = float(config["model"].get("coefficient_positive_tolerance", 1e-8))
        positive = pd.Series(model.estimator.coef_[0], index=selected)
        positive = positive[positive > tolerance]
        if (
            positive.empty
            or not bool(config["model"].get("enforce_expected_coefficient_sign", True))
            or len(selected) == 1
        ):
            break
        remove_feature = str(positive.idxmax())
        selected.remove(remove_feature)
        removed_for_sign.append(remove_feature)
    if removed_for_sign:
        iv_table.loc[
            iv_table["feature"].isin(removed_for_sign), "status"
        ] = "coefficient_sign_removed"
    transformer.features_ = {
        feature: transformer.features_[feature] for feature in selected
    }
    verify_scaling(model)
    return model, iv_table, complete_bin_table


def _fit_reject_inference_family(
    development: pd.DataFrame,
    accepted: np.ndarray,
    accepted_model: ScorecardModel,
    config: dict[str, Any],
    target_name: str,
) -> tuple[ScorecardModel, ScorecardModel, dict[str, ScorecardModel], list[pd.DataFrame]]:
    """Fit parcel sensitivity models and the oracle on one frozen specification."""
    acceptance_settings = config["synthetic_acceptance"]
    observed_target = np.where(
        accepted, development[target_name].to_numpy(dtype=float), np.nan
    )
    accepted_scores = accepted_model.score(development)
    primary_multiplier = float(acceptance_settings["parcel_bad_odds_multiplier"])
    sensitivity_values = {
        float(value) for value in acceptance_settings.get("parcel_bad_odds_sensitivity", [])
    }
    sensitivity_values.add(primary_multiplier)
    parcel_sensitivity: dict[str, ScorecardModel] = {}
    parcel_tables: list[pd.DataFrame] = []
    parcelled_model: ScorecardModel | None = None
    for multiplier in sorted(sensitivity_values):
        parcel = parcel_rejects(
            accepted_scores,
            observed_target,
            accepted,
            multiplier,
            int(acceptance_settings["parcel_bands"]),
        )
        parcel_transformer = accepted_model.transformer.reestimate_woe(
            development[accepted_model.selected_features],
            parcel.expected_target,
            features=accepted_model.selected_features,
        )
        model_name = (
            "parcelled"
            if np.isclose(multiplier, primary_multiplier)
            else f"parcel_lambda_{multiplier:g}"
        )
        sensitivity_model = fit_fractional_scorecard(
            development[accepted_model.selected_features],
            parcel.expected_target,
            parcel_transformer,
            accepted_model.selected_features,
            config["model"],
            model_name,
        )
        verify_scaling(sensitivity_model)
        parcel_sensitivity[f"{multiplier:g}"] = sensitivity_model
        parcel_table = parcel.table.copy()
        hidden_rejected_target = development[target_name].to_numpy(dtype=float)[~accepted]
        hidden_rejected_bands = parcel.score_band[~accepted]
        hidden_diagnostic = (
            pd.DataFrame(
                {
                    "score_band": hidden_rejected_bands,
                    "hidden_target": hidden_rejected_target,
                }
            )
            .groupby("score_band", observed=True)["hidden_target"]
            .agg(["count", "mean"])
            .rename(
                columns={
                    "count": "diagnostic_hidden_rejected_count",
                    "mean": "diagnostic_hidden_rejected_bad_rate",
                }
            )
            .reset_index()
        )
        parcel_table = parcel_table.merge(hidden_diagnostic, on="score_band", how="left")
        parcel_table["diagnostic_inference_error"] = (
            parcel_table["inferred_rejected_bad_rate"]
            - parcel_table["diagnostic_hidden_rejected_bad_rate"]
        )
        parcel_table["diagnostic_only_not_used_for_fitting"] = True
        parcel_table.insert(0, "sensitivity_multiplier", multiplier)
        parcel_tables.append(parcel_table)
        if np.isclose(multiplier, primary_multiplier):
            parcelled_model = sensitivity_model
    if parcelled_model is None:
        raise AssertionError("Primary parcel multiplier was not fitted")

    oracle_transformer = accepted_model.transformer.reestimate_woe(
        development[accepted_model.selected_features],
        development[target_name].to_numpy(dtype=float),
        features=accepted_model.selected_features,
    )
    oracle_model = fit_scorecard(
        development[accepted_model.selected_features],
        development[target_name].astype(int),
        oracle_transformer,
        accepted_model.selected_features,
        config["model"],
        "oracle",
    )
    verify_scaling(oracle_model)
    return parcelled_model, oracle_model, parcel_sensitivity, parcel_tables


def train_models(config: dict[str, Any]) -> ModelBundle:
    """Fit application-only, accepted-only, parcelled, and oracle scorecards."""
    ensure_output_directories(config)
    labeled = _load_labeled_matrix(config)
    target_name = config["data"]["target"]
    split_settings = config["split"]
    split = make_split(
        labeled["SK_ID_CURR"],
        labeled[target_name].astype(int),
        float(split_settings["validation_fraction"]),
        float(split_settings["test_fraction"]),
        int(config["project"]["random_seed"]),
    )
    validate_split(split)
    development = _subset_by_ids(labeled, split.development_ids)

    acceptance_settings = config["synthetic_acceptance"]
    policy = SyntheticAcceptancePolicy(
        target_acceptance_rate=float(acceptance_settings["target_acceptance_rate"]),
        temperature=float(acceptance_settings["temperature"]),
        random_seed=int(config["project"]["random_seed"]),
    ).fit(development)
    accepted = policy.sample(development)
    if np.unique(development.loc[accepted, target_name]).size < 2:
        raise ValueError("Synthetic accepted development sample does not contain both outcomes")

    application_model, application_iv, application_bins = _fit_model_specification(
        development, accepted, config, relational=False, name="application_only"
    )
    accepted_model, enriched_iv, enriched_bins = _fit_model_specification(
        development, accepted, config, relational=True, name="accepted_only"
    )

    while True:
        (
            parcelled_model,
            oracle_model,
            parcel_sensitivity,
            parcel_tables,
        ) = _fit_reject_inference_family(
            development,
            accepted,
            accepted_model,
            config,
            target_name,
        )
        tolerance = float(config["model"].get("coefficient_positive_tolerance", 1e-8))
        positive = pd.Series(
            parcelled_model.estimator.coef_[0],
            index=parcelled_model.selected_features,
        )
        positive = positive[positive > tolerance]
        if (
            positive.empty
            or not bool(config["model"].get("enforce_expected_coefficient_sign", True))
            or len(accepted_model.selected_features) == 1
        ):
            break
        remove_feature = str(positive.idxmax())
        retained = [
            feature
            for feature in accepted_model.selected_features
            if feature != remove_feature
        ]
        enriched_iv.loc[
            enriched_iv["feature"].eq(remove_feature), "status"
        ] = "post_parcel_sign_removed"
        accepted_model.transformer.features_.pop(remove_feature)
        accepted_model = fit_scorecard(
            development.loc[accepted, retained],
            development.loc[accepted, target_name].astype(int),
            accepted_model.transformer,
            retained,
            config["model"],
            "accepted_only",
        )
        verify_scaling(accepted_model)

    config_snapshot = deepcopy(config)
    config_snapshot.pop("_config_path", None)
    config_snapshot.pop("_project_root", None)
    bundle = ModelBundle(
        split,
        policy,
        application_model,
        accepted_model,
        parcelled_model,
        oracle_model,
        parcel_sensitivity,
        development.loc[accepted, "SK_ID_CURR"].to_numpy(),
        accepted_model.selected_features,
        config_snapshot,
    )

    artifacts = project_path(config, "artifacts")
    model_dir = artifacts / "models"
    table_dir = artifacts / "tables"
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model_bundle.joblib", compress=3)
    split.as_frame().to_parquet(artifacts / "split_manifest.parquet", index=False)
    application_iv.to_csv(table_dir / "application_only_iv.csv", index=False)
    enriched_iv.to_csv(table_dir / "enriched_iv.csv", index=False)
    application_bins.to_csv(table_dir / "application_only_binning.csv", index=False)
    enriched_bins.to_csv(table_dir / "accepted_only_binning.csv", index=False)
    parcelled_model.transformer.bin_table().to_csv(
        table_dir / "parcelled_binning.csv", index=False
    )
    pd.concat(parcel_tables, ignore_index=True).to_csv(
        table_dir / "parcel_assumptions.csv", index=False
    )
    for model in (application_model, accepted_model, parcelled_model, oracle_model):
        model.coefficient_table().to_csv(table_dir / f"{model.name}_coefficients.csv", index=False)
        model.points_table().to_csv(table_dir / f"{model.name}_points.csv", index=False)
    with (artifacts / "acceptance_policy.json").open("w", encoding="utf-8") as handle:
        json.dump(policy.metadata(), handle, indent=2)
    with (artifacts / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2)
    return bundle


def _load_bundle(config: dict[str, Any]) -> ModelBundle:
    path = project_path(config, "artifacts") / "models" / "model_bundle.joblib"
    if not path.is_file():
        raise FileNotFoundError("Model bundle is absent. Run train first.")
    return joblib.load(path)


def _metric_rows(
    model: ScorecardModel,
    frame: pd.DataFrame,
    target_name: str,
    split_name: str,
    accepted: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    probability = model.predict_proba(frame)
    rows: list[dict[str, Any]] = []
    calibrations: list[pd.DataFrame] = []
    masks = {
        "full": np.ones(len(frame), dtype=bool),
        "synthetic_accepted": accepted,
        "synthetic_rejected": ~accepted,
    }
    for population, mask in masks.items():
        if mask.sum() < 50 or np.unique(frame.loc[mask, target_name]).size < 2:
            continue
        metrics, calibration = validation_metrics(
            frame.loc[mask, target_name].astype(int),
            probability[mask],
            int(config["validation"]["calibration_bins"]),
        )
        row = {"model": model.name, "split": split_name, "population": population, **metrics}
        if population == "full":
            samples = int(config["validation"]["bootstrap_samples"])
            seed = int(config["project"]["random_seed"])
            row["gini_ci_lower"], row["gini_ci_upper"] = bootstrap_interval(
                frame.loc[mask, target_name].astype(int),
                probability[mask],
                gini_coefficient,
                samples,
                seed,
            )
            row["ks_ci_lower"], row["ks_ci_upper"] = bootstrap_interval(
                frame.loc[mask, target_name].astype(int),
                probability[mask],
                lambda y, p: ks_statistic(y, p)[0],
                samples,
                seed + 1,
            )
        rows.append(row)
        calibration.insert(0, "population", population)
        calibration.insert(0, "split", split_name)
        calibration.insert(0, "model", model.name)
        calibrations.append(calibration)
    return rows, calibrations


def _exposure(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    fallback = float(config["economics"]["exposure_at_default"])
    if "AMT_CREDIT" not in frame:
        return np.full(len(frame), fallback)
    return pd.to_numeric(frame["AMT_CREDIT"], errors="coerce").fillna(fallback).to_numpy()


def _economic_scenarios(
    scores: np.ndarray,
    probability: np.ndarray,
    exposure: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    settings = config["economics"]
    rows = []
    for lgd in settings["scenario_lgd"]:
        for margin in settings["scenario_margin_rate"]:
            scenario = deepcopy(settings)
            scenario["loss_given_default"] = float(lgd)
            scenario["net_margin_rate"] = float(margin)
            curve = cutoff_curve(scores, probability, exposure, scenario, target)
            optimum = optimal_cutoff(curve)
            rows.append(
                {
                    "loss_given_default": float(lgd),
                    "net_margin_rate": float(margin),
                    "optimal_score_cutoff": float(optimum["score_cutoff"]),
                    "approval_rate": float(optimum["approval_rate"]),
                    "average_predicted_bad_rate": float(optimum["average_predicted_bad_rate"]),
                    "expected_profit": float(optimum["expected_profit"]),
                    "realized_profit": float(optimum["realized_profit"]),
                }
            )
    return pd.DataFrame(rows)


def _save_plots(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    curve: pd.DataFrame,
    predictions: pd.DataFrame,
    figures: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(figures / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(figures / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    figures.mkdir(parents=True, exist_ok=True)

    test_full = metrics[(metrics["split"] == "test") & (metrics["population"] == "full")]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=test_full, x="model", y="gini", ax=ax, color="#3267a8")
    ax.set_title("Test Gini by scorecard")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures / "test_gini.png", dpi=180)
    plt.close(fig)
    parcel_cal = calibration[
        (calibration["model"] == "parcelled")
        & (calibration["split"] == "test")
        & (calibration["population"] == "full")
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle=":", color="black", label="Perfect calibration")
    ax.plot(
        parcel_cal["average_probability"],
        parcel_cal["observed_bad"] / parcel_cal["count"],
        marker="o",
        label="Parcelled scorecard",
    )
    ax.set_xlabel("Average predicted bad rate")
    ax.set_ylabel("Observed bad rate")
    ax.set_title("Test calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "test_calibration.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(curve["approval_rate"], curve["average_predicted_bad_rate"])
    axes[0].set_xlabel("Approval rate")
    axes[0].set_ylabel("Average predicted bad rate")
    axes[0].set_title("Approval and risk tradeoff")
    axes[1].plot(curve["score_cutoff"], curve["expected_profit"])
    axes[1].set_xlabel("Score cutoff")
    axes[1].set_ylabel("Expected profit")
    axes[1].set_title("Validation cutoff optimization")
    fig.tight_layout()
    fig.savefig(figures / "decision_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=predictions,
        x="parcelled_score",
        hue="split",
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=ax,
    )
    ax.set_title("Parcelled score distribution")
    fig.tight_layout()
    fig.savefig(figures / "score_distribution.png", dpi=180)
    plt.close(fig)


def _display_path(path: Path, project_root: Path) -> str:
    """Path for the manifest: relative to the project root when inside it,
    otherwise just the last two components (drops the local home directory
    prefix for paths that live outside the repo, e.g. a shared dataset
    folder)."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return "/".join(path.parts[-2:])


def _write_run_manifest(config: dict[str, Any]) -> Path:
    packages = [
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "seaborn",
    ]
    software = {}
    for package in packages:
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    project_root = Path(config["_project_root"])
    raw = project_path(config, "raw_data")
    sources = []
    for filename in config["data"]["required_files"]:
        path = raw / filename
        stat = path.stat()
        sources.append(
            {
                "file": filename,
                "configured_path": _display_path(path, project_root),
                "resolved_path": _display_path(path.resolve(), project_root),
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "software": software,
        "raw_sources": sources,
        "config_snapshot": _display_path(
            project_path(config, "artifacts") / "config_snapshot.json", project_root
        ),
        "split_manifest": _display_path(
            project_path(config, "artifacts") / "split_manifest.parquet", project_root
        ),
        "results_summary": _display_path(
            project_path(config, "artifacts") / "results_summary.json", project_root
        ),
    }
    path = project_path(config, "artifacts") / "run_manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def validate_models(config: dict[str, Any]) -> dict[str, Any]:
    """Run validation, stability, economics, bands, and swap-set analysis."""
    bundle = _load_bundle(config)
    labeled, unlabeled = _load_feature_matrices(config)
    target_name = config["data"]["target"]
    development = _subset_by_ids(labeled, bundle.split.development_ids)
    validation = _subset_by_ids(labeled, bundle.split.validation_ids)
    test = _subset_by_ids(labeled, bundle.split.test_ids)
    models = [bundle.application_only, bundle.accepted_only, bundle.parcelled, bundle.oracle]
    policy_masks = {
        "development": np.isin(development["SK_ID_CURR"], bundle.accepted_development_ids),
        "validation": bundle.acceptance_policy.sample(validation, seed_offset=1),
        "test": bundle.acceptance_policy.sample(test, seed_offset=2),
    }

    metric_rows: list[dict[str, Any]] = []
    calibration_tables: list[pd.DataFrame] = []
    predictions = []
    for split_name, frame in (("validation", validation), ("test", test)):
        prediction_frame = pd.DataFrame(
            {
                "SK_ID_CURR": frame["SK_ID_CURR"].to_numpy(),
                "split": split_name,
                "target": frame[target_name].to_numpy(dtype=int),
                "synthetic_accepted": policy_masks[split_name],
            }
        )
        for model in models:
            rows, calibrations = _metric_rows(
                model,
                frame,
                target_name,
                split_name,
                policy_masks[split_name],
                config,
            )
            metric_rows.extend(rows)
            calibration_tables.extend(calibrations)
            prediction_frame[f"{model.name}_pd"] = model.predict_proba(frame)
            prediction_frame[f"{model.name}_score"] = model.score(frame)
        predictions.append(prediction_frame)
    metrics = pd.DataFrame(metric_rows)
    calibration = pd.concat(calibration_tables, ignore_index=True)
    predictions_frame = pd.concat(predictions, ignore_index=True)

    score_psi_rows = []
    score_psi_details = []
    expected_scores = bundle.parcelled.score(development)
    for population_name, frame in (
        ("validation", validation),
        ("test", test),
        ("application_test_unlabeled", unlabeled),
    ):
        psi, details = population_stability_index(
            expected_scores,
            bundle.parcelled.score(frame),
            int(config["validation"]["psi_bins"]),
        )
        score_psi_rows.append({"population": population_name, "psi": psi})
        details.insert(0, "population", population_name)
        score_psi_details.append(details)
    score_psi = pd.DataFrame(score_psi_rows)
    score_psi_detail = pd.concat(score_psi_details, ignore_index=True)

    expected_bins = bundle.parcelled.transformer.assign_bins(
        development, bundle.parcelled.selected_features
    )
    csi_summaries = []
    csi_details = []
    for population_name, frame in (
        ("validation", validation),
        ("test", test),
        ("application_test_unlabeled", unlabeled),
    ):
        actual_bins = bundle.parcelled.transformer.assign_bins(
            frame, bundle.parcelled.selected_features
        )
        summary, detail = characteristic_stability_index(expected_bins, actual_bins)
        summary.insert(0, "population", population_name)
        detail.insert(0, "population", population_name)
        csi_summaries.append(summary)
        csi_details.append(detail)
    csi_summary = pd.concat(csi_summaries, ignore_index=True)
    csi_detail = pd.concat(csi_details, ignore_index=True)

    validation_probability = bundle.parcelled.predict_proba(validation)
    validation_scores = bundle.parcelled.score(validation)
    validation_curve = cutoff_curve(
        validation_scores,
        validation_probability,
        _exposure(validation, config),
        config["economics"],
        validation[target_name].astype(int),
    )
    optimum = optimal_cutoff(validation_curve)
    locked_cutoff = float(optimum["score_cutoff"])
    test_probability = bundle.parcelled.predict_proba(test)
    test_scores = bundle.parcelled.score(test)
    test_curve = cutoff_curve(
        test_scores,
        test_probability,
        _exposure(test, config),
        config["economics"],
        test[target_name].astype(int),
    )
    test_policy = apply_locked_cutoff(
        test_scores,
        locked_cutoff,
        float(config["decisioning"]["manual_review_width_points"]),
    )
    test_policy_table = pd.DataFrame(
        {
            "SK_ID_CURR": test["SK_ID_CURR"],
            "score": test_scores,
            "probability_bad": test_probability,
            "decision_band": test_policy,
            "target": test[target_name].astype(int),
        }
    )
    band_summary = test_policy_table.groupby("decision_band", observed=True).agg(
        applicants=("SK_ID_CURR", "size"),
        average_score=("score", "mean"),
        predicted_bad_rate=("probability_bad", "mean"),
        observed_bad_rate=("target", "mean"),
    ).reset_index()
    reason_codes = bundle.parcelled.reason_codes(test, top_n=4).reset_index(drop=True)
    test_policy_table = pd.concat(
        [test_policy_table.reset_index(drop=True), reason_codes], axis=1
    )

    swap = swap_set_analysis(
        test["SK_ID_CURR"],
        bundle.application_only.score(test),
        bundle.accepted_only.score(test),
        test[target_name].astype(int),
        _exposure(test, config),
        float(optimum["approval_rate"]),
        config["economics"],
    )
    scenarios = _economic_scenarios(
        validation_scores,
        validation_probability,
        _exposure(validation, config),
        validation[target_name].to_numpy(dtype=int),
        config,
    )
    appetite_options = risk_appetite_options(
        validation_curve,
        config["decisioning"]["maximum_bad_rate_scenarios"],
    )
    parcel_sensitivity_rows = []
    for multiplier_text, sensitivity_model in bundle.parcel_sensitivity.items():
        sensitivity_probability = sensitivity_model.predict_proba(test)
        sensitivity_metrics, _ = validation_metrics(
            test[target_name].astype(int),
            sensitivity_probability,
            int(config["validation"]["calibration_bins"]),
        )
        sensitivity_validation_probability = sensitivity_model.predict_proba(validation)
        sensitivity_validation_scores = sensitivity_model.score(validation)
        sensitivity_curve = cutoff_curve(
            sensitivity_validation_scores,
            sensitivity_validation_probability,
            _exposure(validation, config),
            config["economics"],
            validation[target_name].astype(int),
        )
        sensitivity_optimum = optimal_cutoff(sensitivity_curve)
        parcel_sensitivity_rows.append(
            {
                "bad_odds_multiplier": float(multiplier_text),
                "test_gini": sensitivity_metrics["gini"],
                "test_ks": sensitivity_metrics["ks"],
                "test_brier_score": sensitivity_metrics["brier_score"],
                "validation_optimal_cutoff": float(sensitivity_optimum["score_cutoff"]),
                "validation_optimal_approval_rate": float(
                    sensitivity_optimum["approval_rate"]
                ),
                "validation_optimal_expected_profit": float(
                    sensitivity_optimum["expected_profit"]
                ),
            }
        )
    parcel_sensitivity_table = pd.DataFrame(parcel_sensitivity_rows).sort_values(
        "bad_odds_multiplier"
    )

    tables = project_path(config, "artifacts") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "validation_metrics.csv", index=False)
    calibration.to_csv(tables / "calibration.csv", index=False)
    predictions_frame.to_parquet(tables / "predictions.parquet", index=False)
    score_psi.to_csv(tables / "score_psi.csv", index=False)
    score_psi_detail.to_csv(tables / "score_psi_detail.csv", index=False)
    csi_summary.to_csv(tables / "characteristic_csi.csv", index=False)
    csi_detail.to_csv(tables / "characteristic_csi_detail.csv", index=False)
    validation_curve.to_csv(tables / "validation_cutoff_curve.csv", index=False)
    test_curve.to_csv(tables / "test_cutoff_curve.csv", index=False)
    test_policy_table.to_csv(tables / "test_decisions.csv", index=False)
    band_summary.to_csv(tables / "decision_bands.csv", index=False)
    swap.summary.to_csv(tables / "swap_set_summary.csv", index=False)
    swap.applicants.to_parquet(tables / "swap_set_applicants.parquet", index=False)
    scenarios.to_csv(tables / "economic_sensitivity.csv", index=False)
    appetite_options.to_csv(tables / "risk_appetite_options.csv", index=False)
    parcel_sensitivity_table.to_csv(tables / "parcel_sensitivity.csv", index=False)

    test_full_parcelled = metrics[
        (metrics["model"] == "parcelled")
        & (metrics["split"] == "test")
        & (metrics["population"] == "full")
    ].iloc[0]
    test_full_application = metrics[
        (metrics["model"] == "application_only")
        & (metrics["split"] == "test")
        & (metrics["population"] == "full")
    ].iloc[0]
    test_full_accepted = metrics[
        (metrics["model"] == "accepted_only")
        & (metrics["split"] == "test")
        & (metrics["population"] == "full")
    ].iloc[0]
    test_full_oracle = metrics[
        (metrics["model"] == "oracle")
        & (metrics["split"] == "test")
        & (metrics["population"] == "full")
    ].iloc[0]
    locked_approved = test_scores >= locked_cutoff
    locked_economics = account_economics(
        test_probability, _exposure(test, config), config["economics"]
    )
    locked_realized = np.where(
        test.loc[locked_approved, target_name].to_numpy(dtype=int) == 1,
        locked_economics.loc[locked_approved, "profit_bad"].to_numpy(),
        locked_economics.loc[locked_approved, "profit_good"].to_numpy(),
    )
    locked_summary = pd.DataFrame(
        [
            {
                "score_cutoff": locked_cutoff,
                "approved_count": int(locked_approved.sum()),
                "approval_rate": float(locked_approved.mean()),
                "observed_bad_rate": float(test.loc[locked_approved, target_name].mean())
                if locked_approved.any()
                else np.nan,
                "average_predicted_bad_rate": float(test_probability[locked_approved].mean())
                if locked_approved.any()
                else np.nan,
                "expected_profit": float(
                    locked_economics.loc[locked_approved, "expected_profit"].sum()
                ),
                "realized_profit": float(locked_realized.sum()),
            }
        ]
    )
    locked_summary.to_csv(tables / "test_locked_cutoff_summary.csv", index=False)
    test_at_locked = locked_summary.iloc[0]
    swap_indexed = swap.summary.set_index("segment")
    swap_in_bad_rate = (
        float(swap_indexed.loc["swap_in", "bad_rate"])
        if "swap_in" in swap_indexed.index
        else np.nan
    )
    swap_out_bad_rate = (
        float(swap_indexed.loc["swap_out", "bad_rate"])
        if "swap_out" in swap_indexed.index
        else np.nan
    )
    swap_net_realized_value = (
        float(swap_indexed.loc["swap_in", "realized_value"])
        - float(swap_indexed.loc["swap_out", "realized_value"])
        if {"swap_in", "swap_out"}.issubset(swap_indexed.index)
        else 0.0
    )
    summary = {
        "test_application_only_gini": float(test_full_application["gini"]),
        "test_accepted_only_gini": float(test_full_accepted["gini"]),
        "test_gini": float(test_full_parcelled["gini"]),
        "test_oracle_gini": float(test_full_oracle["gini"]),
        "test_gini_ci_lower": float(test_full_parcelled["gini_ci_lower"]),
        "test_gini_ci_upper": float(test_full_parcelled["gini_ci_upper"]),
        "test_ks": float(test_full_parcelled["ks"]),
        "test_brier_score": float(test_full_parcelled["brier_score"]),
        "test_expected_calibration_error": float(
            test_full_parcelled["expected_calibration_error"]
        ),
        "test_calibration_intercept": float(
            test_full_parcelled["calibration_intercept"]
        ),
        "test_calibration_slope": float(test_full_parcelled["calibration_slope"]),
        "test_hosmer_lemeshow_p_value": float(
            test_full_parcelled["hosmer_lemeshow_p_value"]
        ),
        "validation_optimal_score_cutoff": locked_cutoff,
        "validation_optimal_approval_rate": float(optimum["approval_rate"]),
        "validation_optimal_expected_profit": float(optimum["expected_profit"]),
        "test_locked_cutoff_approval_rate": float(test_at_locked["approval_rate"]),
        "test_locked_cutoff_bad_rate": float(test_at_locked["observed_bad_rate"]),
        "test_locked_cutoff_expected_profit": float(test_at_locked["expected_profit"]),
        "test_locked_cutoff_realized_profit": float(test_at_locked["realized_profit"]),
        "test_score_psi": float(score_psi.loc[score_psi["population"] == "test", "psi"].iloc[0]),
        "application_test_unlabeled_score_psi": float(
            score_psi.loc[
                score_psi["population"] == "application_test_unlabeled", "psi"
            ].iloc[0]
        ),
        "swap_in_bad_rate": swap_in_bad_rate,
        "swap_out_bad_rate": swap_out_bad_rate,
        "swap_net_realized_value": swap_net_realized_value,
        "parcel_bad_odds_multiplier": float(
            config["synthetic_acceptance"]["parcel_bad_odds_multiplier"]
        ),
        "coefficient_sign_review_flags": int(
            sum(
                model.coefficient_table()["review_flag"].fillna("").ne("").sum()
                for model in models
            )
        ),
        "selected_features": bundle.selected_features,
        "reject_inference_is_synthetic": True,
        "temporal_validation_available": False,
    }
    with (project_path(config, "artifacts") / "results_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    _write_run_manifest(config)

    _save_plots(
        metrics,
        calibration,
        validation_curve,
        predictions_frame,
        project_path(config, "figures"),
    )
    from credit_scorecard.reporting import update_readme_results

    update_readme_results(config)
    return summary


def build_features(config: dict[str, Any]) -> dict[str, Path]:
    ensure_output_directories(config)
    return save_feature_matrix(config)


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    from credit_scorecard.data import audit_raw_data
    from credit_scorecard.eda import create_eda

    audit_raw_data(config)
    build_features(config)
    create_eda(config)
    train_models(config)
    return validate_models(config)


def clean_generated(config: dict[str, Any]) -> None:
    """Remove generated outputs only when explicitly invoked by the user."""
    for key in ("processed_data", "artifacts", "figures"):
        directory = project_path(config, key)
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
