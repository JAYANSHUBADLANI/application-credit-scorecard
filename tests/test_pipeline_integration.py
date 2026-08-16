from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_scorecard.pipeline import run_all


def _write_synthetic_home_credit(raw: Path, train_rows: int = 900, test_rows: int = 250) -> None:
    rng = np.random.default_rng(123)
    rows = train_rows + test_rows
    ids = np.arange(100000, 100000 + rows)
    ext_2 = rng.uniform(0.05, 0.95, rows)
    ext_3 = rng.uniform(0.05, 0.95, rows)
    income = rng.lognormal(np.log(120000), 0.35, rows)
    credit_ratio = rng.uniform(0.5, 5.0, rows)
    credit = income * credit_ratio
    delinquent = rng.binomial(1, np.clip(0.05 + 0.45 * (1 - ext_2), 0, 1))
    probability = 1 / (
        1
        + np.exp(
            -(
                -4.0
                + 2.3 * (1 - ext_2)
                + 1.8 * (1 - ext_3)
                + 0.22 * credit_ratio
                + 1.0 * delinquent
            )
        )
    )
    target = rng.binomial(1, probability)
    application = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": target,
            "EXT_SOURCE_1": np.clip((ext_2 + ext_3) / 2 + rng.normal(0, 0.08, rows), 0, 1),
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_3": ext_3,
            "AMT_CREDIT": credit,
            "AMT_INCOME_TOTAL": income,
            "AMT_ANNUITY": credit / rng.integers(12, 48, rows),
            "AMT_GOODS_PRICE": credit * rng.uniform(0.85, 1.0, rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, rows).astype(float),
            "DAYS_EMPLOYED": -rng.integers(100, 8000, rows).astype(float),
            "DAYS_BIRTH": -rng.integers(8000, 24000, rows).astype(float),
            "CODE_GENDER": rng.choice(["F", "M"], rows),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner"], rows),
        }
    )
    application.iloc[:train_rows].to_csv(raw / "application_train.csv", index=False)
    application.iloc[train_rows:].drop(columns="TARGET").to_csv(
        raw / "application_test.csv", index=False
    )

    bureau_rows = []
    balance_rows = []
    previous_rows = []
    installment_rows = []
    card_rows = []
    pos_rows = []
    for index, applicant_id in enumerate(ids):
        bureau_id = 200000 + index
        previous_id = 300000 + index
        is_bad_history = int(delinquent[index])
        bureau_rows.append(
            {
                "SK_ID_CURR": applicant_id,
                "SK_ID_BUREAU": bureau_id,
                "CREDIT_ACTIVE": "Active" if index % 2 else "Closed",
                "CREDIT_TYPE": "Consumer credit",
                "DAYS_CREDIT": -int(rng.integers(30, 2000)),
                "DAYS_CREDIT_ENDDATE": -10,
                "DAYS_ENDDATE_FACT": -5,
                "DAYS_CREDIT_UPDATE": -2,
                "CREDIT_DAY_OVERDUE": 20 * is_bad_history,
                "AMT_CREDIT_MAX_OVERDUE": 2000 * is_bad_history,
                "CNT_CREDIT_PROLONG": is_bad_history,
                "AMT_CREDIT_SUM": 50000.0,
                "AMT_CREDIT_SUM_DEBT": 35000.0 if is_bad_history else 5000.0,
                "AMT_CREDIT_SUM_LIMIT": 0.0,
                "AMT_CREDIT_SUM_OVERDUE": 1000.0 * is_bad_history,
                "AMT_ANNUITY": 5000.0,
            }
        )
        for month in (-2, -1, 1):
            balance_rows.append(
                {
                    "SK_ID_BUREAU": bureau_id,
                    "MONTHS_BALANCE": month,
                    "STATUS": str(2 * is_bad_history) if month <= 0 else "5",
                }
            )
        status = "Refused" if is_bad_history else "Approved"
        previous_rows.append(
            {
                "SK_ID_CURR": applicant_id,
                "SK_ID_PREV": previous_id,
                "NAME_CONTRACT_STATUS": status,
                "DAYS_DECISION": -50,
                "AMT_APPLICATION": 40000.0,
                "AMT_CREDIT": 38000.0,
                "AMT_ANNUITY": 4000.0,
                "AMT_DOWN_PAYMENT": 2000.0,
                "RATE_DOWN_PAYMENT": 0.05,
                "CNT_PAYMENT": 12,
            }
        )
        installment_rows.extend(
            [
                {
                    "SK_ID_CURR": applicant_id,
                    "SK_ID_PREV": previous_id,
                    "DAYS_INSTALMENT": -20,
                    "DAYS_ENTRY_PAYMENT": -20 + 8 * is_bad_history,
                    "AMT_INSTALMENT": 1000.0,
                    "AMT_PAYMENT": 700.0 if is_bad_history else 1000.0,
                },
                {
                    "SK_ID_CURR": applicant_id,
                    "SK_ID_PREV": previous_id,
                    "DAYS_INSTALMENT": 10,
                    "DAYS_ENTRY_PAYMENT": 12,
                    "AMT_INSTALMENT": 1000.0,
                    "AMT_PAYMENT": 1000.0,
                },
            ]
        )
        for month in (-2, -1, 1):
            card_rows.append(
                {
                    "SK_ID_CURR": applicant_id,
                    "SK_ID_PREV": previous_id,
                    "MONTHS_BALANCE": month,
                    "SK_DPD": 5 * is_bad_history if month <= 0 else 20,
                    "SK_DPD_DEF": is_bad_history if month <= 0 else 10,
                    "AMT_BALANCE": 6000.0 if is_bad_history else 2000.0,
                    "AMT_CREDIT_LIMIT_ACTUAL": 10000.0,
                    "AMT_DRAWINGS_CURRENT": 500.0,
                    "AMT_PAYMENT_CURRENT": 300.0,
                    "AMT_INST_MIN_REGULARITY": 250.0,
                }
            )
            pos_rows.append(
                {
                    "SK_ID_CURR": applicant_id,
                    "SK_ID_PREV": previous_id,
                    "MONTHS_BALANCE": month,
                    "SK_DPD": 4 * is_bad_history if month <= 0 else 10,
                    "SK_DPD_DEF": is_bad_history if month <= 0 else 5,
                    "CNT_INSTALMENT": 12,
                    "CNT_INSTALMENT_FUTURE": 4,
                }
            )
    pd.DataFrame(bureau_rows).to_csv(raw / "bureau.csv", index=False)
    pd.DataFrame(balance_rows).to_csv(raw / "bureau_balance.csv", index=False)
    pd.DataFrame(previous_rows).to_csv(raw / "previous_application.csv", index=False)
    pd.DataFrame(installment_rows).to_csv(raw / "installments_payments.csv", index=False)
    pd.DataFrame(card_rows).to_csv(raw / "credit_card_balance.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(raw / "POS_CASH_balance.csv", index=False)


def _config(root: Path) -> dict:
    return {
        "_project_root": str(root),
        "_config_path": str(root / "config" / "project.yaml"),
        "project": {"name": "integration", "random_seed": 42},
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "artifacts": "artifacts",
            "figures": "reports/figures",
        },
        "data": {
            "competition": "home-credit-default-risk",
            "target": "TARGET",
            "id_column": "SK_ID_CURR",
            "required_files": [
                "application_train.csv",
                "application_test.csv",
                "bureau.csv",
                "bureau_balance.csv",
                "previous_application.csv",
                "installments_payments.csv",
                "credit_card_balance.csv",
                "POS_CASH_balance.csv",
            ],
            "chunk_size": 300,
        },
        "features": {
            "exclude_columns": ["TARGET", "SK_ID_CURR", "CODE_GENDER"],
            "exclude_name_patterns": ["^SK_ID_"],
            "minimum_non_null_fraction": 0.05,
            "maximum_unique_categories": 20,
        },
        "split": {"validation_fraction": 0.2, "test_fraction": 0.2},
        "synthetic_acceptance": {
            "target_acceptance_rate": 0.65,
            "temperature": 0.2,
            "parcel_bad_odds_multiplier": 1.35,
            "parcel_bad_odds_sensitivity": [1.0, 1.35, 2.0],
            "parcel_bands": 5,
        },
        "binning": {
            "maximum_prebins": 8,
            "maximum_categorical_bins": 4,
            "minimum_bin_fraction": 0.05,
            "smoothing": 0.5,
            "iv_minimum": 0.0001,
            "iv_maximum": 100.0,
            "correlation_maximum": 0.95,
            "maximum_features": 8,
        },
        "model": {
            "regularization_c": 1.0,
            "maximum_iterations": 1000,
            "class_weight": None,
            "enforce_expected_coefficient_sign": True,
            "coefficient_positive_tolerance": 0.00000001,
            "base_score": 600,
            "base_odds_good_to_bad": 50,
            "points_to_double_odds": 20,
        },
        "validation": {"calibration_bins": 5, "bootstrap_samples": 5, "psi_bins": 5},
        "economics": {
            "exposure_at_default": 100000,
            "ead_factor": 1.0,
            "loss_given_default": 0.55,
            "net_margin_rate": 0.08,
            "pre_default_bad_margin_rate": 0.02,
            "acquisition_cost": 1500,
            "operating_cost": 1000,
            "cutoff_grid_size": 21,
            "scenario_lgd": [0.3, 0.6],
            "scenario_margin_rate": [0.05, 0.1],
        },
        "decisioning": {
            "manual_review_width_points": 20,
            "maximum_bad_rate_scenarios": [0.10, 0.15, 0.20],
        },
    }


def test_tiny_end_to_end_pipeline_creates_required_artifacts(tmp_path) -> None:
    for directory in (
        tmp_path / "data/raw",
        tmp_path / "data/processed",
        tmp_path / "artifacts",
        tmp_path / "reports/figures",
        tmp_path / "config",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (tmp_path / "README.md").write_text(
        "# Test\n\n<!-- RESULTS_START -->pending<!-- RESULTS_END -->\n", encoding="utf-8"
    )
    _write_synthetic_home_credit(tmp_path / "data/raw")
    summary = run_all(_config(tmp_path))
    assert 0 <= summary["test_gini"] <= 1
    assert summary["reject_inference_is_synthetic"] is True
    assert summary["temporal_validation_available"] is False
    assert (tmp_path / "artifacts/models/model_bundle.joblib").is_file()
    assert (tmp_path / "artifacts/tables/swap_set_summary.csv").is_file()
    assert (tmp_path / "artifacts/tables/validation_cutoff_curve.csv").is_file()
    parcel_table = pd.read_csv(tmp_path / "artifacts/tables/parcel_assumptions.csv")
    assert parcel_table["diagnostic_only_not_used_for_fitting"].all()
    assert "diagnostic_hidden_rejected_bad_rate" in parcel_table
    assert (tmp_path / "data/processed/feature_lineage.csv").is_file()
    assert (tmp_path / "reports/figures/decision_curve.png").is_file()
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Results are intentionally pending" not in readme
    assert "Parcelled test Gini" in readme
