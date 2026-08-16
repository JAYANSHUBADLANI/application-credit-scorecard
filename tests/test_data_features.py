from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scorecard.data import make_split, validate_split
from credit_scorecard.features import (
    build_application_features,
    build_installment_features,
    build_previous_features,
)


def test_split_is_disjoint_reproducible_and_stratified() -> None:
    ids = np.arange(1000, 2000)
    target = np.tile([0] * 9 + [1], 100)
    first = make_split(ids, target, 0.2, 0.2, 42)
    second = make_split(ids, target, 0.2, 0.2, 42)
    validate_split(first)
    assert np.array_equal(first.development_ids, second.development_ids)
    assert len(first.development_ids) == 600
    assert len(first.validation_ids) == 200
    assert len(first.test_ids) == 200
    lookup = pd.Series(target, index=ids)
    assert np.isclose(lookup.loc[first.test_ids].mean(), 0.1)


def test_application_features_clean_sentinel_and_calculate_ratios() -> None:
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "DAYS_EMPLOYED": [365243, -1000],
            "DAYS_BIRTH": [-10000, -20000],
            "AMT_CREDIT": [100.0, 300.0],
            "AMT_INCOME_TOTAL": [50.0, 100.0],
            "AMT_ANNUITY": [10.0, 0.0],
            "AMT_GOODS_PRICE": [90.0, 250.0],
            "CNT_FAM_MEMBERS": [2.0, 4.0],
        }
    )
    result, lineage = build_application_features(frame)
    assert np.isnan(result.loc[0, "DAYS_EMPLOYED"])
    assert result.loc[1, "APP_CREDIT_INCOME_RATIO"] == 3.0
    assert np.isnan(result.loc[1, "APP_CREDIT_ANNUITY_RATIO"])
    assert "APP_CREDIT_INCOME_RATIO" in {item.feature for item in lineage}


def test_previous_features_exclude_future_decisions(tmp_path) -> None:
    path = tmp_path / "previous.csv"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_PREV": [10, 11, 12, 20],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused", "Refused"],
            "DAYS_DECISION": [-20, 5, np.nan, -3],
            "AMT_APPLICATION": [100.0, 200.0, 300.0, 100.0],
            "AMT_CREDIT": [90.0, 180.0, 250.0, 80.0],
            "AMT_ANNUITY": [10.0, 20.0, 30.0, 10.0],
            "AMT_DOWN_PAYMENT": [10.0, 20.0, 50.0, 20.0],
            "RATE_DOWN_PAYMENT": [0.1, 0.1, 0.2, 0.2],
            "CNT_PAYMENT": [12, 12, 10, 10],
        }
    ).to_csv(path, index=False)
    features, _ = build_previous_features(path)
    assert features.loc[1, "PREV_APPLICATION_COUNT"] == 1
    assert features.loc[1, "PREV_REFUSED_COUNT"] == 0
    assert features.loc[2, "PREV_REFUSED_RATE"] == 1


def test_installments_enforce_point_in_time_and_dpd_sign(tmp_path) -> None:
    path = tmp_path / "installments.csv"
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_PREV": [10, 10, 10, 20],
            "DAYS_INSTALMENT": [-20, -10, 5, -10],
            "DAYS_ENTRY_PAYMENT": [-15, -12, 8, -4],
            "AMT_INSTALMENT": [100.0, 100.0, 100.0, 50.0],
            "AMT_PAYMENT": [80.0, 100.0, 100.0, 50.0],
        }
    ).to_csv(path, index=False)
    features, lineage = build_installment_features(path, {10}, 2)
    assert list(features.index) == [1]
    assert features.loc[1, "INST_RECORD_COUNT"] == 2
    assert features.loc[1, "INST_DPD_MAX"] == 5
    assert features.loc[1, "INST_DPD_SUM"] == 5
    assert features.loc[1, "INST_LATE_RATE"] == 0.5
    assert features.loc[1, "INST_SHORTFALL_SUM"] == 20
    assert all("DAYS_ENTRY_PAYMENT <= 0" in item.cutoff_rule for item in lineage)
