"""Point-in-time applicant feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from credit_scorecard.config import project_path
from credit_scorecard.data import raw_file, require_raw_files


@dataclass(frozen=True)
class FeatureLineage:
    feature: str
    source: str
    source_columns: str
    cutoff_rule: str
    aggregation: str
    availability: str = "Observable no later than the application decision date"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        column if isinstance(column, str) else "_".join(str(part) for part in column if part)
        for column in frame.columns
    ]
    return frame.rename(columns={column: f"{prefix}{column}" for column in frame.columns})


def _group_named(frame: pd.DataFrame, key: str, definitions: dict[str, tuple[str, str]]) -> pd.DataFrame:
    available = {
        output: pd.NamedAgg(column=source, aggfunc=operation)
        for output, (source, operation) in definitions.items()
        if source in frame.columns
    }
    if not available:
        return pd.DataFrame(index=pd.Index([], name=key))
    return frame.groupby(key, observed=True).agg(**available)


def _merge_applicant_features(base: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return base
    return base.merge(aggregate, how="left", left_on="SK_ID_CURR", right_index=True)


def build_application_features(
    application: pd.DataFrame,
    copy: bool = True,
) -> tuple[pd.DataFrame, list[FeatureLineage]]:
    """Clean application fields and add transparent affordability ratios."""
    frame = application.copy() if copy else application
    lineage: list[FeatureLineage] = [
        FeatureLineage(
            column,
            "application_train/application_test",
            column,
            "Application row only",
            "Direct application characteristic",
        )
        for column in frame.columns
        if column not in {"TARGET", "SK_ID_CURR", "__SOURCE_SET__"}
    ]
    if "DAYS_EMPLOYED" in frame:
        frame["DAYS_EMPLOYED"] = frame["DAYS_EMPLOYED"].replace(365243, np.nan)

    ratios = {
        "APP_CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "APP_ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "APP_CREDIT_ANNUITY_RATIO": ("AMT_CREDIT", "AMT_ANNUITY"),
        "APP_GOODS_CREDIT_RATIO": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        "APP_INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        "APP_EMPLOYED_AGE_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    }
    for output, (numerator, denominator) in ratios.items():
        if numerator in frame and denominator in frame:
            frame[output] = safe_divide(frame[numerator], frame[denominator])
            lineage.append(
                FeatureLineage(
                    output,
                    "application_train/application_test",
                    f"{numerator}, {denominator}",
                    "Application row only",
                    f"{numerator} divided by {denominator}",
                )
            )
    return frame, lineage


def build_bureau_features(
    bureau_path: Path,
    balance_path: Path,
    chunk_size: int,
) -> tuple[pd.DataFrame, list[FeatureLineage]]:
    """Build bureau history using only records available by application day zero."""
    bureau_use = [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_TYPE",
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_MAX_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_ANNUITY",
    ]
    header = pd.read_csv(bureau_path, nrows=0).columns
    bureau = pd.read_csv(bureau_path, usecols=[c for c in bureau_use if c in header])
    before = len(bureau)
    if "DAYS_CREDIT" in bureau:
        bureau = bureau[bureau["DAYS_CREDIT"].notna() & bureau["DAYS_CREDIT"].le(0)]
    if "DAYS_CREDIT_UPDATE" in bureau:
        bureau = bureau[
            bureau["DAYS_CREDIT_UPDATE"].notna()
            & bureau["DAYS_CREDIT_UPDATE"].le(0)
        ]
    bureau["BUREAU_IS_ACTIVE"] = bureau.get("CREDIT_ACTIVE", pd.Series(index=bureau.index)).eq("Active").astype(int)
    bureau["BUREAU_IS_CLOSED"] = bureau.get("CREDIT_ACTIVE", pd.Series(index=bureau.index)).eq("Closed").astype(int)
    bureau["BUREAU_HAS_OVERDUE"] = bureau.get("AMT_CREDIT_SUM_OVERDUE", pd.Series(index=bureau.index, dtype=float)).fillna(0).gt(0).astype(int)
    if {"AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM"}.issubset(bureau):
        bureau["BUREAU_DEBT_CREDIT_RATIO"] = safe_divide(
            bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"]
        )

    balance: pd.DataFrame | None = None
    balance_operations = {
        "BB_MONTH_COUNT": "sum",
        "BB_OLDEST_MONTH": "min",
        "BB_LATEST_MONTH": "max",
        "BB_DELINQUENT_COUNT": "sum",
        "BB_SEVERE_COUNT": "sum",
        "BB_MAX_SEVERITY": "max",
    }
    balance_use = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
    balance_header = pd.read_csv(balance_path, nrows=0).columns
    for chunk in pd.read_csv(
        balance_path,
        usecols=[c for c in balance_use if c in balance_header],
        chunksize=chunk_size,
    ):
        chunk = chunk[chunk["MONTHS_BALANCE"].le(0)]
        severity = pd.to_numeric(chunk["STATUS"], errors="coerce").fillna(0)
        chunk["BB_DELINQUENT"] = severity.gt(0).astype(int)
        chunk["BB_SEVERE"] = severity.ge(3).astype(int)
        chunk["BB_SEVERITY"] = severity
        part = _group_named(
            chunk,
            "SK_ID_BUREAU",
            {
                "BB_MONTH_COUNT": ("MONTHS_BALANCE", "count"),
                "BB_OLDEST_MONTH": ("MONTHS_BALANCE", "min"),
                "BB_LATEST_MONTH": ("MONTHS_BALANCE", "max"),
                "BB_DELINQUENT_COUNT": ("BB_DELINQUENT", "sum"),
                "BB_SEVERE_COUNT": ("BB_SEVERE", "sum"),
                "BB_MAX_SEVERITY": ("BB_SEVERITY", "max"),
            },
        )
        balance = _merge_partial_aggregate(balance, part, balance_operations)
    if balance is not None:
        balance["BB_DELINQUENT_RATE"] = safe_divide(
            balance["BB_DELINQUENT_COUNT"], balance["BB_MONTH_COUNT"]
        )
        balance["BB_SEVERE_RATE"] = safe_divide(balance["BB_SEVERE_COUNT"], balance["BB_MONTH_COUNT"])
        bureau = bureau.merge(balance, how="left", left_on="SK_ID_BUREAU", right_index=True)

    definitions = {
        "BUREAU_RECORD_COUNT": ("SK_ID_BUREAU", "count"),
        "BUREAU_ACTIVE_COUNT": ("BUREAU_IS_ACTIVE", "sum"),
        "BUREAU_CLOSED_COUNT": ("BUREAU_IS_CLOSED", "sum"),
        "BUREAU_DAYS_CREDIT_MOST_RECENT": ("DAYS_CREDIT", "max"),
        "BUREAU_DAYS_CREDIT_OLDEST": ("DAYS_CREDIT", "min"),
        "BUREAU_CREDIT_SUM_TOTAL": ("AMT_CREDIT_SUM", "sum"),
        "BUREAU_CREDIT_SUM_MEAN": ("AMT_CREDIT_SUM", "mean"),
        "BUREAU_DEBT_TOTAL": ("AMT_CREDIT_SUM_DEBT", "sum"),
        "BUREAU_OVERDUE_TOTAL": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        "BUREAU_MAX_OVERDUE": ("AMT_CREDIT_MAX_OVERDUE", "max"),
        "BUREAU_DAYS_OVERDUE_MAX": ("CREDIT_DAY_OVERDUE", "max"),
        "BUREAU_OVERDUE_ACCOUNT_COUNT": ("BUREAU_HAS_OVERDUE", "sum"),
        "BUREAU_PROLONG_TOTAL": ("CNT_CREDIT_PROLONG", "sum"),
        "BUREAU_DEBT_CREDIT_RATIO_MEAN": ("BUREAU_DEBT_CREDIT_RATIO", "mean"),
        "BUREAU_BALANCE_DELINQUENCY_RATE_MAX": ("BB_DELINQUENT_RATE", "max"),
        "BUREAU_BALANCE_SEVERE_RATE_MAX": ("BB_SEVERE_RATE", "max"),
        "BUREAU_BALANCE_MAX_SEVERITY": ("BB_MAX_SEVERITY", "max"),
    }
    features = _group_named(bureau, "SK_ID_CURR", definitions)
    if {"BUREAU_DEBT_TOTAL", "BUREAU_CREDIT_SUM_TOTAL"}.issubset(features):
        features["BUREAU_TOTAL_DEBT_CREDIT_RATIO"] = safe_divide(
            features["BUREAU_DEBT_TOTAL"], features["BUREAU_CREDIT_SUM_TOTAL"]
        )
    if {"BUREAU_ACTIVE_COUNT", "BUREAU_RECORD_COUNT"}.issubset(features):
        features["BUREAU_ACTIVE_SHARE"] = safe_divide(
            features["BUREAU_ACTIVE_COUNT"], features["BUREAU_RECORD_COUNT"]
        )
    lineage = [
        FeatureLineage(
            column,
            "bureau and bureau_balance",
            "Applicant bureau history",
            "DAYS_CREDIT <= 0, DAYS_CREDIT_UPDATE <= 0, MONTHS_BALANCE <= 0",
            "Applicant-level history aggregation",
        )
        for column in features.columns
    ]
    features.attrs["rows_removed_by_cutoff"] = before - len(bureau)
    return features, lineage


def build_previous_features(path: Path) -> tuple[pd.DataFrame, list[FeatureLineage]]:
    """Aggregate prior lender applications known by the current decision date."""
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "DAYS_DECISION",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "RATE_DOWN_PAYMENT",
        "CNT_PAYMENT",
    ]
    header = pd.read_csv(path, nrows=0).columns
    previous = pd.read_csv(path, usecols=[c for c in usecols if c in header])
    previous = previous[
        previous["DAYS_DECISION"].notna() & previous["DAYS_DECISION"].le(0)
    ]
    status = previous.get("NAME_CONTRACT_STATUS", pd.Series(index=previous.index, dtype=object))
    previous["PREV_ACCEPTED"] = status.eq("Approved").astype(int)
    previous["PREV_REFUSED"] = status.eq("Refused").astype(int)
    previous["PREV_CANCELLED"] = status.eq("Canceled").astype(int)
    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(previous):
        previous["PREV_GRANTED_REQUESTED_RATIO"] = safe_divide(
            previous["AMT_CREDIT"], previous["AMT_APPLICATION"]
        )
    definitions = {
        "PREV_APPLICATION_COUNT": ("SK_ID_PREV", "count"),
        "PREV_ACCEPTED_COUNT": ("PREV_ACCEPTED", "sum"),
        "PREV_REFUSED_COUNT": ("PREV_REFUSED", "sum"),
        "PREV_CANCELLED_COUNT": ("PREV_CANCELLED", "sum"),
        "PREV_DAYS_DECISION_RECENT": ("DAYS_DECISION", "max"),
        "PREV_DAYS_DECISION_OLDEST": ("DAYS_DECISION", "min"),
        "PREV_AMT_APPLICATION_MEAN": ("AMT_APPLICATION", "mean"),
        "PREV_AMT_CREDIT_MEAN": ("AMT_CREDIT", "mean"),
        "PREV_AMT_ANNUITY_MEAN": ("AMT_ANNUITY", "mean"),
        "PREV_DOWN_PAYMENT_MEAN": ("AMT_DOWN_PAYMENT", "mean"),
        "PREV_TERM_MEAN": ("CNT_PAYMENT", "mean"),
        "PREV_GRANTED_REQUESTED_RATIO_MEAN": ("PREV_GRANTED_REQUESTED_RATIO", "mean"),
    }
    features = _group_named(previous, "SK_ID_CURR", definitions)
    for name, count in (
        ("PREV_ACCEPTED_RATE", "PREV_ACCEPTED_COUNT"),
        ("PREV_REFUSED_RATE", "PREV_REFUSED_COUNT"),
        ("PREV_CANCELLED_RATE", "PREV_CANCELLED_COUNT"),
    ):
        if count in features:
            features[name] = safe_divide(features[count], features["PREV_APPLICATION_COUNT"])
    lineage = [
        FeatureLineage(
            column,
            "previous_application",
            "Prior applications",
            "DAYS_DECISION <= 0",
            "Applicant-level prior application aggregation",
        )
        for column in features.columns
    ]
    return features, lineage


def _merge_partial_aggregate(
    accumulator: pd.DataFrame | None,
    part: pd.DataFrame,
    operations: dict[str, str],
) -> pd.DataFrame:
    """Incrementally combine chunk summaries to bound peak memory."""
    if accumulator is None or accumulator.empty:
        return part
    return pd.concat([accumulator, part]).groupby(level=0).agg(operations)


def build_installment_features(
    path: Path,
    eligible_previous_ids: set[int],
    chunk_size: int,
) -> tuple[pd.DataFrame, list[FeatureLineage]]:
    """Aggregate only fully observed payments on eligible prior contracts."""
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    header = pd.read_csv(path, nrows=0).columns
    operations = {
        "INST_RECORD_COUNT": "sum",
        "INST_DPD_SUM": "sum",
        "INST_DPD_MAX": "max",
        "INST_LATE_COUNT": "sum",
        "INST_SHORTFALL_SUM": "sum",
        "INST_UNDERPAID_COUNT": "sum",
        "INST_PAYMENT_TOTAL": "sum",
        "INST_DUE_TOTAL": "sum",
        "INST_LATEST_PAYMENT_DAY": "max",
    }
    features: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, usecols=[c for c in usecols if c in header], chunksize=chunk_size):
        chunk = chunk[chunk["SK_ID_PREV"].isin(eligible_previous_ids)]
        chunk = chunk[
            chunk["DAYS_INSTALMENT"].le(0)
            & chunk["DAYS_ENTRY_PAYMENT"].le(0)
            & chunk["DAYS_INSTALMENT"].notna()
            & chunk["DAYS_ENTRY_PAYMENT"].notna()
        ]
        chunk["INST_DPD"] = (chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]).clip(lower=0)
        chunk["INST_LATE"] = chunk["INST_DPD"].gt(0).astype(int)
        chunk["INST_SHORTFALL"] = (chunk["AMT_INSTALMENT"] - chunk["AMT_PAYMENT"]).clip(lower=0)
        chunk["INST_UNDERPAID"] = chunk["INST_SHORTFALL"].gt(0).astype(int)
        chunk["INST_PAYMENT_SUM"] = chunk["AMT_PAYMENT"].fillna(0)
        chunk["INST_DUE_SUM"] = chunk["AMT_INSTALMENT"].fillna(0)
        part = _group_named(
            chunk,
            "SK_ID_CURR",
            {
                "INST_RECORD_COUNT": ("SK_ID_PREV", "count"),
                "INST_DPD_SUM": ("INST_DPD", "sum"),
                "INST_DPD_MAX": ("INST_DPD", "max"),
                "INST_LATE_COUNT": ("INST_LATE", "sum"),
                "INST_SHORTFALL_SUM": ("INST_SHORTFALL", "sum"),
                "INST_UNDERPAID_COUNT": ("INST_UNDERPAID", "sum"),
                "INST_PAYMENT_TOTAL": ("INST_PAYMENT_SUM", "sum"),
                "INST_DUE_TOTAL": ("INST_DUE_SUM", "sum"),
                "INST_LATEST_PAYMENT_DAY": ("DAYS_ENTRY_PAYMENT", "max"),
            },
        )
        features = _merge_partial_aggregate(features, part, operations)
    if features is None:
        features = pd.DataFrame()
    if not features.empty:
        features["INST_LATE_RATE"] = safe_divide(features["INST_LATE_COUNT"], features["INST_RECORD_COUNT"])
        features["INST_DPD_MEAN"] = safe_divide(features["INST_DPD_SUM"], features["INST_RECORD_COUNT"])
        features["INST_UNDERPAID_RATE"] = safe_divide(
            features["INST_UNDERPAID_COUNT"], features["INST_RECORD_COUNT"]
        )
        features["INST_PAYMENT_RATIO"] = safe_divide(features["INST_PAYMENT_TOTAL"], features["INST_DUE_TOTAL"])
    lineage = [
        FeatureLineage(
            column,
            "installments_payments",
            "Prior-contract installment and payment fields",
            "Eligible SK_ID_PREV, DAYS_INSTALMENT <= 0, DAYS_ENTRY_PAYMENT <= 0",
            "Applicant-level repayment aggregation",
        )
        for column in features.columns
    ]
    return features, lineage


def build_balance_features(
    path: Path,
    eligible_previous_ids: set[int],
    chunk_size: int,
    source: str,
) -> tuple[pd.DataFrame, list[FeatureLineage]]:
    """Aggregate prior card or point-of-sale monthly snapshots."""
    is_card = source == "credit_card_balance"
    usecols = ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF"]
    if is_card:
        usecols += [
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_CURRENT",
            "AMT_PAYMENT_CURRENT",
            "AMT_INST_MIN_REGULARITY",
        ]
    else:
        usecols += ["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]
    header = pd.read_csv(path, nrows=0).columns
    features: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, usecols=[c for c in usecols if c in header], chunksize=chunk_size):
        chunk = chunk[chunk["SK_ID_PREV"].isin(eligible_previous_ids)]
        chunk = chunk[chunk["MONTHS_BALANCE"].le(0)]
        prefix = "CC" if is_card else "POS"
        chunk[f"{prefix}_DELINQUENT"] = chunk["SK_DPD"].fillna(0).gt(0).astype(int)
        definitions = {
            f"{prefix}_MONTH_COUNT": ("MONTHS_BALANCE", "count"),
            f"{prefix}_LATEST_MONTH": ("MONTHS_BALANCE", "max"),
            f"{prefix}_DPD_SUM": ("SK_DPD", "sum"),
            f"{prefix}_DPD_MAX": ("SK_DPD", "max"),
            f"{prefix}_DPD_DEF_MAX": ("SK_DPD_DEF", "max"),
            f"{prefix}_DELINQUENT_COUNT": (f"{prefix}_DELINQUENT", "sum"),
        }
        if is_card:
            chunk["CC_UTILIZATION"] = safe_divide(chunk["AMT_BALANCE"], chunk["AMT_CREDIT_LIMIT_ACTUAL"])
            chunk["CC_PAYMENT_MIN_RATIO"] = safe_divide(
                chunk["AMT_PAYMENT_CURRENT"], chunk["AMT_INST_MIN_REGULARITY"]
            )
            definitions.update(
                {
                    "CC_BALANCE_SUM": ("AMT_BALANCE", "sum"),
                    "CC_LIMIT_SUM": ("AMT_CREDIT_LIMIT_ACTUAL", "sum"),
                    "CC_UTILIZATION_SUM": ("CC_UTILIZATION", "sum"),
                    "CC_UTILIZATION_MAX": ("CC_UTILIZATION", "max"),
                    "CC_DRAWINGS_SUM": ("AMT_DRAWINGS_CURRENT", "sum"),
                    "CC_PAYMENT_SUM": ("AMT_PAYMENT_CURRENT", "sum"),
                    "CC_PAYMENT_MIN_RATIO_SUM": ("CC_PAYMENT_MIN_RATIO", "sum"),
                }
            )
        else:
            definitions.update(
                {
                    "POS_TERM_SUM": ("CNT_INSTALMENT", "sum"),
                    "POS_REMAINING_SUM": ("CNT_INSTALMENT_FUTURE", "sum"),
                }
            )
        part = _group_named(chunk, "SK_ID_CURR", definitions)
        operations = {
            column: ("max" if column.endswith(("_MAX", "LATEST_MONTH")) else "sum")
            for column in part.columns
        }
        features = _merge_partial_aggregate(features, part, operations)
    if features is None:
        return pd.DataFrame(), []
    prefix = "CC" if is_card else "POS"
    features[f"{prefix}_DELINQUENCY_RATE"] = safe_divide(
        features[f"{prefix}_DELINQUENT_COUNT"], features[f"{prefix}_MONTH_COUNT"]
    )
    features[f"{prefix}_DPD_MEAN"] = safe_divide(
        features[f"{prefix}_DPD_SUM"], features[f"{prefix}_MONTH_COUNT"]
    )
    if is_card:
        features["CC_UTILIZATION_MEAN"] = safe_divide(
            features["CC_UTILIZATION_SUM"], features["CC_MONTH_COUNT"]
        )
        features["CC_PAYMENT_MIN_RATIO_MEAN"] = safe_divide(
            features["CC_PAYMENT_MIN_RATIO_SUM"], features["CC_MONTH_COUNT"]
        )
    else:
        features["POS_REMAINING_SHARE"] = safe_divide(
            features["POS_REMAINING_SUM"], features["POS_TERM_SUM"]
        )
    lineage = [
        FeatureLineage(
            column,
            source,
            "Prior-contract monthly balances",
            "Eligible SK_ID_PREV and MONTHS_BALANCE <= 0",
            "Applicant-level monthly history aggregation",
        )
        for column in features.columns
    ]
    return features, lineage


def build_feature_matrix(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build labeled and unlabeled applicant matrices plus feature lineage."""
    require_raw_files(config)
    train = pd.read_csv(raw_file(config, "application_train.csv"))
    test = pd.read_csv(raw_file(config, "application_test.csv"))
    train["__SOURCE_SET__"] = "train"
    test["__SOURCE_SET__"] = "test"
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    del train, test
    combined, lineage = build_application_features(combined, copy=False)

    chunk_size = int(config["data"]["chunk_size"])
    bureau, bureau_lineage = build_bureau_features(
        raw_file(config, "bureau.csv"),
        raw_file(config, "bureau_balance.csv"),
        chunk_size,
    )
    combined = _merge_applicant_features(combined, bureau)
    del bureau

    previous, previous_lineage = build_previous_features(
        raw_file(config, "previous_application.csv")
    )
    combined = _merge_applicant_features(combined, previous)
    del previous
    previous_ids_frame = pd.read_csv(
        raw_file(config, "previous_application.csv"),
        usecols=["SK_ID_PREV", "DAYS_DECISION"],
    )
    eligible_previous_ids = set(
        previous_ids_frame.loc[previous_ids_frame["DAYS_DECISION"].le(0), "SK_ID_PREV"].astype(int)
    )
    del previous_ids_frame
    installments, installment_lineage = build_installment_features(
        raw_file(config, "installments_payments.csv"), eligible_previous_ids, chunk_size
    )
    combined = _merge_applicant_features(combined, installments)
    del installments
    cards, card_lineage = build_balance_features(
        raw_file(config, "credit_card_balance.csv"),
        eligible_previous_ids,
        chunk_size,
        "credit_card_balance",
    )
    combined = _merge_applicant_features(combined, cards)
    del cards
    pos, pos_lineage = build_balance_features(
        raw_file(config, "POS_CASH_balance.csv"),
        eligible_previous_ids,
        chunk_size,
        "POS_CASH_balance",
    )
    combined = _merge_applicant_features(combined, pos)
    del pos
    del eligible_previous_ids
    lineage.extend(bureau_lineage + previous_lineage + installment_lineage + card_lineage + pos_lineage)

    if combined["SK_ID_CURR"].duplicated().any():
        raise ValueError("Feature matrix contains duplicate applicants")
    target = config["data"]["target"]
    labeled = combined[combined["__SOURCE_SET__"].eq("train")].drop(columns="__SOURCE_SET__")
    unlabeled = combined[combined["__SOURCE_SET__"].eq("test")].drop(columns="__SOURCE_SET__")
    if unlabeled[target].notna().any():
        raise ValueError("Unlabeled application records unexpectedly contain target values")
    lineage_frame = pd.DataFrame([item.__dict__ for item in lineage]).drop_duplicates("feature")
    modeled_columns = set(labeled.columns) - {target, "SK_ID_CURR"}
    missing_lineage = sorted(modeled_columns - set(lineage_frame["feature"]))
    if missing_lineage:
        raise ValueError(f"Feature lineage is incomplete: {missing_lineage[:10]}")
    return labeled, unlabeled, lineage_frame


def save_feature_matrix(config: dict[str, Any]) -> dict[str, Path]:
    labeled, unlabeled, lineage = build_feature_matrix(config)
    output = project_path(config, "processed_data")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "labeled": output / "applicants_labeled.parquet",
        "unlabeled": output / "applicants_unlabeled.parquet",
        "lineage": output / "feature_lineage.csv",
    }
    labeled.to_parquet(paths["labeled"], index=False)
    unlabeled.to_parquet(paths["unlabeled"], index=False)
    lineage.to_csv(paths["lineage"], index=False)
    return paths
