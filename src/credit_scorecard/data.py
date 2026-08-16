"""Data access, verification, and split utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scorecard.config import project_path


@dataclass(frozen=True)
class DatasetSplit:
    """Disjoint applicant identifiers for model development and evaluation."""

    development_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        parts = []
        for name, values in (
            ("development", self.development_ids),
            ("validation", self.validation_ids),
            ("test", self.test_ids),
        ):
            parts.append(pd.DataFrame({"SK_ID_CURR": values, "split": name}))
        return pd.concat(parts, ignore_index=True)


def raw_file(config: dict[str, Any], filename: str) -> Path:
    return project_path(config, "raw_data") / filename


def require_raw_files(config: dict[str, Any]) -> list[Path]:
    """Return required paths or raise with an actionable message."""
    paths = [raw_file(config, name) for name in config["data"]["required_files"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = "\n".join(f"  - {path.name}" for path in missing)
        raise FileNotFoundError(
            "Home Credit raw data is incomplete. Place these files in "
            f"{project_path(config, 'raw_data')}:\n{names}"
        )
    return paths


def count_csv_rows(path: Path) -> int:
    """Count data rows without loading a full CSV into memory."""
    with path.open("rb") as handle:
        lines = sum(block.count(b"\n") for block in iter(lambda: handle.read(8 << 20), b""))
        if path.stat().st_size:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                lines += 1
    return max(lines - 1, 0)


def audit_raw_data(config: dict[str, Any]) -> dict[str, Any]:
    """Audit real files, columns, null rates, and target balance."""
    paths = require_raw_files(config)
    file_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    target_summary: dict[str, Any] = {}
    audit_chunk_size = min(int(config["data"].get("chunk_size", 250000)), 250000)
    target_name = str(config["data"]["target"])

    for path in paths:
        header = pd.read_csv(path, nrows=0)
        null_counts = pd.Series(0, index=header.columns, dtype="int64")
        data_types: dict[str, str] = {}
        streamed_rows = 0
        target_counts: dict[str, int] = {}
        for chunk in pd.read_csv(path, chunksize=audit_chunk_size, low_memory=False):
            streamed_rows += len(chunk)
            null_counts = null_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")
            if not data_types:
                data_types = {column: str(dtype) for column, dtype in chunk.dtypes.items()}
            if path.name == "application_train.csv" and target_name in chunk:
                counts = chunk[target_name].value_counts(dropna=False)
                for key, value in counts.items():
                    target_counts[str(key)] = target_counts.get(str(key), 0) + int(value)
        line_rows = count_csv_rows(path)
        if streamed_rows != line_rows:
            raise ValueError(
                f"Row audit mismatch for {path.name}: parser={streamed_rows}, lines={line_rows}"
            )
        file_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": streamed_rows,
                "columns": len(header.columns),
            }
        )
        for column in header.columns:
            column_rows.append(
                {
                    "file": path.name,
                    "column": column,
                    "dtype_first_chunk": data_types.get(column, "unknown"),
                    "null_count": int(null_counts[column]),
                    "null_fraction": float(null_counts[column] / streamed_rows)
                    if streamed_rows
                    else np.nan,
                }
            )
        if path.name == "application_train.csv":
            if target_name not in header.columns:
                raise ValueError(f"Target column {target_name!r} is absent from {path.name}")
            bad_count = target_counts.get("1", target_counts.get("1.0", 0))
            target_summary = {
                "target": target_name,
                "counts": target_counts,
                "bad_rate": float(bad_count / streamed_rows),
            }

    output = project_path(config, "artifacts") / "data_audit"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(file_rows).to_csv(output / "files.csv", index=False)
    pd.DataFrame(column_rows).to_csv(output / "source_columns.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"files": file_rows, "target": target_summary}, handle, indent=2)
    return {"files": file_rows, "target": target_summary}


def make_split(
    ids: Iterable[int],
    target: Iterable[int],
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> DatasetSplit:
    """Create reproducible stratified development, validation, and test splits."""
    ids_array = np.asarray(list(ids))
    target_array = np.asarray(list(target))
    holdout_fraction = validation_fraction + test_fraction
    if not 0 < holdout_fraction < 1:
        raise ValueError("Validation plus test fraction must be between zero and one")
    dev_ids, holdout_ids, _, holdout_y = train_test_split(
        ids_array,
        target_array,
        test_size=holdout_fraction,
        stratify=target_array,
        random_state=random_seed,
    )
    relative_test = test_fraction / holdout_fraction
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=relative_test,
        stratify=holdout_y,
        random_state=random_seed + 1,
    )
    return DatasetSplit(dev_ids, val_ids, test_ids)


def validate_split(split: DatasetSplit) -> None:
    """Fail when any applicant appears in more than one split."""
    sets = [
        set(split.development_ids.tolist()),
        set(split.validation_ids.tolist()),
        set(split.test_ids.tolist()),
    ]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Applicant identifiers overlap across data splits")
