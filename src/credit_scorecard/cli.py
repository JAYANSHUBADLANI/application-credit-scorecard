"""Command line interface for the scorecard workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any, Callable

from credit_scorecard.config import load_config, project_path
from credit_scorecard.data import audit_raw_data
from credit_scorecard.eda import create_eda
from credit_scorecard.pipeline import (
    build_features,
    clean_generated,
    run_all,
    train_models,
    validate_models,
)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _download_data(config: dict[str, Any]) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install the project Kaggle extra with "
            "python3 -m pip install -c constraints-tested.txt -e '.[kaggle]', then "
            "configure Kaggle credentials."
        )
    raw = project_path(config, "raw_data")
    raw.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "competitions",
        "download",
        "-c",
        str(config["data"]["competition"]),
        "-p",
        str(raw),
    ]
    subprocess.run(command, check=True)
    archives = sorted(raw.glob("*.zip"))
    if not archives:
        raise RuntimeError("Kaggle command completed but no ZIP archive was found")
    import zipfile

    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(raw)
    print(f"Extracted {len(archives)} archive(s) into {raw}")


def _command_with_config(
    function: Callable[[dict[str, Any]], Any],
    args: argparse.Namespace,
) -> None:
    config = load_config(args.config)
    result = function(config)
    if result is not None:
        _print_json(result)


def _train_summary(config: dict[str, Any]) -> dict[str, Any]:
    bundle = train_models(config)
    return {
        "models": ["application_only", "accepted_only", "parcelled", "oracle"],
        "selected_features": bundle.selected_features,
        "parcel_bad_odds_multipliers": sorted(
            float(value) for value in bundle.parcel_sensitivity
        ),
        "artifact": str(project_path(config, "artifacts") / "models" / "model_bundle.joblib"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-scorecard",
        description="Application scorecard and credit decisioning workflow",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "download-data": _download_data,
        "audit-data": audit_raw_data,
        "build-features": build_features,
        "eda": create_eda,
        "train": _train_summary,
        "validate": validate_models,
        "run-all": run_all,
        "clean-generated": clean_generated,
    }
    for name, function in commands.items():
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="config/project.yaml")
        command.set_defaults(handler=lambda args, fn=function: _command_with_config(fn, args))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
