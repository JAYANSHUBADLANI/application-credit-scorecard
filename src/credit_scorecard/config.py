"""Configuration loading and project path handling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and resolve the project root."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured project path against the repository root."""
    value = Path(config["paths"][key]).expanduser()
    if value.is_absolute():
        return value
    return Path(config["_project_root"]) / value


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create all generated output directories."""
    for key in ("processed_data", "artifacts", "figures"):
        project_path(config, key).mkdir(parents=True, exist_ok=True)

