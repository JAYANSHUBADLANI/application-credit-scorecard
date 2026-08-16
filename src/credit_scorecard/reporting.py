"""README result rendering from completed pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from credit_scorecard.config import project_path


START_MARKER = "<!-- RESULTS_START -->"
END_MARKER = "<!-- RESULTS_END -->"


def _results_markdown(summary: dict[str, Any]) -> str:
    features = ", ".join(f"`{feature}`" for feature in summary["selected_features"])
    return f"""{START_MARKER}
| Measure | Result |
|---|---:|
| Application-only test Gini | {summary['test_application_only_gini']:.4f} |
| Accepted-only relational test Gini | {summary['test_accepted_only_gini']:.4f} |
| Parcelled test Gini | {summary['test_gini']:.4f} |
| Oracle benchmark test Gini | {summary['test_oracle_gini']:.4f} |
| Parcelled Gini 95% bootstrap interval | {summary['test_gini_ci_lower']:.4f} to {summary['test_gini_ci_upper']:.4f} |
| Test KS | {summary['test_ks']:.4f} |
| Test Brier score | {summary['test_brier_score']:.4f} |
| Test calibration slope | {summary['test_calibration_slope']:.4f} |
| Test expected calibration error | {summary['test_expected_calibration_error']:.4f} |
| Test Hosmer-Lemeshow p-value | {summary['test_hosmer_lemeshow_p_value']:.4g} |
| Locked validation score cutoff | {summary['validation_optimal_score_cutoff']:.1f} |
| Validation approval rate at selected cutoff | {summary['validation_optimal_approval_rate']:.2%} |
| Test approval rate at locked cutoff | {summary['test_locked_cutoff_approval_rate']:.2%} |
| Test observed bad rate at locked cutoff | {summary['test_locked_cutoff_bad_rate']:.2%} |
| Test score PSI versus development | {summary['test_score_psi']:.4f} |
| Unlabeled application-test score PSI versus development | {summary['application_test_unlabeled_score_psi']:.4f} |
| Swap-in observed bad rate | {summary['swap_in_bad_rate']:.2%} |
| Swap-out observed bad rate | {summary['swap_out_bad_rate']:.2%} |
| Net realized swap value, illustrative units | {summary['swap_net_realized_value']:,.0f} |
| Coefficient sign review flags | {summary['coefficient_sign_review_flags']} |

Selected scorecard characteristics: {features}

These values were written from `artifacts/results_summary.json` after a completed run.
{END_MARKER}"""


def update_readme_results(config: dict[str, Any]) -> Path:
    root = Path(config["_project_root"])
    readme = root / "README.md"
    summary_path = project_path(config, "artifacts") / "results_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Results summary is absent. Run validation first.")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    text = readme.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise ValueError("README result markers are absent or malformed")
    end += len(END_MARKER)
    readme.write_text(text[:start] + _results_markdown(summary) + text[end:], encoding="utf-8")
    return readme
