# Project progress

Last updated: 2026-08-07

## Completed

- Created a reproducible Python project and configuration.
- Implemented raw-file verification and application schema, null-rate, and target audit.
- Implemented point-in-time features across all six auxiliary history sources.
- Implemented numeric and categorical WOE binning, IV screening, and correlation filtering.
- Implemented logistic fitting, PDO score scaling, points tables, and reason codes.
- Implemented X-only synthetic acceptance, fractional parcelling, and an oracle benchmark.
- Implemented Gini, KS, calibration, Brier, Hosmer-Lemeshow, PSI, and CSI.
- Implemented equal-volume swap sets, cutoff economics, decision bands, and sensitivities.
- Added CLI entry points, staged Make targets, and artifact persistence.
- Added 17 automated tests, including a full synthetic raw-data run through reporting.
- Confirmed all tests pass.
- Linked the complete 2.5 GB Home Credit dataset from the Desktop without duplicating it.
- Audited every source table, including row counts, columns, null rates, and target balance.
- Built 201 applicant-level characteristics for 307,511 labeled applicants.
- Ran the final 17-characteristic scorecard and all validation and decisioning stages.
- Removed all coefficient sign reversals using development information only.
- Generated the final README metrics, figures, decision records, and run manifest.

## Verification completed

- Unit tests cover point-in-time filters, DPD direction, splits, WOE, score scaling,
  synthetic acceptance, hidden rejected outcomes, parcelling, validation metrics,
  stability, economics, and swap-set reconciliation.
- The integration fixture exercises all eight source files and produces models, tables,
  decisions, reason codes, figures, and README results.

## Final verification

- All 17 automated tests pass after the real-data run.
- All four final coefficient tables have zero sign-review flags.
- The validation cutoff was selected before evaluating the locked test policy.
- Raw data remains excluded from version control.

## Important limitations

- Reject inference is synthetic and assumption dependent.
- The dataset has no genuine application timestamp, so validation is not out of time.
- Profit values use illustrative EAD, LGD, margin, and operating-cost assumptions.
- The parcelled model does not outperform the accepted-only or oracle scorecards on Gini.
- Raw-data links depend on the current Desktop dataset location.
