# Application Credit Scorecard and Decisioning Strategy

[![tests](https://github.com/JAYANSHUBADLANI/application-credit-scorecard/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/application-credit-scorecard/actions/workflows/tests.yml)

I built this project to demonstrate an end-to-end consumer credit risk workflow, not a
leaderboard classifier. It converts an interpretable logistic model into a points-based
application scorecard, simulates selection bias and reject inference, validates the model,
and translates predicted risk into an approval strategy.

## Current status

The full pipeline has been run locally against the Home Credit competition files. Raw CSVs
remain outside the repository and are linked from `data/raw/`, so the 2.5 GB source dataset
is not duplicated or committed. Models, validation evidence, decision tables, figures, and
the reproducibility manifest are available under `artifacts/` and `reports/figures/`.

## Results

<!-- RESULTS_START -->
| Measure | Result |
|---|---:|
| Application-only test Gini | 0.4884 |
| Accepted-only relational test Gini | 0.5114 |
| Parcelled test Gini | 0.5065 |
| Oracle benchmark test Gini | 0.5128 |
| Parcelled Gini 95% bootstrap interval | 0.4932 to 0.5189 |
| Test KS | 0.3758 |
| Test Brier score | 0.0682 |
| Test calibration slope | 1.0494 |
| Test expected calibration error | 0.0075 |
| Test Hosmer-Lemeshow p-value | 2.793e-08 |
| Locked validation score cutoff | 543.6 |
| Validation approval rate at selected cutoff | 78.00% |
| Test approval rate at locked cutoff | 77.46% |
| Test observed bad rate at locked cutoff | 4.80% |
| Test score PSI versus development | 0.0005 |
| Unlabeled application-test score PSI versus development | 0.0016 |
| Swap-in observed bad rate | 11.36% |
| Swap-out observed bad rate | 13.75% |
| Net realized swap value, illustrative units | 17,101,861 |
| Coefficient sign review flags | 0 |

Selected scorecard characteristics: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `BUREAU_DEBT_CREDIT_RATIO_MEAN`, `DAYS_EMPLOYED`, `BUREAU_DAYS_CREDIT_MOST_RECENT`, `PREV_GRANTED_REQUESTED_RATIO_MEAN`, `APP_GOODS_CREDIT_RATIO`, `OCCUPATION_TYPE`, `BUREAU_ACTIVE_COUNT`, `CC_UTILIZATION_MEAN`, `INST_LATE_RATE`, `NAME_INCOME_TYPE`, `PREV_REFUSED_RATE`, `PREV_ACCEPTED_RATE`, `REGION_RATING_CLIENT_W_CITY`, `NAME_EDUCATION_TYPE`

These values were written from `artifacts/results_summary.json` after a completed run.
<!-- RESULTS_END -->

The relational scorecard improves discrimination over the application-only baseline. The
parcelled model is slightly less discriminating than the accepted-only model and the oracle
benchmark, which is an important limitation rather than something I hide. Parcelling makes
the assumed rejected population more conservative, and its value depends on the configured
bad-odds multiplier. The calibration slope and expected calibration error remain informative
alongside the small Hosmer-Lemeshow p-value, since that test is highly sensitive at this
sample size.

## What this project demonstrates

- Applicant-level feature engineering from the application, bureau, prior-application,
  installment, credit-card, and point-of-sale tables.
- Point-in-time filtering that removes future relative-date records before aggregation.
- Numeric monotonic coarse classing and bad-rate-ordered categorical grouping.
- Weight of Evidence transformation and Information Value variable screening.
- A logistic scorecard scaled to a base score, base odds, and points to double odds.
- An X-only synthetic acceptance process that creates biased observed outcomes.
- Accepted-only, fractional-parcelling, and oracle scorecard comparisons on frozen bin cuts.
- Parcel bad-odds sensitivity analysis across all configured multipliers.
- Gini, KS, Brier score, calibration, Hosmer-Lemeshow, PSI, and characteristic CSI.
- A genuine equal-volume swap set between application-only and relational scorecards.
- A profit cutoff based on EAD, LGD, margin, acquisition cost, and operating cost assumptions.
- Auto-approve, manual-review, and auto-decline operating bands with reason codes.

## Methodology

### Data design

I split labeled applicants into stratified development, validation, and test samples before
any target-aware transformation. This dataset does not contain a genuine application
timestamp. I therefore call these holdouts, not out-of-time validation. The Kaggle
`application_test.csv` population has no outcomes and is used only for score PSI and
characteristic CSI.

All auxiliary features follow an application-day-zero rule:

- Bureau records require nonpositive credit and update dates.
- Bureau monthly history requires `MONTHS_BALANCE <= 0`.
- Prior applications require `DAYS_DECISION <= 0`.
- Installments require an eligible prior contract plus nonpositive scheduled and actual
  payment dates.
- Card and point-of-sale snapshots require an eligible prior contract and a nonpositive
  month balance.

The run writes `data/processed/feature_lineage.csv` so every engineered characteristic has
its source, cutoff rule, aggregation, and availability rationale.

### WOE and scorecard

The event is `TARGET = 1`, meaning default. I define:

```text
WOE = ln(distribution of good accounts / distribution of bad accounts)
IV  = sum((good distribution - bad distribution) * WOE)
```

Numeric variables start with development-sample quantiles, merge small adjacent bins, and
use pooled adjacent violators to enforce a monotonic bad-rate pattern. Categorical variables
receive missing and unseen handling, rare-level grouping, and bad-rate-ordered groups. All
cut points, category maps, WOE values, and IV decisions are learned on accepted development
records only.

For predicted bad log odds `z`, the points conversion is:

```text
Factor = PDO / ln(2)
Offset = BaseScore - Factor * ln(BaseOddsGoodToBad)
Score  = Offset - Factor * z
```

The default configuration gives 600 points at 50 good accounts to 1 bad account, with 20
additional points whenever good-to-bad odds double. Higher scores mean lower predicted risk.

### Reject inference experiment

Home Credit does not identify genuinely rejected applicants with later repayment outcomes.
I therefore treat reject inference as a controlled synthetic experiment:

1. A transparent policy uses only development-fitted percentiles of observable predictors.
2. The policy samples accepted and rejected applicants without reading `TARGET`.
3. Rejected outcomes are hidden from accepted-only model training.
4. The accepted-only scorecard assigns frozen score bands to rejected applicants.
5. Parcelling increases accepted bad odds within each band by a stated multiplier.
6. Fractional good and bad mass re-estimates WOE values on frozen cut points and refits the
   logistic scorecard.
7. An oracle model uses all development outcomes only as a simulation benchmark.

Because this is a simulation, I retain rejected outcomes in an isolated diagnostics path.
The parcel table compares each assumed rejected bad rate with the hidden realized bad rate by
score band. Those hidden outcomes never choose the configured multiplier, bins, coefficients,
or approval cutoff.

This experiment cannot prove that parcelling would recover outcomes for real rejected
applicants. Its conclusion is conditional on the synthetic policy and the bad-odds multiplier.

### Validation and stability

The validation suite reports discrimination, calibration, and stability separately. It
includes bootstrap intervals for Gini and KS. Hosmer-Lemeshow is included because it is common
in model validation, but I do not use it as a single pass or fail test because it is sensitive
to sample size. PSI and CSI values are diagnostic measures rather than universal regulatory
thresholds.

### Swap set and cutoff

The swap set compares application-only and relationally enriched scorecards that are both
fitted on the same synthetically accepted development population. This isolates the value of
relational characteristics from the separate reject-inference adjustment. Both approve
exactly the same number of test applicants, with applicant ID used as a deterministic tie
breaker. The output reports both-approved, swap-in, swap-out, and both-declined populations
with their bad rates, exposure, and realized backtest value.

Cutoff selection is a separate validation-sample optimization. Expected value uses the
configured exposure factor, LGD, performing-account margin, pre-default margin, acquisition
cost, and operating cost. The selected validation cutoff is locked before test evaluation.
These economics are illustrative because `AMT_CREDIT` is only an EAD proxy and the dataset
does not provide the lender's realized margins or recoveries.

The run also reports the highest validation approval rate that satisfies each configured
maximum bad-rate scenario. A tighter limit raises the score threshold and reduces approvals.
A looser limit permits a lower threshold and more approvals. I keep these constrained policy
options beside the unconstrained profit optimum so the risk-appetite choice is explicit.

## Repository layout

```text
config/project.yaml              Model, validation, and economic assumptions
src/credit_scorecard/            Reusable pipeline implementation
tests/                           Calculation, leakage, and integration tests
data/raw/                        Kaggle CSV files, excluded from Git
data/processed/                  Applicant matrices and lineage, excluded from Git
artifacts/                       Models, tables, predictions, and run metadata
reports/figures/                 Generated validation and decisioning charts
PROGRESS.md                      Work completed and remaining handoffs
```

## Reproduce the project

Use Python 3.12. A machine with at least 8 GB of memory is recommended. The large source
tables are read in chunks with only required columns selected.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested.txt -e ".[dev,kaggle]"
```

Accept the Home Credit competition terms on Kaggle and configure the Kaggle API client. Then:

```bash
python -m credit_scorecard download-data --config config/project.yaml
python -m credit_scorecard run-all --config config/project.yaml
python -m pytest
```

If the files were downloaded elsewhere, place the eight required CSV files directly in
`data/raw/` and start with `run-all`.

The individual stages are also available through `make audit`, `make features`, `make train`,
and `make validate`.

## Generated evidence

A successful run creates:

- Exact source row and column audit tables.
- A frozen split manifest and configuration snapshot.
- Full WOE bin tables, IV decisions, coefficients, and scorecard points.
- Synthetic acceptance metadata and parcel assumptions.
- Validation metrics for full, synthetically accepted, and synthetically rejected samples.
- Calibration, PSI, CSI, cutoff, decision-band, sensitivity, and swap-set tables.
- Applicant predictions, scores, operating decisions, and reason-code support.
- Figures for discrimination, calibration, stability, and cutoff tradeoffs.

## Limitations

- Reject inference is synthetic and assumption dependent.
- The holdout is not genuinely out of time.
- `AMT_CREDIT` is an exposure proxy and the economic inputs are illustrative.
- External source scores are available predictors but their proprietary construction is not
  observable in this dataset.
- Sensitive-variable eligibility depends on jurisdiction and lender policy. `CODE_GENDER` is
  excluded by default, and other variables require governance review before real use.
- This is a portfolio backtest, not a production lending policy or regulatory submission.

## License

This project is available under the MIT License.
