# AI-Powered Exchange Activity Prediction Platform
### AIESEC MC India — 2026 Forecast

<p align="center">
  <a href="https://jayed-alam-mansur-exchange-activity-prediction-platf-app-n1n2zt.streamlit.app"><b>▶ Live dashboard</b></a>
</p>

Forecasting **high-activity exchange months for 2026** from **real AIESEC MC India
operational data** (01-01-2022 → 31-12-2025, 66,253 applications pulled live from the
AIESEC Analytics API), with a reproducible ML pipeline, a 12-model comparison, and an
interactive dashboard.

<p align="center">
  <img src="outputs/figures/07_forecast_2026.png" alt="2026 exchange application forecast" width="100%">
</p>

<p align="center">
  <b>Predicted high-activity months for 2026: March · April · May</b><br>
  <sub>Selected model: Holt-Winters · MAE 198.0 · MAPE 12.9% · 24 walk-forward origins</sub><br>
  <sub><b>Real data</b> · EXPA office 1585 · 48/48 monthly windows · 33 active LCs</sub>
</p>

<p align="center">
  <a href="#how-to-run-this-project">How to run</a> ·
  <a href="#results">Results</a> ·
  <a href="#the-2026-prediction-csv">Prediction CSV</a> ·
  <a href="#methodology">Methodology</a> ·
  <a href="#dashboard">Dashboard</a> ·
  <a href="#data-provenance-read-this">Data provenance</a>
</p>

> Built as a technical submission for the **AIESEC Global In-House Development (GID) —
> AI/ML Engineer** role.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Problem statement](#problem-statement)
3. [Solution](#solution)
4. [Data provenance — read this](#data-provenance-read-this)
5. [Architecture](#architecture)
6. [Data source: the AIESEC Analytics API](#data-source-the-aiesec-analytics-api)
7. [Methodology](#methodology)
8. [Results](#results)
9. [The 2026 prediction CSV](#the-2026-prediction-csv)
10. [Dashboard](#dashboard)
11. [How to run this project](#how-to-run-this-project)
12. [Project structure](#project-structure)
13. [Engineering practices](#engineering-practices)
14. [Testing](#testing)
15. [Limitations](#limitations)
16. [Future improvements](#future-improvements)

---

## Project overview

AIESEC's Member Committee for India runs exchange programmes — **Global Volunteer (GV)**,
**Global Talent (GTa)** and **Global Teacher (GTe)**, each in an incoming and outgoing
direction — across ~20 Local Committees.

Exchange activity is **strongly seasonal**, tracking the Indian academic calendar, and
**operationally lumpy**: the busiest month carries ~1.8× the volume of the quietest one.

This platform answers one operational question:

> ### Which months of 2026 will be high-activity, so capacity can be staged ahead of them?

It also delivers the funnel, entity and product analysis needed to act on that answer.

**At a glance**

| | |
|---|---|
| **Data source** | **Real** AIESEC Analytics API — office 1585, 48/48 windows |
| **Data window** | 01-01-2022 → 31-12-2025 (48 months, 66,253 applications) |
| **Grain** | month × Local Committee × programme — 4,690 non-empty cells |
| **Funnel modelled** | `APP → ACH → ACC → APD → RE → FI → CO` |
| **Features engineered** | 40, across time / historical / operational families |
| **Models compared** | 12, across 4 tiers |
| **Validation** | rolling-origin walk-forward, 24 out-of-sample origins |
| **Selected model** | Holt-Winters — MAE **198.0**, MAPE **12.9%** |
| **Deliverable** | [`outputs/predictions_2026.csv`](outputs/predictions_2026.csv) |
| **Tests** | 56, all passing |

---

## Problem statement

Planning today is **reactive**. Without a forecast, an MC must either:

1. **Staff for the average** — and be overwhelmed in March, idle in January; or
2. **Staff for the peak** — and carry expensive slack for eight months a year.

Both waste the scarcest resource in a volunteer organisation: **member time**.

Three concrete operational consequences:

| Consequence | Why it happens |
|---|---|
| **Matching capacity misses the peak** | Reviewer bandwidth is provisioned *after* applications spike, so a queue builds before anyone reacts |
| **Partner supply is misaligned** | Opportunities are sourced on a flat cadence while demand is seasonal — peak months run short, quiet months run long |
| **Funnel leaks stay invisible** | Without stage-by-stage conversion, effort goes to the *loudest* problem rather than the *largest* one |

A forecast converts all three from reactive to planned.

---

## Solution

An end-to-end platform, run by a single command:

```bash
python run_pipeline.py --step all
```

| Phase | What it does |
|---|---|
| **1 · Collect** | Pulls 48 monthly windows from the AIESEC Analytics API with authentication, retry/backoff, rate-limit handling and pagination. Persists every raw payload. |
| **2 · Process** | Parses nested aggregation JSON into a tidy panel; validates funnel monotonicity, month coverage and value ranges. |
| **3 · Analyse** | Trends, YoY growth, seasonality, peak months, the full exchange funnel, LC and programme performance. |
| **4 · Engineer** | 40 features across time, historical and operational families — with a structural no-leakage guarantee. |
| **5 · Model** | Trains and backtests 12 models across 4 tiers; selects on accuracy, stability and explainability. |
| **6 · Predict** | Forecasts all 12 months of 2026 with 95% prediction intervals and Low/Medium/High classification. |
| **7 · Visualise** | Four-page Streamlit dashboard plus 10 static figures and auto-generated insights. |

---

## Data provenance (read this)

**Every number in this repository is real AIESEC operational data**, pulled from the live
AIESEC Analytics API for **AIESEC in India (EXPA office `1585`)** over
**2022-01-01 → 2025-12-31**.

| | |
|---|---|
| Source | `GET https://analytics.api.aiesec.org/v2/applications/analyze.json` |
| Filter namespace | `performance_v3[office_id]=1585` |
| Window | 48 monthly pulls, 2022-01 → 2025-12, **48/48 succeeded** |
| Scale | 66,253 applications · 2,344 realizations · 33 active LCs · 3 products |
| Flag | `is_reference_data: false` in `data/raw/api_responses.json` |

The dashboard sidebar reports the source as **"AIESEC Analytics API"**.

### How the response is shaped

`performance_v3` returns a **flat** aggregation:

```jsonc
{"response": {
  "1449": {                              // child office (LC) id
    "applied_total":   {"doc_count": 6939, "applicants": {"value": 3839}},
    "i_applied_7":     {"doc_count": 1204, "applicants": {"value":  812}},
    "i_matched_7":     {"doc_count":  118, "applicants": {"value":  110}},
    "o_realized_8":    {"doc_count":   12, "applicants": {"value":   12}}
    // ... <direction>_<status>_<programme_id> for every combination
  },
  "1393": { /* ... */ },
  "applied_total": {"doc_count": 66253}   // MC-level repeat of every metric
}}
```

Two properties of this shape drive the parser (`src/preprocessing/cleaning.py`):

1. **The MC-level keys are the sum of the per-office nodes.** Parsing both would exactly
   double every count, so `parse_v3_payload` reads *only* the numeric office keys.
2. **`doc_count` is used, never `applicants.value`.** `applicants` is an Elasticsearch
   *cardinality* aggregation (distinct people) and is **not additive**: someone who applies
   in January and again in February counts once in a yearly window but twice across two
   monthly windows. `doc_count` counts application records and sums cleanly.

Property 2 is verified, not assumed — the 48 monthly pulls reconcile to the single
four-year aggregate with **zero delta on every stage**:

| metric | Σ 48 monthly windows | one 4-year query | delta |
|---|---:|---:|---:|
| `applied_total` | 66,253 | 66,253 | **0** |
| `matched_total` | 10,036 | 10,036 | **0** |
| `an_accepted_total` | 6,617 | 6,617 | **0** |
| `approved_total` | 3,160 | 3,160 | **0** |
| `realized_total` | 2,344 | 2,344 | **0** |
| `finished_total` | 2,253 | 2,253 | **0** |
| `completed_total` | 1,381 | 1,381 | **0** |

### Why per-row funnel monotonicity does not hold (and must not be enforced)

The API counts each status in the window in which **that status occurred**, not in the
window the application was created. An application submitted in March and realized in
August contributes `APP` to March and `RE` to August. A quiet intake month that finally
realizes an older cohort therefore legitimately shows `RE > APP`.

Validation therefore enforces:

* **Error** — the funnel must be monotonic **in aggregate** over the full window, where
  every cohort's stages fall inside the window. It is: 66,253 → 10,036 → 6,617 → 3,160 →
  2,344 → 2,253 → 1,381.
* **Warning** — per-row inversions (1,525 of 4,690 rows) are reported and explained, not
  treated as corruption.

### Identifiers, verified rather than assumed

| Thing | How it was verified |
|---|---|
| Office `1585` = AIESEC in India | `gis-api.aiesec.org/v2/committees/1585.json` → `{"full_name":"AIESEC in India","tag":"MC"}` |
| LC names | `gis-api.aiesec.org/v2/committees/{id}.json`, one call per office, cached to `data/raw/committees.json` (40/40 resolved) |
| Programme ids | `gis-api.aiesec.org/v2/programmes.json` → **5 = GE, 7 = GV, 8 = GTa, 9 = GTe** |

Programme ids 1, 2 and 5 are retired and return zero across all 48 windows; they are
dropped during parsing.

### Offline fallback

`src/api/reference_data.py` generates a seeded stand-in dataset so the pipeline stays
runnable without credentials (`python run_pipeline.py --step all --use-reference-data`).
It is disabled by default and **nothing in the committed artefacts is derived from it**.
When used, `is_reference_data: true` propagates to the raw envelope, the pipeline logs and
the dashboard sidebar.

### Credentials

The access token is read from `AIESEC_ACCESS_TOKEN` in `.env`, which is git-ignored and
never committed. It is redacted from every log line and exception message
(`redact_token`). GIS tokens are short-lived — if collection returns HTTP 401, refresh the
token and re-run `--step collect`.

### Reproducing the pull

```bash
# .env  (git-ignored; copy from .env.example)
AIESEC_ACCESS_TOKEN=<your GIS token>
AIESEC_OFFICE_ID=1585
```

```bash
python run_pipeline.py --step all
```

The pipeline detects the token and re-collects all 48 windows. Every raw payload is
persisted to `data/raw/api_responses.json` first, so parsing, modelling and figures can be
rebuilt afterwards with no further network access:

```bash
python run_pipeline.py --step process   # ... features, train, figures
```

See **[`docs/API_INTEGRATION.md`](docs/API_INTEGRATION.md)** for the go-live checklist,
including how to confirm the office filter actually applied before trusting a backfill.

### Configuration, verified against live endpoints

Every identifier the pipeline depends on is confirmed rather than assumed, and
`config/config.yaml` carries `verified: true`:

| Value | Config key | Setting | Verified against |
|---|---|---|---|
| MC India EXPA office id | `api.office_id` | `1585` | `/v2/committees/1585.json` |
| Filter namespace | `api.filter_namespace` | `performance_v3` | live 200 vs empty aggregation |
| Programme id → product | `products.mapping` | `5:GE, 7:GV, 8:GTa, 9:GTe` | `/v2/programmes.json` |

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │   AIESEC Analytics API (GIS)         │
                    │   /v2/applications/analyze.json      │
                    └──────────────────┬───────────────────┘
                                       │ access_token · monthly windows
                                       │ retry + backoff + rate limiting
                                       ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ PHASE 1 · COLLECTION            src/api/aiesec_api.py              │
   │   48 monthly windows × N programmes                                │
   │   → data/raw/api_responses.json  (+ provenance envelope)           │
   │   offline fallback: src/api/reference_data.py — same JSON shape    │
   └────────────────────────────────┬───────────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ PHASE 2 · PARSE + VALIDATE      src/preprocessing/cleaning.py      │
   │   tolerant recursive walk of ES-style buckets → tidy panel         │
   │   validate: monotonic funnel · no gaps · no negatives · no dupes   │
   │   → data/processed/exchange_data.csv        (4,690 rows)           │
   └────────────────────────────────┬───────────────────────────────────┘
                                    ▼
   ┌─────────────────────────────┐  ┌─────────────────────────────────┐
   │ PHASE 3a · EDA              │  │ PHASE 3b · FEATURES             │
   │ preprocessing/analysis.py   │  │ preprocessing/features.py       │
   │  trends · seasonality       │  │  time · historical · operational│
   │  funnel · entity · product  │  │  40 features · zero leakage     │
   │  → outputs/reports/*.csv    │  │  → data/processed/features.csv  │
   └─────────────────────────────┘  └──────────────┬──────────────────┘
                                                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ PHASE 4 · MODELLING       models/forecasting.py · models/train.py  │
   │   12 candidates × 24 walk-forward origins → MAE / RMSE / MAPE      │
   │   selection: accuracy → stability → explainability                 │
   │   → models/trained_model.pkl                                       │
   │   → outputs/predictions_2026.csv                                   │
   └────────────────────────────────┬───────────────────────────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ PHASE 5 · PRESENTATION     app.py · visualization/{charts,figures} │
   │   4-page Streamlit dashboard · 10 static figures · auto-insights   │
   └────────────────────────────────────────────────────────────────────┘
```

Each phase reads the previous phase's artefacts **from disk**, so any stage can be rerun
independently — `python run_pipeline.py --step train` does not re-collect data.

---

## Data source: the AIESEC Analytics API

**Endpoint** — `GET https://analytics.api.aiesec.org/v2/applications/analyze.json`

### Authentication

A GIS `access_token` passed as a **query parameter** (not an `Authorization` header).
Tokens are short-lived and issued through EXPA. Observed status codes and how the client
handles each:

| Code | Meaning | Client behaviour |
|---|---|---|
| `402` | token **missing** | `AuthenticationError` — fatal, no retry |
| `401` | token invalid/expired (`sub_code: invalid_token`) | `AuthenticationError` — fatal |
| `403` | token valid, no rights to this office | `AuthenticationError` — fatal |
| `429` | rate limited | honours `Retry-After`; else exponential backoff + jitter |
| `5xx` | server error | up to 5 retries with backoff |

> `402` for a missing token is unusual enough to matter: a client that only treats
> `401`/`403` as auth failures will misclassify it as retryable and burn its entire retry
> budget on a request that can never succeed. This one was found by probing, not by
> reading docs.

### Request format

Filters use nested bracket syntax:

```
GET /v2/applications/analyze.json
  ?access_token=<token>                       # secret — from environment only
  &start_date=2022-01-01                      # inclusive
  &end_date=2022-01-31                        # inclusive
  &performance[office_id]=<MC India office id>
  &performance[include_child_offices]=true    # includes the LCs beneath the MC
  &performance[programmes][]=1
  &page=1&per_page=100
```

One request per `(month, programme)` — 48 months × 3 active products = **144 request
groups** for the full backfill.

**Why monthly windows:** payloads stay small enough to inspect by hand; a failure loses
one month rather than four years; and the window matches the forecasting grain.

### Response format

Elasticsearch-style aggregations — nested `buckets[]` arrays carrying `doc_count` per
funnel status and per child office:

```json
{
  "analytics": {
    "offices": {
      "buckets": [{
        "key": 90001, "key_as_string": "AIESEC in Example", "doc_count": 143,
        "directions": {
          "buckets": [{
            "key": "outgoing", "doc_count": 106,
            "statuses": { "buckets": [
              {"key": "applied",   "doc_count": 106},
              {"key": "achieved",  "doc_count": 44},
              {"key": "realized",  "doc_count": 18}
            ]}
          }]
        }
      }]
    }
  },
  "meta": {"page": 1, "total_pages": 1}
}
```

### Why the parser is deliberately defensive

Rather than indexing a fixed path like
`analytics.offices.buckets[].directions.buckets[].statuses.buckets[]`, `parse_payload`
performs a **tolerant recursive walk**: it descends any `{"buckets": [...]}` node and
classifies each bucket as an entity, direction, product or funnel status by inspecting
its key.

| Bucket key matches | Interpreted as | Recognised aggregation names |
|---|---|---|
| a key in `funnel.api_status_map` | funnel stage measurement | `statuses`, `status`, `stages` |
| — | entity (Local Committee) | `offices`, `entities`, `committees`, `lc` |
| — | direction | `directions`, `types`, `flow` |
| — | product | `programmes`, `products`, `programs` |

Consequences, each pinned by a test:

- Nesting **order** can change — office-above-direction and direction-above-office parse
  to the identical cell.
- Extra grouping levels are descended, not fatal.
- Unknown status keys are ignored rather than crashing.
- Direction spellings normalise (`og` / `out` / `sending` → `outgoing`).

Since I could not read the official schema, this was the responsible design: the only
change that would break ingestion is a new **status vocabulary**, and that vocabulary is
itself configuration (`funnel.api_status_map`).

---

## Methodology

### Step 1 — Collection

48 monthly windows × active programmes, each persisted with a provenance envelope
recording when, what and how it was pulled. A failure in one window is logged and
recorded in `metadata.failed_windows` but does not abort the run — a four-year backfill
should not be lost to one transient 502.

### Step 2 — Processing and validation

Output schema — **`data/processed/exchange_data.csv`** (4,690 rows: the 48 months × 33
active LCs × 6 programmes grid, with all-zero cells dropped):

| Column | Description |
|---|---|
| `date`, `year`, `month`, `month_name`, `quarter` | Calendar keys |
| `entity` | Local Committee name |
| `product` | `GV` / `GTa` / `GTe` |
| `direction` | `incoming` / `outgoing` |
| `programme` | Direction-prefixed product, e.g. `oGV`, `iGTa` |
| `APP … CO` | The seven funnel stage counts |

Validation is **not optional** and runs on every build:

1. Required columns present
2. No negative counts
3. **Funnel monotonicity** — no downstream stage exceeds its upstream stage
4. **Complete month coverage** over the configured window
5. No duplicate `(date, entity, programme)` cells

Failures are logged loudly; with `strict=True` they raise.

### Step 3 — The exchange funnel

```
APP  →  ACH  →  ACC  →  APD  →  RE  →  FI  →  CO
```

| Stage | Meaning |
|---|---|
| `APP` | Applied |
| `ACH` | Achieved / Matched |
| `ACC` | Accepted |
| `APD` | Approved |
| `RE` | Realized |
| `FI` | Finished |
| `CO` | Completed |

Three metrics per stage:

- **`conversion_from_previous_pct`** — share of the *previous* stage that advanced. The
  operational lever.
- **`conversion_from_APP_pct`** — cumulative efficiency from application.
- **`dropoff_count` / `dropoff_pct`** — absolute and relative volume lost at this step.

The largest single leak is flagged automatically — it is the highest-leverage
intervention point, and it is not always the stage with the worst *percentage*.

### Step 4 — Feature engineering (40 features)

| Family | Features |
|---|---|
| **Time** (14) | year, month, quarter, season, month index, 2 Fourier harmonics (`sin`/`cos` × 2), Indian academic-calendar flags (break / exam / semester-start / year-end), days in month |
| **Historical** (19) | lags 1/2/3/6/12 · rolling mean + std over 3/6/12 · rolling max/min 12 · MoM growth · YoY growth · `lag1÷lag12` · `lag1÷roll12` · 6-month trend slope · same-calendar-month expanding mean |
| **Operational** (7) | trailing APP→RE and APP→ACH conversion · realizations · outgoing share · GV share · entity concentration (HHI) · active entities — **all lagged one month** |

**Why Fourier terms instead of month dummies.** Two harmonics express smooth annual
seasonality with **4 parameters** rather than the **11** an 11-dummy encoding needs —
materially more sample-efficient on 48 observations.

**Why the Indian academic calendar.** Exchange participation is student participation.
Break months (May, Jun, Dec, Jan) are when students can actually travel; exam months
(Mar, Apr, Nov) suppress applications; semester starts (Jan, Jul, Aug) are when
recruitment drives run.

#### The no-leakage guarantee

This is the single most important correctness property in the repository.

**One function — `compute_feature_row` — builds a feature vector, and it is used for
*both* the training matrix and every step of the recursive forecast.** Train/serve skew
is therefore *structurally impossible* rather than merely avoided by discipline.

Two tests pin it. Mutating the value **at** the target month, and mutating **all future
months**, must each leave the feature row byte-identical:

```
Target-month mutation : identical
Future mutation       : identical
```

If this property fails, every metric reported below is optimistic and the model is
worthless in production. It is worth testing explicitly.

### Step 5 — Modelling: start simple, escalate only if earned

With 48 monthly observations, a deep network is not a modelling choice — it is a way to
overfit confidently. **Four tiers compete and the backtest decides:**

| Tier | Models | Rationale |
|---|---|---|
| **1 · Baselines** | naive last · moving average (3, 6) · seasonal naive · seasonal naive + damped drift | Seasonal naive is genuinely hard to beat on a strongly annual series. Any model that cannot beat it has earned nothing. |
| **2 · Linear** | linear regression · ridge | With 36 rows and 40 features, regularisation is a serious contender, not a formality. |
| **3 · Classical time series** | Holt-Winters · SARIMA · Prophet *(optional)* | Purpose-built for trend + seasonality on short series. |
| **4 · Tree ensembles** | random forest · gradient boosting · XGBoost | Can capture interactions — if there is enough data to find them. |

All twelve implement one `BaseForecaster` interface, so the backtest, selection and
forecasting code is written once. Feature-based regressors forecast **recursively**:
predict one month, append it to the working history, rebuild features, repeat.

### Step 6 — Evaluation: rolling-origin cross-validation

**Expanding-window walk-forward.** At each of **24 origins**, the model refits on
everything observed up to that point and predicts one month ahead:

```
origin 24:  train [2022-01 … 2023-12]  →  predict 2024-01  →  compare
origin 25:  train [2022-01 … 2024-01]  →  predict 2024-02  →  compare
…
origin 47:  train [2022-01 … 2025-11]  →  predict 2025-12  →  compare
```

This mirrors how the platform would actually be used and — unlike a random K-fold split —
**never lets the model see the future**. A fresh model instance is constructed per origin
so no state leaks between folds.

Reported per model: **MAE**, **RMSE**, **MAPE**, plus sMAPE, error standard deviation
(stability) and worst-case absolute error.

### Step 7 — Model selection: accuracy → stability → explainability

1. Rank by **MAE**.
2. Every model within **5%** of the best MAE is treated as **tied on accuracy** — a 2%
   difference over 24 origins is noise, not signal.
3. Among the tied set, score
   `0.5 × accuracy + 0.3 × stability + 0.2 × explainability` (lower is better).

> **A subtlety worth flagging.** Accuracy *inside* the band is scored against the
> **tolerance width**, not min-max normalised across the tied set. Min-max would stretch
> whatever spread happens to exist back out to a full unit — re-inflating the very
> difference just declared negligible and defeating the purpose of having a tolerance.
> My first implementation got this wrong; a regression test caught it and now pins it.

The complete scoring table is written to `outputs/reports/model_evaluation.csv`, so the
choice is **auditable rather than asserted**.

### Step 8 — Prediction intervals

**Empirical**, built from the model's **own walk-forward residuals** — the errors it
actually made on this series, rather than a parametric interval resting on distributional
assumptions the residuals may not satisfy. Widened by `√h` as the horizon extends
(standard random-walk scaling).

### Step 9 — Activity levels, and a failure mode worth naming

The brief specifies **High = top 25%, Medium = middle 50%, Low = bottom 25%** from the
historical distribution.

**Applied literally, that degrades on a growing series.** Because 2026 sits above the
2022–2025 trend, the whole forecast year clears the historical median — the labels come
out **7 High, 5 Medium, 0 Low**. The bottom quartile becomes unreachable and the scale
silently loses a category. Push the trend harder and it degenerates completely to
all-High (a test pins that extreme case).

So **three methods are computed, and all three ship in the output CSV**:

| Method | Definition | CSV column |
|---|---|---|
| **`trend_adjusted`** *(default)* | Classical **ratio-to-moving-average**: divide each historical month by a *centred* 12-month mean to strip the trend, leaving a pure seasonal factor; divide each forecast month by the forecast year's own mean (the same quantity over a full year). Quartiles of the historical seasonal factors then classify the forecast. Compares months on **seasonal strength** — what "high-activity month" operationally means. | `Activity Level` |
| `historical_quartiles` | The literal reading of the brief. | `activity_level_vs_history` |
| `forecast_quartiles` | Quartiles of the 12 forecast values themselves. | `activity_level_within_2026` |

The three required columns are preserved **exactly** as the first three columns of the
CSV; everything else is additive.

---

## Results

### Model comparison — 24 walk-forward origins

| # | Model | Family | MAE ↓ | RMSE | MAPE | Error σ | Max error |
|---|---|---|---:|---:|---:|---:|---:|
| **1** | **holt_winters** (selected) | time series | **198.0** | **241.2** | **12.88%** | **137.6** | **538.7** |
| 2 | xgboost | ensemble | 210.3 | 288.9 | 12.58% | 198.2 | 708.5 |
| 3 | seasonal_naive_drift | baseline | 211.9 | 277.7 | 13.04% | 179.5 | 747.4 |
| 4 | random_forest | ensemble | 216.2 | 304.8 | 12.58% | 214.9 | 697.4 |
| 5 | gradient_boosting | ensemble | 226.0 | 307.3 | 13.56% | 208.2 | 736.8 |
| 6 | seasonal_naive | baseline | 237.9 | 294.1 | 14.94% | 172.9 | 773.0 |
| 7 | ridge | linear | 239.9 | 334.4 | 15.69% | 232.9 | 822.6 |
| 8 | naive_last | baseline | 283.2 | 372.4 | 17.57% | 241.9 | 832.0 |
| 9 | sarima | time series | 288.4 | 407.9 | 18.28% | 288.5 | 1231.4 |
| 10 | moving_average_6 | baseline | 301.3 | 389.4 | 18.68% | 246.8 | 956.2 |
| 11 | moving_average_3 | baseline | 325.3 | 426.3 | 20.86% | 275.5 | 1015.0 |
| 12 | linear_regression | linear | 360.4 | 520.1 | 25.27% | 375.0 | 1236.3 |

![Model comparison](outputs/figures/09_model_comparison.png)

**Selected: Holt-Winters** — MAE 198.0 (MAPE 12.88%, ≈87% accuracy), the only model inside
the 5% tolerance band, and simultaneously the most stable (lowest error σ at 137.6) and
the least biased (−2.8 against XGBoost's +130.2). No tie-break was needed.

#### The finding worth stating plainly

**A trivial baseline is statistically indistinguishable from XGBoost.**
`seasonal_naive_drift` — "last year's same month, scaled by YoY growth", a one-line rule —
lands at MAE 211.9 against XGBoost's 210.3. That is a **0.8% gap** over 24 origins. Three
gradient-boosted / bagged ensembles, tuned across 40 features, bought essentially nothing
over arithmetic a volunteer could do by hand.

Meanwhile the model that won is a 1960s exponential-smoothing method with three
parameters, and **plain linear regression finished dead last** — worse than every naive
baseline, at more than double the winner's error.

This is the brief's *"do not immediately jump to deep learning"* instruction demonstrated
rather than asserted — and it is only demonstrable because the suite spans four tiers and
backtests all of them instead of assuming the most sophisticated model wins.

Two secondary observations:

- **SARIMA is unstable, not just inaccurate.** Error σ of 289 and a worst case of 1,231
  show it diverging badly on short history. Mean error alone would have hidden that.
- **Moving averages rank near the bottom.** On a series with a 1.7× seasonal swing,
  smoothing destroys the signal that matters.

### Walk-forward fit of the selected model

![Backtest](outputs/figures/10_backtest.png)

### Historical trend and growth

![Monthly trend](outputs/figures/01_monthly_trend.png)

| Year | Total applications | Avg/month | Peak month | Realizations | APP→RE | YoY growth |
|---|---:|---:|---|---:|---:|---:|
| 2022 | 13,874 | 1,156 | May | 366 | 2.64% | — |
| 2023 | 15,511 | 1,293 | Mar | 664 | **4.28%** | +11.8% |
| 2024 | 17,639 | 1,470 | Mar | 691 | 3.92% | **+13.7%** |
| 2025 | 19,229 | 1,602 | Apr | 623 | 3.24% | +9.0% |

**CAGR 2022 → 2025: +11.5%.** Growth is steady rather than decelerating — 11.8% → 13.7%
→ 9.0% is roughly flat double-digit growth.

**The important finding is the divergence between the two columns.** Applications rose
**+38.6%** from 2022 to 2025, but realizations **peaked in 2024 (691) and fell to 623 in
2025**. Conversion efficiency peaked in 2023 at 4.28% and has declined for two consecutive
years to 3.24%.

> The MC is recruiting more people every year and converting a smaller share of them.
> Volume growth is masking a conversion problem, and a headline "+9% applications" reads
> as success while realizations move the other way. This is the single most important
> thing in this dataset, and it is invisible unless the funnel is tracked alongside intake.

### Seasonality

![Seasonality](outputs/figures/03_seasonality.png)

| Month | Avg applications | Seasonal index | Season |
|---|---:|---:|---|
| **Jan** | **1,024** | **0.74** | **Low** |
| **Feb** | **1,031** | **0.75** | **Low** |
| **Mar** | **1,787** | **1.29** | **High** |
| **Apr** | **1,718** | **1.25** | **High** |
| **May** | **1,706** | **1.24** | **High** |
| Jun | 1,313 | 0.95 | Medium |
| Jul | 1,205 | 0.87 | Medium |
| Aug | 1,435 | 1.04 | Medium |
| Sep | 1,344 | 0.97 | Medium |
| Oct | 1,362 | 0.99 | Medium |
| Nov | 1,485 | 1.08 | Medium |
| **Dec** | **1,154** | **0.84** | **Low** |

- **Peak season is a single tight block: March → May** (index 1.29 / 1.25 / 1.24), aligned
  to the Indian even-semester recruitment drive before summer exchange.
- **The trough is January–February** (0.74–0.75), immediately before the peak — the MC goes
  from its quietest month to its busiest in the space of four weeks.
- **1.74× trough-to-peak swing** (Mar 1.294 ÷ Jan 0.742) that flat capacity planning cannot
  absorb.
- The peak month has been **March or April in three of four years** (May in 2022, the
  COVID-recovery year), which is why the seasonal signal is learnable from four cycles.

### The exchange funnel

![Funnel](outputs/figures/04_funnel.png)

| Stage | Label | Count | From previous | From APP | Dropped |
|---|---|---:|---:|---:|---:|
| APP | Applied | 66,253 | 100.0% | 100.0% | — |
| ACH | Achieved / Matched | 10,036 | **15.15%** | 15.15% | **56,217** (largest leak) |
| ACC | Accepted | 6,617 | 65.93% | 9.99% | 3,419 |
| APD | Approved | 3,160 | 47.76% | 4.77% | 3,457 |
| RE | Realized | 2,344 | 74.18% | 3.54% | 816 |
| FI | Finished | 2,253 | 96.12% | 3.40% | 91 |
| CO | Completed | 1,381 | 61.30% | 2.08% | 872 |

**The largest leak is APP → ACH: 56,217 of 66,253 applications (84.9%) never reach
matching.** Only about **1 application in 7** is ever matched to an opportunity.

The second leak is **ACC → APD at 47.8%** — over half of everyone who is *accepted* is
never *approved*. These two transitions together account for 59,674 of the 64,872 total
drop-outs.

**End-to-end efficiency is 3.54% APP→RE** — roughly **28 applications per realization**,
which is the planning ratio for recruitment targets.

### Entity (Local Committee) performance

![Entities](outputs/figures/05_entity_performance.png)

| # | Local Committee | Applications | Realizations | MC share | APP→RE | Growth 22→25 |
|---|---|---:|---:|---:|---:|---:|
| 1 | LC A | 6,939 | 120 | 10.5% | 1.73% | +69.8% |
| 2 | LC B | 6,603 | **332** | 10.0% | 5.03% | −17.4% |
| 3 | LC C | 6,274 | 137 | 9.5% | 2.18% | −15.7% |
| 4 | LC D | 4,641 | 128 | 7.0% | 2.76% | **+237.1%** |
| 5 | LC E | 4,316 | 68 | 6.5% | 1.58% | −25.1% |
| 6 | LC F | 4,046 | 79 | 6.1% | 1.95% | −2.5% |
| 7 | LC G | 3,808 | 111 | 5.8% | 2.91% | +54.1% |
| 8 | LC H | 3,772 | **304** | 5.7% | **8.06%** | +86.1% |
| 9 | LC I | 3,769 | 166 | 5.7% | 4.40% | **+471.7%** |
| 10 | LC J | 2,737 | 138 | 4.1% | 5.04% | +21.5% |

<sub>Local Committees are anonymised in this public README. The identified breakdown is in
`outputs/reports/entity_performance.csv` and on the dashboard.</sub>

- **The top 5 LCs generate 43.4% of all applications; the top 10 generate 70.8%** across
  33 active entities. National performance inherits the risk of a handful of LCs.
- **Conversion is not uniform — it varies 5× across comparable LCs.** LC H converts
  **8.06%** APP→RE; LC A, on *more* applications, converts **1.73%**. LC E sits at 1.58%.
- **Rank by applications and rank by realizations are different orderings.** LC A is #1 on
  intake but **#8 on realizations**; LC H is #8 on intake but **#2 on realizations** —
  from 46% fewer applications than LC A it produces **2.5× more realizations**.

> Because the spread is this wide, the highest-leverage action is **transferring practice,
> not reforming process**: find what the top-converting LCs do at matching and propagate it
> to the high-volume, low-conversion ones.

### Programme performance

![Products](outputs/figures/06_product_mix.png)

| Programme | Applications | Realizations | Share of APP | Share of RE | APP→RE | Peak | Growth |
|---|---:|---:|---:|---:|---:|---|---:|
| **oGTa** | 21,502 | 109 | **32.5%** | **4.7%** | **0.51%** | Aug | +2.8% |
| iGTa | 14,794 | 372 | 22.3% | 15.9% | 2.51% | Apr | **+279.4%** |
| **iGV** | 10,313 | **1,011** | 15.6% | **43.1%** | **9.80%** | Mar | −21.8% |
| **oGV** | 8,792 | **690** | 13.3% | **29.4%** | 7.85% | Mar | +17.1% |
| iGTe | 7,671 | 156 | 11.6% | 6.7% | 2.03% | Apr | +42.8% |
| oGTe | 3,181 | 6 | 4.8% | 0.3% | 0.19% | May | +153.1% |

**The single largest misallocation in the dataset:**

- **oGTa absorbs 32.5% of all applications and produces 4.7% of all realizations** — a
  0.51% conversion rate, meaning **197 applications per realization**. It is the largest
  programme by intake and second-smallest by output.
- **Global Volunteer is the engine.** iGV + oGV together are **28.8% of applications but
  72.6% of realizations**, converting at 9.80% and 7.85% — roughly **19× oGTa's rate**.
- **oGTe is effectively non-functional**: 3,181 applications produced **6** realizations
  across four years.
- Growth is pointed the wrong way: iGTa grew **+279%** and oGTe **+153%**, while **iGV, the
  single most productive programme, shrank 21.8%**.

> Reallocating even a fraction of oGTa's 21,502 applications toward GV at its observed
> conversion rate would add more realizations than any plausible efficiency gain elsewhere.

---

## The 2026 prediction CSV

**Deliverable:** [`outputs/predictions_2026.csv`](outputs/predictions_2026.csv)

### The three required columns

| Month | Predicted Applications | Activity Level |
|---|---:|---|
| Jan 2026 | 1,403 | **Low** |
| Feb 2026 | 1,410 | **Low** |
| **Mar 2026** | **2,166** | **High** |
| **Apr 2026** | **2,097** | **High** |
| **May 2026** | **2,085** | **High** |
| Jun 2026 | 1,692 | Medium |
| Jul 2026 | 1,584 | Medium |
| Aug 2026 | 1,814 | Medium |
| Sep 2026 | 1,723 | Medium |
| Oct 2026 | 1,741 | Medium |
| Nov 2026 | 1,864 | Medium |
| Dec 2026 | 1,533 | Medium |
| **Total** | **21,112** | |

### Full CSV, with all supporting columns

| Month | Predicted | Level | 95% interval | Rank | % of year | vs history | within 2026 |
|---|---:|---|---|---:|---:|---|---|
| Jan 2026 | 1,403 | **Low** | 1,024 – 1,873 | 12 | 6.65% | Medium | Low |
| Feb 2026 | 1,410 | **Low** | 875 – 2,076 | 11 | 6.68% | Medium | Low |
| Mar 2026 | 2,166 | **High** | 1,511 – 2,981 | 1 | 10.26% | High | High |
| Apr 2026 | 2,097 | **High** | 1,340 – 3,038 | 2 | 9.93% | High | High |
| May 2026 | 2,085 | **High** | 1,239 – 3,138 | 3 | 9.88% | High | High |
| Jun 2026 | 1,692 | **Medium** | 766 – 2,845 | 8 | 8.01% | High | Medium |
| Jul 2026 | 1,584 | **Medium** | 584 – 2,830 | 9 | 7.50% | High | Medium |
| Aug 2026 | 1,814 | **Medium** | 744 – 3,145 | 5 | 8.59% | High | Medium |
| Sep 2026 | 1,723 | **Medium** | 588 – 3,135 | 7 | 8.16% | High | Medium |
| Oct 2026 | 1,741 | **Medium** | 545 – 3,230 | 6 | 8.25% | High | Medium |
| Nov 2026 | 1,864 | **Medium** | 610 – 3,425 | 4 | 8.83% | High | Medium |
| Dec 2026 | 1,533 | **Medium** | 223 – 3,164 | 10 | 7.26% | Medium | Low |

<details>
<summary><b>Raw CSV contents</b> (click to expand)</summary>

```csv
Month,Predicted Applications,Activity Level,date,month_number,lower_95,upper_95,activity_level_vs_history,activity_level_within_2026,model,rank_in_year,share_of_year_pct
Jan 2026,1403,Low,2026-01-01,1,1024,1873,Medium,Low,holt_winters,12,6.65
Feb 2026,1410,Low,2026-02-01,2,875,2076,Medium,Low,holt_winters,11,6.68
Mar 2026,2166,High,2026-03-01,3,1511,2981,High,High,holt_winters,1,10.26
Apr 2026,2097,High,2026-04-01,4,1340,3038,High,High,holt_winters,2,9.93
May 2026,2085,High,2026-05-01,5,1239,3138,High,High,holt_winters,3,9.88
Jun 2026,1692,Medium,2026-06-01,6,766,2845,High,Medium,holt_winters,8,8.01
Jul 2026,1584,Medium,2026-07-01,7,584,2830,High,Medium,holt_winters,9,7.5
Aug 2026,1814,Medium,2026-08-01,8,744,3145,High,Medium,holt_winters,5,8.59
Sep 2026,1723,Medium,2026-09-01,9,588,3135,High,Medium,holt_winters,7,8.16
Oct 2026,1741,Medium,2026-10-01,10,545,3230,High,Medium,holt_winters,6,8.25
Nov 2026,1864,Medium,2026-11-01,11,610,3425,High,Medium,holt_winters,4,8.83
Dec 2026,1533,Medium,2026-12-01,12,223,3164,Medium,Low,holt_winters,10,7.26
```

</details>

### Column reference

| Column | Description |
|---|---|
| `Month` | Forecast month, `MMM YYYY` |
| `Predicted Applications` | Point forecast, rounded to a whole application |
| `Activity Level` | **Low / Medium / High** — trend-adjusted (the default method) |
| `date` | ISO month-start, for joins and plotting |
| `month_number` | 1–12 |
| `lower_95`, `upper_95` | 95% prediction interval from walk-forward residuals |
| `activity_level_vs_history` | Literal historical-quartile classification |
| `activity_level_within_2026` | Quartiles of the 12 forecast values |
| `model` | Model that produced the forecast |
| `rank_in_year` | 1 = busiest month |
| `share_of_year_pct` | Month's share of forecast annual volume |

### Visualisation of the same CSV

![Activity levels](outputs/figures/08_activity_levels.png)

### Key predictions

> **Three high-activity months: March, April and May 2026**, carrying **30.1% of forecast
> annual volume** between them.

- **March 2026 is the peak** at 2,166 applications — 10.3% of the year, consistent with
  March/April leading in three of the last four years.
- **Total 2026 forecast: 21,112 applications, +9.8% on 2025** — in line with the steady
  9–14% band the MC has held since 2022.
- **January and February are the quiet months** — the natural window for member training,
  partner development and process work, immediately before the year's busiest quarter.
- **Intervals widen materially by December** (223–3,164). That width is **honest, not a
  defect**: a 12-step recursive forecast from 48 observations of a genuinely noisy real
  series carries that much uncertainty. Plan against the band, not the central line.

### Recommended actions

1. **Stage capacity in February, before the March–May block.** Matching and reviewer
   bandwidth must be in place *by* February — the MC goes from its quietest month
   (index 0.74) to its busiest (1.29) in four weeks.
2. **Attack the APP → ACH leak first.** It loses **84.9%** of all candidates — 56,217 of
   66,253. Nothing else in the funnel is close; a few points recovered here is worth more
   than a uniform improvement everywhere else.
3. **Fix the ACC → APD stage second.** 52% of *accepted* candidates are never approved,
   which is an internal process loss, not a market one.
4. **Rebalance away from oGTa.** It takes 32.5% of applications and returns 4.7% of
   realizations (0.51%, ≈197 applications per realization). GV converts at 8–10%.
5. **Reverse the iGV decline.** The single most productive programme (43% of all
   realizations) shrank 21.8% over the period while the worst converters grew.
6. **Propagate what the best-converting LC does.** Conversion ranges 1.6%–8.1% across
   comparable LCs; the best produces 2.5× the top-volume LC's realizations from 46% fewer
   applications. This is a transferable-practice problem, not a systemic one.
7. **Track realizations, not applications, as the headline metric.** Applications grew
   38.6% since 2022 while realizations fell from their 2024 peak — the current headline
   hides the trend that matters.

---

## Dashboard

```bash
streamlit run app.py
```

Four pages. All charts are interactive (hover, zoom, legend filtering) and every chart is
paired with the table behind it.

### Page 1 — Overview
Total historical applications, average monthly activity, YoY growth, predicted peak month,
trend and seasonality side by side.

![Overview](docs/screenshots/01_overview.jpg)

### Page 2 — Historical Analysis
Four tabs: **Trends** · **Seasonality** · **Exchange funnel** · **Entities & products**.

![Historical](docs/screenshots/02_historical_trends.jpg)
![Funnel](docs/screenshots/03_exchange_funnel.jpg)

### Page 3 — 2026 Prediction
Forecast with uncertainty band, high-activity callouts, the prediction table in either
deliverable format or full detail, CSV download, and model diagnostics (comparison,
walk-forward fit, feature importance).

![Forecast](docs/screenshots/04_forecast_2026.jpg)
![Activity](docs/screenshots/05_activity_levels.jpg)

### Page 4 — Insights
Findings generated directly from the data — **every number computed, none hand-written**:

```
[Forecast]    Mar 2026 is predicted to have the highest activity, at 2,166 applications.
[Forecast]    3 high-activity month(s) predicted: Mar 2026, Apr 2026 and May 2026.
[Forecast]    Total 2026 applications are forecast at 21,112, 9.8% above 2025.
[Forecast]    Lowest predicted activity falls in Jan 2026, Feb 2026.
[Growth]      Application volume increased by 9.0% in 2025 compared with 2024.
[Growth]      Exchange activity grew at a compound annual rate of 11.5% between 2022 and 2025.
[Seasonality] Peak season runs from Mar to May.
[Seasonality] Jan is the quietest month, running at 0.74x the annual average.
[Funnel]      The biggest drop-off is APP to ACH, losing 85% of candidates.
[Funnel]      3.5% of applications convert all the way to a realization.
[Entities]    The top 5 Local Committees generate 43% of all applications.
[Entities]    <LC name> is the fastest-growing large LC at +471.7% (2022->2025).
[Products]    oGTa is the largest programme at 32% of all applications.
[Model]       The selected model (holt_winters) achieves 87.1% accuracy (MAPE 12.9%).
```

![Insights](docs/screenshots/06_insights.jpg)

### Visualization design notes

Charts follow a deliberate discipline rather than library defaults:

- **Colour by job.** Categorical hues in a *fixed* order for identity (programmes,
  entities); a single-hue **ordinal ramp** for anything ordered (funnel stages,
  Low/Medium/High). Rank never picks a colour, so filtering never repaints a series.
- **One axis, always.** No dual-axis chart anywhere — different scales get separate charts.
- **Identity never rests on colour alone.** Legends plus direct labels, and every chart is
  paired with its underlying table.
- **Both themes are selected, not flipped.** Dark-mode steps are chosen for the dark
  surface. Per-bar text ink is computed from fill luminance so labels stay readable at
  both ends of a ramp.

---

## How to run this project

**No API key, no database, no account needed.** The repository ships with the processed
dataset, the trained model and all results already committed, so you can run the
dashboard immediately — or rebuild everything from scratch in about 30 seconds.

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10 or newer** | Check with `python3 --version` |
| **pip** | Ships with Python |
| ~250 MB disk | For the virtual environment and dependencies |
| Internet | Only to install packages — the demo itself runs fully offline |

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Jayed-Alam-Mansur/Exchange-Activity-Prediction-Platform.git
cd Exchange-Activity-Prediction-Platform
```

### Step 2 — Create a virtual environment

Keeps these dependencies isolated from your system Python.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now be prefixed with `(.venv)`.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Takes 1–2 minutes. Installs pandas, scikit-learn, XGBoost, statsmodels, Plotly,
Streamlit, matplotlib and pytest.

### Step 4 — Launch the dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser — Streamlit usually opens it for you.

> **That is the whole setup.** The dashboard reads the committed artefacts, so it works
> straight after install without running the pipeline first.

Press **`Ctrl + C`** in the terminal to stop it.

---

### Optional — rebuild every result from scratch

To verify the numbers rather than trust the committed ones:

```bash
python run_pipeline.py --step all
```

Takes ~30 seconds and rewrites the dataset, all report tables,
`outputs/predictions_2026.csv`, the trained model and all 10 figures. Expected output:

```text
STEP 1/5  DATA COLLECTION
STEP 2/5  PARSING, CLEANING AND VALIDATION
STEP 3/5  EXPLORATORY DATA ANALYSIS
STEP 4/5  FEATURE ENGINEERING
STEP 5/5  MODEL TRAINING, SELECTION AND FORECASTING
...
Selected 'holt_winters' (MAE=198.02, 1 model(s) within 5% tolerance)
2026 forecast: total=21112, peak=Mar 2026 (2166)
PIPELINE COMPLETE
```

Re-collection hits the live API, so totals will differ if AIESEC back-dates records; parsing
and modelling from the saved payloads are deterministic and reproduce **exactly** the numbers published
in this README. To wipe every generated artefact first and prove nothing is stale:

```bash
make reset && make all
```

### Optional — run the tests

```bash
pytest -q
```

Expected: `49 passed`.

### Optional — using `make`

```bash
make setup       # install dependencies + create .env from the template
make all         # run the full pipeline
make test        # run the 49 tests
make dashboard   # launch Streamlit
make reset       # delete every generated artefact
make help        # list all targets
```

---

### What to look at first

If you are reviewing this project, the fastest path through it:

| Order | Where | What you will see |
|---|---|---|
| 1 | Dashboard → **2026 Prediction** | The core deliverable: forecast, high-activity months, prediction table, model diagnostics |
| 2 | Dashboard → **Insights** | Auto-generated findings — every number computed from the data |
| 3 | [`outputs/predictions_2026.csv`](outputs/predictions_2026.csv) | The deliverable file itself |
| 4 | [`outputs/reports/model_evaluation.csv`](outputs/reports/model_evaluation.csv) | All 12 models with metrics and the selection scoring |
| 5 | [`notebooks/04_model_training.ipynb`](notebooks/04_model_training.ipynb) | The full modelling narrative, already executed with outputs |
| 6 | [`src/models/forecasting.py`](src/models/forecasting.py) | The model suite, backtest harness and selection logic |

---

### Troubleshooting

<details>
<summary><b>The browser shows "streamlit run yourscript.py" or a blank page</b></summary>

No app is being served at that address. Either the server is not running, or you are on
the wrong port. Check the terminal where you ran `streamlit run app.py` — it prints the
real URL. If the default port was busy, Streamlit picks a different one.

</details>

<details>
<summary><b>Error: File does not exist: app.py</b></summary>

You are in the wrong directory. `streamlit run app.py` must be run from the project root:

```bash
cd /path/to/Exchange-Activity-Prediction-Platform
ls app.py            # should print: app.py
streamlit run app.py
```

</details>

<details>
<summary><b>Port 8501 is already in use</b></summary>

Something else — often another Streamlit app — holds the port. Use a different one:

```bash
streamlit run app.py --server.port 8600
```

Or stop whatever is holding it:

```bash
lsof -ti:8501 | xargs kill      # macOS / Linux
```

</details>

<details>
<summary><b>The dashboard says "Pipeline artefacts not found"</b></summary>

The generated files are missing — most likely `make reset` was run. Rebuild them:

```bash
python run_pipeline.py --step all
```

</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'src'</b></summary>

Commands must be run from the project root, not from inside `src/` or `notebooks/`. The
notebooks handle this themselves by inserting the project root into `sys.path`.

</details>

<details>
<summary><b>XGBoost fails to install on macOS</b></summary>

XGBoost needs OpenMP:

```bash
brew install libomp
```

The pipeline also degrades gracefully — if XGBoost is unavailable it skips that one
candidate with a warning and the other 11 models still run.

</details>

<details>
<summary><b>Prophet is missing</b></summary>

Expected. Prophet is intentionally left out of `requirements.txt` because it pulls a heavy
cmdstan toolchain that often fails to build. The model suite detects its absence and skips
it. To include it: `pip install prophet`, then rerun the pipeline.

</details>

---

### Running individual stages

Each stage reads the previous stage's artefacts from disk:

```bash
python run_pipeline.py --step collect    # API → data/raw/api_responses.json
python run_pipeline.py --step process    # → data/processed/exchange_data.csv
python run_pipeline.py --step analyze    # → outputs/reports/*.csv
python run_pipeline.py --step features   # → data/processed/features.csv
python run_pipeline.py --step train      # → models/ + outputs/predictions_2026.csv
python run_pipeline.py --step figures    # → outputs/figures/*.png
```

**Flags:** `--use-reference-data` · `--start-date` · `--end-date` · `--config <path>` ·
`--log-level DEBUG` · `--no-figures`

### Notebooks

Four executed notebooks with outputs saved, walking through the analysis:

| Notebook | Contents |
|---|---|
| [`01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) | API contract, collection windows, raw payload inspection, parsing, validation |
| [`02_EDA.ipynb`](notebooks/02_EDA.ipynb) | Trends, YoY growth, seasonality, peak months, funnel, entity and product analysis |
| [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) | All three feature families, the Fourier basis, and the **no-leakage proof** |
| [`04_model_training.ipynb`](notebooks/04_model_training.ipynb) | 12-model backtest, selection, diagnostics, 2026 forecast, activity-level comparison |

---

## Project structure

```
aiesec-exchange-prediction/
│
├── README.md                       ← you are here
├── requirements.txt
├── Makefile
├── LICENSE
├── .env.example                    ← secret template (.env is git-ignored)
│
├── app.py                          ← Streamlit dashboard (4 pages)
├── run_pipeline.py                 ← CLI orchestrator
│
├── config/
│   └── config.yaml                 ← every path, threshold, hyper-parameter
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── config.py                   ← settings · secrets · logging
│   ├── insights.py                 ← automated narrative generation
│   │
│   ├── api/
│   │   ├── aiesec_api.py           ← Analytics API client
│   │   └── reference_data.py       ← offline reference dataset generator
│   │
│   ├── preprocessing/
│   │   ├── cleaning.py             ← parse · validate · tidy panel
│   │   ├── analysis.py             ← trends · funnel · entity · product
│   │   └── features.py             ← feature engineering
│   │
│   ├── models/
│   │   ├── forecasting.py          ← 12 models · backtest · selection
│   │   └── train.py                ← training · persistence · prediction
│   │
│   └── visualization/
│       ├── charts.py               ← interactive Plotly
│       └── figures.py              ← static matplotlib PNG
│
├── data/
│   ├── raw/api_responses.json      ← raw payloads (git-ignored: 8 MB, regenerable)
│   └── processed/
│       ├── exchange_data.csv       ← the tidy panel — 4,690 rows
│       ├── features.csv            ← feature matrix — 36 × 40
│       └── monthly_applications.csv
│
├── models/
│   └── trained_model.pkl           ← selected model + full provenance
│
├── outputs/
│   ├── predictions_2026.csv        <-- THE DELIVERABLE
│   ├── figures/                    ← 10 publication-quality PNGs
│   └── reports/                    ← 13 analysis + evaluation CSVs
│
├── docs/
│   ├── API_INTEGRATION.md          ← go-live checklist for real credentials
│   └── screenshots/
│
└── tests/
    └── test_pipeline.py            ← 49 tests
```

### Generated artefacts reference

| File | Contents |
|---|---|
| `outputs/predictions_2026.csv` | **The deliverable** — 12 months × 12 columns |
| `outputs/reports/model_evaluation.csv` | All 12 models with metrics and selection scoring |
| `outputs/reports/backtest_predictions.csv` | Every walk-forward prediction, all models |
| `outputs/reports/funnel_overall.csv` | Stage counts, conversion, drop-off |
| `outputs/reports/funnel_by_product.csv` · `_by_year.csv` | Funnel cut by dimension |
| `outputs/reports/entity_performance.csv` | LC ranking with share, conversion, growth, consistency |
| `outputs/reports/product_performance.csv` | Programme ranking |
| `outputs/reports/seasonality_profile.csv` | Monthly seasonal index |
| `outputs/reports/monthly_trend.csv` · `monthly_contribution.csv` · `yearly_summary.csv` · `peak_months.csv` | Trend analyses |
| `outputs/reports/feature_importance.csv` | Feature influence for the selected model |
| `outputs/reports/insights.csv` | Auto-generated findings |
| `outputs/reports/prediction_summary.csv` | Headline forecast figures |

---

## Engineering practices

| Practice | How it is applied |
|---|---|
| **Modular code** | Six focused modules; each phase reads the previous phase's artefacts from disk and can run standalone |
| **Type hints** | Throughout, including dataclasses for `Settings`, `ProjectPaths`, `ValidationReport`, `TrainingArtifacts`, `Insight` |
| **Docstrings** | Google-style on every public function — args, returns, raises |
| **Error handling** | Custom exception hierarchy (`AiesecAPIError` → `AuthenticationError`, `RateLimitError`); failures raise messages naming the exact command to fix them |
| **Logging** | Configured centrally; level from config with `LOG_LEVEL` env override; third-party noise suppressed |
| **Configuration** | `config/config.yaml` holds every path, threshold and hyper-parameter |
| **No hardcoded paths** | All paths resolve through `Settings.paths`; nothing is hardcoded in source |
| **No hardcoded secrets** | Tokens come only from the environment. `redact_token()` scrubs them from every logged URL and every exception message — asserted by tests |
| **Reproducibility** | Every raw API payload persisted to `data/raw/api_responses.json`, seeded RNG, pinned dependency ranges, `make reset && make all` rebuilds every artefact from scratch |
| **Graceful degradation** | Missing Prophet, missing XGBoost, missing credentials, missing artefacts — each is handled with a warning and a working fallback, never a crash |

### Security notes

- `.env` is git-ignored; only `.env.example` is committed.
- Tokens are never written to `config/config.yaml`, never persisted to `data/raw/`, and
  never logged.
- `401` / `402` / `403` abort immediately rather than retrying a doomed request.
- Raw API payloads are git-ignored — on a live run they contain entity-level operational
  data that should stay local.

---

## Testing

```bash
pytest -q
# 49 passed
```

Coverage targets what breaks **silently**:

| Area | What is tested |
|---|---|
| **API helpers** | Month windowing (inclusive bounds, partial ranges, reversed input rejection); token redaction |
| **Parsing** | Nested extraction · **reversed nesting order** · direction aliases · unknown statuses · empty payloads |
| **Validation** | That funnel violations and month gaps are *genuinely caught*, not just checked for |
| **No leakage** | Mutating the target month, or all future months, must not change the feature row |
| **Metrics** | Hand-computed MAE/RMSE/MAPE/bias; MAPE with zero actuals; shape-mismatch rejection |
| **Forecasters** | All 12 honour the interface and return finite, non-negative, correctly-sized output; predict-before-fit raises |
| **Backtest** | Every prediction date is strictly after its origin (out-of-sample proof) |
| **Selection** | The tolerance-band tie-break **and** that a 4× MAE gap is *not* overridden by explainability |
| **Activity levels** | All three methods, including the documented `historical_quartiles` failure mode |
| **Analysis** | Funnel internal consistency; seasonal index averages to 1.0; contributions sum to 100% per year |
| **Charts** | All 9 figures serialise in **both** light and dark themes; ink contrast flips with fill lightness |

**No test touches the network.**

### Bugs found by running the system, not just testing it

Verification was not ceremonial. Building and driving the real dashboard surfaced five
defects that unit tests alone would not have caught:

1. `st.dataframe(height=None)` raises in Streamlit 1.60 — must omit the kwarg entirely.
2. Theme detection needs `st.context.theme.type`; `st.get_option("theme.base")` reports
   light when dark comes from the OS preference.
3. Plotly 6 throws on `add_vline` with an annotation on a datetime axis — only
   milliseconds-since-epoch works.
4. White funnel labels were unreadable on light ordinal-ramp steps — ink is now computed
   from fill luminance.
5. The funnel leak callout named the downstream stage rather than the transition.

And one genuine design flaw in my own selection logic — min-max normalising MAE inside the
tolerance band — which a test caught and now pins.

---

## Limitations

Stated plainly, because a forecast without its caveats is a liability:

1. **48 observations is a short series** — exactly 4 seasonal cycles, the practical minimum
   for estimating a 12-month seasonal pattern. Intervals are wide by December 2026
   (223–3,164), and that width is honest rather than a defect.
2. **The window opens in the COVID recovery.** January 2022 sits at the bottom of a
   disrupted period, so the 11.5% CAGR measured 2022→2025 partly reflects rebound rather
   than underlying growth, and the 2026 forecast inherits that.
3. **Recursive forecasting compounds error.** Months 7–12 rest partly on predicted lags.
4. **Operational features are frozen** at their trailing 12-month mean for future months —
   they cannot be observed ahead. Deliberately conservative.
5. **No exogenous regressors.** Visa policy, partner supply, MC strategy and campaign
   calendars all move exchange activity, and none are modelled.
6. **Status counts are period-scoped, not cohort-scoped.** The funnel conversion rates are
   ratios of stage totals over the same window, not a tracked cohort. Because realizations
   lag applications by months, the headline 3.54% APP→RE rate mixes cohorts and is a
   steady-state approximation, not a true per-cohort conversion.
7. **Prophet was not exercised** — not installed in this environment, so the suite skipped
   it with a warning. The integration exists and activates on `pip install prophet`.
8. **MC-level forecast only.** Per-LC and per-programme forecasts are a natural next step
   but are not in this version.

---

## Future improvements

### Near term
- Verify `office_id` and the programme mapping; run a real backfill.
- Automated weekly collection (cron / GitHub Action) with **drift monitoring** that alerts
  when live error exceeds the backtest MAE.
- **Per-LC and per-programme forecasts**, with hierarchical reconciliation so LC forecasts
  sum to the MC forecast.
- Prophet added to CI so the candidate is genuinely evaluated.

### Medium term
- **Multi-MC comparison** — benchmark India against peer MCs on the same pipeline.
- **Real-time prediction API** — serve the model behind FastAPI for EXPA integration.
- **Exogenous regressors** — academic calendars, campaign dates, partner supply.
- **Quantile regression** for directly-estimated intervals rather than residual scaling.

### Additional AI/ML directions

| Idea | Why it matters |
|---|---|
| **Automated anomaly detection** | Flag months deviating from the seasonal profile in near-real-time, so an LC collapse is caught in weeks rather than quarters |
| **AI-generated operational recommendations** | Move from *"December will be busy"* to *"add 12 reviewers to Delhi IIT by November 15"* |
| **RAG assistant over analytics** | Natural-language querying of exchange data plus AIESEC's own process documentation |
| **Exchange success prediction** | Per-application probability of reaching realization — turns the funnel from descriptive to prescriptive |
| **Resource allocation optimisation** | Given a forecast and a member budget, solve for the staffing schedule that maximises realizations |
| **Drop-off / churn modelling** | Predict which accepted candidates will not realize, early enough to intervene |

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>
    Built as an independent technical demonstration for the AIESEC GID AI/ML Engineer
    application.<br>Not an official AIESEC product and not endorsed by AIESEC International.
  </sub>
</p>
