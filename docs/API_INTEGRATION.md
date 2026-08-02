# AIESEC Analytics API — integration notes

Everything needed to switch this project from the offline reference dataset to live MC
India data, plus a record of what was verified against the live endpoint and what still
needs confirming against the GID documentation.

---

## 1. What was verified

Probed directly against the live endpoint on 2026-08-02:

```bash
# No token
$ curl -s -w "HTTP %{http_code}\n" \
    "https://analytics.api.aiesec.org/v2/applications/analyze.json?start_date=2025-01-01&end_date=2025-01-31"
HTTP 402
{"error":"Unauthorized"}

# Invalid token
$ curl -s -w "HTTP %{http_code}\n" \
    "https://analytics.api.aiesec.org/v2/applications/analyze.json?access_token=invalid&start_date=2025-01-01&end_date=2025-01-31"
HTTP 401
{"status":{"code": 401, "sub_code": "invalid_token", "message": "Invalid, missing or expired token"}}
```

Confirmed facts:

| Fact | Evidence |
|---|---|
| The endpoint is live and reachable over HTTPS | both probes returned structured JSON |
| Authentication is via the `access_token` **query parameter** | `sub_code: invalid_token` when supplied, `Unauthorized` when absent |
| A **missing** token returns `402`, not `401` | first probe |
| An **invalid/expired** token returns `401` with `sub_code: invalid_token` | second probe |

`402` for a missing token is unusual enough to be worth calling out — a client that only
treats `401`/`403` as auth failures will misclassify it as a retryable error and burn its
entire retry budget on a request that can never succeed.
`src/api/aiesec_api.py` treats `401`, `402` and `403` as fatal `AuthenticationError`.

---

## 2. What still needs confirming

These are **configuration values**, not assumptions compiled into code. Set them once and
no source file changes.

| # | Unknown | Config key | Env override | Current default |
|---|---|---|---|---|
| 1 | MC India EXPA office id | `api.office_id` | `AIESEC_OFFICE_ID` | `null` (required) |
| 2 | Programme id → product | `products.mapping` | — | `1:GV, 2:GTa, 5:GTe, 7:GE` |
| 3 | Filter namespace | `api.filter_namespace` | — | `performance` |
| 4 | Pagination contract | `api.per_page` | — | `100`, probed defensively |
| 5 | Rate limits | `api.rate_limit_sleep_seconds` | — | `0.35s` between calls |

Both `api.office_id_verified` and `products.verified` are `false` until checked.

### 2.1 Office id

The MC office id scopes the pull. Set `performance[include_child_offices]=true` (the
default) to include the LCs beneath it, which is what makes entity-level analysis
possible. Without a confirmed id, `Settings.require_office_id()` raises a message naming
the exact fix rather than silently pulling the wrong entity — a wrong-but-valid id would
produce a plausible dataset for the wrong office, which is far worse than an error.

### 2.2 Programme mapping

The defaults are the conventional public mapping. If they are wrong, products get
mislabelled but volumes stay correct — a recoverable error, and one the entity/product
report would make visible immediately (e.g. a "GTe" line carrying half of all volume).

### 2.3 Filter namespace

The Analytics API groups filters under a family-specific namespace — `recruitment[...]`
for recruitment analytics, `advancement[...]` for advancement, `position[...]` for
positions. `performance` is the assumption for application analytics. If wrong, the API
will most likely ignore the filter and return unscoped data — **so validate the first
response before trusting a full backfill** (see §5).

---

## 3. Request contract as implemented

```
GET https://analytics.api.aiesec.org/v2/applications/analyze.json
  ?access_token=<token>                          # secret, from env only
  &start_date=YYYY-MM-DD                         # inclusive
  &end_date=YYYY-MM-DD                           # inclusive
  &performance[office_id]=<id>
  &performance[include_child_offices]=true
  &performance[programmes][]=<programme_id>      # omitted when unset
  &page=<n>&per_page=100
```

One request per `(month, programme)`. 48 months × 3 active products = 144 request groups
for the full 2022–2025 backfill.

**Why monthly windows** — payloads stay small enough to read by hand; a failure loses one
month rather than four years; and the window matches the forecasting grain, so no
re-bucketing is needed downstream.

---

## 4. Response contract

Elasticsearch-style aggregations. The shape the reference generator emits, and the shape
the parser is written against:

```json
{
  "analytics": {
    "offices": {
      "buckets": [
        {
          "key": 90001,
          "key_as_string": "AIESEC in Mumbai",
          "doc_count": 143,
          "directions": {
            "buckets": [
              {
                "key": "outgoing",
                "doc_count": 106,
                "statuses": {
                  "buckets": [
                    {"key": "applied",   "doc_count": 106},
                    {"key": "achieved",  "doc_count": 44},
                    {"key": "accepted",  "doc_count": 29},
                    {"key": "approved",  "doc_count": 24},
                    {"key": "realized",  "doc_count": 18},
                    {"key": "finished",  "doc_count": 17},
                    {"key": "completed", "doc_count": 15}
                  ]
                }
              }
            ]
          }
        }
      ]
    }
  },
  "meta": {"page": 1, "total_pages": 1}
}
```

### Why the parser does not depend on this shape

`parse_payload` performs a **tolerant recursive walk** instead of indexing a fixed path.
It descends any `{"buckets": [...]}` node and classifies each bucket by its key:

| Bucket key matches | Interpreted as | Recognised aggregation names |
|---|---|---|
| a key in `funnel.api_status_map` | funnel stage measurement | `statuses`, `status`, `stages`, `stage` |
| — | entity (LC) | `offices`, `office`, `entities`, `committees`, `lc` |
| — | direction | `directions`, `direction`, `types`, `flow` |
| — | product | `programmes`, `products`, `programs` |

Consequences:

- Nesting **order** can change — office-above-direction and direction-above-office parse
  to the identical cell (pinned by `test_parser_is_tolerant_of_reversed_nesting`).
- Extra grouping levels are descended, not fatal.
- Unknown status keys are ignored rather than crashing.
- Direction spellings normalise (`og`/`out`/`sending` → `outgoing`).

The only breaking change would be a new **status vocabulary**, and that vocabulary is
config: `funnel.api_status_map` in `config/config.yaml`.

---

## 5. Go-live checklist

1. **Credentials** — put a fresh token and the office id in `.env`:
   ```
   AIESEC_ACCESS_TOKEN=...
   AIESEC_OFFICE_ID=...
   ```

2. **Cheap connectivity check** before any long pull:
   ```python
   from src.api.aiesec_api import AiesecAnalyticsClient
   from src.config import get_settings
   AiesecAnalyticsClient(get_settings()).check_connection()
   ```

3. **Inspect one real response** and confirm the aggregation shape and status vocabulary:
   ```python
   import json
   from src.api.aiesec_api import AiesecAnalyticsClient
   from src.config import get_settings
   pages = AiesecAnalyticsClient(get_settings()).analyze_applications("2025-01-01", "2025-01-31")
   print(json.dumps(pages[0], indent=2)[:3000])
   ```
   Check: are the status keys the ones in `funnel.api_status_map`? Does an office
   breakdown appear? Is the volume plausible for one month of one MC?

4. **Confirm the filter actually applied.** Compare a request with and without
   `performance[office_id]`. If the totals are identical, the namespace is wrong and you
   are pulling global data — fix `api.filter_namespace` before backfilling.

5. **Single-month trial run:**
   ```bash
   python run_pipeline.py --step collect --start-date 2025-01-01 --end-date 2025-01-31
   python run_pipeline.py --step process
   ```
   Validation must pass, and `data/raw/api_responses.json` should show
   `"is_reference_data": false`.

6. **Full backfill:**
   ```bash
   python run_pipeline.py --step all
   ```

7. **Flip the verification flags** in `config/config.yaml` once confirmed:
   ```yaml
   api:
     office_id_verified: true
   products:
     verified: true
   ```

---

## 6. Reliability behaviour

| Concern | Handling |
|---|---|
| Missing/expired/unauthorised token (`401`/`402`/`403`) | Fatal `AuthenticationError`, no retry — the run aborts rather than hammering a doomed request |
| Rate limit (`429`) | Honours `Retry-After`; otherwise exponential backoff with jitter, up to `api.max_retries` |
| Server errors (`5xx`) | Same retry policy |
| Network errors | Retried with backoff |
| Partial failure | Window is logged and recorded in `metadata.failed_windows`; the run continues (`--fail-fast` behaviour is available via `fail_fast=True`) |
| Request pacing | `api.rate_limit_sleep_seconds` enforces a minimum gap between calls |
| Pagination | Drains `total_pages` / `pages` / `next_page`; aggregation responses without paging correctly stop at one page |
| Token leakage | `redact_token()` scrubs the secret from every logged URL and every exception message; tests assert this |

---

## 7. Token acquisition

Tokens are issued through EXPA (`expa.aiesec.org`) and are short-lived. Two options:

- **Paste a token** into `AIESEC_ACCESS_TOKEN` — simplest, needs periodic refresh.
- **OAuth password grant** — placeholders exist in `.env.example`
  (`AIESEC_AUTH_URL`, `AIESEC_CLIENT_ID`, `AIESEC_CLIENT_SECRET`, `AIESEC_USERNAME`,
  `AIESEC_PASSWORD`). The exchange helper is **not implemented**, because the grant URL
  and payload shape were not available to verify; implementing it against a guess would
  produce code that looks working and silently isn't.

Either way the token is read from the environment only. It is never written to
`config/config.yaml`, never persisted to `data/raw/`, and never logged.
