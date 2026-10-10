# Air Raid Alert Hazard Forecast — Plan

## Context

KSE Agentic AI School, Stage 2: a Python mini-project on time series analysis of Ukrainian air raid alerts. Evaluate on the latest period with zero leakage: nothing used for training or as input may come from after the forecast time. Graded mainly on process (AI log), then on the artifact.

The June 2026 cohort solved the same task with national daily counts + ADF/STL + SARIMA/ETS/Prophet + Streamlit. This project instead answers the questions a person actually has, per oblast, at any moment:

- **Oblast is quiet:** what is the probability an alert starts within the next 1–24 h?
- **Oblast is under alert:** what is the probability it is over within the next 15 min – 6 h?

Each question is asked for two levels, matching the yellow/red alert system introduced on 6 Sep 2026: **any alert** (yellow or red) and **red alert** (missile, ballistic or massed attack).

## Key data findings that shape the design

1. **The alert definition changed on 6 Sep 2026.**
   - Before: one siren-backed alert.
   - Since: yellow (drone threat) and red. 63% of air raid alerts in the test month are yellow.
   - Share of time under alert, before vs. after, any (red-only in brackets):

     | Region | Before (official, Dec–Jul) | After (alerts.in.ua, 10 Sep–10 Oct) |
     |---|---|---|
     | Kyiv City | 5.9% | 31.1% (red 2.6%) |
     | Kyiv oblast | 17% | 67.7% (red 11.4%) |

   - Pre-September data is a different definition. It is used deliberately, never silently mixed in.
2. **Artillery is mixed into the official data.**
   - The official CSV has no type column. Its hromada-level rows are artillery-shelling hromadas (Nikopol, Marhanets, Pokrovsk…).
   - In alerts.in.ua, `artillery_shelling` appears only at hromada/city level, never at raion or oblast level.
3. **The volunteer (eTryvoga) data doesn't match the declarations:**
   - only ~50% of its starts match an official start exactly, and ~30% have no official start within 5 min;
   - it shows alerts in western oblasts that were never declared.

   Not used.
4. **The official CSV has every row twice**, a gap from 29 Aug to 1 Sep, and stops on 7 Sep, because its upstream channel went silent.
5. **No raion is dropped.** Raion coverage has no natural gap between 50% and 100%, and threshold cuts either wipe out whole oblasts or remove city raions. Near-permanent oblasts are shown as they are.

## Product

- **CLI:** `alerts forecast --at "2026-10-09T18:00"` prints, per oblast:
  - its state: quiet, or under yellow/red alert and for how long;
  - the "any" and "red" curves.
- **Static HTML site** (`alerts report` → `site/index.html`, GitHub Pages, no server), four tabs:
  1. **Forecast map** at the latest data time.
     - Toggles: Any / Red, and horizon.
     - Colour: quiet oblasts by P(start within H); oblasts under alert by P(end within H), in a second colour scale.
     - Clicking an oblast shows both curves.
  2. **Replay** over the test periods, readable without explanation:
     - **Timeline chart (main view):** oblast selector, Any/Red toggle, horizon dropdown. The predicted probability is a line; actual alerts are yellow and red bands beneath it on the same time axis. A good forecast rises before a band starts.
     - **Side-by-side maps:** one time slider. Left map = forecast made at that moment; right map = what actually happened in the next H hours.
  3. **Scorecard:** skill against baselines per horizon for evaluations E1 and E2, calibration plots, per-oblast table, the old-data ablation, feature importance.
  4. **Data:**
     - hour-of-day and day-of-week profiles, oblast co-occurrence;
     - per-raion coverage vs. the merged oblast. Finding: Dnipropetrovsk oblast's raw union is near-permanent, while Dniprovskyi raion, where Dnipro city is, is under alert only 23%;
     - regime shifts: Dec 2025 raion-level switch; the Sep 2026 yellow/red switch with the before/after table above.

## Data

| Source | What | Window (UTC) |
|---|---|---|
| Vadimkin `official_data_uk.csv` | Scraped `@air_alert_ua` declarations, old system, no type/level | 2025-12-01 → 2026-08-28 |
| alerts.in.ua `/v1/regions/{uid}/alerts/month_ago.json` | Declarations with `alert_type`, `alert_level` (red/yellow), oblast/raion/hromada/city, new system | ~2026-09-10 → last fetch |

- **alerts.in.ua is a rolling 30-day window.** Each `alerts fetch` adds to `data/alerts_in_ua.parquet` (deduplicated by `id`), never replaces it, or the early September data is lost.
  - The first fetch was made on 2026-10-10 (16,269 alerts, 26 regions); its raw JSON is kept in `data/raw/alerts_in_ua_2026-10-10/`.
  - A final refetch on 12 Oct adds two test days.
- **Token:** `ALERTS_IN_UA_TOKEN` from the environment, never committed. Rate limit 2 requests/min, so a full fetch takes ~15 min.

### Building the timelines

1. **Canonical table:** `source, oblast, raion, location_type, alert_type, level, started_at, finished_at`, all UTC. One adapter per source; official rows get `level = NULL` and `alert_type = NULL`. Duplicate rows are removed.
2. **Same rule in both systems:** only oblast- and raion-level declarations are used.
   - It removes 100% of artillery alerts in the new data and, by the same logic, the artillery hromadas in the old data. "Any alert" is then defined identically before and after September.
   - Documented cost: in the new data, city-only air raid alerts are lost: Kirovohrad 19% of its air-raid time, Kharkiv 10%, Dnipro 4%.
3. **Two event streams per oblast:**
   - **any** = union of all intervals (old alerts, or new red + yellow);
   - **red** = union of red intervals, new system only.

   Quiet gaps shorter than 2 min are bridged so overlapping raion alerts don't flicker.
4. **Units:** 24 oblasts + Kyiv City. Luhansk and Crimea are excluded: permanent alerts, not forecastable.

## Models

Discrete-time hazard (survival) models. For each future time bin k, the model predicts the probability that the event happens in bin k given it hasn't happened before. The cumulative probability is `P(event ≤ H) = 1 − Π_{k≤H}(1 − h_k)`. One model gives the whole curve, and the curve never goes down as H grows.

Four models from one parametrised pipeline:

| | Forecast times | Bins | Event | Reported horizons |
|---|---|---|---|---|
| **any-start** | hourly, oblast quiet (no alert) at t | 24 × 1 h | any alert starts | 1, 3, 6, 12, 24 h |
| **any-end** | every 15 min, oblast under any alert at t | 24 × 15 min | oblast goes quiet | 15, 30, 60, 120 min |
| **red-start** | hourly, oblast not under red at t | 24 × 1 h | red alert starts | 1, 3, 6, 12, 24 h |
| **red-end** | every 15 min, oblast under red at t | 24 × 15 min | red ends | 15, 30, 60, 120 min |

- **Training rows:** each (oblast, t) expands into one row per bin k up to and including the bin where the event happens. Label 1 in that bin, 0 before it.
- **Censoring:** if the data ends before any event, the row stops at the last fully observed bin.
- **Bin index:** `k` is a feature.

### How the old data is used

| Model | Training data |
|---|---|
| any-start, any-end | Old rows + new rows, with a `regime` feature (old/new). The old alert is treated as the closest match to "any" (one signal for any air threat). Red/yellow input features are empty for old rows; LightGBM handles missing values natively. |
| red-start, red-end | New rows only (old rows have no red label). The any-model's forecast at the same t is a feature, so red benefits from the old data indirectly. |

### LightGBM setup

- `objective=binary`, no class weights, so probabilities stay honest.
- `learning_rate=0.05`, early stopping on validation log loss.
- Small grid on validation only, about 8 configurations: `num_leaves` ∈ {15, 31, 63}, `min_data_in_leaf` ∈ {200, 1000}, `feature_fraction` 0.8.
- `oblast` as a categorical feature: one pooled model across oblasts.
- **Calibration:** check reliability on validation. If needed, apply temperature scaling: a single scalar applied to the hazard logits, fitted on validation. It keeps the curve rising with H.
- **Consistency:** the displayed P(red ≤ H) is capped at P(any ≤ H), and the report counts how often the cap is applied.
- **Frozen after validation.** No retraining before the test.

### Baselines (each must be beaten)

- **Start models:**
  - **Climatology:** per oblast, hour of week and bin k, the empirical hazard over the training window.
  - **Logistic regression** on the same rows and features.
- **End models:**
  - **Kaplan–Meier** per oblast, conditioned on elapsed duration ("alerts that have lasted 40 min so far end within 15 min X% of the time").
  - **Logistic regression.**

## Features

One function `features(events, t)` serves training, validation, test, the CLI and the UI. An alert is visible at t only if `started_at ≤ t`. Its end is visible only if `finished_at ≤ t`; otherwise it counts as still active at t.

- **Calendar:** hour of day and day of week in Kyiv time. No date or trend index.
- **Regime:** old/new.
- **Own oblast:**
  - current state duration: minutes since the last end, or minutes elapsed in the current alert;
  - current level (none/yellow/red; empty for old rows);
  - decayed start counts with half-lives 3 h, 24 h, 72 h, separately for any and red;
  - number of starts in the last 24 h;
  - share of raions active now.
- **Neighbours** (hand-written adjacency map of 25 units):
  - number under any alert and under red now;
  - minutes since the last neighbour start;
  - neighbour starts in the last 30, 60 and 120 min.
- **National:**
  - number of oblasts under any alert and under red now;
  - starts nationwide in the last 60 min;
  - hours since the last national wave (≥15 oblasts starting within 30 min);
  - state of each other oblast, so the model can learn routes (Sumy → Poltava → Kyiv).
- **Red models only:** the any-model's P(≤ H) at t.
- **Structure:** `oblast` (categorical) and bin index `k`.
- **Lookback:** at most 72 h, so the first 72 h after any data gap are warm-up and are not scored.

## Evaluation

Two evaluations, because the definition changed:

| | E1: method check, old system | E2: latest period, new system (the task's requirement) |
|---|---|---|
| Train | official 2025-12-01 → 2026-06-29 | old data → 2026-08-28 + new data 2026-09-13 → 09-27 |
| Validation | official 2026-07-01 → 07-28 | new data 2026-09-28 → 10-01 |
| Test | official 2026-07-30 → 08-28 | new data 2026-10-02 → last fetch (~10–12 Oct) |
| Models | any-start, any-end | all four |
| Why | Stable definition and ~4 weeks × 25 oblasts of test data: tight confidence intervals; answers "does the method work at all" | Current system; small test, so wide confidence intervals, stated openly |

- **Split boundaries:** each is separated by at least 24 h. An example at time t is kept only if t + its longest horizon ≤ the end of its split.
- **Old-data ablation on E2 (any-start and any-end):**
  - **(a)** trained on new data only;
  - **(b)** old + new with the regime feature (the main model);
  - **(c)** old data only, applied as is.

  If (b) beats (a), old data has been shown to help. The gap between (c) and the others puts the regime shift into numbers.

## Measuring accuracy

The headline is probability quality: when the model says 30%, the event happens about 30% of the time.

- **Brier score** at each reported horizon: mean of (p − outcome)².
- **Brier skill score:** `1 − Brier_model / Brier_baseline` against climatology (start models) or Kaplan–Meier (end models). This is **the headline number**: >0 means the model beats history alone.
- **Integrated Brier score** over the whole curve, as one number per model.
- **Log loss** at each horizon.
- **PR-AUC** for start models: events are rare in the west, so ROC-AUC would flatter.
- **Calibration:** reliability diagram (10 bins) plus expected calibration error per horizon.
- **Uncertainty:** 95% confidence intervals for the skill score from a day-block bootstrap.
- **Breakdown:**
  - three oblast groups: west, centre/south, frontline. The start-model headline excludes the frontline group, which has few quiet hours; end models are reported in full. Near-permanent oblasts are marked as such, with their trivial forecasts shown as is;
  - per-oblast table;
  - by hour of day.

## Leakage controls

1. **One feature code path:** `features(events, t)` filters the events itself.
2. **Property test** (`tests/test_leakage.py`): for ~200 random t in both regimes, features computed on the full data equal features computed on data truncated at t, where alerts that start after t are removed and end times after t are blanked.
3. **No label crosses a split boundary:** examples whose horizon window crosses the end of their split are dropped; splits are separated by at least 24 h.
4. **Fixed in advance:** the adjacency map and the level rule are written by hand; climatology and Kaplan–Meier come from training data; tuning and calibration use validation only.
5. **Test touched once per evaluation:** `alerts evaluate --eval e1|e2 --split test` writes `reports/<eval>_test_metrics.json` with a timestamp and the frozen model hash, both logged in the README.
6. **Red models:** they read the any-model's forecast at the same t, which is itself built only from data up to t.
7. **Telegram (stretch only):** messages used by `message.date ≤ t`; edited messages are dropped.

## Telegram ablation (stretch)

- **Source:** `@kpszsu` only, through Telethon using the local telegram skill session (read-only, no channel joins), adapting `~/$code/PET/ballistic-kyiv/backend/scripts/pull_corpus.py`.
- **Features:** national counts in the last 30/60/120 min of takeoff/launch posts (MiG-31K, Tu-95/160, ballistic, launch), reusing the regex approach of `ballistic-kyiv/backend/classifier.py`. No geocoding.
- **Reported as:** E2 red-start and any-start skill with vs. without these features.

## Repo

```
kse/
  pyproject.toml          uv, Python 3.12: pandas, numpy, pyarrow, lightgbm, scikit-learn, httpx, plotly, typer, pytest
  src/alerts/
    sources.py            official CSV + alerts.in.ua (append-only store) -> canonical table
    timeline.py           level rule, any/red streams, forecast times, hazard row expansion
    adjacency.py          oblast neighbour map
    features.py           features(events, t)
    models.py             climatology, Kaplan-Meier, logistic regression, LightGBM, temperature scaling
    evaluate.py           E1/E2 splits, metrics, bootstrap, reliability, ablation
    site.py               static HTML (Plotly + oblast GeoJSON)
    cli.py                fetch | train | evaluate | forecast | report
  tests/test_leakage.py
  reports/                generated metrics + plots
  site/                   generated HTML for GitHub Pages
  data/, artifacts/       gitignored; data/raw keeps every alerts.in.ua fetch (append-only store, back it up)
```

- **Reuse:**
  - alerts.in.ua field names follow `~/$code/PET/silent-sky/backend/services/alerts_in_ua.py`;
  - oblast boundaries come from the OCHA admin1 GeoJSON in silent-sky `backend/data/.cache`.
- **Region UIDs:** verified on 2026-10-10. 3 Khmelnytska, 4 Vinnytska, 5 Rivnenska, 8 Volynska, 9 Dnipropetrovska, 10 Zhytomyrska, 11 Zakarpatska, 12 Zaporizka, 13 Ivano-Frankivska, 14 Kyivska, 15 Kirovohradska, 16 Luhanska, 17 Mykolaivska, 18 Odeska, 19 Poltavska, 20 Sumska, 21 Ternopilska, 22 Kharkivska, 23 Khersonska, 24 Cherkaska, 25 Chernihivska, 26 Chernivetska, 27 Lvivska, 28 Donetska, 29 Crimea, 31 Kyiv City.
- **Code style:** no code comments, per Bohdan's conventions.

## Submission

- **GitHub repo** with a README covering:
  - the questions answered;
  - data provenance (why the volunteer data was rejected, the artillery rule, the regime shift);
  - E1 and E2 protocol and results with confidence intervals;
  - the old-data ablation;
  - the site link;
  - known limits.
- **AI log:** full export of this Claude Code session.
- **Reflection:** ~100 words, plain text. What went wrong:
  - the first plan was built on the volunteer data;
  - the official CSV turned out to be duplicated, stalled and mixed with artillery alerts;
  - the yellow/red switch made "alert" a different event.

  How the approach changed, and why the final version beats the first attempt.

## Verification

- `uv run pytest`: the leakage property test passes in both regimes.
- `uv run alerts fetch && uv run alerts train && uv run alerts evaluate --eval e1 --split val` and `--eval e2 --split val`: models beat their baselines on validation before any test set is touched.
- `uv run alerts evaluate --eval e1 --split test`, then `--eval e2 --split test`, each run once.
- `uv run alerts forecast --at <time in E2 test>`: compare by hand with the replay at the same time.
- `uv run alerts report`: open `site/index.html`, check all tabs; check the replay against known events (e.g. the 13–14 Sep mass strike, late-September Kyiv days with 12–18 h of alerts).

## Risks

- **The E2 test is about 8–10 days:** wide confidence intervals. That's why E1 exists; both are reported.
- **Weekly recalibration of yellow/red criteria** by the General Staff adds non-stationarity within the new system. It is reported, not modelled.
- **Red events are rare per oblast:** red models are pooled, and western oblasts are reported as a group.
- **The alerts.in.ua rolling window:** a fetch that replaced the store instead of adding to it would lose data, so the store is append-only and every raw fetch is kept in `data/raw/`.
