# Air raid alert hazard forecast

Per oblast, at any moment, this project answers two questions people actually ask during the war:

- **Oblast is quiet:** what is the chance an alert starts within the next 1–24 h?
- **Oblast is under alert:** what is the chance it is over within the next 15 min – 2 h?

Both are answered for **any alert** (yellow or red) and for **red alerts** (missiles, ballistic, massed attack), matching the two-level system Ukraine introduced on 6 September 2026.

Everything is computed only from alerts declared before the forecast time. That is enforced in code and checked by a test (see *Leakage*).

## Quick start

```bash
uv sync
export ALERTS_IN_UA_TOKEN=...          # https://devs.alerts.in.ua (free, by request)
uv run alerts fetch                    # official CSV + 30-day alerts.in.ua history (append-only)
uv run alerts train --eval e1          # old-system method check: train + validation metrics
uv run alerts train --eval e2          # new-system models: train + validation metrics
uv run alerts evaluate --eval e1 --split test
uv run alerts evaluate --eval e2 --split test
uv run alerts forecast --at 2026-10-09T18:00Z
uv run alerts report                   # site/index.html
uv run pytest
```

## Data

| Source | What | Used for |
|---|---|---|
| [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) `official_data_uk.csv` | Declarations scraped from the official `@air_alert_ua` channel; oblast/raion/hromada level; no type or level | Old system, Dec 2025 – Aug 2026 |
| [alerts.in.ua](https://devs.alerts.in.ua) `regions/{uid}/alerts/month_ago.json` | Declarations with `alert_type` and `alert_level` (red/yellow); rolling 30-day window | New system, 10 Sep 2026 onward |

### Findings that shaped the design

1. **The volunteer feed is not ground truth.**
   - The dataset's `volunteer_data` (eTryvoga) is the only file that is still updated daily.
   - Cross-checked against official declarations (Jun–Oct 2025), only ~50% of its starts match an official start to the minute, and ~30% have no official start within 5 minutes.
   - Kyiv City shows 4.1% of time under alert there vs. 8.4% officially, and western oblasts show alerts that were never declared.
   - Not used.
2. **The official CSV needs cleaning.** Every row appears twice, it has a gap from 29 Aug to 1 Sep, and it stops on 7 Sep 2026, when the upstream channel went silent.
3. **Artillery is mixed in.**
   - The official CSV has no alert type. Its hromada-level rows turned out to be artillery-shelling hromadas (Nikopol, Marhanets, Pokrovsk…).
   - alerts.in.ua confirms that `artillery_shelling` appears only at hromada/city level, never at raion or oblast level.
   - **Rule, applied to both sources:** only oblast- and raion-level declarations are used. This removes 100% of artillery alerts. Cost: some city-only air raid alerts are lost (Kirovohrad 19% of its air-raid time, Kharkiv 10%, Dnipro 4%).
4. **The definition of an alert changed on 6 Sep 2026.** Yellow (drone threat) and red levels replaced the single siren-backed alert, and 63% of air raid alerts are now yellow. Share of time under alert:

   | | Old system (Dec–Jul) | New: any | New: red |
   |---|---|---|---|
   | Kyiv City | 5.7% | 31.6% | 2.7% |
   | Kyiv oblast | 18.6% | 69.1% | 11.6% |
   | Odesa | 18.9% | 40.5% | 6.6% |

   The old data is a different event. It is used deliberately (see *Models*), never silently mixed in.
5. **No raion is dropped.**
   - Each oblast's timeline is the union of its oblast- and raion-level declarations. A few oblasts therefore look near-permanent: Dnipropetrovsk is 88.6% under alert, while Dniprovskyi raion, where Dnipro city is, is under alert only 23%.
   - Threshold cuts were checked and rejected. Raion coverage has no natural gap between 50% and 100%, and a 60% cut would wipe out Donetsk and Zaporizhzhia entirely, because all their raions carry identical oblast-wide declarations.

## Models

Discrete-time hazard (survival) models. For each future bin k, the model predicts the probability that the event happens in bin k given it hasn't happened yet. The forecast curve is `P(event ≤ H) = 1 − Π(1 − h_k)`, so one model gives every horizon, and the curve never decreases.

| Model | Forecast times | Bins | Event |
|---|---|---|---|
| any-start | hourly, oblast quiet | 24 × 1 h | an alert starts |
| any-end | every 15 min, oblast under alert | 24 × 15 min | the oblast is quiet for ≥ 2 min |
| red-start | hourly, oblast not under red | 24 × 1 h | a red alert starts |
| red-end | every 15 min, oblast under red | 24 × 15 min | red ends |

- **Features** (one function, `features.feature_frame`, used for training, evaluation, the CLI and the site):
  - hour and weekday (Kyiv time);
  - current state and how long it has lasted;
  - exponentially decayed alert-start counts (3 h, 24 h, 72 h);
  - share of the oblast's raions under alert;
  - neighbours under alert and their recent starts (hand-written adjacency map);
  - national counts, and time since the last national wave (≥15 oblasts within 30 min);
  - the current state of every oblast.
- **Candidates per model:** LightGBM and logistic regression on the same rows. Logistic regression with a one-hot bin index is the classic discrete-time proportional-hazards model. The **champion is chosen by validation log loss**, and both are reported.
- **Old data:**
  - the any-models train on old + new rows with a `regime` feature;
  - the red models train on new rows only, since old rows have no red label;
  - the any-models do not use red-derived features at all (see *Leakage*).
- **Baselines:**
  - **start models:** climatology, i.e. the empirical hazard per oblast × hour of day × bin, fitted on the evaluated regime's training rows;
  - **end models:** Kaplan–Meier style life tables per oblast × elapsed-duration bucket × bin.

## Evaluation

| | E1: method check (old system) | E2: latest period (new system) |
|---|---|---|
| Train | 2025-12-01 → 2026-06-29 | old 2025-12-01 → 2026-08-28 + new 2026-09-13 → 09-27 |
| Validation | 2026-07-01 → 07-28 | 2026-09-28 → 10-01 |
| Test (touched once) | 2026-07-30 → 08-28 | 2026-10-02 → last fetch |

- **Metric:** the headline is the **Brier skill score** against the baseline: >0 means better than history alone. 95% confidence intervals come from a day-block bootstrap.
- **Also reported:** log loss, PR-AUC, reliability diagrams and calibration error, the integrated Brier score over the whole curve, and per-group / per-oblast tables.
- **Frontline oblasts** are excluded from the start-model headline, because they are almost never quiet; they are reported separately.
- **Old-data ablation (E2, any-models):** trained on new data only vs. old + new vs. old only.

## Results

Brier skill vs. baseline, with 95% day-block bootstrap CI. Full tables, calibration and per-oblast results are in `reports/`.

**E1 test** (old system, 30 Jul – 28 Aug 2026; evaluated once):

| Model (champion) | 1 h / 15 min | 3 h / 30 min | 6 h / 60 min | 12 h / 120 min | 24 h |
|---|---|---|---|---|---|
| any-start (logistic) | +9.9% [7.7, 12.0] | +14.7% [11.2, 18.5] | +12.2% [7.6, 17.0] | +6.8% [1.9, 11.8] | +2.0% [−1.8, 6.8] |
| any-end (LightGBM) | +2.5% [1.1, 3.8] | +2.1% [0.8, 3.3] | +3.1% [1.7, 4.6] | +5.8% [4.0, 7.8] | |

**E2 validation** (new system, 28 Sep – 1 Oct 2026). These days also chose the champion, early stopping and temperature, so the figures lean optimistic; the baseline got no such tuning. With only 4 days the CIs are wide (any-start 1 h: [+2, +13]; 6 h: [−6, +21]). The E2 test runs once on 12 Oct, after the final data fetch.

| Model (champion) | 1 h / 15 min | 3 h / 30 min | 6 h / 60 min | 12 h / 120 min | 24 h |
|---|---|---|---|---|---|
| any-start (LightGBM, old + new) | +7.0% | +6.5% | +7.7% | +12.4% | +15.9% |
| any-start, new data only | +4.7% | +3.0% | +1.4% | −0.5% | −1.0% |
| any-start, old data only | +4.4% | +3.0% | +1.8% | −2.4% | −18.8% |
| any-end (LightGBM) | +0.3% | −0.6% | −0.4% | +4.2% | |
| red-start (logistic) | −0.4% | −3.1% | −0.8% | +0.5% | −9.6% |
| red-end (LightGBM) | −1.2% | −2.3% | −1.8% | +0.8% | |

- **Old data helps.** On validation, old + new data beats either source alone at every horizon (point estimates; the differences have no CIs yet). The old-only collapse at 24 h is the regime shift in numbers.
- **LightGBM vs. logistic.** LightGBM lost to logistic regression for E1 starts: trees learned on December–June did not transfer to July, while the proportional-hazards structure did. Hence champion selection by validation instead of assuming LightGBM.

## Leakage

- **Minute-level state is causal by construction.** An alert counts as active at minute m only if it started at or before m, and as ended only if its end is at or before m (`timeline.build_timeline`).
- **Property test** (`tests/test_leakage.py`):
  - features at random times t in both regimes are identical whether built from the full data or from data truncated at t (later alerts removed, later ends blanked);
  - a deliberately leaky feature (state 30 min ahead) makes the test fail, so the test has teeth.
- **Labels never cross a split boundary.** Every split censors labels at its end.
- **Tuning uses validation only:** model choice, grid search and temperature scaling. Each test set was evaluated once, and the time and model hash are logged in `reports/test_log.jsonl`.
- **Red-level escalation:**
  - The history endpoint keeps one `alert_level` per alert. If a yellow alert is escalated to red *in place*, its history row says red from the start.
  - So the any-models use no red-derived features: dropping them improved any-start validation log loss (0.2066 → 0.2044) and cost any-end 0.6%.
  - A live poller of active alerts checks whether in-place escalation happens.

## Limits

- **The E2 test is short** (days, not weeks), so its confidence intervals are wide. That is why E1 exists.
- **Red models do not beat climatology on validation.** Red events in the new system are rare, and there are only ~2 weeks to learn from. The red forecast is still shown, labelled as no better than historical rates.
- **any-start is overconfident above ~60%** (forecast 65–72% within 1 h → observed ~50% on validation).
- **The General Staff recalibrates yellow/red criteria weekly.** This non-stationarity is reported, not modelled.
- **Telegram text was not used.** Checked against @kpszsu and @war_monitor history (6 Sep – 1 Oct):
  - no post came before the first red start of a wave: in all 24 of 24 cases a red start was already under way;
  - posts made while red was quiet were followed by red within a median of 1 min, so they arrive with the alert, not ahead of it;
  - the only signal is spread *within* a wave already under way (2.6–5× more new red oblasts in the next hour), which the national and neighbour features partly cover;
  - edited and deleted posts make backfilled history differ from what was visible live.

  Four validation days cannot show a gain this small, so it was left out.

## Layout

```
src/alerts/
  sources.py     download + load both sources into one canonical table
  timeline.py    minute-level any/red state per oblast, causal by construction
  adjacency.py   oblast neighbour map and reporting groups
  features.py    features at forecast times, hazard labels, row expansion
  models.py      LightGBM, logistic, climatology/life-table baselines, champion selection
  evaluate.py    E1/E2 splits, metrics, bootstrap, ablation
  forecast.py    forecast at an arbitrary time
  site.py        static HTML (assets/site.html template, assets/oblasts.geojson)
  cli.py         fetch | train | evaluate | forecast | report
tests/test_leakage.py
```
