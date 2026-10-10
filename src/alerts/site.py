import json
from pathlib import Path

import numpy as np
import pandas as pd

from alerts.adjacency import OBLASTS, group_of
from alerts.evaluate import EVALS, HORIZONS
from alerts.forecast import forecast
from alerts.timeline import NEW, OLD, Timeline

ASSETS = Path(__file__).resolve().parents[2] / "assets"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _runs(state: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], state[lo:hi], [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(lo + int(a), lo + int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _iso(timeline: Timeline, minute: int) -> str:
    return timeline.time(np.array([minute]))[0].isoformat()


def replay(timeline: Timeline, artifacts: Path, eval_name: str) -> dict | None:
    config = EVALS[eval_name]
    lo = timeline.minute(pd.Timestamp(config["test"][0], tz="UTC"))
    hi = timeline.size if config["test"][1] is None else timeline.minute(pd.Timestamp(config["test"][1], tz="UTC"))
    hours = np.arange(lo - lo % 60 + 60, hi, 60)
    levels = {}
    for level in ("any", "red"):
        start_path = artifacts / eval_name / f"{level}-start.test.parquet"
        if not start_path.exists():
            continue
        start = pd.read_parquet(start_path)
        end = pd.read_parquet(artifacts / eval_name / f"{level}-end.test.parquet")
        state = timeline.any if level == "any" else timeline.red
        entry = {"active": state[:, hours].astype(int).tolist(), "start": {}, "baseline": {}, "happened": {}, "end": {}}
        position = {m: i for i, m in enumerate(hours)}
        for h in HORIZONS["start"]:
            grids = {key: np.full((len(OBLASTS), hours.size), np.nan) for key in ("start", "baseline", "happened")}
            rows = start[start["minute"].isin(position)]
            col = rows["minute"].map(position).to_numpy()
            obl = rows["oblast"].to_numpy()
            grids["start"][obl, col] = rows[f"model_{h}"]
            grids["baseline"][obl, col] = rows[f"baseline_{h}"]
            happened = (rows["event_bin"] <= h).astype(float).to_numpy()
            evaluable = (rows["event_bin"] <= h) | (rows["observed_bins"] >= h)
            grids["happened"][obl, col] = np.where(evaluable, happened, np.nan)
            for key, grid in grids.items():
                entry[key][h] = np.round(grid, 3).tolist()
        for h in HORIZONS["end"]:
            grid = np.full((len(OBLASTS), hours.size), np.nan)
            rows = end[end["minute"].isin(position)]
            grid[rows["oblast"].to_numpy(), rows["minute"].map(position).to_numpy()] = rows[f"model_{h}"]
            entry["end"][h * 15] = np.round(grid, 3).tolist()
        levels[level] = entry
    if not levels:
        return None
    bands = {
        level: [[[_iso(timeline, a), _iso(timeline, b)] for a, b in _runs(state[i], lo, hi)] for i in range(len(OBLASTS))]
        for level, state in (("any", timeline.any), ("red", timeline.red))
        if level in levels
    }
    return {"times": [_iso(timeline, int(m)) for m in hours], "levels": levels, "bands": bands}


def eda(timeline: Timeline) -> dict:
    regimes = {"old": timeline.regime == OLD, "new": timeline.regime == NEW}
    kyiv = timeline.time(np.arange(timeline.size)).tz_convert("Europe/Kyiv")
    hour, dow = kyiv.hour.to_numpy(), kyiv.dayofweek.to_numpy()
    onsets = timeline.any & ~np.concatenate([np.zeros((len(OBLASTS), 1), dtype=bool), timeline.any[:, :-1]], axis=1)
    out = {"share": {}, "hour": {}, "dow": {}}
    for name, mask in regimes.items():
        out["share"][name] = {
            "any": (timeline.any[:, mask].mean(axis=1) * 100).round(1).tolist(),
            "red": (timeline.red[:, mask].mean(axis=1) * 100).round(1).tolist() if name == "new" else None,
        }
        starts = onsets[:, mask].sum(axis=0)
        days = mask.sum() / 1440
        out["hour"][name] = [float(starts[hour[mask] == h].sum() / days) for h in range(24)]
        out["dow"][name] = [float(starts[dow[mask] == d].sum() / max(1, (dow[mask] == d).sum() / 1440)) for d in range(7)]
    sample = np.flatnonzero(regimes["new"])[::15]
    out["cooccurrence"] = np.round(np.nan_to_num(np.corrcoef(timeline.any[:, sample].astype(float))), 2).tolist()
    out["groups"] = [group_of(o) for o in OBLASTS]
    return out


def scorecard(reports: Path) -> dict:
    out = {}
    for eval_name in EVALS:
        for split in ("val", "test"):
            path = reports / eval_name / f"{split}_metrics.json"
            if path.exists():
                out.setdefault(eval_name, {})[split] = json.loads(path.read_text())
        training = reports / eval_name / "training.json"
        if training.exists():
            out.setdefault(eval_name, {})["training"] = {
                k: {"champion": v["champion"], "val_log_loss": v["val_log_loss"], "train_rows": v["train_rows"], "events": v["events"]}
                for k, v in json.loads(training.read_text()).items()
            }
    return out


def raion_finding(events: pd.DataFrame, lo: str = "2025-12-01", hi: str = "2026-07-29") -> list[dict]:
    start, stop = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
    rows = events[(events["source"] == "official") & (events["location_type"] == "raion")]
    rows = rows[(rows["finished_at"] > start) & (rows["started_at"] < stop)]
    span = (stop - start) / pd.Timedelta(minutes=1)
    out = []
    for (oblast, raion), group in rows.groupby(["oblast", "raion"]):
        intervals = sorted(zip(group["started_at"].clip(lower=start), group["finished_at"].clip(upper=stop)))
        covered, cursor = 0.0, start
        for a, b in intervals:
            a = max(a, cursor)
            if b > a:
                covered += (b - a) / pd.Timedelta(minutes=1)
                cursor = b
        out.append({"oblast": oblast, "raion": raion, "share": round(100 * covered / span, 1)})
    return sorted(out, key=lambda r: -r["share"])


def build(timeline: Timeline, artifacts: Path, reports: Path, extra: dict, target: Path) -> Path:
    latest = int(np.flatnonzero(timeline.known)[-1])
    now = forecast(timeline, latest, artifacts)
    payload = {
        "oblasts": list(OBLASTS),
        "geo": json.loads((ASSETS / "oblasts.geojson").read_text()),
        "forecast": {"at": _iso(timeline, latest), "table": json.loads(now.to_json(orient="records", force_ascii=False))},
        "replay": {name: replay(timeline, artifacts, name) for name in EVALS},
        "scores": scorecard(reports),
        "eda": {**eda(timeline), **extra},
        "horizons": HORIZONS,
    }
    template = (ASSETS / "site.html").read_text()
    html = template.replace("__PLOTLY__", PLOTLY).replace("__DATA__", json.dumps(payload, ensure_ascii=False, default=float).replace("NaN", "null"))
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text(html)
    return target / "index.html"
