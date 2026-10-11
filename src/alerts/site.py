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


def _iso(timeline: Timeline, minute: int) -> str:
    return timeline.time(np.array([minute]))[0].isoformat()


LABEL_SHIFT = {"Київська область": (0.0, -0.45)}


def _area(ring: np.ndarray) -> float:
    x, y = ring.T
    return float(x[:-1] @ y[1:] - x[1:] @ y[:-1]) / 2


def _centroid(ring: np.ndarray) -> tuple[float, float]:
    x, y = ring.T
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    scale = 6 * _area(ring)
    return float(((x[:-1] + x[1:]) * cross).sum() / scale), float(((y[:-1] + y[1:]) * cross).sum() / scale)


def label_points(geo: dict) -> dict[str, list[float]]:
    out = {}
    for feature in geo["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        outer = max((np.array(polygon[0], dtype=float) for polygon in polygons), key=lambda ring: abs(_area(ring)))
        name = feature["properties"]["name"]
        lon, lat = _centroid(outer)
        dx, dy = LABEL_SHIFT.get(name, (0.0, 0.0))
        out[name] = [round(lon + dx, 3), round(lat + dy, 3)]
    return out


def _outcome(rows: pd.DataFrame, h: int) -> np.ndarray:
    evaluable = (rows["event_bin"] <= h) | (rows["observed_bins"] >= h)
    return np.where(evaluable, (rows["event_bin"] <= h).astype(float), np.nan)


def replay(timeline: Timeline, artifacts: Path, now: pd.DataFrame, now_minute: int, eval_name: str = "e2") -> dict | None:
    split = "test" if (artifacts / eval_name / "any-start.test.parquet").exists() else "val"
    window = EVALS[eval_name][split]
    lo = timeline.minute(pd.Timestamp(window[0], tz="UTC"))
    hi = timeline.size if window[1] is None else timeline.minute(pd.Timestamp(window[1], tz="UTC"))
    hours = np.arange(lo - lo % 60 + 60, min(hi, now_minute), 60)
    position = {m: i for i, m in enumerate(hours)}
    levels = {}
    for level in ("any", "red"):
        start_path = artifacts / eval_name / f"{level}-start.{split}.parquet"
        if not start_path.exists():
            continue
        state = timeline.any if level == "any" else timeline.red
        entry = {"active": np.column_stack([state[:, hours], now[f"{level}_active"]]).astype(int).tolist()}
        for phase, horizons, unit in (("start", HORIZONS["start"], 60), ("end", HORIZONS["end"], 15)):
            frame = pd.read_parquet(artifacts / eval_name / f"{level}-{phase}.{split}.parquet")
            rows = frame[frame["minute"].isin(position)]
            obl, col = rows["oblast"].to_numpy(), rows["minute"].map(position).to_numpy()
            entry[phase], entry[f"{phase}_outcome"] = {}, {}
            for h in horizons:
                forecast_grid = np.full((len(OBLASTS), hours.size + 1), np.nan)
                outcome_grid = np.full((len(OBLASTS), hours.size + 1), np.nan)
                forecast_grid[obl, col] = rows[f"model_{h}"]
                forecast_grid[:, -1] = now[f"{level}_{phase}_{h * unit}m"]
                outcome_grid[obl, col] = _outcome(rows, h)
                entry[phase][h * unit] = np.round(forecast_grid, 3).tolist()
                entry[f"{phase}_outcome"][h * unit] = outcome_grid.tolist()
        levels[level] = entry
    if not levels:
        return None
    times = [_iso(timeline, int(m)) for m in hours] + [_iso(timeline, now_minute)]
    return {"split": split, "times": times, "levels": levels}


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


def build(timeline: Timeline, artifacts: Path, extra: dict, target: Path) -> Path:
    latest = int(np.flatnonzero(timeline.known)[-1])
    now = forecast(timeline, latest, artifacts)
    geo = json.loads((ASSETS / "oblasts.geojson").read_text())
    payload = {
        "oblasts": list(OBLASTS),
        "geo": geo,
        "labels": label_points(geo),
        "replay": replay(timeline, artifacts, now, latest),
        "eda": {**eda(timeline), **extra},
        "horizons": {"start": [h * 60 for h in HORIZONS["start"]], "end": [h * 15 for h in HORIZONS["end"]]},
    }
    template = (ASSETS / "site.html").read_text()
    html = template.replace("__PLOTLY__", PLOTLY).replace("__DATA__", json.dumps(payload, ensure_ascii=False, default=float).replace("NaN", "null"))
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text(html)
    return target / "index.html"
