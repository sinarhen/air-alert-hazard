import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from alerts.adjacency import OBLASTS, group_of
from alerts.features import KINDS, Kind, expand, feature_frame, horizon_outcome, outcomes, warm_origins
from alerts.models import Candidates, Climatology, cumulative, curve_rows, fit_candidates, log_loss
from alerts.timeline import NEW, OLD, Timeline

HORIZONS = {"start": (1, 3, 6, 12, 24), "end": (1, 2, 4, 8)}
BOOTSTRAP = 300
OLD_RANGE = ("2025-12-01", "2026-08-29")
NEW_RANGE = ("2026-09-10", "2026-09-28")

EVALS = {
    "e1": {
        "train": {"main": [("2025-12-01", "2026-06-30")]},
        "val": ("2026-07-01", "2026-07-29"),
        "test": ("2026-07-30", "2026-08-29"),
        "kinds": ("any-start", "any-end"),
        "baseline_regime": OLD,
    },
    "e2": {
        "train": {"main": [OLD_RANGE, NEW_RANGE], "new_only": [NEW_RANGE], "old_only": [OLD_RANGE]},
        "val": ("2026-09-28", "2026-10-02"),
        "test": ("2026-10-02", None),
        "kinds": ("any-start", "any-end", "red-start", "red-end"),
        "baseline_regime": NEW,
    },
}


RED_DERIVED = ("level_now", "nb_red_now", "nat_red_now")


def level_columns(frame: pd.DataFrame, kind: Kind) -> pd.DataFrame:
    if kind.level == "red":
        return frame
    return frame.drop(columns=[c for c in frame.columns if c.startswith("red_") or c in RED_DERIVED])


def kind_of(name: str) -> Kind:
    return next(k for k in KINDS if k.name == name)


@dataclass
class Split:
    features: pd.DataFrame
    result: pd.DataFrame


class Workspace:
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.features = feature_frame(timeline, warm_origins(timeline, 0, timeline.size, 15))

    def bound(self, value: str | None) -> int:
        if value is None:
            return self.timeline.size
        return min(self.timeline.size, self.timeline.minute(pd.Timestamp(value, tz="UTC")))

    def split(self, kind: Kind, lo: str, hi: str | None) -> Split:
        start, stop = self.bound(lo), self.bound(hi)
        minute = self.features["minute"]
        chosen = self.features[(minute >= start) & (minute < stop) & (minute % kind.step == 0)]
        result = outcomes(self.timeline, kind, chosen, stop)
        keep = result["at_risk"].to_numpy()
        return Split(chosen[keep], result[keep])

    def training_rows(self, kind: Kind, ranges: list[tuple[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
        parts = [expand(s.features, s.result) for s in (self.split(kind, lo, hi) for lo, hi in ranges)]
        return pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts])


def _artifact(root: Path, eval_name: str, kind: Kind, variant: str) -> Path:
    suffix = "" if variant == "main" else f"-{variant}"
    return root / eval_name / f"{kind.name}{suffix}"


def train(workspace: Workspace, eval_name: str, artifacts: Path, reports: Path) -> dict:
    config = EVALS[eval_name]
    summary = {}
    for name in config["kinds"]:
        kind = kind_of(name)
        variants = config["train"] if kind.level == "any" else {"main": config["train"]["main"]}
        val = workspace.split(kind, *config["val"])
        val_rows, val_y = expand(val.features, val.result)
        for variant, ranges in variants.items():
            rows, y = workspace.training_rows(kind, ranges)
            rows = level_columns(rows, kind)
            candidates = fit_candidates(rows, y, val_rows, val_y)
            path = _artifact(artifacts, eval_name, kind, variant)
            candidates.save(path)
            if variant == "main":
                base_mask = (rows["regime"] == config["baseline_regime"]).to_numpy()
                pd.to_pickle(Climatology.fit(rows[base_mask], y[base_mask], kind), path.with_suffix(".baseline.pkl"))
            summary[f"{name}/{variant}"] = {
                "train_rows": int(len(rows)),
                "events": int(y.sum()),
                "champion": candidates.champion,
                "val_log_loss": candidates.val_log_loss,
                "temperature": candidates.lightgbm.temperature,
                "trials": candidates.trials,
            }
    _write(reports / eval_name / "training.json", summary)
    return summary


def predict(workspace: Workspace, eval_name: str, split_name: str, artifacts: Path) -> dict[str, pd.DataFrame]:
    config = EVALS[eval_name]
    frames = {}
    for name in config["kinds"]:
        kind = kind_of(name)
        split = workspace.split(kind, *config[split_name])
        rows = curve_rows(split.features)
        main = Candidates.load(_artifact(artifacts, eval_name, kind, "main"))
        baseline = pd.read_pickle(_artifact(artifacts, eval_name, kind, "main").with_suffix(".baseline.pkl"))
        curves = {
            "model": cumulative(main.hazard(rows)),
            "baseline": cumulative(baseline.hazard(rows, kind)),
            "lightgbm": cumulative(main.hazard(rows, "lightgbm")),
            "logistic": cumulative(main.hazard(rows, "logistic")),
        }
        variants = config["train"] if kind.level == "any" else {}
        for variant in variants:
            if variant != "main":
                curves[variant] = cumulative(Candidates.load(_artifact(artifacts, eval_name, kind, variant)).hazard(rows))
        frame = split.features[["minute", "oblast"]].reset_index(drop=True).join(split.result.reset_index(drop=True))
        for label, values in curves.items():
            for h in range(values.shape[1]):
                frame[f"{label}_{h + 1}"] = values[:, h].astype(np.float32)
        frames[name] = frame
    return frames


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table, error = [], 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            table.append({"bin": b, "p": float(p[mask].mean()), "y": float(y[mask].mean()), "n": int(mask.sum())})
            error += mask.sum() / p.size * abs(p[mask].mean() - y[mask].mean())
    return float(error), table


def _skill_ci(p: np.ndarray, base: np.ndarray, y: np.ndarray, days: np.ndarray) -> list[float]:
    frame = pd.DataFrame({"day": days, "m": (p - y) ** 2, "b": (base - y) ** 2}).groupby("day").sum()
    rng = np.random.default_rng(7)
    picks = rng.integers(0, len(frame), size=(BOOTSTRAP, len(frame)))
    model, baseline = frame["m"].to_numpy()[picks].sum(axis=1), frame["b"].to_numpy()[picks].sum(axis=1)
    skill = 1 - model / np.maximum(baseline, 1e-12)
    return [float(np.quantile(skill, 0.025)), float(np.quantile(skill, 0.975))]


def score(frame: pd.DataFrame, kind: Kind, timeline: Timeline) -> dict:
    groups = np.array([group_of(OBLASTS[i]) for i in frame["oblast"]])
    days = timeline.time(frame["minute"].to_numpy()).tz_convert("Europe/Kyiv").date
    headline = groups != "frontline" if kind.phase == "start" else np.ones(len(frame), dtype=bool)
    models = [c[:-2] for c in frame.columns if c.endswith("_1")]
    out = {"horizons": {}, "groups": {}, "oblasts": {}}

    def block(mask: np.ndarray, horizon: int, detailed: bool) -> dict | None:
        y, evaluable = horizon_outcome(frame, horizon)
        mask = mask & evaluable
        if mask.sum() < 20:
            return None
        y = y[mask].astype(float)
        result = {"n": int(mask.sum()), "event_rate": float(y.mean())}
        base = frame[f"baseline_{horizon}"].to_numpy()[mask]
        base_brier = _brier(base, y)
        for name in models:
            p = frame[f"{name}_{horizon}"].to_numpy()[mask]
            entry = {"brier": _brier(p, y), "skill": 1 - _brier(p, y) / base_brier if base_brier > 0 else None}
            if detailed:
                entry["log_loss"] = log_loss(y, p)
                entry["pr_auc"] = float(average_precision_score(y, p)) if 0 < y.sum() < y.size else None
                entry["ece"], entry["reliability"] = _ece(p, y)
                if name != "baseline":
                    entry["skill_ci"] = _skill_ci(p, base, y, days[mask])
            result[name] = entry
        return result

    for horizon in range(1, 25):
        entry = block(headline, horizon, horizon in HORIZONS[kind.phase])
        if entry:
            out["horizons"][horizon] = entry
    for name in models:
        values = [out["horizons"][h][name]["brier"] for h in out["horizons"]]
        out.setdefault("integrated_brier", {})[name] = float(np.mean(values)) if values else None
    for group in ("west", "centre_south", "frontline"):
        out["groups"][group] = {h: block(groups == group, h, False) for h in HORIZONS[kind.phase]}
    for i, oblast in enumerate(OBLASTS):
        out["oblasts"][oblast] = {h: block((frame["oblast"] == i).to_numpy(), h, False) for h in HORIZONS[kind.phase]}
    return out


def evaluate(workspace: Workspace, eval_name: str, split_name: str, artifacts: Path, reports: Path) -> dict:
    frames = predict(workspace, eval_name, split_name, artifacts)
    results = {}
    for name, frame in frames.items():
        frame.to_parquet(artifacts / eval_name / f"{name}.{split_name}.parquet")
        results[name] = score(frame, kind_of(name), workspace.timeline)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    digest = hashlib.sha256(b"".join(sorted(p.read_bytes() for p in (artifacts / eval_name).glob("*") if p.suffix in (".lgb", ".pkl")))).hexdigest()[:16]
    _write(reports / eval_name / f"{split_name}_metrics.json", {"evaluated_at": stamp, "model_hash": digest, "results": results})
    if split_name == "test":
        with (reports / "test_log.jsonl").open("a") as log:
            log.write(json.dumps({"eval": eval_name, "evaluated_at": stamp, "model_hash": digest}) + "\n")
    return results


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
