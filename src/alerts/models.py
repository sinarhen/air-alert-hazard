import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from alerts.features import BINS, Kind

EPS = 1e-6
GRID = [{"num_leaves": leaves, "min_data_in_leaf": min_data} for leaves in (15, 31, 63) for min_data in (200, 1000)]
ELAPSED_EDGES = [0, 15, 30, 60, 120, 240, 480, np.inf]
SMOOTHING = 20.0
LOGISTIC_ROWS = 400_000


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "minute"]


def curve_rows(features: pd.DataFrame) -> pd.DataFrame:
    rows = features.loc[features.index.repeat(BINS)].reset_index(drop=True)
    rows.insert(0, "k", np.tile(np.arange(1, BINS + 1, dtype=np.float32), len(features)))
    return rows


def cumulative(hazard: np.ndarray) -> np.ndarray:
    return 1 - np.cumprod(1 - hazard.reshape(-1, BINS), axis=1)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class Booster:
    booster: lgb.Booster
    columns: list[str]
    temperature: float = 1.0
    params: dict = field(default_factory=dict)

    def hazard(self, rows: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(rows[self.columns], raw_score=True)
        return 1 / (1 + np.exp(-raw / self.temperature))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path.with_suffix(".lgb")))
        meta = {"columns": self.columns, "temperature": self.temperature, "params": self.params}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Booster":
        meta = json.loads(path.with_suffix(".json").read_text())
        booster = lgb.Booster(model_file=str(path.with_suffix(".lgb")))
        return cls(booster, meta["columns"], meta["temperature"], meta["params"])


def fit_booster(train: pd.DataFrame, y_train: np.ndarray, val: pd.DataFrame, y_val: np.ndarray) -> tuple[Booster, list[dict]]:
    columns = feature_columns(train)
    train_set = lgb.Dataset(train[columns], y_train, categorical_feature=["oblast"], free_raw_data=False)
    val_set = lgb.Dataset(val[columns], y_val, reference=train_set, categorical_feature=["oblast"])
    trials = []
    best = None
    for grid in GRID:
        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": 7,
            **grid,
        }
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=3000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        score = booster.best_score["valid_0"]["binary_logloss"]
        trials.append({**grid, "rounds": booster.best_iteration, "val_logloss": score})
        if best is None or score < best[0]:
            best = (score, booster, {**params, "rounds": booster.best_iteration})
    model = Booster(best[1], columns, 1.0, best[2])
    raw = best[1].predict(val[columns], raw_score=True)
    fitted = minimize_scalar(lambda t: log_loss(y_val, 1 / (1 + np.exp(-raw / t))), bounds=(0.5, 3.0), method="bounded")
    if log_loss(y_val, 1 / (1 + np.exp(-raw / fitted.x))) < best[0] - 1e-5:
        model.temperature = float(fitted.x)
    return model, trials


@dataclass
class Climatology:
    table: pd.Series
    fallback: pd.Series

    @classmethod
    def fit(cls, rows: pd.DataFrame, y: np.ndarray, kind: Kind) -> "Climatology":
        keys = cls._keys(rows, kind)
        frame = keys.assign(y=y)
        fine_cols = list(keys.columns)
        coarse_cols = [c for c in fine_cols if c != "slot"]
        fallback = frame.groupby(coarse_cols)["y"].mean()
        stats = frame.groupby(fine_cols)["y"].agg(["sum", "count"])
        prior = fallback.reindex(stats.index.droplevel("slot")).to_numpy()
        table = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
        return cls(table, fallback)

    @staticmethod
    def _keys(rows: pd.DataFrame, kind: Kind) -> pd.DataFrame:
        if kind.phase == "start":
            slot = rows["hour"].astype(int)
        else:
            elapsed = rows[f"{kind.level}_since_change"]
            slot = pd.cut(elapsed, ELAPSED_EDGES, right=False, labels=False).fillna(0).astype(int)
        return pd.DataFrame({"oblast": rows["oblast"].astype(int), "slot": slot, "k": rows["k"].astype(int)})

    def hazard(self, rows: pd.DataFrame, kind: Kind) -> np.ndarray:
        keys = self._keys(rows, kind)
        fine = self.table.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        coarse = self.fallback.reindex(pd.MultiIndex.from_frame(keys[["oblast", "k"]])).to_numpy()
        pooled = float(np.nanmean(self.fallback.to_numpy()))
        return np.where(np.isnan(fine), np.where(np.isnan(coarse), pooled, coarse), fine)


@dataclass
class Logistic:
    pipeline: object
    columns: list[str]

    @classmethod
    def fit(cls, rows: pd.DataFrame, y: np.ndarray) -> "Logistic":
        columns = [c for c in feature_columns(rows) if rows[c].notna().any()]
        if len(rows) > LOGISTIC_ROWS:
            pick = np.random.default_rng(7).choice(len(rows), LOGISTIC_ROWS, replace=False)
            rows, y = rows.iloc[pick], y[pick]
        numeric = [c for c in columns if c not in ("oblast", "k")]
        transform = ColumnTransformer(
            [
                ("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), StandardScaler()), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["oblast", "k"]),
            ]
        )
        pipeline = make_pipeline(transform, LogisticRegression(max_iter=500, C=1.0))
        pipeline.fit(rows[columns], y)
        return cls(pipeline, columns)

    def hazard(self, rows: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(rows[self.columns])[:, 1]


@dataclass
class Candidates:
    lightgbm: Booster
    logistic: Logistic
    champion: str
    val_log_loss: dict
    trials: list
    kind: Kind
    climatology: Climatology | None = None

    def hazard(self, rows: pd.DataFrame, name: str | None = None) -> np.ndarray:
        name = name or self.champion
        if name == "climatology":
            return self.climatology.hazard(rows, self.kind)
        return getattr(self, name).hazard(rows)

    def save(self, path: Path) -> None:
        self.lightgbm.save(path)
        pd.to_pickle(self.logistic, path.with_suffix(".logistic.pkl"))
        meta = {"champion": self.champion, "val_log_loss": self.val_log_loss, "trials": self.trials}
        path.with_suffix(".champion.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path, kind: Kind) -> "Candidates":
        meta = json.loads(path.with_suffix(".champion.json").read_text())
        baseline = path.with_suffix(".baseline.pkl")
        climatology = pd.read_pickle(baseline) if baseline.exists() else None
        return cls(Booster.load(path), pd.read_pickle(path.with_suffix(".logistic.pkl")), meta["champion"], meta["val_log_loss"], meta["trials"], kind, climatology)


def fit_candidates(
    train: pd.DataFrame,
    y_train: np.ndarray,
    val: pd.DataFrame,
    y_val: np.ndarray,
    kind: Kind,
    climatology: Climatology | None = None,
) -> Candidates:
    booster, trials = fit_booster(train, y_train, val, y_val)
    logistic = Logistic.fit(train, y_train)
    scores = {"lightgbm": log_loss(y_val, booster.hazard(val)), "logistic": log_loss(y_val, logistic.hazard(val))}
    if climatology is not None:
        scores["climatology"] = log_loss(y_val, climatology.hazard(val, kind))
    return Candidates(booster, logistic, min(scores, key=scores.get), scores, trials, kind, climatology)
