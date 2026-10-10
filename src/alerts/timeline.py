from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alerts.adjacency import OBLASTS
from alerts.sources import Coverage, load_alerts_in_ua, load_official

GRID_START = pd.Timestamp("2025-11-01", tz="UTC")
OFFICIAL_END = pd.Timestamp("2026-08-29", tz="UTC")
OLD, NEW = 0, 1


@dataclass(frozen=True)
class Segment:
    source: str
    regime: int
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class Timeline:
    start: pd.Timestamp
    known: np.ndarray
    regime: np.ndarray
    segment_starts: np.ndarray
    any: np.ndarray
    red: np.ndarray
    raion_share: np.ndarray

    @property
    def size(self) -> int:
        return self.known.size

    def minute(self, ts: pd.Timestamp) -> int:
        return int(np.ceil((ts - self.start) / pd.Timedelta(minutes=1)))

    def time(self, minutes: np.ndarray) -> pd.DatetimeIndex:
        return self.start + pd.to_timedelta(minutes, unit="min")


def segments_for(new_coverage: Coverage) -> list[Segment]:
    return [
        Segment("official", OLD, GRID_START, OFFICIAL_END),
        Segment("alerts_in_ua", NEW, new_coverage.start, new_coverage.end),
    ]


def truncate(events: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    visible = events[events["started_at"] <= cutoff].copy()
    visible.loc[visible["finished_at"] > cutoff, "finished_at"] = pd.NaT
    return visible


def _minutes(values: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return np.ceil((values - origin) / pd.Timedelta(minutes=1)).to_numpy()


def _paint(rows: pd.DataFrame, origin: pd.Timestamp, lo: int, hi: int, size: int) -> np.ndarray:
    diff = np.zeros(size + 1, dtype=np.int32)
    if len(rows):
        begin = np.clip(_minutes(rows["started_at"], origin), lo, hi).astype(np.int64)
        finish = _minutes(rows["finished_at"], origin)
        finish = np.clip(np.where(np.isnan(finish), hi, finish), lo, hi).astype(np.int64)
        keep = finish > begin
        np.add.at(diff, begin[keep], 1)
        np.add.at(diff, finish[keep], -1)
    return np.cumsum(diff[:size]) > 0


def build_timeline(events: pd.DataFrame, segments: list[Segment], end: pd.Timestamp) -> Timeline:
    size = int((end - GRID_START) / pd.Timedelta(minutes=1))
    known = np.zeros(size, dtype=bool)
    regime = np.full(size, -1, dtype=np.int8)
    any_state = np.zeros((len(OBLASTS), size), dtype=bool)
    red_state = np.zeros((len(OBLASTS), size), dtype=bool)
    share = np.zeros((len(OBLASTS), size), dtype=np.float32)
    starts = []
    for segment in segments:
        lo = max(0, int(_minutes(pd.Series([segment.start]), GRID_START)[0]))
        hi = min(size, int((segment.end - GRID_START) / pd.Timedelta(minutes=1)))
        if hi <= lo:
            continue
        known[lo:hi] = True
        regime[lo:hi] = segment.regime
        starts.append(lo)
        rows = events[events["source"] == segment.source]
        for index, oblast in enumerate(OBLASTS):
            mine = rows[rows["oblast"] == oblast]
            whole = _paint(mine[mine["location_type"] == "oblast"], GRID_START, lo, hi, size)
            raions = mine[mine["location_type"] == "raion"]
            painted = [_paint(group, GRID_START, lo, hi, size) | whole for _, group in raions.groupby("raion")]
            union = np.logical_or.reduce(painted + [whole])
            any_state[index, lo:hi] = union[lo:hi]
            share[index, lo:hi] = (np.mean(painted, axis=0) if painted else whole)[lo:hi]
            if segment.regime == NEW:
                red_state[index, lo:hi] = _paint(mine[mine["level"] == "red"], GRID_START, lo, hi, size)[lo:hi]
    return Timeline(GRID_START, known, regime, np.array(sorted(starts)), any_state, red_state, share)


def load(data_dir: Path, cutoff: pd.Timestamp | None = None) -> tuple[pd.DataFrame, Coverage, Timeline]:
    official = load_official(data_dir / "official_data_uk.csv")
    recent, coverage = load_alerts_in_ua(data_dir / "raw")
    events = pd.concat([official, recent], ignore_index=True)
    end = coverage.end
    if cutoff is not None:
        events = truncate(events, cutoff)
        end = min(end, cutoff.ceil("min") + pd.Timedelta(minutes=1))
    return events, coverage, build_timeline(events, segments_for(coverage), end)
