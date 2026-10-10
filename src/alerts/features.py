from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from alerts.adjacency import OBLASTS, neighbour_matrix
from alerts.timeline import NEW, Timeline

LOOKBACK = 72 * 60
CAP = float(LOOKBACK)
HALF_LIVES = {"3h": 180, "24h": 1440, "72h": 4320}
WAVE_OBLASTS = 15
WAVE_WINDOW = 30
QUIET_GAP = 2


@dataclass(frozen=True)
class Kind:
    level: str
    phase: str

    @property
    def bin_minutes(self) -> int:
        return 60 if self.phase == "start" else 15

    @property
    def step(self) -> int:
        return 60 if self.phase == "start" else 15

    @property
    def name(self) -> str:
        return f"{self.level}-{self.phase}"


BINS = 24
KINDS = (Kind("any", "start"), Kind("any", "end"), Kind("red", "start"), Kind("red", "end"))


def _shift(x: np.ndarray, fill=False) -> np.ndarray:
    out = np.empty_like(x)
    out[:, 0] = fill
    out[:, 1:] = x[:, :-1]
    return out


def _onsets(state: np.ndarray, segment_starts: np.ndarray) -> np.ndarray:
    onsets = state & ~_shift(state)
    onsets[:, segment_starts] = False
    return onsets


def _since(events: np.ndarray, resets: np.ndarray) -> np.ndarray:
    columns = np.arange(events.shape[1])
    marks = np.where(events | resets[None, :], columns, -1)
    last = np.maximum.accumulate(marks, axis=1)
    return np.minimum(columns - last, CAP).astype(np.float32)


def _window(events: np.ndarray, width: int) -> np.ndarray:
    total = np.cumsum(events, axis=1, dtype=np.int32)
    lagged = np.zeros_like(total)
    lagged[:, width:] = total[:, :-width]
    return (total - lagged).astype(np.float32)


def _decay(events: np.ndarray, half_life: int, segment_starts: np.ndarray) -> np.ndarray:
    factor = 0.5 ** (1 / half_life)
    out = np.zeros(events.shape, dtype=np.float32)
    bounds = list(segment_starts) + [events.shape[1]]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        out[:, lo:hi] = lfilter([1.0], [1.0, -factor], events[:, lo:hi].astype(np.float32), axis=1)
    return out


def _next_index(mask: np.ndarray) -> np.ndarray:
    size = mask.shape[-1]
    marks = np.where(mask, np.arange(size), size)
    return np.minimum.accumulate(marks[..., ::-1], axis=-1)[..., ::-1]


def warm_origins(timeline: Timeline, lo: int, hi: int, step: int) -> np.ndarray:
    unknown = np.cumsum(~timeline.known, dtype=np.int64)
    minutes = np.arange(lo - lo % step + (step if lo % step else 0), hi, step)
    minutes = minutes[(minutes >= LOOKBACK) & (minutes < timeline.size)]
    clean = unknown[minutes] - unknown[minutes - LOOKBACK] == 0
    return minutes[clean & timeline.known[minutes]]


def feature_frame(timeline: Timeline, minutes: np.ndarray) -> pd.DataFrame:
    end = int(minutes.max()) + 1
    any_state = timeline.any[:, :end]
    red_state = timeline.red[:, :end]
    starts_idx = timeline.segment_starts[timeline.segment_starts < end]
    resets = np.zeros(end, dtype=bool)
    resets[starts_idx] = True
    new_regime = timeline.regime[:end] == NEW
    neighbours = neighbour_matrix().astype(np.float32)

    any_on = _onsets(any_state, starts_idx)
    red_on = _onsets(red_state, starts_idx)
    any_change = any_on | (~any_state & _shift(any_state))
    red_change = red_on | (~red_state & _shift(red_state))
    in_wave = (_window(any_on, WAVE_WINDOW) > 0).sum(axis=0) >= WAVE_OBLASTS

    columns: dict[str, np.ndarray] = {}

    def take(values: np.ndarray, masked: bool = False) -> np.ndarray:
        sampled = values[:, minutes].astype(np.float32)
        if masked:
            sampled[:, ~new_regime[minutes]] = np.nan
        return sampled

    columns["any_now"] = take(any_state)
    columns["red_now"] = take(red_state, masked=True)
    columns["level_now"] = take(any_state.astype(np.int8) + red_state.astype(np.int8), masked=True)
    columns["any_since_change"] = take(_since(any_change, resets))
    columns["red_since_change"] = take(_since(red_change, resets), masked=True)
    columns["any_since_start"] = take(_since(any_on, resets))
    columns["red_since_start"] = take(_since(red_on, resets), masked=True)
    for label, half_life in HALF_LIVES.items():
        columns[f"any_decay_{label}"] = take(_decay(any_on, half_life, starts_idx))
        columns[f"red_decay_{label}"] = take(_decay(red_on, half_life, starts_idx), masked=True)
    columns["any_starts_24h"] = take(_window(any_on, 1440))
    columns["red_starts_24h"] = take(_window(red_on, 1440), masked=True)
    columns["raion_share"] = take(timeline.raion_share[:, :end])

    any_sampled = any_state[:, minutes].astype(np.float32)
    red_sampled = red_state[:, minutes].astype(np.float32)
    columns["nb_any_now"] = neighbours @ any_sampled
    nb_red = neighbours @ red_sampled
    nb_red[:, ~new_regime[minutes]] = np.nan
    columns["nb_red_now"] = nb_red
    since_on = _since(any_on, resets)[:, minutes]
    columns["nb_since_start"] = np.stack(
        [since_on[neighbours[i] > 0].min(axis=0) for i in range(len(OBLASTS))]
    )
    for width in (30, 60, 120):
        columns[f"nb_starts_{width}m"] = neighbours @ _window(any_on, width)[:, minutes]

    national = {
        "nat_any_now": any_sampled.sum(axis=0),
        "nat_red_now": np.where(new_regime[minutes], red_sampled.sum(axis=0), np.nan),
        "nat_starts_60m": _window(any_on, 60)[:, minutes].sum(axis=0),
        "since_wave": _since(in_wave[None, :], resets)[0, minutes],
    }
    for name, values in national.items():
        columns[name] = np.broadcast_to(values, (len(OBLASTS), minutes.size))
    for j, oblast in enumerate(OBLASTS):
        columns[f"any_{j:02d}"] = np.broadcast_to(any_sampled[j], (len(OBLASTS), minutes.size))
        red_j = np.where(new_regime[minutes], red_sampled[j], np.nan)
        columns[f"red_{j:02d}"] = np.broadcast_to(red_j, (len(OBLASTS), minutes.size))

    kyiv = timeline.time(minutes).tz_convert("Europe/Kyiv")
    frame = pd.DataFrame({name: values.reshape(-1) for name, values in columns.items()})
    frame.insert(0, "minute", np.tile(minutes, len(OBLASTS)))
    frame.insert(1, "oblast", np.repeat(np.arange(len(OBLASTS)), minutes.size))
    frame.insert(2, "regime", np.tile(timeline.regime[minutes], len(OBLASTS)).astype(np.float32))
    frame.insert(3, "hour", np.tile(kyiv.hour.to_numpy(), len(OBLASTS)).astype(np.float32))
    frame.insert(4, "dow", np.tile(kyiv.dayofweek.to_numpy(), len(OBLASTS)).astype(np.float32))
    return frame


def outcomes(timeline: Timeline, kind: Kind, frame: pd.DataFrame, censor_at: int) -> pd.DataFrame:
    state = timeline.any if kind.level == "any" else timeline.red
    oblast = frame["oblast"].to_numpy()
    minute = frame["minute"].to_numpy()
    if kind.phase == "start":
        at_risk = ~state[oblast, minute]
        target = state
    else:
        at_risk = state[oblast, minute]
        quiet = ~state
        target = quiet & np.concatenate([quiet[:, QUIET_GAP - 1:], np.zeros((quiet.shape[0], QUIET_GAP - 1), dtype=bool)], axis=1)
    if kind.level == "red":
        at_risk &= timeline.regime[minute] == NEW
    unknown_next = _next_index(~timeline.known)
    event_next = _next_index(target)
    after = np.minimum(minute + 1, timeline.size - 1)
    event = event_next[oblast, after]
    censor = np.minimum(unknown_next[after], censor_at)
    observed = event < censor
    elapsed = (event - minute).astype(np.float64)
    event_bin = np.where(observed, np.ceil(elapsed / kind.bin_minutes), np.inf)
    observed_bins = np.floor((censor - minute) / kind.bin_minutes)
    return pd.DataFrame(
        {"at_risk": at_risk, "event_bin": event_bin, "observed_bins": observed_bins},
        index=frame.index,
    )


def expand(features: pd.DataFrame, result: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    last = np.minimum(np.minimum(result["event_bin"], result["observed_bins"]), BINS).astype(int).to_numpy()
    keep = last > 0
    counts = last[keep]
    base = features[keep]
    rows = base.loc[base.index.repeat(counts)].reset_index(drop=True)
    k = np.concatenate([np.arange(1, n + 1) for n in counts]) if counts.size else np.array([], dtype=int)
    rows.insert(0, "k", k.astype(np.float32))
    event_bin = np.repeat(result["event_bin"].to_numpy()[keep], counts)
    return rows, (k == event_bin).astype(np.int8)


def horizon_outcome(result: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    happened = (result["event_bin"] <= horizon).to_numpy()
    evaluable = happened | (result["observed_bins"] >= horizon).to_numpy()
    return happened, evaluable
