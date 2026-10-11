from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alerts.escalations import lines, place, poller_escalations
from alerts.features import KINDS, feature_frame, outcomes, warm_origins
from alerts.timeline import NEW, OLD, build_timeline, load_events, segments_for, truncate

DATA = Path(__file__).resolve().parents[1] / "data"
CUTOFFS = 12


@pytest.fixture(scope="module")
def corpus():
    if not (DATA / "official_data_uk.csv").exists() or not (DATA / "raw").exists():
        pytest.skip("run `alerts fetch` first")
    events, coverage = load_events(DATA)
    segments = segments_for(coverage)
    return events, segments, build_timeline(events, segments, coverage.end)


def _cutoffs(timeline, regime: int) -> list[int]:
    origins = warm_origins(timeline, 0, timeline.size, 15)
    origins = origins[timeline.regime[origins] == regime]
    return list(np.random.default_rng(regime).choice(origins, CUTOFFS // 2, replace=False))


@pytest.mark.parametrize("regime", [OLD, NEW])
def test_features_see_only_the_past(corpus, regime):
    events, segments, full = corpus
    for minute in _cutoffs(full, regime):
        cutoff = full.time(np.array([minute]))[0]
        partial = build_timeline(truncate(events, cutoff), segments, cutoff + pd.Timedelta(minutes=1))
        expected = feature_frame(full, np.array([minute]))
        actual = feature_frame(partial, np.array([minute]))
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-5)


def test_labels_stop_at_split_end(corpus):
    _, _, full = corpus
    origins = warm_origins(full, 0, full.size, 15)
    frame = feature_frame(full, origins[::50])
    stop = int(np.median(origins))
    for kind in KINDS:
        result = outcomes(full, kind, frame, stop)
        horizon_end = frame["minute"].to_numpy() + result["observed_bins"].to_numpy() * kind.bin_minutes
        assert (horizon_end[frame["minute"].to_numpy() < stop] <= stop).all()
        observed = np.isfinite(result["event_bin"].to_numpy())
        assert (result["event_bin"].to_numpy()[observed] <= result["observed_bins"].to_numpy()[observed] + 1).all()


def test_red_switch_is_a_public_post(corpus):
    events, _, _ = corpus
    posts = lines(DATA / "raw")
    raised = events[events["red_from"] > events["started_at"]]
    published = set(zip(posts.loc[posts["mark"] == "🔴", "place"], posts.loc[posts["mark"] == "🔴", "posted_at"]))
    assert (events["red_from"].dropna() >= events.loc[events["red_from"].notna(), "started_at"]).all()
    assert all(time in {t for name, t in published if name == place(raion)} for raion, time in zip(raised["raion"].fillna(raised["oblast"]), raised["red_from"]) if raion.endswith("район") or raion.startswith("м. "))


def test_red_switch_matches_live_poller(corpus):
    events, _, _ = corpus
    truth = poller_escalations(DATA / "raw" / "active_poll.jsonl")
    history = events[events["source"] == "alerts_in_ua"].assign(title=lambda f: f["raion"].fillna(f["oblast"]))
    joined = truth.merge(history, on=["title", "started_at"])
    joined = joined[joined["level"] == "red"]
    if joined.empty:
        pytest.skip("no poller escalations inside the fetched history yet")
    offset = (joined["red_from"] - joined["switched_at"]).dt.total_seconds()
    assert offset.between(-600, 600).mean() >= 0.95
