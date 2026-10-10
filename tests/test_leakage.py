from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alerts.features import KINDS, feature_frame, outcomes, warm_origins
from alerts.sources import load_alerts_in_ua, load_official
from alerts.timeline import NEW, OLD, build_timeline, segments_for, truncate

DATA = Path(__file__).resolve().parents[1] / "data"
CUTOFFS = 12


@pytest.fixture(scope="module")
def corpus():
    if not (DATA / "official_data_uk.csv").exists() or not (DATA / "raw").exists():
        pytest.skip("run `alerts fetch` first")
    events = pd.concat([load_official(DATA / "official_data_uk.csv"), load_alerts_in_ua(DATA / "raw")[0]], ignore_index=True)
    coverage = load_alerts_in_ua(DATA / "raw")[1]
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
