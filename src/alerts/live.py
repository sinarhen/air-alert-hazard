import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from alerts.forecast import forecast
from alerts.sources import COLUMNS, DECLARATION_LEVELS, EXCLUDED_OBLASTS, Coverage
from alerts.timeline import build_timeline, load_events, segments_for

ACTIVE_URL = "https://api.alerts.in.ua/v1/alerts/active.json"
POLL_LOG = "active_poll.jsonl"


def snapshot(raw_dir: Path, token: str) -> dict:
    response = httpx.get(ACTIVE_URL, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    response.raise_for_status()
    record = {"polled_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "body": response.json()}
    with (raw_dir / POLL_LOG).open("a") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def poll_events(path: Path, after: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    seen: dict[int, dict] = {}
    polled = after
    for text in path.read_text().splitlines():
        record = json.loads(text)
        polled = pd.Timestamp(record["polled_at"])
        if polled < after:
            continue
        present = set()
        for alert in record["body"]["alerts"]:
            if alert["alert_type"] != "air_raid" or alert["location_type"] not in DECLARATION_LEVELS or alert["location_oblast"] in EXCLUDED_OBLASTS:
                continue
            present.add(alert["id"])
            entry = seen.setdefault(alert["id"], {"level": None, "red_from": pd.NaT, "finished_at": pd.NaT})
            if alert["alert_level"] == "red" and pd.isna(entry["red_from"]):
                entry["red_from"] = pd.Timestamp(alert["started_at"] if entry["level"] is None else alert["updated_at"])
            entry["alert"] = alert
            entry["level"] = alert["alert_level"]
            entry["finished_at"] = pd.NaT
        for key, entry in seen.items():
            if key not in present and pd.isna(entry["finished_at"]):
                entry["finished_at"] = polled
    rows = [
        {
            "source": "alerts_in_ua",
            "oblast": e["alert"]["location_oblast"],
            "raion": e["alert"]["location_title"] if e["alert"]["location_type"] == "raion" else None,
            "location_type": e["alert"]["location_type"],
            "level": e["level"],
            "started_at": pd.Timestamp(e["alert"]["started_at"]),
            "finished_at": e["finished_at"],
            "red_from": e["red_from"],
        }
        for e in seen.values()
    ]
    return pd.DataFrame(rows, columns=[*COLUMNS, "red_from"]), polled


def merge(history: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([history, live], ignore_index=True)
    combined["raion_key"] = combined["raion"].fillna("")
    key = ["source", "oblast", "location_type", "raion_key", "started_at"]
    live_wins = combined.duplicated(key, keep="last")
    history_rows = combined[live_wins]
    combined = combined.drop_duplicates(key, keep="last").copy()
    earlier = history_rows.set_index(key)["red_from"]
    joined = combined.set_index(key)
    earlier = earlier.reindex(joined.index)
    joined["red_from"] = earlier.where(earlier.notna(), joined["red_from"])
    return joined.reset_index().drop(columns="raion_key")[[*COLUMNS, "red_from"]]


def run(data_dir: Path, site_dir: Path, artifacts: Path, token: str, interval: int = 60) -> None:
    history, coverage = load_events(data_dir)
    while True:
        started = time.monotonic()
        try:
            snapshot(data_dir / "raw", token)
            live, polled = poll_events(data_dir / "raw" / POLL_LOG, coverage.end)
            now = polled.floor("min")
            events = merge(history, live)
            timeline = build_timeline(events, segments_for(Coverage(coverage.start, now)), now)
            minute = int(np.flatnonzero(timeline.known)[-1])
            table = forecast(timeline, minute, artifacts)
            payload = {"at": timeline.time(np.array([minute]))[0].isoformat(), "polled_at": polled.isoformat(), "table": json.loads(table.to_json(orient="records", force_ascii=False))}
            target = site_dir / "live.json"
            target.with_suffix(".tmp").write_text(json.dumps(payload, ensure_ascii=False).replace("NaN", "null"))
            target.with_suffix(".tmp").replace(target)
            print(f"{polled:%H:%M:%S} live forecast at {payload['at']}", flush=True)
        except (httpx.HTTPError, ValueError) as error:
            print(f"poll failed: {error}", flush=True)
        time.sleep(max(0, interval - (time.monotonic() - started)))
