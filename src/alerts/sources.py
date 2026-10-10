import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

OFFICIAL_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv"
ALERTS_IN_UA_URL = "https://api.alerts.in.ua/v1/regions/{uid}/alerts/month_ago.json"
REGION_UIDS = (3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31)
HISTORY_WINDOW = timedelta(days=30)
EXCLUDED_OBLASTS = ("Луганська область", "Автономна Республіка Крим")
DECLARATION_LEVELS = ("oblast", "raion")
COLUMNS = ["source", "oblast", "raion", "location_type", "level", "started_at", "finished_at"]


@dataclass(frozen=True)
class Coverage:
    start: pd.Timestamp
    end: pd.Timestamp


def download_official(data_dir: Path) -> Path:
    path = data_dir / "official_data_uk.csv"
    response = httpx.get(OFFICIAL_URL, timeout=120, follow_redirects=True)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def fetch_alerts_in_ua(raw_dir: Path, token: str, pause_seconds: float = 31) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M")
    target = raw_dir / f"alerts_in_ua_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=60, headers=headers) as client:
        for index, uid in enumerate(REGION_UIDS):
            if index:
                time.sleep(pause_seconds)
            response = client.get(ALERTS_IN_UA_URL.format(uid=uid))
            response.raise_for_status()
            (target / f"r{uid}.json").write_text(response.text)
    return target


def load_official(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).drop_duplicates()
    frame = frame[frame["level"].isin(DECLARATION_LEVELS) & ~frame["oblast"].isin(EXCLUDED_OBLASTS)]
    return pd.DataFrame(
        {
            "source": "official",
            "oblast": frame["oblast"],
            "raion": frame["raion"],
            "location_type": frame["level"],
            "level": None,
            "started_at": pd.to_datetime(frame["started_at"], utc=True),
            "finished_at": pd.to_datetime(frame["finished_at"], utc=True),
        }
    )[COLUMNS].reset_index(drop=True)


def _fetch_time(payload: dict) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(payload["meta"]["last_updated_at"], "%Y/%m/%d %H:%M:%S %z")).tz_convert(UTC)


def load_alerts_in_ua(raw_dir: Path) -> tuple[pd.DataFrame, Coverage]:
    records = []
    region_fetches: dict[str, list[pd.Timestamp]] = {}
    for path in sorted(raw_dir.glob("alerts_in_ua_*/r*.json")):
        payload = json.loads(path.read_text())
        fetched = _fetch_time(payload)
        region_fetches.setdefault(path.name, []).append(fetched)
        for alert in payload["alerts"]:
            records.append({**alert, "fetched_at": fetched})
    if not records:
        raise FileNotFoundError(f"no alerts.in.ua fetches under {raw_dir}")
    raw = pd.DataFrame(records).sort_values("fetched_at").drop_duplicates("id", keep="last")
    raw = raw[(raw["alert_type"] == "air_raid") & raw["location_type"].isin(DECLARATION_LEVELS)]
    raw = raw[~raw["location_oblast"].isin(EXCLUDED_OBLASTS)]
    frame = pd.DataFrame(
        {
            "source": "alerts_in_ua",
            "oblast": raw["location_oblast"],
            "raion": raw["location_title"].where(raw["location_type"] == "raion"),
            "location_type": raw["location_type"],
            "level": raw["alert_level"],
            "started_at": pd.to_datetime(raw["started_at"], utc=True),
            "finished_at": pd.to_datetime(raw["finished_at"], utc=True),
        }
    )[COLUMNS].reset_index(drop=True)
    starts = [min(times) - HISTORY_WINDOW for times in region_fetches.values()]
    ends = [max(times) for times in region_fetches.values()]
    return frame, Coverage(start=max(starts).ceil("min"), end=min(ends).floor("min"))
