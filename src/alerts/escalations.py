import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

CHANNEL = "air_alert_map"
OFFICIAL = {
    "kyivoda": "Київська область",
    "VinnytsiaODA": "Вінницька область",
    "khersonskaODA": "Херсонська область",
    "ODA_RV": "Рівненська область",
}
PULL_SINCE = datetime(2026, 9, 6, tzinfo=UTC)
START_TOLERANCE = (pd.Timedelta(minutes=-1), pd.Timedelta(minutes=5))
LINE = re.compile(r"^(🟡|🔴|🟢\S*)\s+(?:\[(\d+)\]\s+)?(.+?)(?:\s+\(([^)]*)\))?$")
OFFICIAL_LINE = re.compile(r"^(🟡|🔴|🟢)\s*(.+?)\s+[—-]\s+(?:повітряна тривога, (жовтий|червоний) рівень|відбій повітряної тривоги)")


def place(name: str) -> str:
    return name.replace("’", "'").split(" та ")[0].strip()


async def _pull(raw_dir: Path, channels: list[str], session: str, api_id: int, api_hash: str) -> list[Path]:
    from telethon import TelegramClient

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M")
    written = []
    async with TelegramClient(session, api_id, api_hash) as client:
        for channel in channels:
            path = raw_dir / f"tg_{channel}_{stamp}.jsonl"
            with path.open("w") as out:
                async for message in client.iter_messages(channel):
                    if message.date < PULL_SINCE:
                        break
                    edited = message.edit_date.isoformat() if message.edit_date else None
                    out.write(json.dumps({"id": message.id, "date": message.date.isoformat(), "edit": edited, "text": message.message or ""}, ensure_ascii=False) + "\n")
            written.append(path)
    return written


def pull(raw_dir: Path, session: str, api_id: int, api_hash: str) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_pull(raw_dir, [CHANNEL, *OFFICIAL], session, api_id, api_hash))


def messages(raw_dir: Path, channel: str) -> pd.DataFrame:
    paths = sorted(raw_dir.glob(f"tg_{channel}_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no {channel} pulls under {raw_dir}; run `alerts fetch --channel`")
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line]
    frame = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values("date").reset_index(drop=True)


def lines(raw_dir: Path) -> pd.DataFrame:
    out = []
    for message in messages(raw_dir, CHANNEL).itertuples():
        for text in message.text.splitlines():
            match = LINE.match(text.strip())
            if match:
                out.append({"posted_at": message.date, "message": message.id, "mark": match[1], "started": match[2] is not None, "place": place(match[3])})
    return pd.DataFrame(out)


def official_lines(raw_dir: Path, channel: str) -> pd.DataFrame:
    out = []
    for message in messages(raw_dir, channel).itertuples():
        match = OFFICIAL_LINE.match(message.text)
        if match:
            level = {"жовтий": "yellow", "червоний": "red"}.get(match[3], "clear")
            out.append({"posted_at": message.date, "place": place(match[2]), "level": level})
    return pd.DataFrame(out, columns=["posted_at", "place", "level"])


def official_escalations(frame: pd.DataFrame) -> pd.DataFrame:
    state, out = {}, []
    for row in frame.itertuples():
        if row.level == "red" and state.get(row.place) == "yellow":
            out.append({"posted_at": row.posted_at, "place": row.place})
        state[row.place] = row.level
    return pd.DataFrame(out, columns=["posted_at", "place"])


def red_onsets(events: pd.DataFrame, channel: pd.DataFrame) -> pd.Series:
    red_lines = channel[channel["mark"] == "🔴"]
    by_place = {name: group for name, group in red_lines.groupby("place")}
    raions = events.dropna(subset=["raion"]).groupby("oblast")["raion"].unique()
    onsets = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
    for index, row in events[(events["source"] == "alerts_in_ua") & (events["level"] == "red")].iterrows():
        if row["location_type"] == "raion":
            names = [place(row["raion"])]
        elif row["oblast"].startswith("м. "):
            names = [row["oblast"]]
        else:
            names = [place(name) for name in raions.get(row["oblast"], [])]
        candidates = pd.concat([by_place[name] for name in names if name in by_place]) if any(name in by_place for name in names) else red_lines.iloc[:0]
        end = row["finished_at"] if pd.notna(row["finished_at"]) else pd.Timestamp.max.tz_localize(UTC)
        opened = candidates[candidates["started"]]
        lo, hi = row["started_at"] + START_TOLERANCE[0], row["started_at"] + START_TOLERANCE[1]
        if opened["posted_at"].between(lo, hi).any():
            onsets[index] = row["started_at"]
            continue
        raised = candidates[~candidates["started"] & candidates["posted_at"].between(row["started_at"], end)]
        if len(raised):
            onsets[index] = raised["posted_at"].min()
    return onsets


def poller_escalations(path: Path) -> pd.DataFrame:
    seen, out = {}, []
    for text in path.read_text().splitlines():
        snapshot = json.loads(text)
        for alert in snapshot["body"]["alerts"]:
            if alert["alert_type"] != "air_raid" or alert["location_type"] not in ("raion", "oblast"):
                continue
            previous = seen.get(alert["id"])
            if previous == "yellow" and alert["alert_level"] == "red":
                out.append({"id": alert["id"], "title": alert["location_title"], "switched_at": pd.Timestamp(alert["updated_at"]), "started_at": pd.Timestamp(alert["started_at"])})
            seen[alert["id"]] = alert["alert_level"]
    return pd.DataFrame(out, columns=["id", "title", "switched_at", "started_at"])


def check(events: pd.DataFrame, raw_dir: Path, window: tuple[str, str]) -> dict:
    channel = lines(raw_dir)
    onsets = red_onsets(events, channel)
    lo, hi = (pd.Timestamp(day, tz="UTC") for day in window)
    red = events[(events["source"] == "alerts_in_ua") & (events["level"] == "red") & events["started_at"].between(lo, hi)]
    matched = onsets[red.index].notna()
    coverage = {
        "window": list(window),
        "red_alerts": int(len(red)),
        "matched": round(float(matched.mean()), 4),
        "escalated": int((onsets[red.index] > red["started_at"]).sum()),
        "by_oblast": {oblast: round(float(group.mean()), 3) for oblast, group in matched.groupby(red["oblast"])},
    }

    poll = poller_escalations(raw_dir / "active_poll.jsonl")
    history = events[events["source"] == "alerts_in_ua"].assign(title=lambda f: f["raion"].fillna(f["oblast"]))
    history = history.reset_index().rename(columns={"index": "row"})[["row", "title", "started_at", "level"]]
    joined = poll.merge(history, on=["title", "started_at"], how="inner")
    offsets = [(onsets[row.row] - row.switched_at).total_seconds() for row in joined.itertuples() if pd.notna(onsets[row.row])]
    poller = {
        "escalations": int(len(poll)),
        "in_history": int(len(joined)),
        "final_level_red": int((joined["level"] == "red").sum()),
        "matched": int(len(offsets)),
        "early": int(sum(o < 0 for o in offsets)),
        "offset_seconds": {"min": min(offsets, default=None), "median": float(np.median(offsets)) if offsets else None, "max": max(offsets, default=None)},
    }

    official = {}
    escalation_lines = channel[(channel["mark"] == "🔴") & ~channel["started"]]
    for handle, oblast in OFFICIAL.items():
        theirs = official_escalations(official_lines(raw_dir, handle))
        theirs = theirs[theirs["posted_at"].between(lo, hi)]
        agree = [
            ((escalation_lines["place"] == row.place) & ((escalation_lines["posted_at"] - row.posted_at).abs() <= pd.Timedelta(minutes=3))).any()
            for row in theirs.itertuples()
        ]
        official[handle] = {"oblast": oblast, "escalations": int(len(theirs)), "agree_within_3min": int(sum(agree))}
    return {"coverage": coverage, "poller": poller, "official": official}
