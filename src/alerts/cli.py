import json
import os
from pathlib import Path

import pandas as pd
import typer

from alerts import escalations, evaluate as ev
from alerts.forecast import describe, forecast as make_forecast
from alerts.live import run as run_live
from alerts.site import build, raion_finding
from alerts.sources import download_official, fetch_alerts_in_ua
from alerts.timeline import load, load_events

app = typer.Typer(no_args_is_help=True)
DATA = Path("data")
ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")


@app.command()
def fetch(official: bool = True, recent: bool = True, channel: bool = True) -> None:
    DATA.mkdir(exist_ok=True)
    if official:
        typer.echo(f"official -> {download_official(DATA)}")
    if recent:
        token = os.environ.get("ALERTS_IN_UA_TOKEN")
        if not token:
            raise typer.BadParameter("set ALERTS_IN_UA_TOKEN")
        typer.echo(f"alerts.in.ua -> {fetch_alerts_in_ua(DATA / 'raw', token)}")
    if channel:
        missing = [name for name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION") if not os.environ.get(name)]
        if missing:
            raise typer.BadParameter(f"set {', '.join(missing)}")
        paths = escalations.pull(DATA / "raw", os.environ["TELEGRAM_SESSION"], int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"])
        typer.echo("telegram -> " + ", ".join(str(p) for p in paths))


@app.command()
def check_escalations(start: str = "2026-09-10", end: str = "2026-10-02", name: str = "escalations") -> None:
    events, _ = load_events(DATA)
    result = escalations.check(events, DATA / "raw", (start, end))
    (REPORTS / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))


@app.command()
def train(eval_name: str = typer.Option("e2", "--eval"), kinds: str = typer.Option("", help="comma-separated, e.g. red-start,red-end")) -> None:
    _, coverage, timeline = load(DATA)
    workspace = ev.Workspace(timeline)
    ev.train(workspace, eval_name, ARTIFACTS, REPORTS, [k for k in kinds.split(",") if k] or None)
    ev.evaluate(workspace, eval_name, "val", ARTIFACTS, REPORTS)
    typer.echo(f"{eval_name}: trained, validation metrics in {REPORTS / eval_name}")


@app.command()
def evaluate(eval_name: str = typer.Option("e2", "--eval"), split: str = "test") -> None:
    _, coverage, timeline = load(DATA)
    ev.evaluate(ev.Workspace(timeline), eval_name, split, ARTIFACTS, REPORTS)
    typer.echo(f"{eval_name} {split}: metrics in {REPORTS / eval_name}")


@app.command()
def forecast(at: str = typer.Option(..., help="UTC or offset-aware time, e.g. 2026-10-09T18:00Z")) -> None:
    moment = pd.Timestamp(at)
    moment = (moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")).floor("min")
    _, _, timeline = load(DATA, cutoff=moment)
    minute = timeline.minute(moment)
    if not timeline.known[minute]:
        raise typer.BadParameter(f"no alert data covering {moment}")
    typer.echo(f"forecast at {moment} (only alerts started by then are used)")
    typer.echo(describe(make_forecast(timeline, minute, ARTIFACTS)))


@app.command()
def live(interval: int = 60, target: Path = Path("site")) -> None:
    token = os.environ.get("ALERTS_IN_UA_TOKEN")
    if not token:
        raise typer.BadParameter("set ALERTS_IN_UA_TOKEN")
    run_live(DATA, target, ARTIFACTS, token, interval)


@app.command()
def report(target: Path = Path("site")) -> None:
    events, _, timeline = load(DATA)
    path = build(timeline, ARTIFACTS, {"raions": raion_finding(events)}, target)
    typer.echo(f"site -> {path}")


if __name__ == "__main__":
    app()
