import os
from pathlib import Path

import pandas as pd
import typer

from alerts import evaluate as ev
from alerts.forecast import describe, forecast as make_forecast
from alerts.site import build, raion_finding
from alerts.sources import download_official, fetch_alerts_in_ua
from alerts.timeline import load

app = typer.Typer(no_args_is_help=True)
DATA = Path("data")
ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")


@app.command()
def fetch(official: bool = True, recent: bool = True) -> None:
    DATA.mkdir(exist_ok=True)
    if official:
        typer.echo(f"official -> {download_official(DATA)}")
    if recent:
        token = os.environ.get("ALERTS_IN_UA_TOKEN")
        if not token:
            raise typer.BadParameter("set ALERTS_IN_UA_TOKEN")
        typer.echo(f"alerts.in.ua -> {fetch_alerts_in_ua(DATA / 'raw', token)}")


@app.command()
def train(eval_name: str = typer.Option("e2", "--eval")) -> None:
    _, coverage, timeline = load(DATA)
    workspace = ev.Workspace(timeline)
    ev.train(workspace, eval_name, ARTIFACTS, REPORTS)
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
def report(target: Path = Path("site")) -> None:
    events, _, timeline = load(DATA)
    path = build(timeline, ARTIFACTS, REPORTS, {"raions": raion_finding(events)}, target)
    typer.echo(f"site -> {path}")


if __name__ == "__main__":
    app()
