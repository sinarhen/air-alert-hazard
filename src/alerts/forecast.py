from pathlib import Path

import numpy as np
import pandas as pd

from alerts.adjacency import OBLASTS
from alerts.evaluate import HORIZONS, _artifact, kind_of
from alerts.features import feature_frame
from alerts.models import Candidates, cumulative, curve_rows
from alerts.timeline import Timeline


def forecast(timeline: Timeline, minute: int, artifacts: Path, eval_name: str = "e2") -> pd.DataFrame:
    features = feature_frame(timeline, np.array([minute]))
    rows = curve_rows(features)
    out = pd.DataFrame({"oblast": OBLASTS})
    for level, state in (("any", timeline.any), ("red", timeline.red)):
        active = state[:, minute]
        out[f"{level}_active"] = active
        out[f"{level}_for_min"] = features[f"{level}_since_change"].to_numpy()
        for phase in ("start", "end"):
            kind = kind_of(f"{level}-{phase}")
            curve = cumulative(Candidates.load(_artifact(artifacts, eval_name, kind, "main")).hazard(rows))
            for h in HORIZONS[phase]:
                column = f"{level}_{phase}_{h * kind.bin_minutes}m"
                out[column] = np.where(active == (phase == "end"), curve[:, h - 1], np.nan)
    return out


def describe(table: pd.DataFrame) -> str:
    lines = []
    for _, row in table.iterrows():
        parts = []
        for level in ("any", "red"):
            if row[f"{level}_active"]:
                probs = ", ".join(f"≤{c.rsplit('_', 1)[1]} {row[c]:.0%}" for c in table.columns if c.startswith(f"{level}_end_"))
                parts.append(f"{level}: ACTIVE {row[f'{level}_for_min']:.0f}m, ends {probs}")
            else:
                probs = ", ".join(f"≤{int(c.rsplit('_', 1)[1][:-1]) // 60}h {row[c]:.0%}" for c in table.columns if c.startswith(f"{level}_start_"))
                parts.append(f"{level}: quiet, starts {probs}")
        lines.append(f"{row['oblast']:<28} " + " | ".join(parts))
    return "\n".join(lines)
