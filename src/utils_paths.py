from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    fig_dir: Path
    met_dir: Path

def make_run_paths(project_root: Path, tag: str = "step4") -> RunPaths:
    run_id = f"{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = project_root / "outputs" / "runs" / run_id
    fig_dir = run_dir / "figures"
    met_dir = run_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=run_id, run_dir=run_dir, fig_dir=fig_dir, met_dir=met_dir)