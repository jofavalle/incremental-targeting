"""The memory probe end to end on the synthetic file, so it does not rot between runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import N_ROWS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "peak_memory.py"


def test_probe_reports_three_stages_and_cleans_up(tmp_path, synthetic_csv_gz):
    out = tmp_path / "peak_memory.json"
    work_parent = tmp_path / "work"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--src", str(synthetic_csv_gz), "--out", str(out)]
        + ["--work-parent", str(work_parent), "--threads", "1", "--rounds", "2"],
        check=True,
        capture_output=True,
    )
    report = json.loads(out.read_text())
    assert set(report["stages"]) == {"ingest", "split", "fit"}  # keys are sorted on write
    assert all(stage["peak_rss_mib"] > 0 for stage in report["stages"].values())
    assert report["stages"]["ingest"]["rows"] == N_ROWS
    assert report["stages"]["fit"]["train_rows"] == report["stages"]["split"]["rows"]["train"]
    assert str(tmp_path) not in out.read_text()
    assert list(work_parent.iterdir()) == []
