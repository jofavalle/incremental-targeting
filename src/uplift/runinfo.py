"""What every run records next to its numbers: machine, versions and threads.

A published figure is only as good as the record of the run that produced it.
This module collects that record once, so the ingest, the split and the memory
probe write the same fields. It never records a path or a host name: a JSON
under reports/ is versioned, and an absolute path in it would publish the home
directory of whoever ran it.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import pyarrow as pa


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_metadata(packages: tuple[str, ...] = ("numpy", "pyarrow")) -> dict:
    """Machine, interpreter, package versions and the Arrow thread pool size."""
    page = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_gib": round(page * pages / 2**30, 2),
        "python": sys.version.split()[0],
        "packages": {name: _version(name) for name in packages},
        "arrow_cpu_threads": pa.cpu_count(),
    }


def write_json(path: Path, payload: dict) -> None:
    """Write `payload` as stable, diffable JSON (sorted keys, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
