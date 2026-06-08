#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src").exists() and (parent / "outputs").exists():
            return parent
    return start


def main() -> None:
    root = find_repo_root(Path(__file__).resolve())
    scripts_dir = root / "scripts"
    steps = [
        scripts_dir / "cluster_tabular.py",
        scripts_dir / "cluster_ts2vec.py",
        scripts_dir / "analyze_clusters.py",
    ]

    for step in steps:
        if not step.exists():
            raise FileNotFoundError(f"Missing script: {step}")
        print(f"[run] {step}")
        subprocess.check_call([sys.executable, str(step)])


if __name__ == "__main__":
    main()
