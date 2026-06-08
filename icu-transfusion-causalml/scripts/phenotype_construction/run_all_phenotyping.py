from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from lib.utils import repo_root, write_latest_run


def parse_args():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--window", type=int, choices=[24, 48, 72], default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--caliper", type=float, default=0.3)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--replace", dest="replace", action="store_true", default=True)
    parser.add_argument("--no-replace", dest="replace", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_stays", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_step(script: Path, args_list, env=None):
    cmd = [sys.executable, script.as_posix()] + args_list
    subprocess.check_call(cmd, env=env)


def main():
    args = parse_args()
    root = repo_root()
    scripts_dir = Path(__file__).resolve().parent

    windows = [args.window] if args.window else [24, 48, 72]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    write_latest_run(root, run_id)

    run_step(scripts_dir / "build_outcomes_cohort.py", ["--dry-run"] if args.dry_run else [], env=env)

    for window in windows:
        base_args = ["--window", str(window)]
        baseline_args = base_args[:]
        match_args = base_args[:] + ["--seed", str(args.seed), "--caliper", str(args.caliper), "--ratio", str(args.ratio)]
        embed_args = base_args[:] + ["--seed", str(args.seed)]
        report_args = base_args[:]

        if args.limit_stays:
            baseline_args += ["--limit_stays", str(args.limit_stays)]
            match_args += ["--limit_stays", str(args.limit_stays)]
            embed_args += ["--limit_stays", str(args.limit_stays)]
        if args.dry_run:
            baseline_args += ["--dry-run"]
            match_args += ["--dry-run"]
            embed_args += ["--dry-run"]
            report_args += ["--dry-run"]
        if args.replace:
            match_args += ["--replace"]
        else:
            match_args += ["--no-replace"]

        run_step(scripts_dir / "build_baseline_features.py", baseline_args, env=env)
        run_step(scripts_dir / "match_controls.py", match_args, env=env)
        run_step(scripts_dir / "embed_minirocket_temporal.py", embed_args, env=env)
        run_step(scripts_dir / "embed_ts2vec_temporal.py", embed_args, env=env)
        run_step(scripts_dir / "generate_phenotype_reports.py", report_args, env=env)


if __name__ == "__main__":
    main()


