from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"
    print(f"[run] {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        code = proc.wait()
        handle.write(f"\nRETURN_CODE: {code}\n")
    if code != 0:
        raise SystemExit(f"{name} failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust target-trial causal inference pipeline.")
    parser.add_argument("--config-dir", default="configs")
    args = parser.parse_args()
    py = sys.executable
    config = ["--config-dir", args.config_dir]
    log_dir = ROOT / "outputs" / "target_trial" / "logs"
    steps = [
        ("build_target_trial", [py, "scripts/target_trial/build_target_trial.py", *config]),
        ("doubly_robust", [py, "scripts/causal_estimands/doubly_robust.py", *config]),
        ("final_group_inference", [py, "scripts/heterogeneity/final_group_inference.py", *config]),
    ]
    for name, command in steps:
        run_step(name, command, log_dir)
    print("[done] robust target-trial causal pipeline complete", flush=True)


if __name__ == "__main__":
    main()
