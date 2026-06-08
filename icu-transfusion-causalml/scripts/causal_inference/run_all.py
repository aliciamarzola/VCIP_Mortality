from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reuse-based causal inference workflow.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--phenotype-root", default="outputs/phenotype_construction")
    parser.add_argument("--run-id", default="run_cal03_replace_full_w48")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--embedding", choices=["minirocket", "ts2vec"], default="minirocket")
    parser.add_argument("--k-list", default="2,3,5,6")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--keep-post-t0-features", action="store_true", default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: list[str], log_dir: Path, continue_on_error: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {name}", flush=True)
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{started}_{name}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"COMMAND: {' '.join(command)}\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        return_code = process.wait()
        handle.write(f"\nRETURN_CODE: {return_code}\n")

    if return_code != 0:
        print(f"[error] {name} failed. Log: {log_path}", flush=True)
        if not continue_on_error:
            raise SystemExit(return_code)


def main() -> None:
    args = parse_args()
    py = sys.executable
    config_arg = ["--config-dir", args.config_dir]
    log_dir = ROOT / "outputs" / "causal_inference" / "logs"
    phenotype_root = args.phenotype_root

    import_cmd = [
        py,
        "scripts/causal_inference/import_phenotype_outputs.py",
        *config_arg,
        "--phenotype-root",
        phenotype_root,
        "--run-id",
        args.run_id,
        "--window",
        str(args.window),
        "--embedding",
        args.embedding,
    ]
    if args.keep_post_t0_features:
        import_cmd.append("--keep-post-t0-features")

    steps = [
        ("import_phenotype_outputs", import_cmd),
        ("causal_forest", [py, "scripts/causal_inference/causal_forest.py", *config_arg]),
        ("counterfactual_model", [py, "scripts/causal_inference/counterfactual_model.py", *config_arg]),
        ("counterfactual_evaluation", [py, "scripts/causal_inference/counterfactual_evaluation.py", *config_arg]),
        ("evaluation", [py, "scripts/causal_inference/evaluation.py", *config_arg]),
        (
            "scan_group_inference",
            [
                py,
                "scripts/causal_inference/scan_group_inference.py",
                *config_arg,
                "--phenotype-root",
                phenotype_root,
                "--run-id",
                args.run_id,
                "--window",
                str(args.window),
                "--embedding",
                args.embedding,
                "--k-list",
                args.k_list,
                "--bootstrap",
                str(args.bootstrap),
            ],
        ),
        ("generate_report", [py, "scripts/causal_inference/generate_report.py", *config_arg]),
    ]

    for name, command in steps:
        run_step(name, command, log_dir, args.continue_on_error)

    print("[done] reuse-based causal workflow complete", flush=True)


if __name__ == "__main__":
    main()
