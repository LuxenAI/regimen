#!/usr/bin/env python3
"""Run the repo's full model evaluation suite and collect one report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"runs": {}, "notes": []}
    report["runs"]["remaining_subroutines"] = run_remaining(args)
    report["runs"]["mcp_subroutines"] = run_mcp_subroutines(args)
    report["runs"]["verifier"] = run_verifier(args)

    summary_path = args.output_dir / "all_model_evals.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_dir / "all_model_evals.md")
    print(json.dumps({"summary": str(summary_path), "runs": list(report["runs"])}, indent=2, sort_keys=True))


def run_remaining(args: argparse.Namespace) -> dict[str, Any]:
    output_json = args.output_dir / "remaining_subroutines.json"
    command = [sys.executable, "scripts/benchmark_remaining_subroutines.py", "--output-json", str(output_json)]
    if args.remaining_cases_jsonl:
        command.extend(["--cases-jsonl", str(args.remaining_cases_jsonl)])
    if usable_model_dir(args.failure_model_dir):
        command.extend(["--failure-model-dir", str(args.failure_model_dir)])
    if usable_model_dir(args.patch_risk_model_dir):
        command.extend(["--patch-risk-model-dir", str(args.patch_risk_model_dir)])
    return run_json_command(command, output_json)


def run_mcp_subroutines(args: argparse.Namespace) -> dict[str, Any]:
    env = {
        "SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR": str(args.failure_model_dir)
        if usable_model_dir(args.failure_model_dir)
        else "",
        "SLM_HARNESS_PATCH_RISK_MODEL_DIR": str(args.patch_risk_model_dir)
        if usable_model_dir(args.patch_risk_model_dir)
        else "",
    }
    command = [sys.executable, "scripts/benchmark_mcp_subroutines.py"]
    result = run_command(command, env=env)
    output_json = args.output_dir / "mcp_subroutines_stdout.json"
    output_json.write_text(result["stdout"], encoding="utf-8")
    result["output_json"] = str(output_json)
    result["summary_csv"] = str(ROOT / "artifacts" / "results" / "summary.csv")
    result["summary_md"] = str(ROOT / "artifacts" / "results" / "summary.md")
    return result


def run_verifier(args: argparse.Namespace) -> dict[str, Any]:
    output_json = args.output_dir / "verifier_escalation.json"
    command = [sys.executable, "scripts/benchmark_verifier_models.py", "--output-json", str(output_json)]
    if usable_model_dir(args.verifier_model_dir):
        command.extend(["--tiny-model-dir", str(args.verifier_model_dir)])
    return run_json_command(command, output_json)


def run_json_command(command: list[str], output_json: Path) -> dict[str, Any]:
    result = run_command(command)
    result["output_json"] = str(output_json)
    if output_json.exists():
        try:
            result["metrics"] = json.loads(output_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result["parse_error"] = str(exc)
    return result


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = None
    if env is not None:
        import os

        merged_env = os.environ.copy()
        merged_env.update({key: value for key, value in env.items() if value})
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# All model evaluation report",
        "",
        "| Suite | Return code | Output |",
        "| --- | ---: | --- |",
    ]
    for name, payload in report["runs"].items():
        lines.append(f"| `{name}` | {payload.get('returncode')} | `{payload.get('output_json', '')}` |")
    lines.append("")
    lines.append("See the JSON outputs in this directory for full per-case metrics.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def usable_model_dir(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    weight_files = [candidate for candidate in path.iterdir() if candidate.suffix in {".safetensors", ".bin"}]
    if not weight_files:
        return False
    return not all(is_lfs_pointer(candidate) for candidate in weight_files)


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "results" / "all_models")
    parser.add_argument("--remaining-cases-jsonl", type=Path)
    parser.add_argument("--failure-model-dir", type=Path, default=ROOT / "models" / "trained" / "failure_classifier_v1" / "model")
    parser.add_argument("--patch-risk-model-dir", type=Path, default=ROOT / "models" / "trained" / "patch_risk_classifier_v1" / "model")
    parser.add_argument("--verifier-model-dir", type=Path, default=ROOT / "models" / "trained" / "verifier_escalation_v2" / "model")
    return parser.parse_args()


if __name__ == "__main__":
    main()
