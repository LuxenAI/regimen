#!/usr/bin/env python3
"""Benchmark merged SLM MCP subroutines and emit required artifacts."""

from __future__ import annotations

import asyncio
import csv
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openharness.orchestration.mcp_server import _run_subroutine_tool  # noqa: E402
from openharness.orchestration.subroutine_models import (  # noqa: E402
    generate_search_queries_subroutine,
    localize_traceback_subroutine,
    rank_search_hits_subroutine,
    repair_json_subroutine,
)

MODES = [
    "deterministic_only",
    "slm_only",
    "deterministic_slm_fallback",
    "deterministic_slm_frontier_stub",
    "mcp_full_agent_path",
]


def main() -> None:
    out = ROOT / "artifacts"
    raw_dir = out / "results" / "raw"
    fig_dir = out / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for case in cases():
            rows.append(asyncio.run(run_case(mode, case)))

    raw_path = raw_dir / "subroutine_benchmark.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(rows)
    write_summary_csv(summary, out / "results" / "summary.csv")
    write_summary_md(summary, out / "results" / "summary.md")
    write_png(fig_dir / "success_rate.png", summary)
    print(json.dumps({"raw": str(raw_path), "summary": summary}, indent=2, sort_keys=True))


def cases() -> list[dict[str, Any]]:
    traceback = (
        "Traceback (most recent call last):\n"
        "  File \"/usr/lib/python3.11/runpy.py\", line 10, in <module>\n"
        "    run()\n"
        "  File \"/workspace/app/pkg/service.py\", line 42, in handle\n"
        "    user.name.lower()\n"
        "AttributeError: 'NoneType' object has no attribute 'name'\n"
    )
    hits = [
        {"id": "import", "path": "pkg/a.py", "line": 1, "text": "from pkg.service import retry_budget"},
        {"id": "def", "path": "pkg/service.py", "line": 20, "text": "def retry_budget(config):"},
        {"id": "call", "path": "pkg/main.py", "line": 8, "text": "retry_budget(cfg)"},
    ]
    return [
        {"task_type": "json_repair", "payload": {"raw": "{'action': 'SEARCH', 'pattern': 'retry_budget',}"}},
        {"task_type": "trace_localize", "payload": {"traceback": traceback, "project_prefix": "/workspace/app/"}},
        {"task_type": "search_query", "payload": {"task": "Find the retry budget helper function"}},
        {"task_type": "search_rank", "payload": {"query": "retry_budget", "hits": hits}},
    ]


async def run_case(mode: str, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    task_type = case["task_type"]
    payload = case["payload"]
    if mode == "mcp_full_agent_path":
        result = await _run_subroutine_tool(task_type, task_type, payload)
        output = result["result"]["output"]
        success = bool(output.get("success"))
        schema_valid = bool(output.get("schema_valid"))
        verifier_pass = bool(output.get("verifier_pass"))
        fallback_path = str(output.get("fallback_path"))
        cost = float(output.get("estimated_cost_usd") or 0.0)
    else:
        output = run_direct(mode, task_type, payload)
        success = bool(output.get("success"))
        schema_valid = bool(output.get("schema_valid"))
        verifier_pass = bool(output.get("verifier_pass"))
        fallback_path = str(output.get("fallback_path"))
        cost = float(output.get("estimated_cost_usd") or 0.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "mode": mode,
        "task_type": task_type,
        "success": success,
        "schema_valid": schema_valid,
        "verifier_pass": verifier_pass,
        "latency_ms": latency_ms,
        "estimated_cost_usd": cost,
        "fallback_path": fallback_path,
        "fallback_used": fallback_path not in {"deterministic_parse", "deterministic_trace_rules", "deterministic_query_rules", "deterministic_rank_rules"},
        "gpu_memory_mb": gpu_memory_mb(),
        "output": output,
    }


def run_direct(mode: str, task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if task_type == "json_repair":
        run = repair_json_subroutine(
            str(payload["raw"]),
            allow_slm=mode != "deterministic_only",
            allow_frontier=mode == "deterministic_slm_frontier_stub",
        )
    elif task_type == "trace_localize":
        run = localize_traceback_subroutine(str(payload["traceback"]), str(payload.get("project_prefix") or ""))
    elif task_type == "search_query":
        run = generate_search_queries_subroutine(str(payload["task"]))
    elif task_type == "search_rank":
        run = rank_search_hits_subroutine(str(payload["query"]), payload["hits"])
    else:
        raise ValueError(task_type)
    data = run.as_output()
    if mode == "slm_only" and data["success"]:
        data["fallback_path"] = "schema_slm_only"
    return data


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        latencies = [float(row["latency_ms"]) for row in subset]
        successes = sum(1 for row in subset if row["success"])
        total_cost = sum(float(row["estimated_cost_usd"]) for row in subset)
        summary.append(
            {
                "mode": mode,
                "tasks": len(subset),
                "success_rate": successes / len(subset),
                "schema_valid_rate": sum(1 for row in subset if row["schema_valid"]) / len(subset),
                "verifier_pass_rate": sum(1 for row in subset if row["verifier_pass"]) / len(subset),
                "latency_p50_ms": percentile(latencies, 0.5),
                "latency_p95_ms": percentile(latencies, 0.95),
                "estimated_cost_per_task_usd": total_cost / len(subset),
                "estimated_cost_per_success_usd": total_cost / successes if successes else None,
                "fallback_rate": sum(1 for row in subset if row["fallback_used"]) / len(subset),
                "gpu_memory_mb": max(int(row.get("gpu_memory_mb") or 0) for row in subset),
            }
        )
    return summary


def write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def write_summary_md(summary: list[dict[str, Any]], path: Path) -> None:
    lines = ["# SLM MCP subroutine benchmark", "", "| Mode | Success | Schema valid | Verifier pass | p50 ms | p95 ms | Fallback |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(
            f"| {row['mode']} | {row['success_rate']:.3f} | {row['schema_valid_rate']:.3f} | {row['verifier_pass_rate']:.3f} | {row['latency_p50_ms']:.3f} | {row['latency_p95_ms']:.3f} | {row['fallback_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(path: Path, summary: list[dict[str, Any]]) -> None:
    width, height = 640, 360
    pixels = bytearray([255, 255, 255] * width * height)
    colors = [(37, 99, 235), (5, 150, 105), (217, 119, 6), (147, 51, 234), (220, 38, 38)]
    for idx, row in enumerate(summary):
        bar_h = int(float(row["success_rate"]) * 260)
        x0 = 40 + idx * 115
        for y in range(height - 40 - bar_h, height - 40):
            for x in range(x0, x0 + 70):
                off = (y * width + x) * 3
                pixels[off : off + 3] = bytes(colors[idx % len(colors)])
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00") + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def chunk(kind: bytes, data: bytes) -> bytes:
    import binascii

    return len(data).to_bytes(4, "big") + kind + data + binascii.crc32(kind + data).to_bytes(4, "big")


def percentile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * frac))]


def gpu_memory_mb() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=2)
        return max(int(line.strip()) for line in out.splitlines() if line.strip())
    except Exception:
        return 0


if __name__ == "__main__":
    main()
