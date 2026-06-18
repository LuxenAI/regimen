#!/usr/bin/env python3
"""Compile model weights, training metadata, and eval outputs into one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".onnx", ".gguf"}
META_NAMES = {"train_meta.json", "training_summary.json", "config.json", "generation_config.json"}


def main() -> None:
    args = parse_args()
    manifest = {
        "models": collect_models(ROOT / "models"),
        "eval_outputs": collect_eval_outputs(ROOT / "artifacts" / "results"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(manifest, args.output_md)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2, sort_keys=True))


def collect_models(models_root: Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in models_root.glob("*/*") if path.is_dir()):
        weights = [weight_info(path) for path in sorted(model_dir.rglob("*")) if path.suffix in WEIGHT_SUFFIXES]
        metadata = {}
        for meta_path in sorted(path for path in model_dir.rglob("*") if path.name in META_NAMES):
            try:
                metadata[str(meta_path.relative_to(ROOT))] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata[str(meta_path.relative_to(ROOT))] = {"unparsed": True}
        if weights or metadata:
            models.append(
                {
                    "name": model_dir.name,
                    "path": str(model_dir.relative_to(ROOT)),
                    "family": model_dir.parent.name,
                    "weights": weights,
                    "metadata": metadata,
                }
            )
    return models


def weight_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    pointer = parse_lfs_pointer(data)
    info: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "size_bytes_on_disk": path.stat().st_size,
    }
    if pointer:
        info.update(pointer)
        info["sha256"] = pointer["lfs_oid_sha256"]
        info["storage"] = "git_lfs_pointer"
    else:
        info["sha256"] = hashlib.sha256(data).hexdigest()
        info["storage"] = "file"
    return info


def parse_lfs_pointer(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        return None
    text = data.decode("utf-8", errors="replace")
    payload: dict[str, Any] = {"storage": "git_lfs_pointer"}
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            payload["lfs_oid_sha256"] = line.split("sha256:", 1)[1].strip()
        elif line.startswith("size "):
            try:
                payload["lfs_size_bytes"] = int(line.split(" ", 1)[1])
            except ValueError:
                payload["lfs_size_bytes"] = line.split(" ", 1)[1]
    return payload if "lfs_oid_sha256" in payload else None


def collect_eval_outputs(results_root: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    if not results_root.exists():
        return outputs
    for path in sorted(results_root.rglob("*")):
        if path.suffix not in {".json", ".jsonl", ".csv", ".md"} or not path.is_file():
            continue
        outputs.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return outputs


def write_markdown(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# Model run summary",
        "",
        "## Models",
        "",
        "| Model | Family | Weight files | Metadata files |",
        "| --- | --- | ---: | ---: |",
    ]
    for model in manifest["models"]:
        lines.append(
            f"| `{model['name']}` | `{model['family']}` | {len(model['weights'])} | {len(model['metadata'])} |"
        )
    lines.extend(["", "## Eval Outputs", "", "| Path | Size bytes |", "| --- | ---: |"])
    for output in manifest["eval_outputs"]:
        lines.append(f"| `{output['path']}` | {output['size_bytes']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "model_manifest.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "artifacts" / "MODEL_RUN_SUMMARY.md")
    return parser.parse_args()


if __name__ == "__main__":
    main()
