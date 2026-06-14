#!/usr/bin/env python3
"""Download Hugging Face model snapshots listed in models/huggingface/manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "huggingface" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "models" / "huggingface"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Run with: "
            "uv run --with huggingface-hub python scripts/download_hf_models.py"
        ) from exc

    manifest = _load_manifest(Path(args.manifest))
    selected = set(args.task_type)
    output_dir = Path(args.output_dir)
    for item in manifest["models"]:
        if selected and item["task_type"] not in selected:
            continue
        destination = output_dir / item["local_dir"]
        print(
            json.dumps(
                {
                    "repo_id": item["repo_id"],
                    "revision": item["revision"],
                    "destination": str(destination),
                    "dry_run": args.dry_run,
                },
                sort_keys=True,
            )
        )
        if args.dry_run:
            continue
        snapshot_download(
            repo_id=item["repo_id"],
            revision=item["revision"],
            local_dir=destination,
            local_dir_use_symlinks=False,
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError(f"Invalid manifest: {path}")
    return payload


if __name__ == "__main__":
    sys.exit(main())
