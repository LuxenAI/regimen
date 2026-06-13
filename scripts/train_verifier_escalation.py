#!/usr/bin/env python3
"""Train a tiny verifier/escalation classifier for slm-harness.

This script is intentionally teacher-model agnostic. Pass `--teacher-jsonl` with
frontier-labeled examples when available; otherwise it generates synthetic
teacher-style labels for smoke training and self-training validation.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openharness.orchestration.verifier_model import (  # noqa: E402
    DEFAULT_VERIFIER_LABELS,
    VerifierInput,
    quantize_transformer_linear_layers,
)


@dataclass(frozen=True)
class Example:
    text: str
    label: str


class TextDataset:
    """Minimal torch dataset for tokenized verifier examples."""

    def __init__(self, examples: list[Example], tokenizer: Any, label_to_id: dict[str, int], max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.label_to_id[example.label], dtype=torch.long)
        return item


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch and transformers before training this verifier.") from exc

    labels = list(DEFAULT_VERIFIER_LABELS)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    examples = load_examples(args.teacher_jsonl, args.samples)
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.82))
    train_examples = examples[:split]
    dev_examples = examples[split:] or examples[-1:]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    param_count = sum(parameter.numel() for parameter in model.parameters())
    if param_count > args.max_params:
        raise SystemExit(f"Base model has {param_count:,} params, above limit {args.max_params:,}.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    round_summaries: list[dict[str, Any]] = []
    for round_index in range(args.self_training_rounds + 1):
        train_model(
            model,
            tokenizer,
            train_examples,
            label_to_id,
            optimizer,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        accuracy = evaluate_model(
            model,
            tokenizer,
            dev_examples,
            label_to_id,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        round_summaries.append(
            {
                "round": round_index,
                "train_examples": len(train_examples),
                "dev_examples": len(dev_examples),
                "dev_accuracy": accuracy,
            }
        )
        if round_index < args.self_training_rounds:
            pseudo = pseudo_label_examples(
                model,
                tokenizer,
                make_synthetic_examples(max(20, args.samples // 3)),
                id_to_label,
                device,
                confidence_threshold=args.pseudo_label_threshold,
                max_length=args.max_length,
            )
            train_examples.extend(pseudo)
            random.shuffle(train_examples)

    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    summary: dict[str, Any] = {
        "base_model": args.base_model,
        "teacher_model": args.teacher_model,
        "labels": labels,
        "parameter_count": param_count,
        "parameter_limit": args.max_params,
        "device": device,
        "rounds": round_summaries,
        "model_dir": str(model_dir),
    }

    if args.quantize_dynamic:
        quantized_dir = args.output_dir / "quantized"
        quantized_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, quantized_dir / "model_metadata", dirs_exist_ok=True)
        summary["quantized"] = quantize_transformer_linear_layers(
            model_dir,
            quantized_dir / "pytorch_model_int8_dynamic_state.pt",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def train_model(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    label_to_id: dict[str, int],
    optimizer: Any,
    device: str,
    *,
    epochs: int,
    batch_size: int,
    max_length: int,
) -> None:
    import torch
    from torch.utils.data import DataLoader

    dataset = TextDataset(examples, tokenizer, label_to_id, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate_model(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    label_to_id: dict[str, int],
    device: str,
    *,
    batch_size: int,
    max_length: int,
) -> float:
    import torch
    from torch.utils.data import DataLoader

    dataset = TextDataset(examples, tokenizer, label_to_id, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions = logits.argmax(dim=-1)
            total += int(labels.numel())
            correct += int((predictions == labels).sum().item())
    return correct / total if total else 0.0


def pseudo_label_examples(
    model: Any,
    tokenizer: Any,
    candidates: list[Example],
    id_to_label: dict[int, str],
    device: str,
    *,
    confidence_threshold: float,
    max_length: int,
) -> list[Example]:
    import torch

    pseudo: list[Example] = []
    model.eval()
    with torch.no_grad():
        for candidate in candidates:
            encoded = tokenizer(
                candidate.text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probs = torch.softmax(model(**encoded).logits, dim=-1)[0]
            score, index = torch.max(probs, dim=-1)
            if float(score.item()) >= confidence_threshold:
                pseudo.append(Example(text=candidate.text, label=id_to_label[int(index.item())]))
    return pseudo


def load_examples(path: Path | None, samples: int) -> list[Example]:
    if path is None:
        return make_synthetic_examples(samples)
    examples: list[Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = str(record["label"])
        if "text" in record:
            text = str(record["text"])
        else:
            text = VerifierInput(
                task=str(record.get("task", "")),
                task_type=record.get("task_type"),
                executor_name=str(record.get("executor_name", "external.candidate")),
                executor_kind=record.get("executor_kind", "mcp_tool"),
                executor_output=record.get("output", ""),
                executor_confidence=float(record.get("confidence", 0.0)),
                logs=str(record.get("logs", "")),
                diff=str(record.get("diff", "")),
                screenshot_summary=str(record.get("screenshot_summary", "")),
            ).to_model_text()
        examples.append(Example(text=text, label=label))
    return examples


def make_synthetic_examples(count: int) -> list[Example]:
    factories: dict[str, list[Any]] = {
        "accept": [
            lambda i: VerifierInput(
                task=f"Extract email address from customer note {i}",
                task_type="extract",
                executor_name="local.regex_extractor",
                executor_kind="deterministic",
                executor_output={"emails": [f"user{i}@example.com"]},
                executor_confidence=0.88,
                logs="1 passed",
            ),
            lambda i: VerifierInput(
                task=f"Extract email {i}",
                task_type="extract",
                executor_name="external.candidate",
                executor_kind="mcp_tool",
                executor_output={"emails": [f"admin{i}@example.com"]},
                executor_confidence=0.91,
                logs="1 passed",
            ),
            lambda i: VerifierInput(
                task=f"Patch auth middleware {i}",
                task_type="code",
                executor_name="external.candidate",
                executor_kind="mcp_tool",
                executor_output="Patch added and tests pass",
                executor_confidence=0.89,
                diff="auth middleware update",
                logs="12 passed, 0 failed",
            ),
        ],
        "escalate_empty_output": [
            lambda i: VerifierInput(
                task=f"Classify CI failure {i}",
                task_type="classify",
                executor_output="",
                executor_confidence=0.81,
            ),
            lambda i: VerifierInput(
                task=f"Verify extraction {i}",
                task_type="verify",
                executor_output={},
                executor_confidence=0.9,
                logs="tool completed with no extracted fields",
            ),
        ],
        "escalate_error": [
            lambda i: VerifierInput(
                task=f"Run unit tests {i}",
                task_type="tool",
                executor_output="pytest output",
                executor_confidence=0.93,
                logs="Traceback: TypeError: expected str, got None",
            ),
            lambda i: VerifierInput(
                task=f"Run tests {i}",
                task_type="tool",
                executor_output="pytest output",
                executor_confidence=0.95,
                logs="Traceback: ModuleNotFoundError: No module named fastapi",
            ),
        ],
        "escalate_low_confidence": [
            lambda i: VerifierInput(
                task=f"Route subtask {i}",
                task_type="route",
                executor_output={"task_type": "unknown"},
                executor_confidence=0.38,
            ),
            lambda i: VerifierInput(
                task=f"Classify failure {i}",
                task_type="classify",
                executor_output={"label": "type_error"},
                executor_confidence=0.41,
            ),
        ],
        "escalate_risky_diff": [
            lambda i: VerifierInput(
                task=f"Review patch {i}",
                task_type="code",
                executor_output="Patch created",
                executor_confidence=0.86,
                diff="subprocess.run(user_command, shell=True)",
            ),
            lambda i: VerifierInput(
                task=f"Review payment patch {i}",
                task_type="code",
                executor_output="Patch updated billing flow",
                executor_confidence=0.87,
                diff="payment token delete path changed",
            ),
        ],
        "escalate_visual_failure": [
            lambda i: VerifierInput(
                task=f"Check browser screenshot {i}",
                task_type="verify",
                executor_output="Rendered page",
                executor_confidence=0.89,
                screenshot_summary="The page is blank with a loading spinner.",
            ),
            lambda i: VerifierInput(
                task=f"Check browser screenshot {i}",
                task_type="verify",
                executor_output="Rendered page",
                executor_confidence=0.89,
                screenshot_summary="Main button is clipped and text overlaps.",
            ),
        ],
        "escalate_incomplete": [
            lambda i: VerifierInput(
                task=f"Summarize implementation result {i}",
                task_type="reason",
                executor_output="TODO: wire the final handler",
                executor_confidence=0.78,
            ),
            lambda i: VerifierInput(
                task=f"Implement feature {i}",
                task_type="code",
                executor_output="Placeholder only; not implemented",
                executor_confidence=0.79,
            ),
        ],
    }
    examples: list[Example] = []
    for index in range(count):
        label = DEFAULT_VERIFIER_LABELS[index % len(DEFAULT_VERIFIER_LABELS)]
        variants = factories[label]
        factory = variants[(index // len(DEFAULT_VERIFIER_LABELS)) % len(variants)]
        examples.append(Example(text=factory(index).to_model_text(), label=label))
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-jsonl", type=Path)
    parser.add_argument("--teacher-model", default="frontier-teacher-compatible")
    parser.add_argument("--base-model", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--samples", type=int, default=280)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--self-training-rounds", type=int, default=1)
    parser.add_argument("--pseudo-label-threshold", type=float, default=0.92)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-params", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quantize-dynamic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
