#!/usr/bin/env python3
"""Train the two remaining learned subroutine classifiers.

The script trains independent sequence classifiers for:

- failure_classify: log/trace/test-output category classification.
- patch_risk: diff/test-evidence risk classification.

It accepts mixed JSONL teacher data and can fall back to synthetic examples for
smoke validation before the Lambda GPU run.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


FAILURE_LABELS = [
    "dependency_issue",
    "syntax_error",
    "type_error",
    "missing_fixture",
    "flaky_test",
    "frontend_render_issue",
    "sandbox_network_issue",
    "assertion_failure",
    "unknown",
]

PATCH_RISK_LABELS = ["low", "medium", "high"]


@dataclass(frozen=True)
class SubroutineSpec:
    task_type: str
    labels: list[str]
    default_output_name: str
    max_length: int


@dataclass(frozen=True)
class Example:
    task_type: str
    text: str
    label: str
    source: str = "unknown"


TASK_SPECS = {
    "failure_classify": SubroutineSpec(
        task_type="failure_classify",
        labels=FAILURE_LABELS,
        default_output_name="failure_classifier_v1",
        max_length=384,
    ),
    "patch_risk": SubroutineSpec(
        task_type="patch_risk",
        labels=PATCH_RISK_LABELS,
        default_output_name="patch_risk_classifier_v1",
        max_length=512,
    ),
}


class TextDataset:
    """Small torch dataset for tokenized subroutine examples."""

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

    task_types = list(TASK_SPECS) if args.task == "all" else [args.task]
    all_summaries: dict[str, Any] = {}
    for task_type in task_types:
        spec = TASK_SPECS[task_type]
        examples = load_examples(args.teacher_jsonl, args.samples, task_type)
        if args.dry_run:
            all_summaries[task_type] = dry_run_summary(examples, spec)
            continue
        output_dir = args.output_dir / spec.default_output_name if args.task == "all" else args.output_dir
        all_summaries[task_type] = train_task(args, spec, examples, output_dir)

    rendered = json.dumps(all_summaries, indent=2, sort_keys=True)
    print(rendered)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered + "\n", encoding="utf-8")


def train_task(
    args: argparse.Namespace,
    spec: SubroutineSpec,
    examples: list[Example],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install torch and transformers before training remaining subroutines.") from exc

    if not examples:
        raise SystemExit(f"No examples available for {spec.task_type}.")

    label_to_id = {label: index for index, label in enumerate(spec.labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    train_examples, dev_examples = stratified_split(examples, dev_ratio=args.dev_ratio, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(spec.labels),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    param_count = sum(parameter.numel() for parameter in model.parameters())
    if param_count > args.max_params:
        raise SystemExit(f"{args.base_model} has {param_count:,} params, above limit {args.max_params:,}.")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    max_length = args.max_length or spec.max_length
    started = time.time()
    round_summaries: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            tokenizer,
            train_examples,
            label_to_id,
            optimizer,
            device,
            batch_size=args.batch_size,
            max_length=max_length,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
        )
        dev_predictions = predict_examples(
            model,
            tokenizer,
            dev_examples,
            id_to_label,
            device,
            batch_size=args.batch_size,
            max_length=max_length,
        )
        metrics = classification_metrics([example.label for example in dev_examples], dev_predictions, spec.labels)
        round_summaries.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_accuracy": metrics["accuracy"],
                "dev_macro_f1": metrics["macro_f1"],
            }
        )

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    final_predictions = predict_examples(
        model,
        tokenizer,
        dev_examples,
        id_to_label,
        device,
        batch_size=args.batch_size,
        max_length=max_length,
    )
    final_metrics = classification_metrics([example.label for example in dev_examples], final_predictions, spec.labels)

    summary: dict[str, Any] = {
        "task_type": spec.task_type,
        "base_model": args.base_model,
        "labels": spec.labels,
        "parameter_count": param_count,
        "parameter_limit": args.max_params,
        "device": device,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "max_length": max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "metrics": final_metrics,
        "rounds": round_summaries,
        "model_dir": str(model_dir),
        "runtime_env_var": runtime_env_var(spec.task_type),
        "elapsed_seconds": round(time.time() - started, 3),
    }

    if args.quantize_dynamic:
        from openharness.orchestration.verifier_model import quantize_transformer_linear_layers

        quantized_dir = output_dir / "quantized"
        quantized_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, quantized_dir / "model_metadata", dirs_exist_ok=True)
        summary["quantized"] = quantize_transformer_linear_layers(
            model_dir,
            quantized_dir / "pytorch_model_int8_dynamic_state.pt",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def train_epoch(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    label_to_id: dict[str, int],
    optimizer: Any,
    device: str,
    *,
    batch_size: int,
    max_length: int,
    gradient_accumulation_steps: int,
    fp16: bool,
) -> float:
    import torch
    from torch.utils.data import DataLoader

    dataset = TextDataset(examples, tokenizer, label_to_id, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            output = model(**batch)
            loss = output.loss / gradient_accumulation_steps
        loss.backward()
        losses.append(float(loss.item() * gradient_accumulation_steps))
        if step % gradient_accumulation_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses) if losses else 0.0


def predict_examples(
    model: Any,
    tokenizer: Any,
    examples: list[Example],
    id_to_label: dict[int, str],
    device: str,
    *,
    batch_size: int,
    max_length: int,
) -> list[str]:
    import torch
    from torch.utils.data import DataLoader

    label_to_id = {label: index for index, label in id_to_label.items()}
    dataset = TextDataset(examples, tokenizer, label_to_id, max_length)
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            for index in logits.argmax(dim=-1).detach().cpu().tolist():
                predictions.append(id_to_label[int(index)])
    return predictions


def load_examples(path: Path | None, samples: int, task_type: str) -> list[Example]:
    if path is None:
        return make_synthetic_examples(samples, task_type)
    examples: list[Example] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        record_task = str(record.get("task_type") or record.get("kind") or "")
        if record_task and record_task != task_type:
            continue
        try:
            examples.append(example_from_record(record, task_type))
        except KeyError as exc:
            raise SystemExit(f"{path}:{line_number} missing required field for {task_type}: {exc}") from exc
    return examples


def example_from_record(record: dict[str, Any], task_type: str) -> Example:
    if task_type == "failure_classify":
        label = str(record.get("label") or record.get("category") or record.get("expected", {}).get("category"))
        text = render_text(task_type, record)
    elif task_type == "patch_risk":
        label = str(record.get("label") or record.get("risk_level") or record.get("expected", {}).get("risk_level"))
        text = render_text(task_type, record)
    else:
        raise KeyError("task_type")
    if label not in TASK_SPECS[task_type].labels:
        raise KeyError(f"label={label!r}")
    return Example(task_type=task_type, text=text, label=label, source=str(record.get("source") or "teacher_jsonl"))


def render_text(task_type: str, record: dict[str, Any]) -> str:
    payload = record.get("input") if isinstance(record.get("input"), dict) else record
    if task_type == "failure_classify":
        text = str(payload.get("text") or payload.get("logs") or payload.get("traceback") or record.get("prompt") or "")
        return f"TASK: classify software failure\nLOG_OR_TRACE:\n{text.strip()}"
    if task_type == "patch_risk":
        diff = str(payload.get("diff") or record.get("diff") or "")
        tests = str(payload.get("tests") or record.get("tests") or record.get("logs") or "")
        prompt = str(record.get("prompt") or "")
        return f"TASK: classify patch risk\nPROMPT:\n{prompt.strip()}\nDIFF:\n{diff.strip()}\nTEST_EVIDENCE:\n{tests.strip()}"
    raise KeyError(task_type)


def make_synthetic_examples(count: int, task_type: str) -> list[Example]:
    if task_type == "failure_classify":
        factories = {
            "dependency_issue": lambda i: f"ModuleNotFoundError: No module named 'rich_{i}'",
            "syntax_error": lambda i: f"SyntaxError: invalid syntax at parser.py line {i}",
            "type_error": lambda i: f"TypeError: expected str but got None in handler {i}",
            "missing_fixture": lambda i: f"pytest: fixture 'client_{i}' not found",
            "flaky_test": lambda i: f"test timed out after {i + 10}s; intermittent failure observed",
            "frontend_render_issue": lambda i: f"Hydration failed; button not visible; screenshot blank screen {i}",
            "sandbox_network_issue": lambda i: f"network unreachable ECONNREFUSED permission denied sandbox {i}",
            "assertion_failure": lambda i: f"AssertionError: expected {i}, actual {i + 1}; assert failed",
            "unknown": lambda i: f"Task finished with ambiguous warning code {i}",
        }
    elif task_type == "patch_risk":
        factories = {
            "low": lambda i: {
                "diff": f"+ update copy for settings label {i}\n+ add unit test assertion\n",
                "tests": "12 passed, 0 failed",
            },
            "medium": lambda i: {
                "diff": f"+ alter retry timeout and cache fallback path {i}\n" + "\n".join("+ touched helper" for _ in range(35)),
                "tests": "lint passed",
            },
            "high": lambda i: {
                "diff": f"+ return jwt.decode(token, SECRET_{i})\n+ subprocess.run(user_command, shell=True)\n",
                "tests": "not run",
            },
        }
    else:
        raise KeyError(task_type)

    labels = TASK_SPECS[task_type].labels
    examples: list[Example] = []
    for index in range(count):
        label = labels[index % len(labels)]
        payload = factories[label](index)
        if isinstance(payload, dict):
            record = {"task_type": task_type, **payload, "label": label, "source": "synthetic"}
        else:
            record = {"task_type": task_type, "text": payload, "label": label, "source": "synthetic"}
        examples.append(example_from_record(record, task_type))
    return examples


def stratified_split(examples: list[Example], *, dev_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    buckets: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        buckets[example.label].append(example)
    rng = random.Random(seed)
    train: list[Example] = []
    dev: list[Example] = []
    for label_examples in buckets.values():
        rng.shuffle(label_examples)
        dev_count = max(1, int(len(label_examples) * dev_ratio)) if len(label_examples) > 1 else 0
        dev.extend(label_examples[:dev_count])
        train.extend(label_examples[dev_count:])
    rng.shuffle(train)
    rng.shuffle(dev)
    if not dev and train:
        dev.append(train.pop())
    return train, dev


def classification_metrics(gold: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold/predicted length mismatch")
    total = len(gold)
    correct = sum(1 for expected, actual in zip(gold, predicted) if expected == actual)
    per_label: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for expected, actual in zip(gold, predicted) if expected == label and actual == label)
        fp = sum(1 for expected, actual in zip(gold, predicted) if expected != label and actual == label)
        fn = sum(1 for expected, actual in zip(gold, predicted) if expected == label and actual != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for expected in gold if expected == label),
        }
    confusion = Counter(f"{expected}->{actual}" for expected, actual in zip(gold, predicted))
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "correct": correct,
        "total": total,
        "per_label": per_label,
        "confusion": dict(sorted(confusion.items())),
    }


def dry_run_summary(examples: list[Example], spec: SubroutineSpec) -> dict[str, Any]:
    counts = Counter(example.label for example in examples)
    train, dev = stratified_split(examples, dev_ratio=0.18, seed=7)
    return {
        "task_type": spec.task_type,
        "labels": spec.labels,
        "examples": len(examples),
        "label_counts": dict(sorted(counts.items())),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "sample_text": examples[0].text if examples else "",
    }


def runtime_env_var(task_type: str) -> str:
    if task_type == "failure_classify":
        return "SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR"
    if task_type == "patch_risk":
        return "SLM_HARNESS_PATCH_RISK_MODEL_DIR"
    raise KeyError(task_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *TASK_SPECS.keys()], default="all")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "models" / "trained")
    parser.add_argument("--teacher-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--base-model", default="microsoft/codebert-base")
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-params", type=int, default=200_000_000)
    parser.add_argument("--dev-ratio", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--quantize-dynamic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
