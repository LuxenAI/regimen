#!/usr/bin/env bash
set -euo pipefail

# Lambda Labs target: one A100 80GB instance is enough for CodeBERT-sized
# classifiers and leaves room for larger DeBERTa/StarEncoder ablations.

cd "$(dirname "$0")/.."

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install torch transformers accelerate

TEACHER_JSONL="${TEACHER_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-models/trained}"
RESULTS_DIR="${RESULTS_DIR:-artifacts/results}"

TRAIN_ARGS=(
  --task all
  --output-dir "$OUTPUT_DIR"
  --summary-json "$RESULTS_DIR/remaining_subroutines_training.json"
  --base-model "${BASE_MODEL:-microsoft/codebert-base}"
  --epochs "${EPOCHS:-3}"
  --batch-size "${BATCH_SIZE:-24}"
  --fp16
  --quantize-dynamic
)

if [[ -n "$TEACHER_JSONL" && -f "$TEACHER_JSONL" ]]; then
  TRAIN_ARGS+=(--teacher-jsonl "$TEACHER_JSONL")
fi

python scripts/train_remaining_subroutines.py "${TRAIN_ARGS[@]}"

BENCH_ARGS=(
  --failure-model-dir "$OUTPUT_DIR/failure_classifier_v1/model"
  --patch-risk-model-dir "$OUTPUT_DIR/patch_risk_classifier_v1/model"
  --output-json "$RESULTS_DIR/remaining_subroutines_benchmark.json"
)

if [[ -n "$TEACHER_JSONL" && -f "$TEACHER_JSONL" ]]; then
  BENCH_ARGS+=(--cases-jsonl "$TEACHER_JSONL")
fi

python scripts/benchmark_remaining_subroutines.py "${BENCH_ARGS[@]}"

SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR="$OUTPUT_DIR/failure_classifier_v1/model" \
SLM_HARNESS_PATCH_RISK_MODEL_DIR="$OUTPUT_DIR/patch_risk_classifier_v1/model" \
python scripts/benchmark_mcp_subroutines.py

python scripts/eval_all_models.py \
  --output-dir "$RESULTS_DIR/all_models" \
  --failure-model-dir "$OUTPUT_DIR/failure_classifier_v1/model" \
  --patch-risk-model-dir "$OUTPUT_DIR/patch_risk_classifier_v1/model" \
  --verifier-model-dir "$OUTPUT_DIR/verifier_escalation_v2/model"

python scripts/compile_model_artifacts.py \
  --output-json artifacts/model_manifest.json \
  --output-md artifacts/MODEL_RUN_SUMMARY.md
