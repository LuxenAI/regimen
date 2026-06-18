# Model Artifacts

This folder is reserved for model artifacts used by `slm-harness`.

## Layout

- `trained/`: locally trained project models.
- `huggingface/`: vendored or referenced Hugging Face specialist subroutine checkpoints.

Large binary weights are tracked with Git LFS. Do not add `.safetensors`, `.pt`, `.pth`,
`.bin`, `.onnx`, or `.gguf` files outside LFS tracking.

## Included Trained Model

`trained/verifier_escalation_v2` contains the trained verifier/escalation classifier:

- Base model: `google/bert_uncased_L-2_H-128_A-2`
- Parameter count: `4,386,823`
- Labels: accept/escalation classes for empty output, errors, low confidence, risky diffs,
  visual failures, and incomplete outputs.
- Quantized artifact: `quantized/pytorch_model_int8_dynamic_state.pt`

Use the model with:

```bash
SLM_HARNESS_VERIFIER_BACKEND=transformers \
SLM_HARNESS_VERIFIER_MODEL_DIR=models/trained/verifier_escalation_v2/model \
slm-harness-mcp
```

## Planned Remaining Subroutine Models

`failure_classify` and `patch_risk` are still rule-backed by default, but the runtime can now load
trained classifier artifacts when these environment variables are set:

```bash
SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR=models/trained/failure_classifier_v1/model
SLM_HARNESS_PATCH_RISK_MODEL_DIR=models/trained/patch_risk_classifier_v1/model
```

See [docs/REMAINING_SUBROUTINE_TRAINING_AND_EVALS.md](../docs/REMAINING_SUBROUTINE_TRAINING_AND_EVALS.md)
for the A100 training runbook, teacher-data schema, and eval matrix.
