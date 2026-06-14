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
