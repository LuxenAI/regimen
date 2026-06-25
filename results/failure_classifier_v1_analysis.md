# failure_classifier_v1: Negative Generalization Finding

**Branch:** yc-agent-session/slm-context-research  
**Date:** 2026-06-25  
**Checkpoint:** `models/trained/failure_classifier_v1/model/model.safetensors` (17 MB, real weights)  
**Base model:** `google/bert_uncased_L-2_H-128_A-2` (BERT-tiny, 4.4M params)  
**GPU tested on:** NVIDIA A100-SXM4-40GB

---

## What the checkpoint claims

The `failure_classifier_v1` was trained to classify software failure logs and tracebacks into 9
categories: `dependency_issue`, `syntax_error`, `type_error`, `missing_fixture`, `flaky_test`,
`frontend_render_issue`, `sandbox_network_issue`, `assertion_failure`, `unknown`.

The training summary (`models/trained/failure_classifier_v1/training_summary.json`) records:
- accuracy: **1.0**, macro_F1: **1.0** on 324 dev examples
- 1,476 train / 324 dev examples, trained on CUDA in 10.4 seconds
- base model: `google/bert_uncased_L-2_H-128_A-2`

The intended runtime path (`subroutine_models.py:148–179`) loads this checkpoint via
`_predict_text_classifier()` when `SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR` is set, replacing
the deterministic regex classifier `classify_failure_rules()`.

---

## What real inference showed

Ten hand-written, non-synthetic failure log snippets were run through both paths on the A100.
These covered: ImportError, assertion failure, network timeout, syntax error, flaky/intermittent
failure, permission denied, type mismatch, missing pytest fixture, frontend hydration error, and
an ambiguous OOM-kill log.

**Model confidence distribution (softmax max across all 10 cases):**

| Case | Expected | rules\_pred | model\_pred | model\_conf |
|---|---|---|---|---|
| pip install, no matching distribution | dependency\_issue | unknown | unknown | 0.136 |
| AssertionError: assert 401 == 200 | assertion\_failure | assertion\_failure | **unknown** | 0.139 |
| ConnectTimeout to api.stripe.com | sandbox\_network\_issue | **flaky\_test** | **dependency\_issue** | 0.133 |
| SyntaxError: expected ")" | syntax\_error | syntax\_error | syntax\_error | 0.153 |
| RuntimeError: Event loop closed (race) | flaky\_test | flaky\_test | **syntax\_error** | 0.144 |
| PermissionError /etc/ssl/private/key.pem | sandbox\_network\_issue | sandbox\_network\_issue | **dependency\_issue** | 0.145 |
| TypeError: int + NoneType | type\_error | type\_error | **dependency\_issue** | 0.139 |
| fixture "mock\_s3\_bucket" not found | missing\_fixture | missing\_fixture | **syntax\_error** | 0.139 |
| Hydration failed (React SSR) | frontend\_render\_issue | frontend\_render\_issue | **dependency\_issue** | 0.128 |
| Worker exited code 137, 94% memory | unknown | unknown | **syntax\_error** | 0.163 |

**Agreement rate:** 2/10  
**Rules accuracy:** 8/10 correct (wrong on cases 1 and 3)  
**Model accuracy:** 2/10 correct (cases 1 and 4 only; case 1 is a trivial `unknown` tie)

The random-chance baseline for 9 classes is 1/9 ≈ **0.111**. Every model confidence reading
falls in the 0.128–0.163 range — only marginally above chance, and not discriminating. The
logit distribution is effectively uniform across all 9 classes on every real-world input.

---

## Root cause

The training and evaluation data are both generated entirely by
`make_synthetic_examples()` in `scripts/train_remaining_subroutines.py` (lines 364–403).
This function constructs examples by injecting the exact keyword strings that the regex rules
already match:

```python
"dependency_issue": lambda i: f"ModuleNotFoundError: No module named 'rich_{i}'",
"syntax_error":     lambda i: f"SyntaxError: invalid syntax at parser.py line {i}",
"type_error":       lambda i: f"TypeError: expected str but got None in handler {i}",
...
```

The `_FAILURE_PATTERNS` regex in `subroutine_models.py` (lines 35–44) matches precisely these
keyword strings. The trained model learns to recognize the same keywords — achieving 1.0 accuracy
on dev examples that were sampled from the same factory as the training examples. This is a
closed-loop tautology: the "test set" is a different sample from the same generator as the
training set, not an independent held-out distribution.

When presented with real failure logs — which use varied phrasing, mix multiple signal words,
or lack the exact factory keywords — the model produces near-uniform logits because it has only
learned to respond to the precise synthetic patterns. The `stratified_split()` at line 416
splits within this single synthetic pool, so the 1.0 dev accuracy measures in-distribution
memorization, not generalization.

---

## Conclusion

`failure_classifier_v1` should not be deployed as a replacement for `classify_failure_rules()`:
on real-world failure logs it produces near-uniform confidence distributions (max ≈ 0.13–0.16
vs. random baseline 0.11) and is correct on only 2/10 hand-written cases, while the
deterministic rules it was intended to replace are correct on 8/10 of the same cases.
