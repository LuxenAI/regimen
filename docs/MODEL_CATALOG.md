# SLM model catalog

This repo tracks candidate small-model checkpoints separately from runtime executor wiring. A
model belongs in the runtime only after it has an adapter, schema tests, and benchmark coverage
against the deterministic or heuristic executor it would replace.

## Hugging Face collections

### Parameter Floors for Developer-Agent Subroutines

- Collection: https://huggingface.co/collections/ishaanranjan/parameter-floors-for-developer-agent-subroutines
- Canonical API slug: `ishaanranjan/parameter-floors-for-developer-agent-subroutines-6a2cb686f9f1d089a21b6039`
- Description from Hugging Face: full-parameter SFT checkpoints for eight schema-verified
  coding-agent subroutines across 135M to 1.5B parameters.
- Status in `slm-harness`: linked as candidate model source; no checkpoint is loaded by default.

Important constraint: these are not `<10M` models. The smallest collection checkpoints are
SmolLM2 135M. They are useful as local/remote SLM executor candidates, but they do not replace
the current BERT-Tiny verifier target or any future strict sub-10M classifier.

## Checkpoint families and harness fit

| Collection family | Best reported candidate | Existing overlap | Integration stance |
| --- | --- | --- | --- |
| Agent Action Router | `ishaanranjan/slm-agent-action-router-qwen2-5-1-5b` at 97.2% held-out success | Overlaps `WorkflowDecomposer`, `KeywordClassifierExecutor`, and `CheapestReliableRouter` | Candidate learned router, but keep deterministic router as fallback and baseline. |
| Code Evidence Judge | `ishaanranjan/slm-agent-evidence-judge-qwen2-5-1-5b` at 91.2% | Overlaps `VerifierEscalationClassifierExecutor` and `ResultVerifier` | Candidate verifier backend; benchmark before replacing BERT-Tiny verifier or heuristic fallback. |
| JSON Tool-Call Repair | 360M, 0.5B, and 1.5B variants report 100.0% | No dedicated executor in orchestration package | Good new executor candidate because deterministic JSON parsing/repair is currently absent. |
| Repository Path Normalizer | `ishaanranjan/slm-agent-path-normalizer-qwen2-5-1-5b` at 96.8% | Partial overlap with `RegexExtractorExecutor` path extraction | Add only if it normalizes ambiguous repo paths better than deterministic path handling. |
| Code Read-Span Selector | `ishaanranjan/slm-agent-read-span-selector-qwen2-5-1-5b` at 83.2% | No current context-selector executor | Good candidate for context selection; lower-size variants are weak and should not be default. |
| Code Search-Hit Ranker | `ishaanranjan/slm-agent-search-hit-ranker-qwen2-5-1-5b` at 98.0% | No current ranker executor | Good candidate for search result ranking before escalating to Codex/Claude. |
| Code Search-Query Generator | `ishaanranjan/slm-agent-search-query-gen-qwen2-5-0-5b` at 88.4% | Partial overlap with decomposition only | Good candidate for repo search planning; 0.5B outperforms 1.5B in the collection note. |
| Python Traceback Localizer | `ishaanranjan/slm-agent-trace-localizer-qwen2-5-1-5b` at 100.0% | Partial conceptual overlap with failure classification, but no localizer executor exists | Strong candidate for a failure-triage/localization executor. |

## Redundancy audit

Current real overlap is limited:

- `Agent Action Router` duplicates behavior already handled by the rule-based decomposer,
  keyword classifier, and cheapest-reliable router. This should be benchmarked as an optional
  learned router, not added as another default router.
- `Code Evidence Judge` duplicates the verifier/escalation classifier surface. It should plug
  into the existing `VerifierClassifier` protocol instead of creating a parallel verifier path.
- `Repository Path Normalizer` partially overlaps path extraction in `RegexExtractorExecutor`,
  but the current executor only extracts paths; it does not normalize ambiguous repo paths.
- `Python Traceback Localizer` overlaps the product idea of a failure classifier, but the current
  orchestration package does not yet expose a trace-localizer executor.

No current runtime executor covers JSON repair, read-span selection, search-hit ranking, or
search-query generation. Those are non-redundant additions if they are added behind the existing
`BaseExecutor` contract.

## Recommended next adapters

1. `local.json_repair_slm`: deterministic JSON parser first, SLM fallback only for malformed
   tool-call payloads.
2. `local.trace_localizer_slm`: classify traceback frames and return file/line/symbol candidates.
3. `local.search_hit_ranker_slm`: rank `rg`/symbol search hits before sending context to a
   frontier coding agent.
4. `local.action_router_slm`: optional learned router benchmarked against the existing router,
   never replacing the deterministic fallback.

Every adapter should expose a stable JSON schema, record cost/latency in `ExecutorResult`, and
ship with benchmark cases showing it beats the deterministic baseline on its own task.
