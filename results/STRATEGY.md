# Strategy: Where the Token Savings Actually Are

**Branch:** `yc-agent-session/slm-context-research`
**Date:** 2026-06-25
**Goal of the project:** make SLMs / cheap tooling reduce the tokens (and cost) a frontier
coding agent burns.

---

## TL;DR verdict

After auditing the existing harness and running a scaled, GPU-backed study, the evidence points
to one conclusion:

> **The token-saving lever is context reduction via cheap deterministic tooling — not trained
> small language models.** Every place we measured an SLM against a deterministic/lexical
> baseline on real inputs, the cheap baseline won or tied. The SLMs as trained do not generalize.

The broad thesis ("route narrow work to the cheapest reliable executor and only escalate to a
frontier model when needed") is **correct and worth pursuing**. The specific bet that the cheap
executor should be a *trained small model* is **not supported** by what we found. The cheap
executor that wins is regex / AST / BM25 — which costs zero inference.

---

## What was found (compressed)

| Component | Claimed | Reality (measured on A100) |
|---|---|---|
| Production "SLM" subroutines | SLM-backed | Deterministic rules mislabeled as SLM; HF checkpoints were LFS pointers, never loaded |
| `failure_classifier_v1` (BERT-tiny) | acc 1.0 | Near-random on 10 real logs (conf 0.13–0.16 vs 0.11 baseline); 2/10 vs rules 8/10. Train/eval from same synthetic keyword factory |
| `search_query_gen` (Qwen2.5-0.5B) | 88.4% success | 0/10 on real held-out repos; collapses to `{"pattern":"<regex>"}`. The 88.4% was a schema-validation artifact (now fixed) |
| Deployed rules query pipeline | works | 0/10 recall — generates behavioral descriptions, not identifiers |
| AST + BM25 + stemming (built here) | — | **recall@5 0.966, ~87% context reduction** on 1,406 queries / 6 repos. Deterministic, no GPU |
| Neural retrieval (jina-code 160M, real GPU) | — | recall@5 0.627 — **loses to BM25 by 34pp** |
| Hybrid (RRF, then gated cascade) | — | Both lose; union ceiling of any fusion is only +0.7pp and is **not capturable** |

The retrieval study's headline: a free, deterministic retriever cuts a coding agent's context
from ~158k tokens to ~16–24k (top-5 files) while keeping the right code 96.6% of the time.

---

## What's been built (reusable assets)

- `evals/context_retrieval_queries.jsonl` — 1,406 ground-truth retrieval queries, 6 pinned repos.
- `src/openharness/orchestration/code_retriever.py` — dependency-light BM25 retriever (camel/snake
  split, Porter stem, compound-token split, def-site weighting) + content-hash index cache
  (734× speedup warm). Exposed as opt-in MCP tool `slm_code_context_search`.
- A measurement harness: lexical / neural / hybrid / cascade eval scripts, all with raw per-query
  predictions saved.
- The fixed `_schema_valid()` (item-type checking) so "schema valid" can no longer mask garbage.
- Honest write-ups: `retrieval_eval_full.md`, `hybrid_cascade_and_caching.md`, `final_summary.md`.

---

## Where the cost actually is (and isn't)

In an agent loop, **input tokens dominate** — context the model reads (files, tool dumps,
prior turns), not what it writes. So the highest-ROI savings come from putting *less* in front of
the frontier model, accurately:

1. **Context reduction (retrieval) — the big lever. PROVEN.** Reading 5 relevant files instead of
   a repo is an ~87% input cut at 0.97 recall. This is already built and measured (retrieval-side).
2. **Deterministic-first routing of trivial sub-decisions** — JSON repair, path extraction, failure
   triage. Cheap when a rule works; the discipline is to *measure the rule first* before reaching
   for a model.
3. **Prompt caching** — orthogonal but enormous; reuse stable context across turns.
4. **NOT a win on this evidence:** running a 160M–500M model for structured subroutines where
   regex/AST/BM25 already wins. The model's inference cost isn't repaid by tokens saved, and it's
   less reliable.

---

## Recommended direction (ranked)

### 1. Make context-reduction the product. (Highest ROI, mostly built.)
Ship `slm_code_context_search` as a first-class step the agent calls *before* reading files.
Extend beyond function/class retrieval to: call-graph neighbors of a hit, import-aware file
expansion, and a token budget (return top-k files up to N tokens). This directly attacks the
dominant cost and is deterministic/free.

### 2. Answer the one unanswered economic question: does the cut preserve task success?
The downstream value test was skipped (no API key). This is **the** validation: with an API key,
ask a frontier model ~30 locate-and-explain questions under (a) full/truncated context vs
(b) retriever's top-5, and score answer correctness + count tokens. Until this runs, "87%
reduction" is a retrieval number, not a proven cost win. **Do this first once a key exists.**

### 3. Re-pose the SLM bet honestly: only where rules provably can't, and only on real data.
Tiny classifiers (routing, "should I escalate?", patch-risk) *could* gate expensive calls — but:
   - Train on **real** distributions (mined CI logs, real PR diffs/outcomes), never a keyword
     factory that the eval also samples from.
   - Gate every model behind a deterministic baseline in CI: it ships only if it beats regex on a
     held-out, independently-sourced set. The current classifiers fail this bar.
   - Expect most structured subroutines to be won by rules. Reserve models for genuinely fuzzy
     triage where a frontier call would otherwise fire.

### 4. Stop spending GPU on generative SLMs for schema-bound subroutines.
`search_query_gen`, `trace_localize`, `json_repair` are better served by BM25 / AST / `ast.literal_eval`
+ schema validation. The 0.5B checkpoints don't earn their inference. Redirect that effort to #1/#2.

### 5. Make measurement the moat.
The most valuable thing produced here is the discipline: every "X saves tokens" claim is checked
against a deterministic baseline *and* a downstream task-success metric, with raw outputs kept.
Wire this into CI as a regression gate so no future "SLM" lands on schema-validity alone.

---

## Concrete next experiments

1. **Downstream token-vs-correctness test** (needs API key) — the decisive economic measurement.
2. **End-to-end agent token accounting** — run one real multi-file task two ways (full-context
   agent vs retrieval-augmented agent); report total tokens + task success, not just retrieval recall.
3. **Real-data classifier retraining** — mine actual failure logs / PR outcomes; retrain
   failure_classify & patch_risk; gate against regex on an independent test set; keep only if it wins.
4. **Retriever v2** — call-graph/import expansion + token-budgeted output; re-measure recall and
   reduction.
5. **Paraphrased query set** — fairer neural comparison (current queries are docstring-derived,
   which favors lexical). Decides whether neural ever earns its place.

---

## What to stop doing

- Reporting schema-validity or synthetic-set accuracy as "success." Both masked failures here.
- Training small generative models for tasks a rule already solves.
- Calling deterministic paths "SLM." It hid that nothing neural was running.
- Shipping any executor without a deterministic baseline and a downstream check.

---

## One-line framing for the project

> The cheapest reliable executor for narrow developer-agent work is usually *not a model* — it's
> retrieval and rules. Spend the model budget on the frontier call you can't avoid, and spend
> engineering on making that call read less. Prove every savings claim against a free baseline and
> a real task.
