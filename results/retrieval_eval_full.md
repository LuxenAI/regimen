# Code-Context Retrieval: Full Evaluation

**Branch:** `yc-agent-session/slm-context-research`  
**Date:** 2026-06-25  
**GPU:** NVIDIA A100-SXM4-40GB  
**Eval set:** `evals/context_retrieval_queries.jsonl` — 1,406 auto-generated ground-truth queries  
**Token counter:** tiktoken cl100k_base  
**Task:** per-repo retrieval — each query searches within its own repo for the function/class
its docstring was taken from. Recall@k = ground-truth (identifier + file) appears in top-k.
Context tokens = tiktoken count over the unique files of the top-5 results.

---

## Methods

| Method | Description | Model / deps |
|---|---|---|
| BM25+stem baseline | BM25 over (name + docstring) tokens, camel/snake split, Porter stem | nltk Porter, pure-python BM25 |
| Improved lexical | + wordninja compound split + test-file down-weight (0.5) | nltk Porter, wordninja |
| Neural | cosine over code-embedding vectors of (name + signature + docstring) | jina-embeddings-v2-base-code (160.9M, 768-d) |
| Hybrid RRF | reciprocal-rank fusion (k=60) of BM25+stem and neural | both above |
| Oracle (n=10 only) | grep on the exact ground-truth identifier (Stage 6 upper bound) | — |

Neural inference was verified real: model `jinaai/jina-embeddings-v2-base-code`, device
`cuda:0`, sample embedding shape `(1, 768)` float32, 3,541 MiB VRAM, 1,581 embeddings/sec
over 9,424 symbols.

---

## Headline results (full set, n=1406)

| Method | recall@1 | recall@3 | recall@5 | recall@10 | mean ctx tokens | reduction |
|---|---:|---:|---:|---:|---:|---:|
| **BM25+stem baseline** | **0.780** | **0.930** | **0.966** | **0.987** | 23,980 | 86.9% |
| Improved lexical | 0.723 | 0.888 | 0.942 | 0.972 | 22,842 | 87.8% |
| Neural (jina-code) | 0.421 | 0.567 | 0.627 | 0.699 | 21,769 | 87.9% |
| Hybrid RRF | 0.289 | 0.663 | 0.790 | 0.892 | 25,486 | 85.9% |

Full-repo mean context: ~158,400 tokens (range: requests 94k → rich 515k).

**The plain BM25+stem lexical baseline wins outright on the full set.** Neural and hybrid both
lose; the improved lexical loses on the full set but wins on the implementation subset (below).

---

## Implementation vs test-target subsets

The auto-generated query set includes 360 queries whose ground truth is itself a **test**
function (its docstring described a test). The test-file down-weight in the improved retriever
is the correct heuristic for finding *implementation*, so it helps impl targets and hurts test
targets. Reported honestly:

| Method | impl recall@5 (n=1046) | test recall@5 (n=360) |
|---|---:|---:|
| BM25+stem baseline | 0.962 | 0.978 |
| **Improved lexical** | **0.977** | 0.842 |
| Neural (jina-code) | 0.635 | 0.606 |

On the realistic implementation-target subset, the improved retriever beats the baseline at
**every** k (r@1 0.769→0.793, r@3 0.923→0.941, r@5 0.962→0.977, r@10 0.986→0.993). The full-set
regression (−2.3pp r@5) is entirely the test-penalty cost on test-target queries.

---

## Per-repo recall@5

| Repo | full-repo tokens | BM25+stem | Improved | Neural |
|---|---:|---:|---:|---:|
| httpx | 133,516 | 170/177 = 0.960 | 161/177 = 0.910 | 120/177 = 0.678 |
| jinja2 | 183,375 | 230/241 = 0.954 | 235/241 = 0.975 | 167/241 = 0.693 |
| requests | 94,437 | 147/148 = 0.993 | 144/148 = 0.973 | 113/148 = 0.764 |
| flask | 134,469 | 168/178 = 0.944 | 169/178 = 0.949 | 91/178 = 0.511 |
| click | 205,742 | 305/313 = 0.974 | 278/313 = 0.888 | 189/313 = 0.604 |
| rich | 514,597 | 338/349 = 0.968 | 338/349 = 0.968 | 202/349 = 0.579 |

(click's improved-lexical drop is driven by test-target queries demoted by the test penalty.)

---

## Why neural loses here (honest caveat)

Neural retrieval underperforming lexical by ~34pp at recall@5 is partly an artifact of the
**query-generation method**: each query is the first sentence of the target function's own
docstring (with identifier tokens stripped). That gives lexical BM25 a large surface-overlap
advantage — the query words literally appear in the candidate's indexed docstring. A code
embedding model has to win on semantic similarity, where a docstring-derived query is
paraphrastically close to many sibling functions.

This does **not** prove lexical > neural in general. It shows that for docstring-anchored
queries on these libraries, exact-token BM25 is hard to beat. A fairer neural test needs
human-written queries that paraphrase intent without reusing docstring vocabulary (e.g. the
n=10 hand-built set, or the downstream test in Workstream E). That test was **not run** (no API
key — see below), so the question is left open rather than answered in neural's favor or against.

## Why the hybrid was dropped

Reciprocal-rank fusion of a strong method (BM25+stem, r@5 0.966) with a weak one (neural,
r@5 0.627) produced r@5 0.790 — worse than the strong method alone, and recall@1 collapsed to
0.289 because the two methods disagree on the top result and equal-weight RRF averages them.
Per the plan, the hybrid did not beat both single methods and is **not integrated**.

---

## Workstream E — downstream value test: SKIPPED

No `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` was present locally or on the GPU. Per the run
plan's guard, the downstream accuracy test (does an ~87% context cut preserve a frontier
model's ability to answer?) was skipped and **no downstream numbers were fabricated**. Details
and a ready-to-run procedure: `results/downstream_value_test_SKIPPED.md`. The retrieval-side
token savings it depends on are measured above; the answer-preservation question is unmeasured.

---

## Continuity with the n=10 hand-built set (Stages 5–8)

The original 10 hand-written queries (httpx/jinja2, paraphrased intent) remain in
`results/bm25_retrieval_baseline.md`. On that set BM25+stem scored 7/10 recall@5; at scale
(1,406 queries) the same method scores 0.966 — the small set understated it because 3 of its 10
queries were single-token private identifiers (`_compile`, `_tokenize`), which are rarer at
scale. The improved retriever puts `_merge_queryparams` (the Stage-5 compound miss) at rank 3.

---

## Winner

**BM25+stem** for the general case (full-set r@5 0.966, 86.9% context reduction); **improved
lexical** when the goal is specifically implementation code (impl r@5 0.977). Both are
deterministic, sub-second, no GPU. The neural model — despite real, fast GPU inference — does
not beat lexical retrieval on this docstring-anchored benchmark.

---

## Raw artifacts

- `results/raw/lexical_bm25_baseline_predictions.jsonl` (1,406 rows)
- `results/raw/lexical_improved_predictions.jsonl` (1,406 rows)
- `results/raw/neural_predictions.jsonl` (1,406 rows)
- `results/raw/hybrid_rrf_predictions.jsonl` (1,406 rows)
- `results/raw/{lexical,neural,hybrid}_eval_metrics.json`
- eval scripts: `scripts/eval_lexical_retrieval.py`, `scripts/eval_neural_retrieval.py`,
  `scripts/eval_hybrid_retrieval.py`, `scripts/build_context_retrieval_queries.py`
