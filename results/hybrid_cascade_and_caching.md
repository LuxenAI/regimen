# Closing the "Can Neural Help at the Margin?" Question + Retriever Caching

**Branch:** `yc-agent-session/slm-context-research`  
**Date:** 2026-06-25  
**GPU:** NVIDIA A100-SXM4-40GB

This follows up the retrieval study (`results/retrieval_eval_full.md`). The PR left one
question open: the naive equal-weight RRF hybrid lost to lexical, but *could a smarter
combination capture the cases neural gets right?* This document answers it (no) and adds a
real engineering improvement to the integrated tool (index caching).

---

## 1. Upper bound: how much could any hybrid gain?

Cross-tabulating the committed per-query predictions (BM25+stem vs neural, n=1406):

- BM25+stem misses only **48/1406 (3.4%)** at recall@5.
- Neural recovers **10** of those 48 in its own top-5 (13 in top-10).
- **Union ceiling** (GT in lexical top-5 OR neural top-5) = **0.973**, vs lexical alone 0.966.
- Neural ranks the ground truth strictly higher than lexical in only **70** queries; lexical
  beats neural in **747**.

So the absolute most any lexical+neural fusion can add is **+0.7pp (10 queries)**. The
recoverable cases are genuinely semantic — the query is a docstring sentence whose words don't
lexically overlap the identifier:

| Query (excerpt) | Ground truth | lexical rank | neural rank |
|---|---|---:|---:|
| "Returns True for 4xx status codes, False otherwise" | `is_client_error` | 6 | 3 |
| "A `{% %}` tag that dumps the available variables" | `DebugExtension` | — | 1 |
| "all values from the query param for a given key" | `get_list` | 6 | 1 |
| "Iterates over the response data" | `iter_content` | 7 | 4 |

---

## 2. Confidence-gated cascade: does not work

Design: run lexical first; when lexical is **uncertain** (top-1 / top-2 BM25 margin below a
gate), inject neural candidates into the top-5; otherwise leave lexical untouched. The intent is
to capture the 10 recoverable misses without disturbing the 96.6% lexical already gets.

Sweep over the gate threshold (`scripts/eval_cascade_retrieval.py`):

| Gate (margin) | recall@5 | recovered | regressed |
|---|---:|---:|---:|
| pure lexical (baseline) | 0.958 | 0 | 0 |
| 0.05 | 0.947 | 3 | 19 |
| 0.10 | 0.943 | 5 | 26 |
| 0.15 | 0.940 | 5 | 30 |
| 0.20 | 0.940 | 5 | 30 |
| 0.30 | 0.939 | 5 | 32 |

(Baseline here is 0.958 rather than the headline 0.966 because this script indexes 500-char
docstrings vs 400; the cascade comparison is internally consistent.)

**At every gate the cascade regresses more than it recovers.** The lexical margin is a poor
confidence signal: a low margin usually means several lexical candidates are *all relevant*
(including the right one), not that lexical is wrong. Injecting the much-weaker neural ranking
then displaces correct lexical hits. The +0.7pp ceiling is real but **not capturable** with a
margin gate — and capturing it would require already knowing which queries lexical will miss,
which is the retrieval problem itself.

**Verdict:** neural adds no usable value on this benchmark. Ship lexical alone. The hybrid path
remains un-integrated (consistent with the earlier RRF finding, now explained rather than just
observed).

> Caveat carried forward: this benchmark's queries are docstring-derived, which structurally
> favors lexical overlap. The verdict is specific to this evaluation; a human-paraphrased query
> set or the (skipped, no-API-key) downstream test could still surface neural value.

---

## 3. Engineering: cached retriever index

The `slm_code_context_search` MCP tool previously called `build_retriever([root])` on every
invocation, re-walking the repo, re-parsing every `.py` file with `ast`, and re-fitting BM25.
For a large repo that is ~2 seconds per query.

Added `get_cached_retriever(root, **kwargs)` (in `code_retriever.py`):

- Keyed by `(abspath(root), retriever kwargs)`.
- Invalidated by `repo_signature(root)` — a SHA-256 over every `.py` file's path + `st_mtime_ns`
  + size, so any add/remove/edit forces a rebuild.
- Bounded LRU (8 repos).

Measured on `rich` (2,096 symbols), A100 box:

```
cold build: 1928 ms
warm cache: 2.6 ms   (same object returned)
speedup:    734x
```

The MCP tool now uses the cached builder. Covered by
`tests/test_code_retriever.py::test_cached_retriever_reuses_index_until_files_change`
(asserts reuse on an unchanged repo and rebuild after a file is added). Full suite: **11 passed**,
ruff clean.

---

## Artifacts

- `scripts/eval_cascade_retrieval.py` — cascade sweep
- `results/raw/cascade_eval_metrics.json` — union ceiling + sweep numbers
- `src/openharness/orchestration/code_retriever.py` — `repo_signature`, `get_cached_retriever`,
  `clear_retriever_cache`
