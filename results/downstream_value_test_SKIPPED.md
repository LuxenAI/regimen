# Workstream E — Downstream Value Test: SKIPPED (no API key)

**Status:** SKIPPED — no frontier API key available.

**Guard check (2026-06-25):**

```
ANTHROPIC_API_KEY: absent (local)
OPENAI_API_KEY:    absent (local)
ANTHROPIC_API_KEY: absent (gpu)
OPENAI_API_KEY:    absent (gpu)
```

Neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is set in the local environment
or on `slm-gpu`. Per the run plan's explicit guard, this workstream is skipped
and **no downstream accuracy numbers are reported** — fabricating them would
violate the honesty constraint.

## What this test would have measured

The downstream test would verify that the winning retriever's reduced context
(top-5 files, ~22–24k tokens) preserves a frontier model's ability to answer
"locate-and-explain" questions versus full-repo-truncated context (~158k tokens
mean). Procedure, ready to run once a key is present:

1. Sample ~30 queries from `evals/context_retrieval_queries.jsonl` whose answer
   requires the ground-truth function (prefer implementation-target queries).
2. For each, ask a frontier model the question twice:
   - (i) with full-repo-truncated context
   - (ii) with the retriever's top-5 reduced context
3. Score correctness automatically: does the answer name the correct
   function/file (`ground_truth` + `gt_file`)?
4. Report downstream accuracy (full vs reduced) and tokens used for each.

A runnable harness is **not** committed because it cannot be validated without a
key; writing untested API-calling code that has never executed would be a
different kind of fabrication. The retrieval-side token-saving numbers it depends
on are already measured honestly in Workstreams B and C:

- Winning retriever (BM25+stem) mean reduced context: ~23,980 tokens
- Full-repo mean: ~158,000 tokens
- Context reduction: ~86.9% at recall@5 = 0.966

The open question this test would answer — *does an 86.9% context cut preserve
answer correctness?* — remains **unmeasured** in this run.
