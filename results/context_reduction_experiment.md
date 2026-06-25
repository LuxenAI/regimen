# Context-Reduction Experiment: Three-Condition Evaluation

**Branch:** yc-agent-session/slm-context-research  
**Date:** 2026-06-25  
**Repos:** httpx @ `b5addb64` (60 .py files, 133,516 tokens), jinja2 @ `5ef70112` (60 .py files, 183,375 tokens)  
**Token counter:** tiktoken cl100k_base 0.13.0  
**grep tool:** GNU grep 3.7 `grep -r -l --include=*.py` (ripgrep not available on GPU instance)

---

## Experiment Setup

Three conditions were evaluated on 10 ground-truth queries (5 from httpx/_client.py, 5 from
jinja2/environment.py) with known correct identifiers drawn directly from AST-parsed source:

**Condition A — Full-repo baseline**  
Tokenize every `.py` file in the repo. This is the worst-case context cost: giving the LLM
everything.

**Condition B — Rules pipeline (currently deployed)**  
1. `generate_search_queries_rules(query)` → list of snake_case / regex / CamelCase terms  
2. `grep -r -l <term> <repo>` for each term → union of matching `.py` files  
3. `rank_search_hits_rules(query, hits)` on matched files → ranked list  
4. Take top-5 files. Count tokens. Check if ground-truth file is in top-5.

**Condition C — Oracle upper bound**  
1. `grep -r -l <ground_truth_identifier> <repo>` → files containing the exact identifier  
2. `rank_search_hits_rules` → top-5. Count tokens. Check recall.  
This is the ceiling: what a perfect query generator would achieve.

---

## Per-Query Results Table

| # | ground\_truth | A\_tokens | B\_files\_found | B\_tokens | B\_recall | C\_files\_found | C\_tokens | C\_recall |
|---|---|---:|---:|---:|:---:|---:|---:|:---:|
| 1 | `_is_https_redirect` | 133,516 | **0** | **0** | **N** | 2 | 15,840 | Y |
| 2 | `_port_or_default` | 133,516 | **0** | **0** | **N** | 1 | 13,659 | Y |
| 3 | `_same_origin` | 133,516 | **0** | **0** | **N** | 2 | 15,840 | Y |
| 4 | `_merge_queryparams` | 133,516 | **0** | **0** | **N** | 1 | 13,659 | Y |
| 5 | `_build_auth` | 133,516 | **0** | **0** | **N** | 2 | 16,328 | Y |
| 6 | `get_spontaneous_environment` | 183,375 | **0** | **0** | **N** | 2 | 18,791 | Y |
| 7 | `load_extensions` | 183,375 | **0** | **0** | **N** | 1 | 13,004 | Y |
| 8 | `_tokenize` | 183,375 | **0** | **0** | **N** | 2 | 21,380 | Y |
| 9 | `_compile` | 183,375 | **0** | **0** | **N** | 3 | 22,431 | Y |
| 10 | `_load_template` | 183,375 | **0** | **0** | **N** | 1 | 13,004 | Y |

Generated terms for representative queries (Condition B):

| Query | Terms generated (all produced 0 grep matches) |
|---|---|
| Q1 (\_is\_https\_redirect) | `['checks_if_redirect_https', 'checks.*if.*redirect.*HTTPS', 'ChecksIfRedirectHTTPS']` |
| Q5 (\_build\_auth) | `['method_builds_auth_object_request', 'method.*builds.*auth.*object', 'MethodBuildsAuthObjectRequest']` |
| Q6 (get\_spontaneous\_environment) | `['creates_spontaneous_jinja2_environment', 'creates.*spontaneous.*Jinja2.*environment', 'CreatesSpontaneousJinja2Environment']` |

---

## Aggregate Results

| Metric | Value |
|---|---|
| Mean A tokens (full repo, across 10 queries) | 158,446 |
| Mean B tokens (rules pipeline, top-5) | **0** |
| Mean C tokens (oracle, top-5) | 16,394 |
| B context reduction vs A | **100.0%** (trivially — no files matched) |
| C context reduction vs A | **89.7%** |
| B recall@5 | **0 / 10** |
| C recall@5 | **10 / 10** |

---

## Plain-Language Interpretation

**The rules pipeline as currently deployed is not fit for context reduction.**

`generate_search_queries_rules()` (subroutine_models.py:231–248) builds search terms by
filtering stop words from the natural-language query and joining the survivors as snake_case,
a regex of the first four words, and CamelCase. On every one of the 10 queries, these terms
produced **zero grep matches** in the target repo. The root cause is structural: the query
says "find the function that checks if a redirect is HTTPS" and the rules emit
`checks_if_redirect_https` — a description of the behavior, not the actual Python identifier
`_is_https_redirect`. Real Python identifiers are often shorter, prefixed with underscores,
or use different words than the query that describes them.

A B recall of 0/10 means the pipeline currently provides no context reduction at all on
queries like these: it returns an empty file list, leaving the downstream LLM with either
nothing or the full repo (A). The 100% "reduction" figure is misleading — it reduces context
to zero tokens because it finds nothing.

**Condition C (oracle) demonstrates the ceiling is achievable:** with the correct identifier,
top-5 results from grep reduce context by 89.7% (from ~158k to ~16k tokens) with 10/10
recall. The oracle is not a realistic runtime path since it requires knowing the identifier
before searching, but it establishes that the *search infrastructure* (grep + rank_search_hits_rules)
works correctly once given a good term. The bottleneck is query generation.

---

## What an SLM Upgrade Would Need to Achieve to Beat Condition C

An SLM-based query generator would need to produce at least one search term per query that
matches an actual Python identifier in the repo (recall@5 > 0 on fresh queries), and it would
need to do this with tokens-in-context lower than Condition C's 16,394 mean — meaning it must
identify the relevant file without needing to match every file that mentions the identifier
(Condition C naturally includes test files and secondary definitions). To genuinely beat the
oracle upper bound, the SLM would need to predict the primary definition file directly, not
just the identifier string.
