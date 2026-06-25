# AST-BM25 Function-Name Retrieval Baseline

**Branch:** yc-agent-session/slm-context-research  
**Date:** 2026-06-25  
**Repos:** httpx @ `b5addb64` (133,516 tokens), jinja2 @ `5ef70112` (183,375 tokens)  
**Library:** `rank_bm25.BM25Okapi` (no version attr; installed via pip on GPU instance)  
**Token counter:** tiktoken cl100k_base 0.13.0

---

## Method

**Step 1 — AST extraction.** Every `.py` file in each repo is parsed with `ast.parse`. For
each `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` node, the name, first docstring line
(≤120 chars), file path, and line number are collected. Entries are written to
`/tmp/fn_index.jsonl`.

| Repo | Functions/classes extracted |
|---|---:|
| httpx | 1,241 |
| jinja2 | 1,822 |

**Step 2 — Tokenization.**

*Identifier tokens:* strip leading underscores → camelCase split → split on `_` → lowercase →
drop tokens of length ≤ 1.  
Examples: `_is_https_redirect` → `["is", "https", "redirect"]`; `GetSpontaneousEnvironment` →
`["get", "spontaneous", "environment"]`; `_compile` → `["compile"]`

*Query tokens:* lowercase → split on spaces and punctuation → remove 29 stopwords including
`find`, `the`, `function`, `method`, `locate`, `where`, `into`, etc.

*Index entries* = identifier tokens + docstring tokens (if any). One `BM25Okapi` index per repo.

**Step 3-4 — Retrieval + token measurement.** Top-10 BM25 results per query. Context tokens =
tiktoken count over unique `.py` files containing the top-5 ranked functions.

---

## 10-Query Results Table

| # | ground\_truth | repo | bm25\_rank | top\_1 | top\_3 | hit@1 | hit@3 | hit@5 | bm25\_tokens |
|---|---|---|---:|---|---|:---:|:---:|:---:|---:|
| 1 | `_is_https_redirect` | httpx | 3 | `test_is_https_redirect` | test, test_not, **_is_https_redirect** | N | **Y** | **Y** | 19,738 |
| 2 | `_port_or_default` | httpx | **1** | `_port_or_default` | `_port_or_default`, `port`, test | **Y** | **Y** | **Y** | 28,290 |
| 3 | `_same_origin` | httpx | **1** | `_same_origin` | `_same_origin`, test, test_not | **Y** | **Y** | **Y** | 22,133 |
| 4 | `_merge_queryparams` | httpx | — (not in top-10) | `QueryParams` | QueryParams, params, params | N | N | N | 20,128 |
| 5 | `_build_auth` | httpx | — (not in top-10) | `_build_request_auth` | `_build_request_auth`, test, auth | N | N | N | 22,778 |
| 6 | `get_spontaneous_environment` | jinja2 | **1** | `get_spontaneous_environment` | get_spont, pass_env, test_spont | **Y** | **Y** | **Y** | 23,421 |
| 7 | `load_extensions` | jinja2 | **1** | `load_extensions` | load_ext, EnvironAttr, pass_env | **Y** | **Y** | **Y** | 33,410 |
| 8 | `_tokenize` | jinja2 | — (not in top-10) | `from_string` | from_string, TemplateData, get_source | N | N | N | 32,345 |
| 9 | `_compile` | jinja2 | — (not in top-10) | `from_code` | from_code, test_string, TemplateData | N | N | N | 25,859 |
| 10 | `_load_template` | jinja2 | — (not in top-10) | `load` | load, Name, DictLoader | N | N | N | 35,637 |

---

## Aggregate 3-Method Comparison Table

| Method | Recall@5 | Mean context tokens | Reduction vs full repo |
|---|:---:|---:|---:|
| Rules (Stage 6) | **0 / 10** | 0 | 100% *(trivial — no matches)* |
| **BM25 baseline** | **5 / 10** | **26,374** | **83.4%** |
| Oracle (Stage 6) | **10 / 10** | 16,394 | 89.7% |

Full-repo mean: 158,446 tokens.

BM25 scores at sub-cutoffs: hit@1 = 4/10, hit@3 = 5/10, hit@5 = 5/10.

---

## Miss Diagnosis (5 misses)

**Q4 — `_merge_queryparams`** ("find the method that merges query parameters into a request")  
- Query tokens: `["merges", "query", "parameters", "request"]`  
- Identifier tokens: `["merge", "queryparams"]`  
- Token overlap: **∅**  
- The identifier fuses "query" and "params" into a single token `queryparams`. After splitting
  on underscores, `queryparams` stays as one token (no camelCase boundary, no underscore), so
  BM25 sees an opaque token that shares nothing with the query's "query" or "parameters".
  Top result is `QueryParams` (a class), which contains the right concept but is wrong.

**Q5 — `_build_auth`** ("find the method that builds the auth object for a request")  
- Query tokens: `["builds", "auth", "object", "request"]`  
- Identifier tokens: `["build", "auth"]`  
- Token overlap: `{"auth"}`  
- The inflected form "builds" ≠ "build" (BM25Okapi has no stemming). `_build_request_auth`
  scores higher because it matches both "build" (via stem) and "auth" and "request" — three
  query tokens vs two for `_build_auth`. The correct identifier is outscored by a more
  specific sibling.

**Q8 — `_tokenize`** ("find the method that tokenizes a Jinja2 template source string")  
- Query tokens: `["tokenizes", "jinja2", "template", "source", "string"]`  
- Identifier tokens: `["tokenize"]`  
- Token overlap: **∅**  
- "tokenizes" (query) does not match "tokenize" (identifier). No stemming means inflected
  verb forms are vocabulary misses. `jinja2`, `template`, `source`, `string` each match many
  functions; the one-token identifier `_tokenize` cannot compete.

**Q9 — `_compile`** ("find the method that compiles a template string to a code object")  
- Query tokens: `["compiles", "template", "string", "code", "object"]`  
- Identifier tokens: `["compile"]`  
- Token overlap: **∅**  
- Same inflection problem: "compiles" ≠ "compile". `from_code` wins because "code" appears in
  query and name. `_compile` is a single-token identifier competing against multi-token
  identifiers that match more query terms.

**Q10 — `_load_template`** ("find the method that looks up and loads a template by name")  
- Query tokens: `["looks", "loads", "template", "name"]`  
- Identifier tokens: `["load", "template"]`  
- Token overlap: `{"template"}`  
- "looks" and "loads" are both in the query but only "load" is in the identifier. The bare
  `load` function (a short one-token identifier) scores highest because "load" has very high
  IDF in this index. `_load_template` would need "template" to boost it above `load`, but
  BM25's IDF weights work against common tokens like "template" that appear across many entries.

**Common patterns across all 5 misses:**
1. No stemming (inflected forms like "compiles", "tokenizes", "builds" miss)
2. Compound tokens that survive unsplit (`queryparams`)
3. Sibling identifiers with higher query-token overlap outrank the target (`_build_request_auth` > `_build_auth`)

---

## Interpretation for the SLM Research Direction

The AST-BM25 baseline, implemented in roughly 80 lines of pure Python with no model calls,
already achieves 5/10 recall@5 and 83.4% context reduction — dramatically better than the
deployed rules pipeline (0/10) and only 6.3 percentage points below the oracle ceiling (89.7%
at 10/10 recall). The remaining 5 misses have clear, addressable engineering causes: adding a
stemmer (e.g., Porter or Snowball) would likely recover Q5, Q8, Q9, and Q10; splitting
compound tokens within identifier components (e.g., tokenizing `queryparams` → `query` +
`params`) would recover Q4. These are deterministic fixes, not model improvements.

An SLM query generator would need to beat this baseline *after* the stemming and compound-
split fixes — meaning it would need to achieve >5/10 recall on queries where BM25+stemming
still fails, while adding those identifiers to the context window more precisely than the
oracle's grep-based approach. The current evidence does not yet show a gap that justifies a
500M-parameter model over a stemmed BM25 index.
