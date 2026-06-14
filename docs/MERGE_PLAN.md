# SLM MCP research merge plan

## Scope

Runtime remains the product repository. The research repository is kept external in `~/slm-mcp-merged/research` and supplies subroutine schemas, parameter-floor evidence, benchmark framing, and paper material. The runtime receives a thin integration layer under `openharness.orchestration` plus tests, smoke scripts, benchmarks, and generated artifacts.

## Copied or adapted from research

- Subroutine contracts for JSON repair, traceback localization, search query generation, and search-hit ranking.
- Schema/guard ideas from `slm_harness.mcp.tools`.
- Parameter-floor model mapping from the Hugging Face collection and `slm_harness/results/subroutines/parameter_floor.*`.
- Paper section structure and distinction between research evidence and newly generated runtime results.

## Kept external

- Full Phase 1/Phase 2 training pipeline.
- LoRA/SFT training scripts and large datasets.
- Vendored research OpenHarness subset.
- Research PDFs and figure sources, except summarized references in generated runtime artifacts.

## Executor API mapping

All new subroutines implement the existing runtime `BaseExecutor` contract and return `ExecutorResult` with telemetry metadata: executor name, model id, local/remote flag, success, verifier result, fallback path, and trace id. Deterministic logic runs first where available, schema-SLM fallback second, frontier stub third.

| Research subroutine | Runtime executor | Task type | MCP tool |
| --- | --- | --- | --- |
| `json_repair` | `local.json_repair_slm` | `json_repair` | `slm_repair_json` |
| `trace_localizer` | `local.trace_localizer_slm` | `trace_localize` | `slm_localize_traceback` |
| `search_query_gen` | `local.search_query_gen_slm` | `search_query` | `slm_generate_search_queries` |
| `search_hit_ranker` | `local.search_hit_ranker_slm` | `search_rank` | `slm_rank_search_hits` |

## MCP tool mapping

Existing tools remain unchanged. New tools are additive and also route through `slm_run_task` when the explicit task type is supplied.

- `slm_repair_json(raw)`
- `slm_localize_traceback(traceback, project_prefix="")`
- `slm_generate_search_queries(task)`
- `slm_rank_search_hits(query, hits)`

## Benchmark matrix

Modes:

A. deterministic only  
B. SLM only  
C. deterministic + SLM fallback  
D. deterministic + SLM fallback + frontier fallback stub  
E. MCP full agent path

Task types:

- JSON repair
- traceback localization
- search query generation
- search-hit ranking

Metrics:

- success rate
- schema-valid rate
- verifier-pass rate
- latency p50/p95
- estimated cost per task
- estimated cost per successful task
- fallback rate
- GPU memory usage when `nvidia-smi` is available
- qualitative failure modes

## Artifact output paths

- `artifacts/results/raw/*.jsonl`
- `artifacts/results/summary.csv`
- `artifacts/results/summary.md`
- `artifacts/figures/*.png`
- `artifacts/whitepaper/slm_mcp_harness_whitepaper.md`
- `artifacts/whitepaper/slm_mcp_harness_whitepaper.tex`
- `artifacts/whitepaper/slm_mcp_harness_whitepaper.pdf`
- `~/slm-mcp-merged/final_artifacts/`
- `~/slm-mcp-merged/RUN_SUMMARY.md`

## Shutdown procedure

1. Run tests and benchmarks.
2. Generate white paper and final artifacts.
3. Commit runtime changes on `mcp-research-merge-a40`.
4. Push branch to GitHub.
5. If push fails, write the exact error to `RUN_SUMMARY.md`, create `final_artifacts/mcp-research-merge-a40.patch`, and archive `final_artifacts.tar.gz`.
6. Only after push success or fallback artifact creation, shut down the Lambda instance.
