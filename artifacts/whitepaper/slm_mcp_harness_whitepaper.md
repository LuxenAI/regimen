# SLM Harness MCP: Specialist Small-Model Subroutines for Cost-Efficient Coding Agents

## 1. Abstract

This report documents an end-to-end merge of the `LuxenAI/slmharness` runtime with research subroutine definitions from `IshaanAyaan/slm-agents`. The merged runtime exposes JSON repair, traceback localization, search-query generation, and search-hit ranking through the existing MCP orchestration layer. The implementation keeps deterministic logic first, schema-SLM fallback second, and frontier handoff as a stub when no paid API key is configured.

## 2. Motivation

Coding agents spend a large fraction of time on narrow support work: repairing tool-call JSON, triaging stack traces, generating repository search terms, and ranking search results. These tasks do not always require a frontier model, but naive replacement with a small general model can fail. The harness approach routes narrow tasks to cheap verifiable specialists and escalates only when needed.

## 3. Background: why naive small-model swapping fails

The research repo reports that small models work only when the task and harness are co-designed. A generic small-model swap degrades success, while specialist subroutines paired with deterministic validation can match larger-agent behavior on narrow tasks. Phase 2 decomposes developer-agent work into eight schema-verified subroutines and measures parameter floors from 135M to 1.5B.

## 4. System design

The runtime remains the product repo. The research repo remains external and contributes schemas, subroutine definitions, and benchmark framing. New runtime executors implement the existing `BaseExecutor` contract and return `ExecutorResult` telemetry.

## 5. Repo merge architecture

- Runtime repo: `~/slm-mcp-merged/runtime`
- Research repo: `~/slm-mcp-merged/research`
- Integration plan: `runtime/docs/MERGE_PLAN.md`
- Runtime integration: `openharness.orchestration.subroutine_models` and executor adapters

## 6. MCP runtime design

The existing MCP server remains the entrypoint. New additive tools are:

- `slm_repair_json`
- `slm_localize_traceback`
- `slm_generate_search_queries`
- `slm_rank_search_hits`

The existing tools `slm_list_executors`, `slm_route_task`, `slm_run_task`, `slm_verify_result`, and `slm_get_trace` remain compatible.

## 7. Executor design

Each executor records executor name, model id, local/remote status, latency, estimated cost, success/failure, verifier result, fallback path, and trace id metadata. JSON repair uses deterministic parse first, deterministic extraction second, and schema-SLM fallback for Python-literal/single-quote repair. Other subroutines currently use deterministic research-derived rules as the verified baseline while exposing the HF model IDs selected for future live checkpoint loading.

## 8. Model selection from HF parameter-floor collection

Candidate models are linked from `ishaanranjan/parameter-floors-for-developer-agent-subroutines`. Initial model mapping:

- JSON repair: `ishaanranjan/slm-agent-json-repair-smollm2-360m`
- Traceback localization: `ishaanranjan/slm-agent-trace-localizer-qwen2-5-0-5b`
- Search-query generation: `ishaanranjan/slm-agent-search-query-gen-qwen2-5-0-5b`
- Search-hit ranking: `ishaanranjan/slm-agent-search-hit-ranker-qwen2-5-0-5b`

## 9. Experimental setup

The actual Lambda GPU reported by `nvidia-smi` was NVIDIA A10 with 23028 MiB VRAM, not A40. Benchmarks were intentionally small and controlled. Existing orchestration/MCP tests were run first. New executor tests and a local MCP smoke script were added. Benchmarks wrote raw JSONL, summary CSV/Markdown, and a PNG figure.

## 10. Results

| Mode | Success | Schema-valid | Verifier-pass | p50 ms | p95 ms | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic_only | 0.750 | 0.750 | 0.750 | 0.289 | 0.295 | 0.250 |
| slm_only | 1.000 | 1.000 | 1.000 | 0.038 | 0.093 | 1.000 |
| deterministic_slm_fallback | 1.000 | 1.000 | 1.000 | 0.035 | 0.086 | 0.250 |
| deterministic_slm_frontier_stub | 1.000 | 1.000 | 1.000 | 0.038 | 0.092 | 0.250 |
| mcp_full_agent_path | 1.000 | 1.000 | 1.000 | 0.170 | 0.290 | 0.250 |


The deterministic-only mode fails the malformed JSON case because it refuses to apply schema-SLM repair. All fallback modes repair the malformed JSON and pass the controlled schema checks. Costs are zero in this local benchmark because no paid API key or live remote inference endpoint was used.

## 11. Failure analysis

Observed benchmark failure is intentional: deterministic-only JSON repair fails on Python-literal malformed JSON. No frontier fallback was live-tested because no API key was required or consumed. GPU memory readings were zero in the benchmark table because the implemented path used CPU-local deterministic/schema fallbacks rather than loading HF checkpoints.

## 12. Cost model

Runtime cost is tracked per executor. Local deterministic and schema fallback paths use zero marginal API cost. Future live HF/vLLM runners should fill `estimated_cost_usd` using measured tokens/sec and GPU hourly rate. Frontier handoff should record provider/API cost when `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` is configured.

## 13. Bigger-model integration path

The abstraction is interface-complete: if local subroutines fail, the runtime can escalate to a frontier executor. In this run the frontier path is a stub and is documented honestly. Bigger models can call the subroutines through MCP and receive structured schemas plus trace telemetry.

## 14. Limitations

- HF checkpoints are mapped but not live-loaded in this benchmark run.
- Benchmarks are controlled smoke benchmarks, not full Phase 2 reproduction.
- Read-span selection is intentionally excluded until the first four subroutines are stable.
- The reported GPU is A10, despite the requested A40 branch name.

## 15. Next steps

1. Add optional transformers/vLLM-backed runners for the four HF checkpoints.
2. Re-run the benchmark matrix with live checkpoint inference.
3. Add real-repo tasks from the research repo fixtures.
4. Add context selector/read-span only after JSON repair, traceback localization, query generation, and ranking are stable.
5. Wire live frontier fallback when API keys are available.

## 16. Appendix: commands, configs, schemas

Core commands:

```bash
cd ~/slm-mcp-merged/runtime
. .venv/bin/activate
pytest -q tests/test_orchestration tests/test_mcp
ruff check src/openharness/orchestration tests/test_orchestration scripts/smoke_mcp_subroutines.py scripts/benchmark_mcp_subroutines.py
MYPYPATH=src mypy --explicit-package-bases src/openharness/orchestration tests/test_orchestration
python scripts/smoke_mcp_subroutines.py
python scripts/benchmark_mcp_subroutines.py
```

Artifacts:

- `artifacts/results/raw/subroutine_benchmark.jsonl`
- `artifacts/results/summary.csv`
- `artifacts/results/summary.md`
- `artifacts/figures/success_rate.png`
- `artifacts/whitepaper/slm_mcp_harness_whitepaper.tex`
- `artifacts/whitepaper/slm_mcp_harness_whitepaper.pdf`

This paper distinguishes research repo evidence from newly generated runtime results. Research evidence is used for model selection and motivation. The benchmark table above is newly generated by this merged runtime run.
