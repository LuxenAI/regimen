# SLM MCP subroutine benchmark

| Mode | Success | Schema valid | Verifier pass | p50 ms | p95 ms | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic_only | 0.750 | 0.750 | 0.750 | 0.289 | 0.295 | 0.250 |
| slm_only | 1.000 | 1.000 | 1.000 | 0.038 | 0.093 | 1.000 |
| deterministic_slm_fallback | 1.000 | 1.000 | 1.000 | 0.035 | 0.086 | 0.250 |
| deterministic_slm_frontier_stub | 1.000 | 1.000 | 1.000 | 0.038 | 0.092 | 0.250 |
| mcp_full_agent_path | 1.000 | 1.000 | 1.000 | 0.170 | 0.290 | 0.250 |
