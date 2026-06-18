# SLM MCP subroutine benchmark

| Mode | Success | Schema valid | Verifier pass | p50 ms | p95 ms | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic_only | 0.833 | 0.833 | 0.833 | 0.503 | 4103.388 | 0.500 |
| slm_only | 1.000 | 1.000 | 1.000 | 0.480 | 2.379 | 1.000 |
| deterministic_slm_fallback | 1.000 | 1.000 | 1.000 | 0.056 | 2.457 | 0.500 |
| deterministic_slm_frontier_stub | 1.000 | 1.000 | 1.000 | 0.049 | 2.613 | 0.500 |
| mcp_full_agent_path | 0.833 | 0.833 | 0.833 | 0.532 | 791.348 | 0.500 |
