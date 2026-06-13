# SLM MCP subroutine benchmark

| Mode | Success | Schema valid | Verifier pass | p50 ms | p95 ms | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic_only | 0.833 | 0.833 | 0.833 | 0.194 | 0.462 | 0.500 |
| slm_only | 1.000 | 1.000 | 1.000 | 0.052 | 0.264 | 1.000 |
| deterministic_slm_fallback | 1.000 | 1.000 | 1.000 | 0.047 | 0.116 | 0.500 |
| deterministic_slm_frontier_stub | 1.000 | 1.000 | 1.000 | 0.044 | 0.100 | 0.500 |
| mcp_full_agent_path | 1.000 | 1.000 | 1.000 | 0.422 | 0.667 | 0.500 |
