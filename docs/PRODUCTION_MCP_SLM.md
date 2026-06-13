# Production MCP SLM Harness

`slm-harness` runs as an MCP server around Codex or Claude Code. The coding agent remains the main brain; the harness handles narrow typed subtasks, records telemetry, and escalates only when cheaper paths are not reliable enough.

## Runtime Shape

- Local Codex: use `slm_codex_probe` or `build_codex_mcp_config_snippet()` for `~/.codex/config.toml`.
- Local Claude Code: use `slm_eval_manifest`, `build_claude_mcp_json()`, or `build_claude_mcp_add_command()`.
- Cloud SLMs: keep Codex/Claude connected to the local MCP server, then point individual subroutines at `remote_http` URLs in `.slm-harness.toml`.
- Remote MCP: run `SLM_HARNESS_MCP_TRANSPORT=streamable-http slm-harness-mcp streamable-http` when Claude Code or another MCP client should connect over HTTP.

## Model Routing

Config loads from `SLM_HARNESS_CONFIG` or `.slm-harness.toml`. See `.slm-harness.example.toml`.

Supported backends:

- `auto`: deterministic first, then local/cloud if configured, otherwise CI-safe fake fallback.
- `deterministic`: rules only.
- `local_transformers`: optional `torch`/`transformers` model path.
- `local_onnx`: optional ONNX slot for tiny classifiers.
- `remote_http`: hosted SLM/vLLM-style endpoint.
- `fake`: deterministic fake model for CI and smoke tests.

## Eval Loop

Run replay evals without paid agent calls:

```bash
uv run python scripts/eval_agent_sessions.py --driver replay
```

Run a focused subroutine benchmark:

```bash
uv run python scripts/benchmark_mcp_subroutines.py
```

Real agent-session deltas should compare:

- `no_mcp`
- `mcp_deterministic_only`
- `mcp_local_slm`
- `mcp_cloud_slm`
- `mcp_cloud_slm_frontier_fallback`

Primary metrics are success rate, schema-valid rate, verifier-pass rate, escalation/fallback rate, tool/MCP call count, latency p50/p95, token savings estimate, and cost per successful task.
