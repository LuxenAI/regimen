# slm-harness orchestration

`slm-harness` adds a local-first orchestration layer on top of OpenHarness. It decomposes agent workflows into typed subtasks, routes each subtask to the cheapest reliable executor, verifies the output, escalates when confidence is low, and records cost/latency/escalation telemetry.

## Runtime shape

```text
Codex / Claude Code
  -> slm-harness MCP server
      -> typed subtask decomposer
      -> cheapest-reliable router
      -> deterministic executors
      -> classifier / tiny-SLM adapters
      -> frontier OpenHarness LLM handoff
      -> trace store
```

The default executors are CPU-local placeholders with production-shaped contracts. Real `<10M` models can replace `local.tiny_slm_stub` or add new executors without changing the MCP tools.

Candidate SLM checkpoints and redundancy notes are tracked in
[MODEL_CATALOG.md](MODEL_CATALOG.md). The linked Hugging Face collection is useful for
developer-agent subroutine experiments, but those checkpoints start at 135M parameters, so they
are optional SLM executors rather than replacements for strict `<10M` classifier work.

## Codex setup

Print the local config snippet:

```bash
oh mcp codex-config
```

Add the printed block to `~/.codex/config.toml` or a trusted project `.codex/config.toml`. Codex will start the stdio MCP server and expose these tools:

- `slm_list_executors`
- `slm_decompose_workflow`
- `slm_route_task`
- `slm_run_task`
- `slm_verify_result`
- `slm_get_trace`
- `slm_codex_probe`

For local testing, run the server directly:

```bash
oh mcp serve-orchestrator
```

## What to replace next

Add real executors behind the existing contracts:

- tiny classifier/router model
- prompt-injection classifier
- extraction model
- result verifier
- live OpenHarness `QueryEngine` frontier executor

The router only needs each executor's supported task types, reliability estimate, cost, latency, and local/remote flag.
