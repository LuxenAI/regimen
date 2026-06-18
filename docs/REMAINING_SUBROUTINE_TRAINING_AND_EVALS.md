# Remaining subroutine training and eval methodology

This plan turns the two rule-only runtime subroutines into trained classifiers:

- `failure_classify`: classify logs, test output, stack traces, and browser/CI errors.
- `patch_risk`: classify whether a proposed diff is low, medium, or high risk, while retaining deterministic subsystem extraction for `auth`, `billing`, `secrets`, `database`, command execution, and permissions.

## GPU choice

Use one Lambda Labs A100 80GB instance for the first full run. H100 is faster, but A100 80GB is enough for CodeBERT-sized classifiers, leaves room for DeBERTa/StarEncoder ablations, and should be cheaper per iteration. Use H100 only if we expand into full generative SFT for both subroutines or run many ablations in parallel.

Runbook:

```bash
git clone <repo-url> slm-harness
cd slm-harness
export TEACHER_JSONL=data/subroutines/remaining_teacher.jsonl
bash scripts/lambda_train_remaining_subroutines.sh
```

The script writes:

- `models/trained/failure_classifier_v1/model`
- `models/trained/patch_risk_classifier_v1/model`
- `artifacts/results/remaining_subroutines_training.json`
- `artifacts/results/remaining_subroutines_benchmark.json`
- refreshed MCP benchmark artifacts from `scripts/benchmark_mcp_subroutines.py`

Activate trained runtime classifiers locally:

```bash
export SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR=models/trained/failure_classifier_v1/model
export SLM_HARNESS_PATCH_RISK_MODEL_DIR=models/trained/patch_risk_classifier_v1/model
slm-harness-mcp
```

## Training data

Use mixed JSONL teacher data so both generated and human/audit labels can land in one file.

`failure_classify` accepted fields:

```json
{"task_type":"failure_classify","text":"ModuleNotFoundError: No module named 'rich'","label":"dependency_issue"}
```

Labels:

- `dependency_issue`
- `syntax_error`
- `type_error`
- `missing_fixture`
- `flaky_test`
- `frontend_render_issue`
- `sandbox_network_issue`
- `assertion_failure`
- `unknown`

`patch_risk` accepted fields:

```json
{"task_type":"patch_risk","diff":"+ return jwt.decode(token, SECRET)\n","tests":"not run","label":"high"}
```

Labels:

- `low`
- `medium`
- `high`

Recommended teacher sources:

- Existing OpenHarness traces, CI logs, pytest/npm failures, MCP tool failures, and browser screenshot summaries.
- SWE-Gym/SWE-smith style task trajectories where tests, edits, and outcomes are available.
- Frontier-teacher labels from Codex/Claude/Mistral agents, but only after normalizing to the fixed schemas above.
- Human spot-checks for high-risk labels, especially auth, billing, secrets, database, command execution, and permission diffs.

## Evaluation matrix

Report both subroutine-local and downstream harness metrics.

Subroutine-local metrics:

- Accuracy and macro F1 for `failure_classify` category and `patch_risk` risk level.
- Per-label precision/recall/F1, because high-risk false accepts matter more than raw accuracy.
- Confusion matrix, especially `high -> low`, `dependency_issue -> unknown`, and `flaky_test -> assertion_failure`.
- p50/p95 latency and GPU memory.
- Rule baseline delta, using `scripts/benchmark_remaining_subroutines.py`.

Harness metrics:

- Success rate, schema-valid rate, verifier-pass rate.
- Escalation/fallback rate.
- Tool/MCP call count where available.
- Latency p50/p95.
- Cost per task and cost per successful task.
- Downstream issue resolution on SWE-bench-style subsets when connected to a real agent scaffold.

## Literature review implications

The core lesson from frontier coding-agent evaluation is that subroutines should not be judged only by isolated label accuracy. They should also be judged by whether they improve an agent's end-to-end ability to resolve real tasks with fewer calls, lower cost, and fewer unsafe accepts.

Frontier-agent reporting pattern:

- OpenAI/Codex reports coding-agent performance with SWE-bench Verified or SWE-Bench Pro, Terminal-Bench, token/tool-call efficiency, and internal refactoring-style evaluations.
- Anthropic/Claude Code emphasizes that an agent eval measures the harness and model together, then reports SWE-bench, Terminal-Bench, latency/cost, and task-resolution behavior.
- Mistral/Devstral reports SWE-bench Verified under a named agent scaffold such as OpenHands, which is the right comparison style for our MCP harness because scaffold quality changes the measured result.

| Source | What it evaluates | Metrics to reuse here |
| --- | --- | --- |
| Codex / HumanEval | Functional correctness for code synthesis from docstrings. | `pass@k`, executable tests, repeated sampling curves. |
| SWE-bench and SWE-bench Verified | Real GitHub issue resolution with repo context and tests. | Resolved rate, pass-to-pass/fail-to-pass tests, human-validated task quality. |
| SWE-agent | Agent-computer interface for repo navigation, editing, and testing. | `pass@1` on SWE-bench/HumanEvalFix, interface ablations, action/error analysis. |
| SWE-Gym | Training agents and verifiers from executable SWE trajectories. | Resolve-rate lift, verifier best-of-N, training/inference scaling curves. |
| SWE-smith | Synthetic executable task generation for SWE agents. | Dataset scale, generated task validity, downstream SWE-bench Verified `pass@1`. |
| R2E-Gym / AgentGym | Procedural executable environments plus hybrid verifiers. | Execution-based verifier score, execution-free verifier score, hybrid best-of-N. |
| Terminal-Bench and OSWorld | Long-horizon terminal/computer-use agent tasks. | Task success, custom verification scripts, completion steps, error categories. |
| AgentBench | Multi-environment interactive agent benchmark. | Task success and categorized failure reasons. |
| BFCL | Function/tool-call correctness. | AST accuracy, executable accuracy, relevance/abstention accuracy. |
| tau-bench | Tool-agent-user conversations with stateful APIs. | Final database-state success and `pass^k` consistency over repeated trials. |
| RepoBench | Repository-level retrieval and completion. | Retrieval top-k/MRR-style ranking, completion accuracy, pipeline accuracy. |
| c-CRAB / CR-Bench / security evals | Code review, vulnerability, and patch-quality assessment. | Precision/recall on review findings, severity-aware false accept rate, executable tests for review claims. |

## How this maps to our two classifiers

`failure_classify` should borrow from SWE-agent, SWE-Gym, Terminal-Bench, and AgentBench:

- Train on real failure traces and agent trajectories.
- Evaluate category accuracy plus actionability: does the predicted category route the agent to the right next step?
- Report confusion and downstream rerun success after the agent receives the category.

`patch_risk` should borrow from SWE-bench, BFCL, tau-bench, c-CRAB, and security benchmarks:

- Treat false-low risk as the primary safety failure.
- Keep deterministic subsystem extraction even after training, because explicit risk markers are useful review evidence.
- Evaluate with severity-weighted precision/recall and downstream acceptance/escalation behavior.

## Source links

- Codex / HumanEval: https://arxiv.org/abs/2107.03374
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-bench Verified: https://openai.com/index/introducing-swe-bench-verified/
- OpenAI on SWE-bench Verified contamination and SWE-bench Pro: https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- OpenAI GPT-5 developer benchmarks: https://openai.com/index/introducing-gpt-5-for-developers/
- OpenAI Codex upgrades: https://openai.com/index/introducing-upgrades-to-codex/
- OpenAI GPT-5.3-Codex benchmarks: https://openai.com/index/introducing-gpt-5-3-codex/
- SWE-agent: https://arxiv.org/abs/2405.15793
- SWE-Gym: https://arxiv.org/abs/2412.21139
- SWE-smith: https://arxiv.org/abs/2504.21798
- R2E-Gym / AgentGym: https://arxiv.org/abs/2504.07164
- Terminal-Bench: https://arxiv.org/abs/2601.11868
- OSWorld: https://arxiv.org/abs/2404.07972
- AgentBench: https://arxiv.org/abs/2308.03688
- BFCL: https://proceedings.mlr.press/v267/patil25a.html
- tau-bench: https://arxiv.org/abs/2406.12045
- RepoBench: https://arxiv.org/abs/2306.03091
- LiveCodeBench: https://arxiv.org/abs/2403.07974
- Anthropic agent eval guidance: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Anthropic Claude SWE-bench post: https://www.anthropic.com/research/swe-bench-sonnet
- Mistral Devstral: https://mistral.ai/news/devstral/
- Devstral technical report: https://arxiv.org/html/2509.25193v1
- c-CRAB code review benchmark: https://arxiv.org/html/2603.23448v2
- JIT vulnerability benchmark: https://aclanthology.org/2025.acl-long.1490.pdf
