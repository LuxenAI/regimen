"""Tests for research-derived SLM subroutine executors."""

from __future__ import annotations

import json

import pytest

from openharness.orchestration.engine import build_default_orchestration_engine
from openharness.orchestration.subroutine_models import (
    classify_failure_subroutine,
    classify_patch_risk_subroutine,
    generate_search_queries_subroutine,
    localize_traceback_subroutine,
    rank_search_hits_subroutine,
    repair_json_subroutine,
)


def test_valid_json_does_not_invoke_slm() -> None:
    run = repair_json_subroutine('{"action":"SEARCH","pattern":"foo"}')

    assert run.success is True
    assert run.output["action"] == "SEARCH"
    assert run.fallback_path == "deterministic_parse"


def test_malformed_json_invokes_slm_fallback_and_returns_valid_json() -> None:
    run = repair_json_subroutine("{'action': 'SEARCH', 'pattern': 'foo',}")

    assert run.success is True
    assert run.output == {"action": "SEARCH", "pattern": "foo"}
    assert run.fallback_path == "schema_slm_fallback"


def test_unrecoverable_json_fails_cleanly() -> None:
    run = repair_json_subroutine("not json at all", allow_slm=True)

    assert run.success is False
    assert run.schema_valid is False
    assert run.error


def test_traceback_localizer_returns_required_schema() -> None:
    tb = (
        "Traceback (most recent call last):\n"
        "  File \"/usr/lib/python3.11/runpy.py\", line 10, in <module>\n"
        "    run()\n"
        "  File \"/workspace/app/pkg/service.py\", line 42, in handle\n"
        "    user.name.lower()\n"
        "AttributeError: 'NoneType' object has no attribute 'name'\n"
    )

    run = localize_traceback_subroutine(tb, project_prefix="/workspace/app/")

    assert run.success is True
    assert run.output["likely_file"] == "pkg/service.py"
    assert run.output["likely_symbol"] == "handle"
    assert run.output["category"] == "null_attribute"


def test_search_query_generator_returns_non_empty_query_list() -> None:
    run = generate_search_queries_subroutine("Find the retry budget helper function")

    assert run.success is True
    assert run.output["queries"]


def test_search_hit_ranker_returns_stable_ranking() -> None:
    hits = [
        {"id": "import", "path": "pkg/a.py", "line": 1, "text": "from pkg.service import retry_budget"},
        {"id": "def", "path": "pkg/service.py", "line": 20, "text": "def retry_budget(config):"},
        {"id": "call", "path": "pkg/main.py", "line": 8, "text": "retry_budget(cfg)"},
    ]

    run = rank_search_hits_subroutine("retry_budget", hits)

    assert run.success is True
    assert run.output["ranked_hit_ids"][0] == "def"


def test_failure_classifier_returns_category() -> None:
    run = classify_failure_subroutine("ModuleNotFoundError: No module named 'rich'")

    assert run.success is True
    assert run.output["category"] == "dependency_issue"


def test_patch_risk_classifier_flags_auth_risk() -> None:
    run = classify_patch_risk_subroutine("+ return jwt.decode(token, SECRET)\n")

    assert run.success is True
    assert run.output["risk_level"] == "high"
    assert run.output["human_review_needed"] is True


@pytest.mark.asyncio
async def test_run_task_works_for_new_task_types() -> None:
    engine = build_default_orchestration_engine()
    trace = await engine.run_goal(
        "Repair this broken json",
        task_type="json_repair",
        context_data={"raw": "{'action': 'SEARCH', 'pattern': 'foo'}"},
    )

    assert trace.results[0].executor_name == "local.json_repair_slm"
    assert trace.results[0].output["success"] is True


@pytest.mark.asyncio
async def test_mcp_server_exposes_subroutine_tools() -> None:
    from openharness.orchestration.mcp_server import _run_subroutine_tool, create_server

    create_server()
    payload = await _run_subroutine_tool(
        "Generate search queries",
        "search_query",
        {"task": "Find retry budget helper"},
    )

    assert payload["decision"]["selected_executor"] == "local.search_query_gen_slm"
    assert payload["result"]["output"]["success"] is True
    json.dumps(payload)
