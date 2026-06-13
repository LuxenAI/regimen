"""Tests for SLM provider config and runner behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import httpx

from openharness.orchestration.slm_config import load_slm_harness_config
from openharness.orchestration.slm_runner import ConfiguredSlmRunner, SlmRunRequest


def test_load_config_file_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / ".slm-harness.toml"
    config_path.write_text(
        "\n".join(
            [
                "[defaults]",
                'backend = "deterministic"',
                "timeout_ms = 1234",
                "",
                "[subroutines.json_repair]",
                'backend = "remote_http"',
                'remote_url = "https://models.example.test/infer"',
                "cost_per_call_usd = 0.001",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLM_HARNESS_JSON_REPAIR_BACKEND", "fake")

    config = load_slm_harness_config(cwd=tmp_path)

    route = config.route_for("json_repair")
    assert route.backend == "fake"
    assert route.remote_url == "https://models.example.test/infer"
    assert route.timeout_ms == 1234
    assert route.cost_per_call_usd == 0.001


@pytest.mark.asyncio
async def test_configured_runner_uses_fake_after_deterministic_json_failure() -> None:
    response = await ConfiguredSlmRunner().run(
        SlmRunRequest(
            task_type="json_repair",
            input={"raw": "{'action': 'SEARCH', 'pattern': 'foo',}"},
            schema={"type": "object", "required": ["action", "pattern"]},
        )
    )

    assert response.backend == "fake"
    assert response.schema_valid is True
    assert response.output == {"action": "SEARCH", "pattern": "foo"}
    assert len(response.metadata["runner_attempts"]) == 2


@pytest.mark.asyncio
async def test_remote_http_runner_accepts_slm_response_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / ".slm-harness.toml"
    config_path.write_text(
        "\n".join(
            [
                "[subroutines.search_query]",
                'backend = "remote_http"',
                'remote_url = "https://models.example.test/infer"',
                'fallback_policy = "model_only"',
            ]
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "output": {"queries": ["retry_budget"], "confidence": 0.91},
                "schema_valid": True,
                "confidence": 0.91,
                "model_id": "fake-remote",
                "raw_text": json.dumps({"queries": ["retry_budget"]}),
            }

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args

        async def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
            del args, kwargs
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    config = load_slm_harness_config(cwd=tmp_path)

    response = await ConfiguredSlmRunner(config=config).run(
        SlmRunRequest(
            task_type="search_query",
            input={"task": "Find retry budget helper"},
            schema={"type": "object", "required": ["queries", "confidence"]},
        )
    )

    assert response.backend == "remote_http"
    assert response.model_id == "fake-remote"
    assert response.output["queries"] == ["retry_budget"]
