
"""Research-derived specialist subroutine helpers for the MCP runtime."""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

HF_SUBROUTINE_MODELS: dict[str, str] = {
    "json_repair": "ishaanranjan/slm-agent-json-repair-smollm2-360m",
    "trace_localize": "ishaanranjan/slm-agent-trace-localizer-qwen2-5-0-5b",
    "search_query": "ishaanranjan/slm-agent-search-query-gen-qwen2-5-0-5b",
    "search_rank": "ishaanranjan/slm-agent-search-hit-ranker-qwen2-5-0-5b",
}

_CATEGORY_BY_EXCEPTION = {
    "AttributeError": "null_attribute",
    "KeyError": "missing_key",
    "TypeError": "type_mismatch",
    "IndexError": "bad_index",
    "ValueError": "bad_value",
    "ImportError": "import_error",
    "ModuleNotFoundError": "import_error",
    "ZeroDivisionError": "arithmetic",
    "SyntaxError": "syntax_error",
}

@dataclass
class SubroutineRun:
    """Normalized result from a specialist subroutine."""

    subroutine: str
    output: dict[str, Any]
    success: bool
    schema_valid: bool
    verifier_pass: bool
    fallback_path: str
    model_id: str
    local: bool = True
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def as_output(self) -> dict[str, Any]:
        return {
            "subroutine": self.subroutine,
            "output": self.output,
            "success": self.success,
            "schema_valid": self.schema_valid,
            "verifier_pass": self.verifier_pass,
            "fallback_path": self.fallback_path,
            "model_id": self.model_id,
            "local": self.local,
            "estimated_cost_usd": self.estimated_cost_usd,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "attempts": self.attempts,
        }


def repair_json_subroutine(raw: str, *, allow_slm: bool = True, allow_frontier: bool = False) -> SubroutineRun:
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    model_id = HF_SUBROUTINE_MODELS["json_repair"]

    parsed, err = _strict_json(raw)
    attempts.append({"stage": "deterministic_parse", "ok": parsed is not None, "error": err})
    if parsed is not None:
        return _run("json_repair", parsed, True, True, "deterministic_parse", model_id, started, attempts)

    candidate = _extract_json_object(raw)
    if candidate != raw:
        parsed, err = _strict_json(candidate)
        attempts.append({"stage": "deterministic_extract", "ok": parsed is not None, "error": err})
        if parsed is not None:
            return _run("json_repair", parsed, True, True, "deterministic_extract", model_id, started, attempts)

    if allow_slm:
        repaired = _slm_like_json_repair(candidate)
        attempts.append({"stage": "schema_slm_fallback", "ok": repaired is not None})
        if repaired is not None:
            return _run("json_repair", repaired, True, True, "schema_slm_fallback", model_id, started, attempts)

    if allow_frontier:
        attempts.append({"stage": "frontier_stub", "ok": False, "error": "no live frontier API configured"})
        return _run("json_repair", {}, False, False, "frontier_stub", "frontier.stub", started, attempts, "frontier stub not live")

    return _run("json_repair", {}, False, False, "failed", model_id, started, attempts, "could not repair JSON")


def localize_traceback_subroutine(traceback: str, project_prefix: str = "") -> SubroutineRun:
    started = time.perf_counter()
    model_id = HF_SUBROUTINE_MODELS["trace_localize"]
    output = localize_traceback_rules(traceback, project_prefix=project_prefix)
    success = bool(output.get("likely_file"))
    return _run("trace_localize", output, success, success, "deterministic_trace_rules", model_id, started, [{"stage": "deterministic_trace_rules", "ok": success}], None if success else "no project frame found")


def generate_search_queries_subroutine(task: str) -> SubroutineRun:
    started = time.perf_counter()
    model_id = HF_SUBROUTINE_MODELS["search_query"]
    queries = generate_search_queries_rules(task)
    success = bool(queries)
    output = {"queries": queries, "confidence": 0.74 if success else 0.0}
    return _run("search_query", output, success, success, "deterministic_query_rules", model_id, started, [{"stage": "deterministic_query_rules", "ok": success}], None if success else "no query terms found")


def rank_search_hits_subroutine(query: str, hits: list[dict[str, Any]]) -> SubroutineRun:
    started = time.perf_counter()
    model_id = HF_SUBROUTINE_MODELS["search_rank"]
    ranked = rank_search_hits_rules(query, hits)
    success = len(ranked) == len(hits) and bool(ranked)
    output = {"ranked_hit_ids": ranked, "confidence": 0.82 if success else 0.0}
    return _run("search_rank", output, success, success, "deterministic_rank_rules", model_id, started, [{"stage": "deterministic_rank_rules", "ok": success}], None if success else "no hits supplied")


def localize_traceback_rules(traceback: str, *, project_prefix: str = "") -> dict[str, Any]:
    last_chain = traceback.split("During handling of the above exception")[-1]
    frames = re.findall(r'File "([^"]+)", line (\d+), in ([^\n]+)', last_chain)
    candidates: list[tuple[str, int, str]] = []
    for path, line, symbol in frames:
        if project_prefix and path.startswith(project_prefix):
            candidates.append((path[len(project_prefix):], int(line), symbol.strip()))
        elif _looks_project_path(path):
            candidates.append((path, int(line), symbol.strip()))
    if not candidates:
        return {"likely_file": "", "likely_symbol": "", "line": 0, "category": "unknown", "confidence": 0.0, "rationale": "no project frame found"}
    path, line, symbol = candidates[-1]
    exc = _extract_exception(last_chain)
    category = _CATEGORY_BY_EXCEPTION.get(exc, "unknown")
    return {"likely_file": path, "likely_symbol": symbol, "line": line, "category": category, "confidence": 0.86, "rationale": "selected deepest project frame in final traceback chain"}


def generate_search_queries_rules(task: str) -> list[str]:
    stop = {"find", "the", "that", "this", "in", "is", "a", "an", "of", "for", "with", "where", "which", "file", "files", "repo", "repository", "function", "class", "constant", "module", "defined", "declared", "implemented", "locate", "located", "i", "need", "used", "to", "handles", "handling", "responsible", "routine", "helper", "type", "global", "setting", "please", "show", "me", "code"}
    words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", task) if w.lower() not in stop]
    seen: list[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    if not seen:
        return []
    snake = "_".join(w.lower() for w in seen[:5])
    camel = "".join(w[:1].upper() + w[1:] for w in seen[:5])
    queries = [snake]
    if len(seen) == 1:
        queries.append(rf"(def|class)\s+{re.escape(seen[0])}\b")
    else:
        queries.append(".*".join(re.escape(w) for w in seen[:4]))
        queries.append(camel)
    return [q for q in queries if q]


def rank_search_hits_rules(query: str, hits: list[dict[str, Any]]) -> list[Any]:
    scored: list[tuple[float, Any]] = []
    q = re.escape(query)
    for index, hit in enumerate(hits):
        hit_id = hit.get("id", hit.get("i", index))
        text = str(hit.get("text", ""))
        path = str(hit.get("path", ""))
        line = int(hit.get("line", 0) or 0)
        score = 0.0
        if re.search(rf"(async\s+)?def\s+{q}\b|class\s+{q}\b|{q}\s*(:[^=]*)?=", text):
            score += 10.0
        if query and query in text:
            score += 2.0
        if re.match(r"\s*(from|import)\s", text):
            score -= 4.0
        if "test" in path.lower():
            score -= 2.0
        score -= line * 1e-6
        scored.append((score, hit_id))
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [hit_id for _, hit_id in scored]


def _run(subroutine: str, output: dict[str, Any], success: bool, schema_valid: bool, fallback_path: str, model_id: str, started: float, attempts: list[dict[str, Any]], error: str | None = None) -> SubroutineRun:
    latency_ms = max(0, int((time.perf_counter() - started) * 1000))
    return SubroutineRun(subroutine=subroutine, output=output, success=success, schema_valid=schema_valid, verifier_pass=success and schema_valid, fallback_path=fallback_path, model_id=model_id, latency_ms=latency_ms, error=error, attempts=attempts)


def _strict_json(raw: str) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "payload is not a JSON object"
    return payload, ""


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _slm_like_json_repair(raw: str) -> dict[str, Any] | None:
    candidate = _extract_json_object(raw)
    try:
        payload = ast.literal_eval(candidate)
        if isinstance(payload, dict):
            return _jsonable_dict(payload)
    except Exception:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    repaired = repaired.replace("None", "null").replace("True", "true").replace("False", "false")
    parsed, _ = _strict_json(repaired)
    return parsed


def _jsonable_dict(payload: dict[Any, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    if not isinstance(normalized, dict):
        return {}
    return {str(key): value for key, value in normalized.items()}


def _looks_project_path(path: str) -> bool:
    lowered = path.lower()
    blocked = ("/usr/lib/", "site-packages", "dist-packages", "<frozen", "python3.")
    return not any(part in lowered for part in blocked) and (path.endswith(".py") or "/" in path)


def _extract_exception(traceback: str) -> str:
    for line in reversed(traceback.strip().splitlines()):
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", line.strip())
        if match:
            return match.group(1)
    return ""
