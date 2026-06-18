
"""Research-derived specialist subroutine helpers for the MCP runtime."""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

HF_SUBROUTINE_MODELS: dict[str, str] = {
    "json_repair": "ishaanranjan/slm-agent-json-repair-smollm2-360m",
    "trace_localize": "ishaanranjan/slm-agent-trace-localizer-qwen2-5-0-5b",
    "search_query": "ishaanranjan/slm-agent-search-query-gen-qwen2-5-0-5b",
    "search_rank": "ishaanranjan/slm-agent-search-hit-ranker-qwen2-5-0-5b",
    "failure_classify": "local/failure-classifier-rules-v1",
    "patch_risk": "local/patch-risk-classifier-rules-v1",
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

_FAILURE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (r"\b(modulenotfounderror|importerror|cannot find module|no module named)\b", "dependency_issue", 0.9),
    (r"\b(syntaxerror|unexpected token|invalid syntax|parse error)\b", "syntax_error", 0.9),
    (r"\b(typeerror|mypy|ts\d{4}|is not assignable|has no attribute)\b", "type_error", 0.84),
    (r"\b(fixture .* not found|unknown fixture|missing fixture)\b", "missing_fixture", 0.88),
    (r"\b(flaky|timed out|timeout|race condition|intermittent)\b", "flaky_test", 0.76),
    (r"\b(blank screen|white screen|hydration|console error|not visible|overlap)\b", "frontend_render_issue", 0.82),
    (r"\b(network unreachable|permission denied|sandbox|econnrefused|dns|forbidden)\b", "sandbox_network_issue", 0.8),
    (r"\b(assertionerror|expected .* actual|assert .*\bfailed\b)\b", "assertion_failure", 0.78),
)

_PATCH_RISK_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(auth|authentication|authorization|oauth|jwt|session|cookie)\b", "auth"),
    (r"\b(payment|billing|invoice|checkout|stripe|paypal)\b", "billing"),
    (r"\b(password|secret|token|private key|credential|api[_-]?key)\b", "secrets"),
    (r"\b(drop table|delete from|truncate|migration|schema)\b", "database"),
    (r"\b(subprocess|shell=True|eval\(|exec\(|os\.system|rm -rf)\b", "command_execution"),
    (r"\b(permission|sandbox|policy|admin|role)\b", "permissions"),
)

_CLASSIFIER_CACHE: dict[str, tuple[Any, Any, Any, str]] = {}


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


def classify_failure_subroutine(text: str) -> SubroutineRun:
    started = time.perf_counter()
    model_id = HF_SUBROUTINE_MODELS["failure_classify"]
    model_dir = os.environ.get("SLM_HARNESS_FAILURE_CLASSIFIER_MODEL_DIR")
    prediction = (
        _predict_text_classifier(model_dir, f"TASK: classify software failure\nLOG_OR_TRACE:\n{text}")
        if model_dir
        else None
    )
    if prediction is not None:
        label, confidence, loaded_model_id = prediction
        output = {
            "category": label,
            "confidence": confidence,
            "rationale": "trained classifier prediction",
        }
        model_id = loaded_model_id
        fallback_path = "trained_failure_classifier"
    else:
        output = classify_failure_rules(text)
        fallback_path = "deterministic_failure_rules"
    success = output["category"] != "unknown"
    return _run(
        "failure_classify",
        output,
        True,
        True,
        fallback_path,
        model_id,
        started,
        [{"stage": fallback_path, "ok": success}],
    )


def classify_patch_risk_subroutine(diff: str, *, tests: str = "") -> SubroutineRun:
    started = time.perf_counter()
    model_id = HF_SUBROUTINE_MODELS["patch_risk"]
    output = classify_patch_risk_rules(diff, tests=tests)
    fallback_path = "deterministic_patch_risk_rules"
    model_dir = os.environ.get("SLM_HARNESS_PATCH_RISK_MODEL_DIR")
    text = f"TASK: classify patch risk\nDIFF:\n{diff}\nTEST_EVIDENCE:\n{tests}"
    prediction = _predict_text_classifier(model_dir, text) if model_dir else None
    if prediction is not None:
        label, confidence, loaded_model_id = prediction
        output = {
            **output,
            "risk_level": label,
            "confidence": confidence,
            "tests_needed": label != "low" or output["tests_needed"],
            "human_review_needed": label == "high" or output["human_review_needed"],
            "rationale": "trained classifier risk prediction with deterministic subsystem extraction",
        }
        model_id = loaded_model_id
        fallback_path = "trained_patch_risk_classifier"
    return _run(
        "patch_risk",
        output,
        True,
        True,
        fallback_path,
        model_id,
        started,
        [{"stage": fallback_path, "ok": True}],
    )


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


def classify_failure_rules(text: str) -> dict[str, Any]:
    lowered = text.lower()
    for pattern, category, confidence in _FAILURE_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return {
                "category": category,
                "confidence": confidence,
                "rationale": f"matched {category} failure signature",
            }
    return {
        "category": "unknown",
        "confidence": 0.42 if text.strip() else 0.0,
        "rationale": "no high-confidence failure signature matched",
    }


def classify_patch_risk_rules(diff: str, *, tests: str = "") -> dict[str, Any]:
    combined = f"{diff}\n{tests}".lower()
    affected: list[str] = []
    for pattern, subsystem in _PATCH_RISK_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE) and subsystem not in affected:
            affected.append(subsystem)

    added_removed = len(re.findall(r"^[+-](?![+-])", diff, re.MULTILINE))
    passed = bool(re.search(r"\b(0 failed|passed|all tests pass|lint passed|typecheck passed)\b", tests, re.I))
    failed = bool(re.search(r"\b(failed|traceback|exception|error:|npm err)\b", tests, re.I))

    if "secrets" in affected or "command_execution" in affected:
        risk = "high"
    elif failed or "auth" in affected or "billing" in affected or added_removed > 120:
        risk = "high"
    elif affected or added_removed > 30:
        risk = "medium"
    else:
        risk = "low"

    tests_needed = risk != "low" or not passed
    human_review_needed = risk == "high" or ("database" in affected and not passed)
    confidence = {"low": 0.76, "medium": 0.81, "high": 0.88}[risk]
    return {
        "risk_level": risk,
        "confidence": confidence,
        "affected_subsystems": affected or ["general"],
        "tests_needed": tests_needed,
        "human_review_needed": human_review_needed,
        "rationale": "risk based on diff-sensitive subsystem markers and test evidence",
    }


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


def _predict_text_classifier(model_dir: str | None, text: str) -> tuple[str, float, str] | None:
    if not model_dir:
        return None
    try:
        tokenizer, model, torch, device = _load_text_classifier(model_dir)
        encoded = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1)[0]
        score, index = torch.max(probs, dim=-1)
        label = model.config.id2label.get(int(index.item()), str(int(index.item())))
        return str(label), float(score.item()), model_dir
    except Exception:
        return None


def _load_text_classifier(model_dir: str) -> tuple[Any, Any, Any, str]:
    cached = _CLASSIFIER_CACHE.get(model_dir)
    if cached is not None:
        return cached
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for trained subroutine classifiers") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    loaded = (tokenizer, model, torch, device)
    _CLASSIFIER_CACHE[model_dir] = loaded
    return loaded
