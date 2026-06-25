"""Unit tests for the lexical code-context retriever and the schema fix."""

from __future__ import annotations

from openharness.orchestration.code_retriever import (
    CodeRetriever,
    SymbolEntry,
    _dp_segment,
    build_corpus_splitter,
    light_stem,
    split_identifier,
    tokenize_query,
)


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #

def test_split_identifier_snake_and_camel() -> None:
    assert split_identifier("_is_https_redirect") == ["is", "https", "redirect"]
    assert split_identifier("GetSpontaneousEnvironment") == [
        "get",
        "spontaneous",
        "environment",
    ]
    # leading underscores stripped, single-char tokens dropped
    assert split_identifier("_x") == []


def test_light_stem_normalizes_inflections() -> None:
    # query verb and identifier root collapse to the same stem
    assert light_stem("builds") == light_stem("build")
    assert light_stem("loads") == light_stem("load")
    assert light_stem("compiles") == light_stem("compile")
    assert light_stem("tokenizes") == light_stem("tokenize")


def test_split_identifier_applies_stemmer() -> None:
    toks = split_identifier("load_extensions", stem=light_stem)
    # "extensions" -> stemmed; "load" stays
    assert "load" in toks
    assert all(t == light_stem(t) for t in toks)


def test_tokenize_query_drops_stopwords() -> None:
    toks = tokenize_query("find the function that loads the template")
    assert "find" not in toks and "the" not in toks and "function" not in toks
    assert "load" in toks or "loads" in toks
    assert "template" in toks


# --------------------------------------------------------------------------- #
# Compound splitting
# --------------------------------------------------------------------------- #

def test_dp_segment_splits_known_compound() -> None:
    costs = {"query": 1.0, "params": 1.0}
    assert _dp_segment("queryparams", costs, maxlen=6) == ["query", "params"]
    # no valid full segmentation -> None
    assert _dp_segment("zzqxweird", costs, maxlen=6) is None


def test_corpus_splitter_splits_fused_token() -> None:
    # a corpus where "query" and "params" are frequent atomic tokens
    atomic = ["query"] * 10 + ["params"] * 10 + ["merge"] * 5 + ["queryparams"]
    splitter, name = build_corpus_splitter(atomic)
    assert name in {"wordninja", "corpus_dp"}
    assert splitter("queryparams") == ["query", "params"]
    # a real single word should not be shredded
    assert splitter("redirect") == ["redirect"]


# --------------------------------------------------------------------------- #
# Retrieval on a tiny fixture
# --------------------------------------------------------------------------- #

def _fixture_entries() -> list[SymbolEntry]:
    return [
        SymbolEntry("_is_https_redirect", "httpx/_client.py",
                    "Return True if a redirect points at an https url.", "FunctionDef"),
        SymbolEntry("_merge_queryparams", "httpx/_client.py",
                    "Merge query parameters into the request url.", "FunctionDef"),
        SymbolEntry("load_extensions", "jinja2/environment.py",
                    "Load the extensions for this environment.", "FunctionDef"),
        SymbolEntry("test_redirects", "tests/test_redirects.py",
                    "Exercise redirect handling end to end.", "FunctionDef"),
        SymbolEntry("unrelated_helper", "pkg/util.py",
                    "Format a number with thousands separators.", "FunctionDef"),
    ]


def test_retriever_ranks_correct_symbol_first() -> None:
    retr = CodeRetriever(stemmer=light_stem).fit(_fixture_entries())
    top = retr.query("merge query parameters into the request", top_k=3)
    assert top[0].entry.name == "_merge_queryparams"


def test_retriever_recall_at_3_on_fixture() -> None:
    retr = CodeRetriever(stemmer=light_stem).fit(_fixture_entries())
    cases = [
        ("find where a redirect is checked for https", "_is_https_redirect"),
        ("load the environment extensions", "load_extensions"),
    ]
    hits = 0
    for query, gt in cases:
        names = [r.entry.name for r in retr.query(query, top_k=3)]
        hits += gt in names
    assert hits == len(cases)


def test_test_file_penalty_demotes_tests() -> None:
    entries = _fixture_entries()
    # query that matches both the impl and the test by docstring vocabulary
    strong = CodeRetriever(stemmer=light_stem, test_penalty=0.1).fit(entries)
    weak = CodeRetriever(stemmer=light_stem, test_penalty=1.0).fit(entries)
    q = "redirect handling"
    strong_top = strong.query(q, top_k=5)[0].entry.file_path
    # with a heavy test penalty the non-test site should not be a test file
    assert not strong_top.startswith("tests/")
    # sanity: both return results
    assert weak.query(q, top_k=5)


# --------------------------------------------------------------------------- #
# Schema-validator fix (from the search_query_gen investigation)
# --------------------------------------------------------------------------- #

def test_schema_valid_rejects_dict_items_for_string_array() -> None:
    from openharness.orchestration.slm_runner import _schema_valid

    schema = {
        "type": "object",
        "required": ["queries", "confidence"],
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
    }
    # the search_query_gen placeholder-collapse output: items are dicts, not strings
    assert _schema_valid({"queries": [{"pattern": "x"}], "confidence": 0.9}, schema) is False
    # correctly typed output passes
    assert _schema_valid({"queries": ["foo_bar"], "confidence": 0.9}, schema) is True
    # missing required key still fails
    assert _schema_valid({"confidence": 0.9}, schema) is False
    # empty list is valid
    assert _schema_valid({"queries": [], "confidence": 0.9}, schema) is True
