"""Lexical code-symbol retriever for context reduction.

A dependency-light retriever that ranks function/class definitions against a
natural-language query. Built to reduce the amount of source a coding agent has
to read before answering a "where is X / find the thing that does Y" question.

Pipeline (all deterministic, no model):

1. Tokenize identifiers by splitting snake_case and camelCase, then optionally
   segment fused compound tokens ("queryparams" -> "query" + "params").
2. Stem index and query tokens with the same stemmer so inflected query verbs
   ("compiles") match identifier roots ("compile").
3. Rank candidates with BM25 over (name tokens + docstring tokens).
4. Apply definition-site weighting: prefer real def/class sites, down-weight
   test files which are usually not the implementation a search is after.

The stemmer and compound splitter are pluggable. Defaults degrade gracefully:
the stemmer uses NLTK's PorterStemmer when importable and a light deterministic
stemmer otherwise; compound splitting uses ``wordninja`` when importable and a
corpus-frequency DP segmenter otherwise. The active strategy is recorded on the
retriever (``stemmer_name`` / ``splitter_name``) so evaluations can report it.

This module imports nothing from the rest of the orchestration package and has
no hard third-party dependency, so it can be unit-tested in isolation.
"""

from __future__ import annotations

import ast
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

__all__ = [
    "SymbolEntry",
    "RetrievalResult",
    "CodeRetriever",
    "split_identifier",
    "tokenize_query",
    "light_stem",
    "resolve_stemmer",
    "iter_python_symbols",
    "build_retriever",
]

_CAMEL_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_2 = re.compile(r"([a-z\d])([A-Z])")
_SPLIT = re.compile(r"[_\s]+")
_WORD = re.compile(r"[^\W\d_]+|\d+")

# Query-side stopwords. Deliberately small; we want to keep domain words.
STOPWORDS: frozenset[str] = frozenset(
    {
        "find", "the", "a", "an", "where", "how", "that", "which", "all", "into",
        "of", "for", "from", "by", "is", "are", "was", "and", "or", "to",
        "method", "function", "class", "module", "locate", "in", "it", "its",
        "this", "be", "do", "does", "if", "up", "at", "on", "out", "as", "with",
        "via", "per", "no", "not", "but", "their", "return", "returns", "given",
        "use", "used", "using", "when", "then", "else", "will", "can", "should",
        "must", "may", "also", "you", "your", "we", "our",
    }
)


# --------------------------------------------------------------------------- #
# Stemming
# --------------------------------------------------------------------------- #

def light_stem(token: str) -> str:
    """A deterministic, dependency-free stemmer.

    Not Porter; it only normalizes the common inflections that matter for
    matching query verbs to identifier roots (plurals, gerunds, past tense).
    The exact root is irrelevant — what matters is that the same surface form
    maps both sides of a match to the same stem.
    """
    t = token.lower()
    if len(t) <= 3:
        return t
    for suf, repl in (
        ("ization", "ize"),
        ("izations", "ize"),
        ("izing", "ize"),
        ("izes", "ize"),
        ("ied", "y"),
        ("ies", "y"),
        ("sses", "ss"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
    ):
        if t.endswith(suf) and len(t) - len(suf) + len(repl) >= 2:
            return t[: len(t) - len(suf)] + repl
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def resolve_stemmer(stemmer: Callable[[str], str] | None) -> tuple[Callable[[str], str], str]:
    """Return ``(stem_fn, name)``. None -> NLTK Porter if available, else light."""
    if stemmer is not None:
        return stemmer, getattr(stemmer, "__qualname__", "custom")
    try:  # pragma: no cover - exercised only when nltk is installed
        from nltk.stem import PorterStemmer

        return PorterStemmer().stem, "nltk.PorterStemmer"
    except Exception:
        return light_stem, "light_stem"


# --------------------------------------------------------------------------- #
# Compound splitting
# --------------------------------------------------------------------------- #

def _dp_segment(token: str, costs: dict[str, float], maxlen: int) -> list[str] | None:
    """Segment ``token`` into known words minimizing summed cost.

    ``costs`` maps a word to a cost (lower = more frequent). Returns the best
    full segmentation into >=2 pieces (each len>=2) or None if none exists.
    """
    n = len(token)
    # best[i] = (cost, pieces) for token[:i]
    best: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    for i in range(1, n + 1):
        for j in range(max(0, i - maxlen), i):
            piece = token[j:i]
            if len(piece) < 2 or piece not in costs:
                continue
            prev = best[j]
            if prev is None:
                continue
            cand = (prev[0] + costs[piece], prev[1] + [piece])
            if best[i] is None or cand[0] < best[i][0]:  # type: ignore[index]
                best[i] = cand
    final = best[n]
    if final is None or len(final[1]) < 2:
        return None
    return final[1]


def build_corpus_splitter(
    atomic_tokens: Iterable[str],
) -> tuple[Callable[[str], list[str]], str]:
    """Build a compound splitter. Prefer wordninja; else corpus-frequency DP.

    The DP fallback only splits a token when a full segmentation into >=2 known
    corpus words exists AND each piece is individually more frequent than the
    fused token, which avoids shredding real words like "redirect".
    """
    try:  # pragma: no cover - exercised only when wordninja is installed
        import wordninja

        def split_wn(token: str) -> list[str]:
            if len(token) < 6 or not token.isalpha():
                return [token]
            parts = wordninja.split(token)
            parts = [p for p in parts if len(p) >= 2]
            return parts if len(parts) >= 2 else [token]

        return split_wn, "wordninja"
    except Exception:
        pass

    freq = Counter(t for t in atomic_tokens if t)
    total = sum(freq.values()) or 1
    maxlen = max((len(t) for t in freq), default=1)
    costs = {w: -math.log((c + 1) / total) for w, c in freq.items()}

    def split_dp(token: str) -> list[str]:
        if len(token) < 6 or not token.isalpha():
            return [token]
        whole = freq.get(token, 0)
        seg = _dp_segment(token, costs, maxlen)
        if not seg:
            return [token]
        # every piece must be strictly more frequent than the fused token
        if all(freq.get(p, 0) > whole for p in seg):
            return seg
        return [token]

    return split_dp, "corpus_dp"


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #

def split_identifier(
    name: str,
    *,
    splitter: Callable[[str], list[str]] | None = None,
    stem: Callable[[str], str] | None = None,
) -> list[str]:
    """Split an identifier into normalized tokens (camel/snake + compound + stem)."""
    n = name.lstrip("_")
    n = _CAMEL_1.sub(r"\1_\2", n)
    n = _CAMEL_2.sub(r"\1_\2", n)
    pieces = [p for p in _SPLIT.split(n) if len(p) > 1]
    out: list[str] = []
    for p in pieces:
        sub = splitter(p) if splitter else [p]
        for s in sub:
            s = s.lower()
            if len(s) <= 1:
                continue
            out.append(stem(s) if stem else s)
    return out


def tokenize_query(
    text: str,
    *,
    stem: Callable[[str], str] | None = None,
) -> list[str]:
    """Tokenize a natural-language query: words, drop stopwords, optional stem."""
    out: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if raw in STOPWORDS or len(raw) <= 1:
            continue
        out.append(stem(raw) if stem else raw)
    return out


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #

@dataclass
class SymbolEntry:
    """A retrieval candidate: one function/class definition."""

    name: str
    file_path: str
    docstring: str = ""
    kind: str = "FunctionDef"
    line: int = 0


@dataclass
class RetrievalResult:
    entry: SymbolEntry
    score: float
    rank: int


@dataclass
class CodeRetriever:
    """BM25 retriever over symbol name + docstring tokens with site weighting."""

    k1: float = 1.5
    b: float = 0.75
    test_penalty: float = 0.5
    class_boost: float = 1.0
    docstring_weight: int = 1
    stemmer: Callable[[str], str] | None = None
    splitter: Callable[[str], list[str]] | None = None

    entries: list[SymbolEntry] = field(default_factory=list)
    _docs: list[list[str]] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avgdl: float = field(default=0.0, repr=False)
    stemmer_name: str = ""
    splitter_name: str = ""

    def _is_test_site(self, path: str) -> bool:
        low = path.replace("\\", "/").lower()
        base = low.rsplit("/", 1)[-1]
        return (
            "/test" in low
            or low.startswith("test")
            or base.startswith("test_")
            or base.startswith("conftest")
        )

    def fit(self, entries: Sequence[SymbolEntry]) -> "CodeRetriever":
        self.entries = list(entries)
        stem_fn, self.stemmer_name = resolve_stemmer(self.stemmer)
        # build compound splitter from the corpus's atomic (camel/snake) tokens
        atomic: list[str] = []
        for e in self.entries:
            n = e.name.lstrip("_")
            n = _CAMEL_1.sub(r"\1_\2", n)
            n = _CAMEL_2.sub(r"\1_\2", n)
            atomic.extend(p.lower() for p in _SPLIT.split(n) if len(p) > 1)
        split_fn = self.splitter
        if split_fn is None:
            split_fn, self.splitter_name = build_corpus_splitter(atomic)
        else:
            self.splitter_name = getattr(split_fn, "__qualname__", "custom")

        self._stem_fn = stem_fn
        self._split_fn = split_fn

        self._docs = []
        self._df = Counter()
        for e in self.entries:
            toks = split_identifier(e.name, splitter=split_fn, stem=stem_fn)
            if e.docstring and self.docstring_weight:
                dtoks = tokenize_query(e.docstring, stem=stem_fn)
                toks = toks + dtoks * self.docstring_weight
            self._docs.append(toks)
            for t in set(toks):
                self._df[t] += 1

        n = len(self._docs)
        self._avgdl = (sum(len(d) for d in self._docs) / n) if n else 0.0
        self._idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self._df.items()
        }
        return self

    def _score(self, q_tokens: Sequence[str], doc: Sequence[str], path: str, kind: str) -> float:
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = self._idf.get(t, 0.0)
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1)
        if self._is_test_site(path):
            score *= self.test_penalty
        if kind == "ClassDef":
            score *= self.class_boost
        return score

    def query(self, text: str, top_k: int = 10) -> list[RetrievalResult]:
        q = tokenize_query(text, stem=self._stem_fn)
        scored = [
            (self._score(q, self._docs[i], e.file_path, e.kind), i)
            for i, e in enumerate(self.entries)
        ]
        scored.sort(key=lambda x: (-x[0], self.entries[x[1]].name))
        out: list[RetrievalResult] = []
        for rank, (sc, i) in enumerate(scored[:top_k], 1):
            out.append(RetrievalResult(entry=self.entries[i], score=sc, rank=rank))
        return out


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #

def iter_python_symbols(root: str) -> Iterator[SymbolEntry]:
    """Yield a SymbolEntry for every function/class def under ``root``."""
    root = os.path.abspath(root)
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                tree = ast.parse(open(fpath, errors="replace").read(), filename=fpath)
            except (SyntaxError, OSError):
                continue
            rel = os.path.relpath(fpath, root)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    yield SymbolEntry(
                        name=node.name,
                        file_path=rel,
                        docstring=(ast.get_docstring(node) or "")[:400],
                        kind=type(node).__name__,
                        line=node.lineno,
                    )


def build_retriever(roots: Iterable[str], **kwargs: object) -> CodeRetriever:
    """AST-index one or more repo roots and return a fitted ``CodeRetriever``."""
    entries: list[SymbolEntry] = []
    for root in roots:
        entries.extend(iter_python_symbols(root))
    return CodeRetriever(**kwargs).fit(entries)  # type: ignore[arg-type]
