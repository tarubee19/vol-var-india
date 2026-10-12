"""BM25 retrieval over the markdown knowledge base in ``docs/kb``.

Documents are split into sections (one chunk per ``##`` heading). Retrieval can abstain: if the
best chunk shares too little vocabulary with the question, the knowledge base is treated as not
covering it, instead of returning the least-bad match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no
    nor not now of off on once only or other our out over own same she should so some such than
    that the their them then there these they this those through to too under until up very was
    we were what when where which while who whom why will with would you your yours tell give
    show me please""".split()
)


def stem(word: str) -> str:
    """Very small suffix stripper; enough to join plurals and -ed/-ing/-ised forms."""
    for suffix in ("isation", "ations", "ation", "ising", "ised", "izing", "ized"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4 and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    doc: str  # file stem, e.g. "value_at_risk"
    title: str  # H1 of the document
    section: str  # H2 heading
    text: str  # body of the section

    @property
    def ref(self) -> str:
        return f"{self.doc} > {self.section}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    coverage: float  # share of distinct query terms present in the chunk


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    abstained: bool


def load_chunks(kb_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        title = path.stem.replace("_", " ")
        sections: list[tuple[str, list[str]]] = []
        for line in path.read_text().splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                sections.append((line[3:].strip(), []))
            elif line.strip() and sections:
                sections[-1][1].append(line.strip())
        for heading, lines in sections:
            if lines:
                chunks.append(Chunk(path.stem, title, heading, " ".join(lines)))
    return chunks


class KnowledgeBase:
    """BM25 index over section chunks."""

    def __init__(self, kb_dir: str | Path, min_coverage: float = 0.6):
        self.chunks = load_chunks(kb_dir)
        if not self.chunks:
            raise ValueError(f"No knowledge-base chunks found in {kb_dir}")
        self.min_coverage = min_coverage
        self._chunk_tokens = [set(tokenize(self._index_text(c))) for c in self.chunks]
        self._bm25 = BM25Okapi([tokenize(self._index_text(c)) for c in self.chunks])

    @staticmethod
    def _index_text(chunk: Chunk) -> str:
        return f"{chunk.title} {chunk.section} {chunk.text}"

    def rank(self, query: str) -> list[Hit]:
        """All chunks, best first, with no abstention applied."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        distinct = set(q_tokens)
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        return [
            Hit(
                self.chunks[i],
                float(scores[i]),
                len(distinct & self._chunk_tokens[i]) / len(distinct),
            )
            for i in order
        ]

    def search(self, query: str, k: int = 3) -> SearchResult:
        ranked = self.rank(query)
        if not ranked or ranked[0].score <= 0 or ranked[0].coverage < self.min_coverage:
            return SearchResult([], True)
        return SearchResult([h for h in ranked[:k] if h.score > 0], False)
