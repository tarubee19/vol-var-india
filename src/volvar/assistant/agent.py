"""The ``Assistant`` facade: run a backend, then check its numbers against the evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import KB_DIR, RESULTS_DIR
from .backends import MAX_QUESTION_CHARS, Backend, RuleBackend
from .grounding import verify_answer
from .retriever import KnowledgeBase
from .store import ResultsStore
from .tools import ToolExecutor


@dataclass
class AssistantReply:
    question: str
    answer: str
    backend: str
    dataset: str
    grounded: bool
    unsupported: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Assistant:
    def __init__(
        self,
        store: ResultsStore,
        kb: KnowledgeBase,
        backend: Backend | None = None,
    ):
        self.store = store
        self.kb = kb
        self.backend = backend or RuleBackend()

    @classmethod
    def from_dirs(
        cls,
        results_dir: str | Path | None = None,
        kb_dir: str | Path | None = None,
        backend: Backend | None = None,
    ) -> Assistant:
        return cls(
            ResultsStore(results_dir or RESULTS_DIR), KnowledgeBase(kb_dir or KB_DIR), backend
        )

    def ask(self, question: str) -> AssistantReply:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question is longer than {MAX_QUESTION_CHARS} characters.")

        tools = ToolExecutor(self.store, self.kb)
        answer = self.backend.respond(question, tools)
        report = verify_answer(answer, question, tools.evidence)
        unsupported = report.unsupported_numbers + report.unsupported_dates
        if unsupported:
            answer += (
                "\n\nWarning: these figures could not be matched to tool output and should not "
                f"be relied on: {', '.join(unsupported)}."
            )
        return AssistantReply(
            question=question,
            answer=answer,
            backend=self.backend.name,
            dataset=self.store.dataset,
            grounded=report.ok,
            unsupported=unsupported,
            tool_calls=[asdict(c) for c in tools.calls],
            sources=list(tools.sources),
        )
