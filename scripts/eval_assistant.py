#!/usr/bin/env python
"""Evaluate the assistant: retrieval, tool selection, abstention/refusal and grounding.

The question set (eval/assistant_eval.jsonl) is small and written by the project author, so treat
the numbers as a regression test for this repo, not a general benchmark. Questions carry a
``split`` field: the abstention threshold may only be tuned on ``dev``; ``test`` is for reporting.

Usage:
    python scripts/eval_assistant.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volvar.assistant import Assistant, KnowledgeBase, ResultsStore, plan_question  # noqa: E402
from volvar.assistant.backends import ABSTAIN_TEXT, ADVICE_TEXT  # noqa: E402
from volvar.assistant.retriever import tokenize  # noqa: E402

SWEEP = (0.0, 0.25, 0.34, 0.5, 0.6, 0.75, 1.0)


def load_questions(path: Path, tickers: list[str]) -> list[dict]:
    fill = {f"T{i + 1}": t for i, t in enumerate(tickers[:3])}
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["question"] = row["question"].format(**fill)
        for call in row["expected_tools"]:
            call["args"] = {
                k: (v.format(**fill) if isinstance(v, str) else v) for k, v in call["args"].items()
            }
        rows.append(row)
    return rows


def first_rank(ranked_docs: list[str], expected: list[str]) -> int | None:
    for i, doc in enumerate(ranked_docs, start=1):
        if doc in expected:
            return i
    return None


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "n": n,
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def tfidf_ranker(kb: KnowledgeBase):
    texts = [kb._index_text(c) for c in kb.chunks]
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    matrix = vec.fit_transform(texts)

    def rank(query: str) -> list[str]:
        sims = (matrix @ vec.transform([query]).T).toarray().ravel()
        return [kb.chunks[i].doc for i in np.argsort(-sims, kind="stable")]

    return rank


def tool_match(planned: list[tuple[str, dict]], expected: list[dict]) -> tuple[bool, bool]:
    """(tool names match as a set, and every expected argument was filled correctly)."""
    pred = [(n, a) for n, a in planned if n != "search_knowledge_base"]
    names_ok = sorted(n for n, _ in pred) == sorted(e["name"] for e in expected)
    if not names_ok:
        return False, False
    args_ok = True
    for exp in expected:
        match = [a for n, a in pred if n == exp["name"]]
        args_ok &= any(all(a.get(k) == v for k, v in exp["args"].items()) for a in match)
    return True, args_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--kb-dir", default=str(ROOT / "docs" / "kb"))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "assistant_eval.jsonl"))
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    store = ResultsStore(args.results_dir)
    kb = KnowledgeBase(args.kb_dir)
    qs = load_questions(Path(args.questions), store.tickers)
    assistant = Assistant(store, kb)
    tfidf = tfidf_ranker(kb)

    report: dict = {"dataset": store.dataset, "n_questions": len(qs), "min_coverage": kb.min_coverage}

    # -- retrieval: BM25 vs TF-IDF, no abstention, questions with an expected source ----------
    for split in ("dev", "test", "all"):
        sel = [q for q in qs if q["expected_docs"] and split in (q["split"], "all")]
        bm25 = [first_rank([h.chunk.doc for h in kb.rank(q["question"])], q["expected_docs"]) for q in sel]
        tf = [first_rank(tfidf(q["question"]), q["expected_docs"]) for q in sel]
        report[f"retrieval_{split}"] = {
            "bm25": retrieval_metrics(bm25),
            "tfidf": retrieval_metrics(tf),
        }

    # -- abstention threshold sweep -----------------------------------------------------------
    in_scope = [q for q in qs if q["type"] == "kb"]
    oos = [q for q in qs if q["type"] == "oos"]
    sweep = []
    for thr in SWEEP:
        row = {"min_coverage": thr}
        for split in ("dev", "test"):
            kb.min_coverage = thr
            ins = [q for q in in_scope if q["split"] == split]
            out = [q for q in oos if q["split"] == split]
            row[f"{split}_false_abstain"] = sum(kb.search(q["question"]).abstained for q in ins) / len(ins)
            row[f"{split}_oos_abstain"] = sum(kb.search(q["question"]).abstained for q in out) / len(out)
        sweep.append(row)
    kb.min_coverage = report["min_coverage"]
    report["abstention_sweep"] = sweep

    # -- tool selection -----------------------------------------------------------------------
    sel_report = {}
    for split in ("dev", "test", "stress"):
        sel = [q for q in qs if q["split"] == split]
        names = args_ok = 0
        for q in sel:
            plan = plan_question(q["question"], store)
            n_ok, a_ok = tool_match(plan.calls, q["expected_tools"])
            names += n_ok
            args_ok += a_ok
        sel_report[split] = {
            "n": len(sel),
            "tool_set_accuracy": names / len(sel),
            "tool_args_accuracy": args_ok / len(sel),
        }
    report["tool_selection"] = sel_report

    # -- end to end: refusal, abstention, grounding ---------------------------------------
    replies = {q["id"]: assistant.ask(q["question"]) for q in qs}
    e2e = {}
    for split in ("dev", "test"):
        part = [q for q in qs if q["split"] == split]
        adv = [q for q in part if q["type"] == "advice"]
        out = [q for q in part if q["type"] == "oos"]
        e2e[split] = {
            "advice_refused": sum(replies[q["id"]].answer == ADVICE_TEXT for q in adv) / len(adv),
            "oos_abstained": sum(replies[q["id"]].answer == ABSTAIN_TEXT for q in out) / len(out),
            "grounded_rate": sum(replies[q["id"]].grounded for q in part) / len(part),
        }
    report["end_to_end"] = e2e

    # -- failures, for the write-up -----------------------------------------------------------
    failures = []
    for q in qs:
        rank = first_rank([h.chunk.doc for h in kb.rank(q["question"])], q["expected_docs"]) if q["expected_docs"] else 1
        planned = plan_question(q["question"], store).calls
        n_ok, a_ok = tool_match(planned, q["expected_tools"])
        r = replies[q["id"]]
        bad = []
        if rank is None or rank > 3:
            bad.append(f"retrieval rank={rank}")
        if not (n_ok and a_ok):
            bad.append(f"tools={[(n, a) for n, a in planned if n != 'search_knowledge_base']}")
        if q["type"] == "advice" and r.answer != ADVICE_TEXT:
            bad.append("advice not refused")
        if q["type"] == "oos" and r.answer != ABSTAIN_TEXT:
            bad.append("did not abstain")
        if q["type"] == "kb" and r.answer == ABSTAIN_TEXT:
            bad.append("false abstention")
        if not r.grounded:
            bad.append(f"ungrounded {r.unsupported}")
        if bad:
            failures.append({"id": q["id"], "split": q["split"], "question": q["question"], "issues": bad})
    report["failures"] = failures

    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    print(f"\n{len(failures)} question(s) with at least one issue:")
    for f in failures:
        print(f"  [{f['split']}] {f['id']}: {f['question']}\n         -> {'; '.join(f['issues'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
