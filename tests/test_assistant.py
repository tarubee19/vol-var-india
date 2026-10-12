"""Tests for the assistant: retrieval, tools, planning, grounding, backends."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from volvar.assistant import (
    AnthropicBackend,
    Assistant,
    KnowledgeBase,
    ResultsNotFoundError,
    ResultsStore,
    plan_question,
)
from volvar.assistant.backends import ABSTAIN_TEXT, ADVICE_TEXT
from volvar.assistant.grounding import verify_answer
from volvar.assistant.retriever import stem, tokenize
from volvar.assistant.tools import TOOL_SPECS, ToolExecutor
from volvar.config import KB_DIR


@pytest.fixture(scope="module")
def store(pipeline_run):
    return ResultsStore(pipeline_run[1])


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(KB_DIR)


@pytest.fixture(scope="module")
def assistant(store, kb):
    return Assistant(store, kb)


# -- retrieval -------------------------------------------------------------------------------
def test_tokenizer_drops_stopwords_and_joins_plurals():
    assert tokenize("What are the violations?") == [stem("violations")]
    assert stem("annualised") == "annual"
    assert stem("class") == "class"  # no stripping of 'ss'


def test_kb_loads_every_note_section(kb):
    docs = {c.doc for c in kb.chunks}
    assert {"value_at_risk", "expected_shortfall", "var_backtesting", "garch_models"} <= docs
    assert all(c.text and c.section for c in kb.chunks)


@pytest.mark.parametrize(
    "question,doc",
    [
        ("What does the Kupiec test check?", "var_backtesting"),
        ("What value of lambda does RiskMetrics use?", "ewma_riskmetrics"),
        ("How does filtered historical simulation work?", "filtered_historical_simulation"),
        ("What is Expected Shortfall?", "expected_shortfall"),
    ],
)
def test_retrieval_finds_the_right_note(kb, question, doc):
    assert kb.rank(question)[0].chunk.doc == doc


def test_retrieval_abstains_on_off_topic_question(kb):
    assert kb.search("What is the capital of France?").abstained
    assert kb.search("").abstained


# -- planning --------------------------------------------------------------------------------
def test_plan_var_question_extracts_arguments(store):
    t = store.tickers[0]
    plan = plan_question(f"What is {t}'s 95% VaR with the normal method using GARCH?", store)
    assert plan.calls == [
        ("get_var_es", {"ticker": t, "level": 0.95, "model": "garch", "method": "normal"})
    ]


def test_plan_concept_question_uses_only_the_knowledge_base(store):
    plan = plan_question("What is Value-at-Risk?", store)
    assert [name for name, _ in plan.calls] == ["search_knowledge_base"]


def test_plan_asks_for_ticker_when_missing(store):
    assert plan_question("What is the VaR tomorrow?", store).needs_ticker


def test_plan_filtered_historical_simulation_is_not_the_historical_model(store):
    t = store.tickers[0]
    plan = plan_question(f"{t} VaR with filtered historical simulation", store)
    assert plan.calls[0][1].get("model") is None
    assert plan.calls[0][1]["method"] == "fhs"


def test_plan_refuses_investment_advice(store):
    assert plan_question(f"Should I buy {store.tickers[0]}?", store).advice
    # a risk question using the word 'rise' is not advice
    assert not plan_question("Will the VaR rise when volatility rises?", store).advice


# -- tools -----------------------------------------------------------------------------------
def test_tool_specs_are_wellformed():
    names = [s["name"] for s in TOOL_SPECS]
    assert len(names) == len(set(names))
    for spec in TOOL_SPECS:
        assert spec["input_schema"]["type"] == "object" and spec["description"]


def test_var_tool_matches_the_results_file(store, kb):
    t = store.tickers[0]
    out = ToolExecutor(store, kb).run("get_var_es", {"ticker": t, "level": 0.99, "model": "ewma"})
    row = store.next_day_var.query(
        "ticker == @t and model == 'ewma' and method == 'fhs' and level == 0.99"
    ).iloc[0]
    assert out["ok"] and out["dataset"] == "simulated"
    assert out["var_pct"] == pytest.approx(row["var_pct"], abs=5e-4)
    assert out["es_pct"] >= out["var_pct"]


def test_tools_reject_bad_arguments_without_raising(store, kb):
    ex = ToolExecutor(store, kb)
    assert not ex.run("get_var_es", {"ticker": "NOPE"})["ok"]
    assert not ex.run("get_var_es", {"ticker": store.tickers[0], "level": 0.5})["ok"]
    assert not ex.run("get_var_es", {"ticker": store.tickers[0], "model": "magic"})["ok"]
    assert not ex.run("get_var_es", {"ticker": store.tickers[0], "bogus": 1})["ok"]
    assert not ex.run("rm_rf", {})["ok"]
    assert not ex.run("compare_models", {"metric": "sharpe"})["ok"]
    assert [c.ok for c in ex.calls] == [False] * 6


def test_level_accepts_percent_or_fraction(store, kb):
    ex = ToolExecutor(store, kb)
    t = store.tickers[0]
    assert ex.run("get_var_es", {"ticker": t, "level": 99})["level"] == 0.99
    assert ex.run("get_var_es", {"ticker": t, "level": "0.95"})["level"] == 0.95


def test_compare_models_is_sorted_and_flags_benchmark(store, kb):
    out = ToolExecutor(store, kb).run("compare_models", {})
    losses = [r["qlike"] for r in out["ranking"]]
    assert losses == sorted(losses)
    bench = [r for r in out["ranking"] if r["is_benchmark"]]
    assert len(bench) == 1 and bench[0]["model"] == "ewma"
    assert bench[0]["dm_p_value"] is None


# -- grounding -------------------------------------------------------------------------------
def test_grounding_accepts_rounded_and_percent_forms():
    ev = [{"var_pct": 2.5612, "rate": 0.0148, "date": "2024-12-19", "label": "GARCH(1,1)-t"}]
    ok = verify_answer("VaR 2.56% on 2024-12-19, rate 1.48%, GARCH(1,1)-t, p<0.001", "q", ev)
    assert ok.ok


def test_grounding_flags_invented_numbers_and_dates():
    ev = [{"var_pct": 2.5612, "date": "2024-12-19"}]
    bad = verify_answer("VaR is 3.40% on 2024-12-20", "q", ev)
    assert not bad.ok
    assert "3.40" in bad.unsupported_numbers and "2024-12-20" in bad.unsupported_dates


def test_grounding_allows_numbers_the_user_typed():
    assert verify_answer("At 97.5 percent...", "what about 97.5 percent?", []).ok


# -- end to end (rule backend) ---------------------------------------------------------------
def test_var_answer_is_grounded_and_labelled_simulated(assistant, store):
    r = assistant.ask(f"What is {store.tickers[0]}'s 99% VaR for tomorrow?")
    assert r.grounded and r.dataset == "simulated"
    assert "SIMULATED" in r.answer
    assert [c["name"] for c in r.tool_calls] == ["get_var_es"]


def test_every_eval_question_is_grounded(assistant, store):
    import json
    from pathlib import Path

    fill = {f"T{i + 1}": t for i, t in enumerate(store.tickers[:3])}
    path = Path(__file__).resolve().parents[1] / "eval" / "assistant_eval.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 50
    for row in rows:
        r = assistant.ask(row["question"].format(**fill))
        assert r.grounded, (row["question"], r.unsupported)


def test_advice_and_out_of_scope_are_handled(assistant, store):
    assert assistant.ask(f"Should I buy {store.tickers[0]} tomorrow?").answer == ADVICE_TEXT
    reply = assistant.ask("What is the capital of France?")
    assert reply.answer == ABSTAIN_TEXT and not reply.tool_calls[1:]


def test_mixed_question_returns_numbers_and_a_cited_source(assistant, store):
    r = assistant.ask(f"What is {store.tickers[0]}'s 99% VaR and what does that number mean?")
    assert "get_var_es" in [c["name"] for c in r.tool_calls]
    assert any(s.startswith("value_at_risk") for s in r.sources)
    assert "(source:" in r.answer


def test_input_validation(assistant):
    with pytest.raises(ValueError):
        assistant.ask("   ")
    with pytest.raises(ValueError):
        assistant.ask("x" * 2000)


def test_unsupported_numbers_in_a_backend_answer_are_flagged(store, kb):
    class Liar:
        name = "liar"

        def respond(self, question, tools):
            tools.run("get_var_es", {"ticker": store.tickers[0]})
            return "The VaR is 99.99% of position value."

    reply = Assistant(store, kb, backend=Liar()).ask("What is the VaR?")
    assert not reply.grounded and "99.99" in reply.unsupported
    assert "Warning" in reply.answer


def test_missing_results_directory_has_helpful_error(tmp_path):
    with pytest.raises(ResultsNotFoundError, match="run_pipeline"):
        ResultsStore(tmp_path)


def test_store_picks_lowest_qlike_model(store):
    best = store.model_comparison.sort_values("qlike").iloc[0]["model"]
    assert store.best_model == best
    assert isinstance(store.model_comparison, pd.DataFrame)


# -- live backend loop, with a scripted fake client (no network) -----------------------------
class FakeClient:
    """Mimics ``anthropic.Anthropic().messages.create`` with a scripted conversation."""

    def __init__(self, ticker):
        self.ticker = ticker
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            block = SimpleNamespace(
                type="tool_use", id="tu_1", name="get_var_es", input={"ticker": self.ticker}
            )
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        results = kwargs["messages"][-1]["content"]
        assert results[0]["type"] == "tool_result" and results[0]["tool_use_id"] == "tu_1"
        text = SimpleNamespace(type="text", text="Done.")
        return SimpleNamespace(stop_reason="end_turn", content=[text])


def test_anthropic_backend_runs_the_tool_loop(store, kb):
    client = FakeClient(store.tickers[0])
    reply = Assistant(store, kb, backend=AnthropicBackend(client=client)).ask("VaR please")
    assert reply.answer == "Done." and reply.backend == "anthropic"
    assert [c["name"] for c in reply.tool_calls] == ["get_var_es"]
    assert client.calls[0]["tools"] == TOOL_SPECS and "simulated" in client.calls[0]["system"]


def test_anthropic_backend_stops_after_max_turns(store, kb):
    class Looping:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", id="t", name="list_available", input={})
            return SimpleNamespace(stop_reason="tool_use", content=[block])

    backend = AnthropicBackend(client=Looping(), max_turns=3)
    ex = ToolExecutor(store, kb)
    assert "budget" in backend.respond("loop", ex) and len(ex.calls) == 3
