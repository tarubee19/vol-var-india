"""Answer backends.

``RuleBackend`` is a deterministic planner + template writer. It is *not* a language model: it
routes the question to tools with regular expressions and fills fixed templates from the tool
output. It exists so the tool layer, retrieval, grounding check and API can be tested and run
offline and reproducibly.

``AnthropicBackend`` is the live path: a standard tool-use loop with the Anthropic Messages API.
It shares the tools and the grounding check with the rule backend.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .store import ResultsStore
from .tools import TOOL_SPECS, ToolExecutor

MAX_QUESTION_CHARS = 1000


class Backend(Protocol):
    name: str

    def respond(self, question: str, tools: ToolExecutor) -> str: ...


# --------------------------------------------------------------------------------------------
# Rule-based planning
# --------------------------------------------------------------------------------------------
_STRONG_ADVICE = re.compile(
    r"\bshould i (buy|sell|invest|hold|trade|short)\b|\b(buy|sell|hold)\s+(or|and)\s+(buy|sell|hold)\b"
    r"|\bgood (investment|stock to buy)\b|\bworth (buying|investing)\b|\bprice target\b"
    r"|\bpredict\b.{0,20}\bprice\b|\bwhich stocks? .{0,20}\b(buy|invest)\b"
)
_WEAK_ADVICE = re.compile(r"\bwill\b.{0,40}\b(go up|go down|rise|fall|drop|crash|rally)\b")
_RISK_WORD = re.compile(r"\b(var|volatility|variance|shortfall|violations?|breach\w*)\b")

_FHS_PHRASE = re.compile(r"filtered historical( simulation)?|historical simulation")
_VAR = re.compile(r"\b(var|value[- ]at[- ]risk|expected shortfall|es|cvar|tail loss)\b")
_VOL = re.compile(r"\b(volatility|volatile|vol|sigma|how risky|riskiness|standard deviation)\b")
_BACKTEST = re.compile(
    r"\b(backtest\w*|back-test\w*|violations?|breach\w*|exceptions?|exceed\w*|kupiec|christoffersen"
    r"|traffic[- ]light|basel zone|coverage|calibrat\w*|pass(ed|es)?|fail(ed|s)?)\b"
)
_COMPARE = re.compile(
    r"\b(which|best|better) models?\b|\bbeat\w*\b|\boutperform\w*|\bcompar\w*|\brank\w*"
    r"|\bwinner\b|\blowest (qlike|mse|loss)\b|\b(qlike|mse|loss) (of|for)\b|\bsignificantly\b"
)
_LIST = re.compile(
    r"real or simulated|simulated or real|real or synthetic|\bis (this|the) (data|result\w*) "
    r"(real|simulated|synthetic)|\b(which|what) (tickers|stocks|companies)\b|\bwhich dataset\b"
    r"|\b(list|available) (tickers|stocks|models)\b"
)
_EXPLAIN = re.compile(
    r"\b(explain|define|definition|meaning|means?|why|difference|intuition|interpret\w*)\b"
    r"|\bhow (does|do|is|are)\b|\bwhat does\b.{0,40}\b(mean|measure|show|tell)\b"
)
_DEFINE = re.compile(r"\bwhat (is|are)\b|\bwhat'?s\b")
_FORWARD = re.compile(r"\b(tomorrow|next[- ]day|next trading day|today|forecast|current|now)\b")
_NOW = re.compile(r"\b(tomorrow|next[- ]day|next trading day|today|right now)\b")
_SUMMARY = re.compile(
    r"\b(overall|across|all (the )?(tickers|stocks)|on average|average|summary|pass rate|how often"
    r"|how many (tickers|stocks))\b"
)
_LEVEL = re.compile(
    r"(?<![\d.])(95|99)(?:\.0)?\s?(?:%|percent|per cent|th percentile|confidence)|(?<!\d)0\.(95|99)\b"
)
_METHOD_NORMAL = re.compile(r"\bnormal\b|\bgaussian\b|\bparametric\b")
_METRIC_MSE = re.compile(r"\bmse\b|squared error")
_MODEL_PATTERNS = {
    "gjr": re.compile(r"\bgjr\b"),
    "garch": re.compile(r"\bgarch\b"),
    "ewma": re.compile(r"\bewma\b|riskmetrics|exponential\w*"),
    "har": re.compile(r"\bhar\b|heterogeneous"),
    "lgbm": re.compile(r"lightgbm|lgbm|gradient boost\w*"),
    "hist": re.compile(r"\bhistorical\b|\brolling\b"),
}
_TICKER_ALIASES = {"NIFTY50": re.compile(r"\bnifty ?50\b(?! (stocks|companies|constituents))")}


@dataclass
class Plan:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    advice: bool = False
    needs_ticker: bool = False


def detect_tickers(text: str, store: ResultsStore, limit: int = 3) -> list[str]:
    found: list[tuple[int, str]] = []
    for ticker in store.tickers:
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text, re.IGNORECASE)
        if m:
            found.append((m.start(), ticker))
        elif ticker in _TICKER_ALIASES:
            m = _TICKER_ALIASES[ticker].search(text.lower())
            if m:
                found.append((m.start(), ticker))
    return [t for _, t in sorted(found)][:limit]


def detect_model(text: str, available: list[str]) -> str | None:
    cleaned = _FHS_PHRASE.sub(" ", text.lower())
    best: tuple[int, str] | None = None
    for model, pattern in _MODEL_PATTERNS.items():
        if model not in available:
            continue
        m = pattern.search(cleaned)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), model)
    return best[1] if best else None


def _kb_query(question: str, tickers: list[str]) -> str:
    """Retrieval query: the question minus ticker names and bare numbers.

    Both would only dilute term overlap with the methodology notes, which never mention them.
    """
    text = question
    for ticker in tickers:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?:'s)?(?![A-Za-z0-9])", " ", text, flags=re.I)
    return re.sub(r"\d+(?:\.\d+)?\s?%?", " ", text)


def plan_question(question: str, store: ResultsStore) -> Plan:
    """Decide which tools to call for a question. Pure function; nothing is executed."""
    q = question.lower()
    if _STRONG_ADVICE.search(q) or (_WEAK_ADVICE.search(q) and not _RISK_WORD.search(q)):
        return Plan(advice=True)

    tickers = detect_tickers(question, store)
    level = None
    if (lm := _LEVEL.search(q)) is not None:
        level = float(lm.group(1) or lm.group(2)) / 100
    model = detect_model(q, store.models)
    method = "fhs" if _FHS_PHRASE.search(q) else ("normal" if _METHOD_NORMAL.search(q) else None)

    def args(**extra: Any) -> dict[str, Any]:
        return {k: v for k, v in extra.items() if v is not None}

    wants_var = bool(_VAR.search(q))
    wants_vol = bool(_VOL.search(q))
    wants_backtest = bool(_BACKTEST.search(q))
    wants_compare = bool(_COMPARE.search(q))
    explain = bool(_EXPLAIN.search(q))
    define = bool(_DEFINE.search(q))

    plan = Plan()
    if _LIST.search(q):
        plan.calls.append(("list_available", {}))

    if tickers:
        for t in tickers:
            if wants_var and (not wants_backtest or _FORWARD.search(q)):
                plan.calls.append(
                    ("get_var_es", args(ticker=t, level=level, model=model, method=method))
                )
            if wants_vol and not (wants_var and wants_backtest):
                plan.calls.append(("get_volatility_forecast", args(ticker=t, model=model)))
            if wants_backtest:
                plan.calls.append(
                    ("get_backtest", args(ticker=t, level=level, model=model, method=method))
                )
    elif wants_backtest and _SUMMARY.search(q):
        plan.calls.append(("get_backtest_summary", args(level=level, model=model, method=method)))
    elif (wants_var or wants_vol or wants_backtest) and not explain and (
        _NOW.search(q) or not define
    ):
        plan.needs_ticker = not wants_compare

    if wants_compare:
        plan.calls.append(("compare_models", {"metric": "mse"} if _METRIC_MSE.search(q) else {}))

    has_data = any(name != "search_knowledge_base" for name, _ in plan.calls)
    if not plan.needs_ticker and (not has_data or explain):
        plan.calls.append(("search_knowledge_base", {"query": _kb_query(question, tickers)}))
    return plan


# --------------------------------------------------------------------------------------------
# Rule-based answer composition (number-free canned text; every figure comes from a tool)
# --------------------------------------------------------------------------------------------
ADVICE_TEXT = (
    "I can't give investment advice or predict price direction. This project forecasts risk "
    "(volatility, Value-at-Risk and Expected Shortfall), not returns. I can show the risk "
    "figures for a ticker instead, for example its next-day Value-at-Risk."
)
ABSTAIN_TEXT = (
    "I can only answer from this project's results and methodology notes (volatility forecasts, "
    "Value-at-Risk, Expected Shortfall, backtests and the models behind them), and I couldn't "
    "find anything relevant to that question."
)
_METHOD_NAMES = {"fhs": "filtered historical simulation", "normal": "normal distribution"}


def _p(value: float | None) -> str:
    """Format a p-value with its operator, e.g. ``p=0.368`` or ``p<0.001``."""
    if value is None:
        return "p=n/a"
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def _compose_one(name: str, r: dict[str, Any]) -> str:
    if not r.get("ok"):
        return f"I couldn't run {name}: {r.get('error', 'unknown error')}"
    if name == "list_available":
        models = ", ".join(r["models"].values())
        return (
            f"The loaded results cover {r['n_tickers']} series ({', '.join(r['tickers'])}) from "
            f"{r['data_start']} to {r['data_end']}, with out-of-sample evaluation from "
            f"{r['oos_start']}. Models: {models}. Value-at-Risk is available by filtered "
            f"historical simulation and by the normal distribution, at the "
            f"{' and '.join(f'{lv * 100:g}%' for lv in r['levels'])} levels."
        )
    if name == "get_volatility_forecast":
        return (
            f"{r['ticker']} ({r['model_label']}): the forecast volatility for {r['target_date']} "
            f"is {r['vol_daily_pct']:.2f}% per day, or about {r['vol_annual_pct']:.1f}% "
            f"annualised (forecast made after the {r['as_of']} close)."
        )
    if name == "get_var_es":
        vol = (
            f" The model's volatility forecast is {r['vol_daily_pct']:.2f}% per day."
            if r.get("vol_daily_pct") is not None
            else ""
        )
        return (
            f"{r['ticker']} ({r['model_label']}, {_METHOD_NAMES[r['method']]}): for "
            f"{r['target_date']} the one-day {r['level_pct']:g}% Value-at-Risk is "
            f"{r['var_pct']:.2f}% of position value and the Expected Shortfall is "
            f"{r['es_pct']:.2f}%.{vol} Forecast made after the {r['as_of']} close."
        )
    if name == "get_backtest":
        verdict = "passes" if r["passes"] else "does not pass"
        zone = f" (Basel zone: {r['basel_zone']})" if r.get("basel_zone") else ""
        return (
            f"{r['ticker']} ({r['model_label']}, {_METHOD_NAMES[r['method']]}), "
            f"{r['level_pct']:g}% Value-at-Risk backtest: {r['violations']} violations in "
            f"{r['n_days']} days, a rate of {r['violation_rate_pct']:.2f}% against "
            f"{r['expected_rate_pct']:.2f}% expected. Kupiec {_p(r['kupiec_p'])}; conditional "
            f"coverage {_p(r['conditional_coverage_p'])}. The model {verdict} the backtest at "
            f"the 5% level. In the most recent trading year it had "
            f"{r['latest_250d_exceptions']} exceptions{zone}."
        )
    if name == "get_backtest_summary":
        green = (
            f" The latest year is in the Basel green zone for {r['share_latest_window_green_pct']:.1f}% "
            "of tickers."
            if r.get("share_latest_window_green_pct") is not None
            else ""
        )
        return (
            f"Across {r['n_tickers']} tickers ({r['model_label']}, {_METHOD_NAMES[r['method']]}, "
            f"{r['level_pct']:g}% Value-at-Risk): the mean violation rate is "
            f"{r['mean_violation_rate_pct']:.2f}% against {r['expected_rate_pct']:.2f}% expected, "
            f"{r['pass_rate_pct']:.1f}% of tickers pass the backtest, and the mean ES ratio is "
            f"{r['mean_es_ratio']:.2f}.{green}"
        )
    if name == "compare_models":
        metric = r["metric"].upper()
        lines = [
            f"Out-of-sample {metric} over {r['n_observations']} forecasts per model "
            f"(lower is better):"
        ]
        for row in r["ranking"]:
            extra = ""
            if r["metric"] == "qlike":
                if row["is_benchmark"]:
                    extra = " (benchmark)"
                else:
                    sig = "significant" if row["dm_p_value"] < 0.05 else "not significant"
                    extra = (
                        f" (gain vs EWMA {row['qlike_gain_vs_ewma']:+.4f}, "
                        f"DM {_p(row['dm_p_value'])}, {sig})"
                    )
            value = row["qlike"] if r["metric"] == "qlike" else row["mse"]
            lines.append(f"- {row['label']}: {value:.3f}{extra}")
        best = next(x for x in r["ranking"] if x["model"] == r["best_model"])
        lines.append(f"Best by {metric}: {best['label']}.")
        return "\n".join(lines)
    return json.dumps(r)


class RuleBackend:
    name = "rule-based"

    def respond(self, question: str, tools: ToolExecutor) -> str:
        plan = plan_question(question, tools.store)
        if plan.advice:
            return ADVICE_TEXT
        if plan.needs_ticker:
            return (
                "Which ticker do you mean? Available: "
                f"{', '.join(tools.store.tickers)}."
            )
        parts: list[str] = []
        kb_parts: list[str] = []
        for name, arguments in plan.calls:
            result = tools.run(name, arguments)
            if name == "search_knowledge_base":
                if result["hits"]:
                    top = result["hits"][0]
                    kb_parts.append(f"{top['text']} (source: {top['source']})")
            else:
                parts.append(_compose_one(name, result))
        body = list(parts)
        if body:
            body.insert(0, f"Data: {tools.store.dataset_note}.")
        body.extend(kb_parts)
        return "\n\n".join(body) if body else ABSTAIN_TEXT


# --------------------------------------------------------------------------------------------
# Live LLM backend (Anthropic Messages API tool-use loop)
# --------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a risk-analytics assistant for a volatility-forecasting and Value-at-Risk \
project on Indian equities.

Rules:
- Get every number from the tools. Never estimate, recall or compute figures yourself.
- Say which dataset the figures come from (the tool results carry a 'dataset' field). If it is \
'simulated', say clearly that it is simulated data and not market performance.
- Use search_knowledge_base for definitions and methodology, and cite the source it returns.
- If a tool returns an error or the knowledge base has no hits, say you do not have that \
information. Do not guess.
- Do not give investment advice or predict price direction; this project only measures risk.
- Be concise: at most about 150 words."""


class AnthropicBackend:
    """Tool-use loop against the Anthropic Messages API.

    Requires the optional ``anthropic`` package and an API key. A ``client`` can be injected for
    testing.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        max_turns: int = 6,
        max_tokens: int = 800,
    ):
        self.model = model or os.environ.get("VOLVAR_LLM_MODEL", "claude-sonnet-5-5")
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError("Install the optional dependency: pip install anthropic") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY to use the live assistant backend.")
            self._client = anthropic.Anthropic()
        return self._client

    def respond(self, question: str, tools: ToolExecutor) -> str:
        client = self._get_client()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        for _ in range(self.max_turns):
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_SPECS,
                messages=messages,
            )
            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text").strip()
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = tools.run(block.name, dict(block.input))
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(output),
                        }
                    )
            messages.append({"role": "user", "content": results})
        return "I could not finish within the tool-call budget, so I have no answer."


__all__ = [
    "ABSTAIN_TEXT",
    "ADVICE_TEXT",
    "AnthropicBackend",
    "Backend",
    "MAX_QUESTION_CHARS",
    "Plan",
    "RuleBackend",
    "detect_model",
    "detect_tickers",
    "plan_question",
]
