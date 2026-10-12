"""Tools the assistant can call, and the executor that runs them.

Every tool returns a plain JSON-serialisable dict that carries the dataset label, so a
simulated-data answer can never be mistaken for a market one. Arguments are validated strictly
(the live LLM backend supplies them), and bad input yields ``{"ok": False, "error": ...}``
instead of an exception.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..config import MODEL_LABELS, VAR_LEVELS
from .retriever import KnowledgeBase
from .store import ResultsStore

METHODS = ("fhs", "normal")
METRICS = ("qlike", "mse")

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_available",
        "description": (
            "Describe what the loaded results cover: dataset label (simulated or real), tickers, "
            "models, VaR methods and levels, and the date of the next-day forecast."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_volatility_forecast",
        "description": (
            "Next-trading-day volatility forecast for one ticker from one model, as daily and "
            "annualised percent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "model": {
                    "type": "string",
                    "enum": list(MODEL_LABELS),
                    "description": "Omit to use the model with the lowest out-of-sample QLIKE.",
                },
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_var_es",
        "description": (
            "Next-day 1-day Value-at-Risk and Expected Shortfall (positive percent of position "
            "value) for one ticker, model, method and confidence level."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "level": {"type": "number", "enum": list(VAR_LEVELS)},
                "model": {"type": "string", "enum": list(MODEL_LABELS)},
                "method": {"type": "string", "enum": list(METHODS)},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_models",
        "description": (
            "Out-of-sample forecast accuracy of every volatility model (QLIKE and MSE, lower is "
            "better), with the gain over the EWMA benchmark and its Diebold-Mariano p-value "
            "(the test is on QLIKE)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"metric": {"type": "string", "enum": list(METRICS)}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_backtest",
        "description": (
            "Historical VaR backtest for one ticker: violations, violation rate against the "
            "expected rate, Kupiec and Christoffersen p-values, ES ratio and Basel zone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "level": {"type": "number", "enum": list(VAR_LEVELS)},
                "model": {"type": "string", "enum": list(MODEL_LABELS)},
                "method": {"type": "string", "enum": list(METHODS)},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_backtest_summary",
        "description": (
            "VaR backtest results averaged across all tickers for one model, method and level: "
            "mean violation rate, share of tickers passing, mean ES ratio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "number", "enum": list(VAR_LEVELS)},
                "model": {"type": "string", "enum": list(MODEL_LABELS)},
                "method": {"type": "string", "enum": list(METHODS)},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the project's methodology notes (volatility, GARCH, EWMA, HAR, LightGBM, "
            "VaR, Expected Shortfall, backtesting, data limits). Use for definitions and "
            "'why/how' questions. Returns no hits when the notes do not cover the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

DATA_TOOLS = frozenset(spec["name"] for spec in TOOL_SPECS if spec["name"] != "search_knowledge_base")


class LookupFailure(ValueError):
    """The request was well-formed but names something that does not exist in the results."""


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool


@dataclass
class ToolExecutor:
    """Runs tools against a store + knowledge base and remembers what was returned."""

    store: ResultsStore
    kb: KnowledgeBase
    calls: list[ToolCallRecord] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = dict(arguments or {})
        handler = getattr(self, f"_tool_{name}", None) if name in _ALL_TOOL_NAMES else None
        if handler is None:
            result: dict[str, Any] = {"ok": False, "error": f"Unknown tool '{name}'."}
        else:
            try:
                result = handler(**arguments)
            except TypeError as exc:
                result = {
                    "ok": False,
                    "error_type": "invalid",
                    "error": f"Bad arguments for {name}: {exc}",
                }
            except LookupFailure as exc:
                result = {"ok": False, "error_type": "not_found", "error": str(exc)}
            except ValueError as exc:
                result = {"ok": False, "error_type": "invalid", "error": str(exc)}
        self.calls.append(ToolCallRecord(name, arguments, bool(result.get("ok"))))
        self.evidence.append(result)
        return result

    # -- argument resolution ----------------------------------------------------------------
    def _ticker(self, ticker: str) -> str:
        resolved = self.store.resolve_ticker(str(ticker))
        if resolved is None:
            raise LookupFailure(
                f"Unknown ticker '{ticker}'. Available: {', '.join(self.store.tickers)}."
            )
        return resolved

    def _model(self, model: str | None) -> str:
        if model is None:
            return self.store.best_model
        if model not in self.store.models:
            raise ValueError(f"Unknown model '{model}'. Available: {', '.join(self.store.models)}.")
        return model

    @staticmethod
    def _level(level: float | None) -> float:
        if level is None:
            return 0.99
        level = float(level)
        if level > 1:  # accept 99 as well as 0.99
            level = level / 100
        for allowed in VAR_LEVELS:
            if math.isclose(level, allowed):
                return allowed
        raise ValueError(f"Unsupported level {level}. Available: {', '.join(map(str, VAR_LEVELS))}.")

    @staticmethod
    def _method(method: str | None) -> str:
        method = "fhs" if method is None else method
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {', '.join(METHODS)}.")
        return method

    def _base(self, **extra: Any) -> dict[str, Any]:
        return {"ok": True, "dataset": self.store.dataset, **extra}

    # -- tools ------------------------------------------------------------------------------
    def _tool_list_available(self) -> dict[str, Any]:
        meta = self.store.meta
        return self._base(
            tickers=self.store.tickers,
            n_tickers=len(self.store.tickers),
            models={m: MODEL_LABELS[m] for m in self.store.models},
            methods=list(METHODS),
            levels=list(VAR_LEVELS),
            data_start=meta.get("data_start"),
            data_end=meta.get("data_end"),
            oos_start=meta.get("oos_start"),
            oos_end=meta.get("oos_end"),
            next_day_target=meta.get("next_day_target"),
            best_model_by_qlike=self.store.best_model,
        )

    def _tool_get_volatility_forecast(
        self, ticker: str, model: str | None = None
    ) -> dict[str, Any]:
        ticker, model = self._ticker(ticker), self._model(model)
        df = self.store.next_day
        row = df[(df.ticker == ticker) & (df.model == model)]
        if row.empty:
            raise LookupFailure(f"No forecast for {ticker} with model {model}.")
        r = row.iloc[0]
        return self._base(
            ticker=ticker,
            model=model,
            model_label=MODEL_LABELS[model],
            as_of=str(r["as_of"]),
            target_date=str(r["target_date"]),
            vol_daily_pct=round(float(r["vol_daily_pct"]), 3),
            vol_annual_pct=round(float(r["vol_annual_pct"]), 2),
        )

    def _tool_get_var_es(
        self,
        ticker: str,
        level: float | None = None,
        model: str | None = None,
        method: str | None = None,
    ) -> dict[str, Any]:
        ticker, model = self._ticker(ticker), self._model(model)
        level, method = self._level(level), self._method(method)
        df = self.store.next_day_var
        mask = (
            (df.ticker == ticker)
            & (df.model == model)
            & (df.method == method)
            & ((df.level - level).abs() < 1e-9)
        )
        row = df[mask]
        if row.empty:
            raise LookupFailure(f"No VaR for {ticker}/{model}/{method}/{level}.")
        r = row.iloc[0]
        vol = self.store.next_day
        v = vol[(vol.ticker == ticker) & (vol.model == model)]
        return self._base(
            ticker=ticker,
            model=model,
            model_label=MODEL_LABELS[model],
            method=method,
            level=level,
            level_pct=round(level * 100, 1),
            var_pct=round(float(r["var_pct"]), 3),
            es_pct=round(float(r["es_pct"]), 3),
            vol_daily_pct=round(float(v.iloc[0]["vol_daily_pct"]), 3) if not v.empty else None,
            as_of=str(r["as_of"]),
            target_date=str(r["target_date"]),
        )

    def _tool_compare_models(self, metric: str | None = None) -> dict[str, Any]:
        metric = "qlike" if metric is None else metric
        if metric not in METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Available: {', '.join(METRICS)}.")
        df = self.store.model_comparison.sort_values(metric).reset_index(drop=True)
        rows = []
        for i, r in df.iterrows():
            is_benchmark = math.isnan(r["dm_p"]) if "dm_p" in df else False
            rows.append(
                {
                    "rank": int(i) + 1,
                    "model": str(r["model"]),
                    "label": str(r["label"]),
                    "qlike": round(float(r["qlike"]), 4),
                    "mse": round(float(r["mse"]), 3),
                    "qlike_gain_vs_ewma": None
                    if is_benchmark
                    else round(float(r["qlike_gain_vs_ewma"]), 4),
                    "dm_p_value": None if is_benchmark else round(float(r["dm_p"]), 4),
                    "is_benchmark": bool(is_benchmark),
                }
            )
        return self._base(
            metric=metric,
            best_model=rows[0]["model"],
            n_observations=int(df.iloc[0]["n"]),
            ranking=rows,
        )

    def _tool_get_backtest(
        self,
        ticker: str,
        level: float | None = None,
        model: str | None = None,
        method: str | None = None,
    ) -> dict[str, Any]:
        ticker, model = self._ticker(ticker), self._model(model)
        level, method = self._level(level), self._method(method)
        df = self.store.var_backtest
        mask = (
            (df.ticker == ticker)
            & (df.model == model)
            & (df.method == method)
            & ((df.level - level).abs() < 1e-9)
        )
        row = df[mask]
        if row.empty:
            raise LookupFailure(f"No backtest for {ticker}/{model}/{method}/{level}.")
        r = row.iloc[0]
        zone = str(r["basel_zone"])
        return self._base(
            ticker=ticker,
            model=model,
            model_label=MODEL_LABELS[model],
            method=method,
            level=level,
            level_pct=round(level * 100, 1),
            n_days=int(r["n"]),
            violations=int(r["violations"]),
            violation_rate_pct=round(float(r["violation_rate"]) * 100, 2),
            expected_rate_pct=round(float(r["expected_rate"]) * 100, 2),
            kupiec_p=round(float(r["kupiec_p"]), 4),
            independence_p=_opt(r["independence_p"], 4),
            conditional_coverage_p=_opt(r["cc_p"], 4),
            es_ratio=_opt(r["es_ratio"], 3),
            latest_250d_exceptions=int(r["latest_250d_exceptions"]),
            basel_zone=None if zone in ("n/a", "nan") else zone,
            passes=bool(r["passes"]),
        )

    def _tool_get_backtest_summary(
        self,
        level: float | None = None,
        model: str | None = None,
        method: str | None = None,
    ) -> dict[str, Any]:
        model = self._model(model)
        level, method = self._level(level), self._method(method)
        df = self.store.var_summary
        mask = (
            (df.model == model) & (df.method == method) & ((df.level - level).abs() < 1e-9)
        )
        row = df[mask]
        if row.empty:
            raise LookupFailure(f"No summary for {model}/{method}/{level}.")
        r = row.iloc[0]
        green = r.get("share_latest_green")
        return self._base(
            model=model,
            model_label=MODEL_LABELS[model],
            method=method,
            level=level,
            level_pct=round(level * 100, 1),
            n_tickers=int(r["n_tickers"]),
            mean_violation_rate_pct=round(float(r["mean_violation_rate"]) * 100, 2),
            expected_rate_pct=round(float(r["expected_rate"]) * 100, 2),
            pass_rate_pct=round(float(r["pass_rate"]) * 100, 1),
            mean_es_ratio=_opt(r["mean_es_ratio"], 3),
            share_latest_window_green_pct=None
            if green is None or _isnan(green)
            else round(float(green) * 100, 1),
        )

    def _tool_search_knowledge_base(self, query: str, k: int = 3) -> dict[str, Any]:
        found = self.kb.search(str(query), k=max(1, min(int(k), 5)))
        for hit in found.hits:
            if hit.chunk.ref not in self.sources:
                self.sources.append(hit.chunk.ref)
        return {
            "ok": True,
            "abstained": found.abstained,
            "hits": [
                {
                    "source": h.chunk.ref,
                    "text": h.chunk.text,
                    "score": round(h.score, 3),
                    "coverage": round(h.coverage, 2),
                }
                for h in found.hits
            ],
        }


_ALL_TOOL_NAMES = frozenset(spec["name"] for spec in TOOL_SPECS)


def _isnan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _opt(value: Any, digits: int) -> float | None:
    return None if _isnan(value) else round(float(value), digits)
