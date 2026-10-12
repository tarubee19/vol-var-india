"""FastAPI service over the pipeline results and the assistant.

Run with ``uvicorn volvar.api:app``. The service is read-only: it serves files produced by
``scripts/run_pipeline.py`` (directory set by ``VOLVAR_RESULTS_DIR``) and never recomputes models.
Every response carries an ``X-Data-Label`` header (``simulated`` or ``real``) so simulated numbers
cannot be mistaken for market ones.

This is a demo service: there is no authentication or rate limiting, so do not expose it to the
internet as is.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .assistant import (
    AnthropicBackend,
    Assistant,
    ResultsNotFoundError,
    RuleBackend,
)
from .assistant.backends import MAX_QUESTION_CHARS, Backend
from .assistant.tools import ToolExecutor
from .config import KB_DIR, RESULTS_DIR

ModelName = Literal["hist", "ewma", "garch", "gjr", "har", "lgbm"]
MethodName = Literal["fhs", "normal"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    question: str
    answer: str
    backend: str
    dataset: str
    grounded: bool
    unsupported: list[str]
    tool_calls: list[dict[str, Any]]
    sources: list[str]


def _default_backend() -> Backend:
    if os.environ.get("VOLVAR_ASSISTANT_BACKEND", "rule").lower() == "anthropic":
        return AnthropicBackend()
    return RuleBackend()


def create_app(
    results_dir: str | Path | None = None,
    kb_dir: str | Path | None = None,
    backend: Backend | None = None,
) -> FastAPI:
    """Build the app. Results are loaded at startup; a missing directory yields 503s, not a crash."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.assistant = None
        app.state.load_error = None
        try:
            app.state.assistant = Assistant.from_dirs(
                results_dir or RESULTS_DIR, kb_dir or KB_DIR, backend or _default_backend()
            )
        except ResultsNotFoundError as exc:
            app.state.load_error = str(exc)
        yield

    app = FastAPI(
        title="volvar - volatility and VaR service",
        version=__version__,
        description=(
            "Next-day volatility, Value-at-Risk and Expected Shortfall forecasts, backtests and a "
            "grounded assistant for Indian equities. Check the `dataset` field and `X-Data-Label` "
            "header: results from simulated data are pipeline validation only."
        ),
        lifespan=lifespan,
    )

    def assistant_of(request: Request) -> Assistant:
        assistant: Assistant | None = request.app.state.assistant
        if assistant is None:
            raise HTTPException(503, request.app.state.load_error or "Results not loaded.")
        return assistant

    def call(request: Request, tool: str, **kwargs: Any) -> dict[str, Any]:
        assistant = assistant_of(request)
        args = {k: v for k, v in kwargs.items() if v is not None}
        result = ToolExecutor(assistant.store, assistant.kb).run(tool, args)
        if not result["ok"]:
            status = 404 if result.get("error_type") == "not_found" else 422
            raise HTTPException(status, result["error"])
        return result

    @app.middleware("http")
    async def label_responses(request: Request, call_next):
        response = await call_next(request)
        assistant = getattr(request.app.state, "assistant", None)
        if assistant is not None:
            response.headers["X-Data-Label"] = assistant.store.dataset
        return response

    @app.get("/health", tags=["meta"])
    def health(request: Request):
        assistant = request.app.state.assistant
        if assistant is None:
            return JSONResponse(
                {"status": "unavailable", "detail": request.app.state.load_error}, status_code=503
            )
        meta = assistant.store.meta
        return {
            "status": "ok",
            "dataset": assistant.store.dataset,
            "n_tickers": len(assistant.store.tickers),
            "next_day_target": meta.get("next_day_target"),
            "version": __version__,
        }

    @app.get("/meta", tags=["meta"])
    def meta(request: Request):
        """Dataset label, tickers, models, evaluation period."""
        return call(request, "list_available")

    @app.get("/forecast/{ticker}", tags=["forecasts"])
    def forecast(request: Request, ticker: str, model: ModelName | None = None):
        """Next-trading-day volatility forecast (daily and annualised percent)."""
        return call(request, "get_volatility_forecast", ticker=ticker, model=model)

    @app.get("/var/{ticker}", tags=["forecasts"])
    def var(
        request: Request,
        ticker: str,
        level: float | None = Query(None, description="0.95 or 0.99 (default 0.99)"),
        model: ModelName | None = None,
        method: MethodName | None = None,
    ):
        """Next-day one-day Value-at-Risk and Expected Shortfall in percent of position value."""
        return call(request, "get_var_es", ticker=ticker, level=level, model=model, method=method)

    @app.get("/backtest/{ticker}", tags=["backtests"])
    def backtest(
        request: Request,
        ticker: str,
        level: float | None = Query(None, description="0.95 or 0.99 (default 0.99)"),
        model: ModelName | None = None,
        method: MethodName | None = None,
    ):
        """Historical VaR backtest for one ticker."""
        return call(request, "get_backtest", ticker=ticker, level=level, model=model, method=method)

    @app.get("/backtest-summary", tags=["backtests"])
    def backtest_summary(
        request: Request,
        level: float | None = Query(None, description="0.95 or 0.99 (default 0.99)"),
        model: ModelName | None = None,
        method: MethodName | None = None,
    ):
        """VaR backtest averaged over all tickers."""
        return call(request, "get_backtest_summary", level=level, model=model, method=method)

    @app.get("/models", tags=["backtests"])
    def models(request: Request, metric: Literal["qlike", "mse"] = "qlike"):
        """Out-of-sample accuracy of every volatility model, with Diebold-Mariano vs EWMA."""
        return call(request, "compare_models", metric=metric)

    @app.post("/assistant", response_model=AskResponse, tags=["assistant"])
    def ask(request: Request, body: AskRequest):
        """Ask a question; numbers in the answer are checked against tool output."""
        assistant = assistant_of(request)
        try:
            return assistant.ask(body.question).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:  # live backend without a key / package
            raise HTTPException(503, str(exc)) from exc

    return app


app = create_app()
