"""Streamlit dashboard: model comparison, next-day forecasts, VaR backtests and the assistant.

Run with ``streamlit run app/dashboard.py`` after ``python scripts/run_pipeline.py``. The results
folder is read from ``VOLVAR_RESULTS_DIR`` (default ``results/``). Every chart has a table view,
and simulated results carry a banner so they cannot be mistaken for market results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volvar.assistant import (  # noqa: E402
    AnthropicBackend,
    Assistant,
    ResultsNotFoundError,
    ResultsStore,
    RuleBackend,
)
from volvar.config import BENCHMARK_MODEL, KB_DIR, MODEL_LABELS, RESULTS_DIR  # noqa: E402
from volvar.reporting import MODEL_COLORS, MODEL_COLORS_DARK  # noqa: E402
from volvar.risk import var_es  # noqa: E402

st.set_page_config(page_title="Volatility & VaR - Nifty", layout="wide")


# -- theme ---------------------------------------------------------------------------------------
def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older Streamlit or no theme info: assume light
        return False


DARK = is_dark()
COLORS = MODEL_COLORS_DARK if DARK else MODEL_COLORS
MUTED = "#6f6f6a" if DARK else "#9a9a94"  # returns / reference marks
CRITICAL = "#e66767" if DARK else "#d03b3b"  # violations (always paired with a marker shape)


# -- data ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_store(results_dir: str) -> ResultsStore:
    return ResultsStore(results_dir)


@st.cache_data(show_spinner=False)
def load_series(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Daily forecast/return series; absent when only the small summary files were kept."""
    d = Path(results_dir)
    try:
        return pd.read_parquet(d / "forecasts.parquet"), pd.read_parquet(d / "returns.parquet")
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_assistant(results_dir: str, live: bool) -> Assistant:
    backend = AnthropicBackend() if live else RuleBackend()
    return Assistant.from_dirs(results_dir, KB_DIR, backend)


def var_frame(
    returns: pd.DataFrame, forecasts: pd.DataFrame, cfg: dict, ticker, model, level, method
) -> pd.DataFrame:
    ret = returns[returns.ticker == ticker].set_index("date")["ret"]
    var_ = forecasts[(forecasts.ticker == ticker) & (forecasts.model == model)]
    variance = var_.set_index("date")["var"]
    v, e = var_es(ret, variance, level, method, cfg["fhs_window"], cfg["fhs_min_obs"])
    out = pd.DataFrame({"ret": ret, "var": v, "es": e}).dropna(subset=["var"])
    out["violation"] = out["ret"] < -out["var"]
    return out


def fig_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


# -- page ----------------------------------------------------------------------------------------
results_dir = os.environ.get("VOLVAR_RESULTS_DIR", str(RESULTS_DIR))
try:
    store = load_store(results_dir)
except ResultsNotFoundError as exc:
    st.error(str(exc))
    st.stop()

series = load_series(results_dir)
forecasts, returns = series if series is not None else (None, None)
cfg = store.meta["config"]
NO_SERIES = (
    "Daily series (forecasts.parquet, returns.parquet) are not in this results folder. "
    "Re-run `python scripts/run_pipeline.py` to enable this chart."
)

st.title("Volatility forecasting & Value-at-Risk")
if store.is_simulated:
    st.error(
        "SIMULATED DATA - pipeline validation only, not market performance. "
        "Run the pipeline on real Nifty data to see market results.",
        icon=":material/science:",
    )
else:
    st.caption(
        f"Real daily data, {store.meta['data_start']} to {store.meta['data_end']}; "
        f"out-of-sample from {store.meta['oos_start']}."
    )

with st.sidebar:
    st.header("Selection")
    ticker = st.selectbox("Ticker", store.tickers)
    model = st.selectbox(
        "Model",
        store.models,
        index=store.models.index(store.best_model),
        format_func=lambda m: MODEL_LABELS[m],
        help="Defaults to the model with the lowest out-of-sample QLIKE.",
    )
    level = st.radio("VaR confidence", [0.99, 0.95], format_func=lambda x: f"{x:.0%}", horizontal=True)
    method = st.radio(
        "VaR method",
        ["fhs", "normal"],
        format_func={"fhs": "Filtered historical simulation", "normal": "Normal"}.get,
    )
    st.caption(f"Results folder: `{results_dir}`")

tab_models, tab_fc, tab_bt, tab_ai = st.tabs(
    ["Model comparison", "Next-day forecast", "VaR backtest", "Ask the assistant"]
)

# -- model comparison ------------------------------------------------------------------------
with tab_models:
    comp = store.model_comparison.copy()
    st.subheader("Does any model beat the EWMA benchmark?")
    st.caption(
        f"Gain in out-of-sample QLIKE over EWMA (lambda=0.94), {int(comp['n'].iloc[0]):,} forecasts "
        "per model on a common sample. Higher is better; significance is a Newey-West "
        "Diebold-Mariano test at the 5% level."
    )
    bars = comp[comp.model != BENCHMARK_MODEL].sort_values("qlike_gain_vs_ewma")
    labels = [
        f"{g:+.4f} ({'significant' if p < 0.05 else 'not significant'})"
        for g, p in zip(bars.qlike_gain_vs_ewma, bars.dm_p, strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=bars.qlike_gain_vs_ewma,
            y=[MODEL_LABELS[m] for m in bars.model],
            orientation="h",
            marker_color=[COLORS[m] for m in bars.model],
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>QLIKE gain %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=MUTED)
    fig.update_layout(bargap=0.45, showlegend=False, hovermode="closest")
    fig_layout(fig, 300)
    fig.update_layout(margin=dict(l=8, r=150, t=8, b=8), legend=None)
    lo, hi = min(bars.qlike_gain_vs_ewma.min(), 0.0), max(bars.qlike_gain_vs_ewma.max(), 0.0)
    pad = 0.5 * (hi - lo)  # room for the value label beside the longest negative bar
    fig.update_xaxes(title="QLIKE gain over EWMA (higher = better)", range=[lo - pad, hi + 0.1 * pad])
    st.plotly_chart(fig, width="stretch")
    with st.expander("Table view"):
        st.dataframe(
            comp.rename(columns={"dm_p": "dm_p_value"})[
                ["label", "qlike", "mse", "qlike_gain_vs_ewma", "dm_t", "dm_p_value", "n"]
            ],
            hide_index=True,
            width="stretch",
        )
    if store.is_simulated:
        st.info(
            "The data were generated by a GJR-GARCH process, so GARCH-family models winning here is "
            "expected by construction. It validates the code, not the models on markets.",
            icon=":material/info:",
        )

# -- next-day forecast -----------------------------------------------------------------------
with tab_fc:
    nd = store.next_day[(store.next_day.ticker == ticker) & (store.next_day.model == model)].iloc[0]
    ndv = store.next_day_var[
        (store.next_day_var.ticker == ticker)
        & (store.next_day_var.model == model)
        & (store.next_day_var.method == method)
        & ((store.next_day_var.level - level).abs() < 1e-9)
    ].iloc[0]
    st.subheader(f"{ticker}: forecast for {nd['target_date']}")
    st.caption(f"{MODEL_LABELS[model]}, made after the {nd['as_of']} close.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volatility (daily)", f"{nd['vol_daily_pct']:.2f}%")
    c2.metric("Volatility (annualised)", f"{nd['vol_annual_pct']:.1f}%")
    c3.metric(f"{level:.0%} VaR", f"{ndv['var_pct']:.2f}%", help="One-day loss exceeded on ~1 - level of days.")
    c4.metric(f"{level:.0%} Expected Shortfall", f"{ndv['es_pct']:.2f}%")

    if series is None:
        st.info(NO_SERIES)
    else:
        window = st.select_slider("History shown (trading days)", [125, 250, 500, 1000], value=500)
        f = forecasts[(forecasts.ticker == ticker) & (forecasts.model == model)].set_index("date")["var"]
        r = returns[returns.ticker == ticker].set_index("date")["ret"]
        hist = pd.DataFrame({"forecast_vol": f**0.5, "abs_return": r.abs()}).dropna().tail(window)
        st.markdown("**Forecast volatility vs realised absolute return** (percent per day)")
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=hist.index, y=hist.abs_return, mode="markers", name="|daily return|",
                marker=dict(size=4, color=MUTED, opacity=0.7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist.forecast_vol, mode="lines", name=f"{MODEL_LABELS[model]} forecast",
                line=dict(color=COLORS[model], width=2),
            )
        )
        st.plotly_chart(fig_layout(fig), width="stretch")
        with st.expander("Table view"):
            st.dataframe(hist.round(4).reset_index().tail(60), hide_index=True, width="stretch")


    st.markdown("**All models, next day**")
    allm = store.next_day[store.next_day.ticker == ticker].copy()
    allm["model"] = allm["model"].map(MODEL_LABELS)
    st.dataframe(
        allm[["model", "vol_daily_pct", "vol_annual_pct"]].round(3), hide_index=True, width="stretch"
    )

# -- VaR backtest ----------------------------------------------------------------------------
with tab_bt:
    row = store.var_backtest[
        (store.var_backtest.ticker == ticker)
        & (store.var_backtest.model == model)
        & (store.var_backtest.method == method)
        & ((store.var_backtest.level - level).abs() < 1e-9)
    ].iloc[0]
    st.subheader(f"{ticker}: {level:.0%} VaR backtest, {MODEL_LABELS[model]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Violations", f"{int(row.violations)} of {int(row.n)}")
    c2.metric(f"Violation rate (expected {row.expected_rate:.2%})", f"{row.violation_rate:.2%}")
    c3.metric("Kupiec p-value", f"{row.kupiec_p:.3f}")
    c4.metric("Conditional coverage p", f"{row.cc_p:.3f}")
    if row.passes:
        st.badge("Passes at the 5% level", icon=":material/check_circle:", color="green")
    else:
        st.badge("Fails at the 5% level", icon=":material/error:", color="red")
    zone = str(row.basel_zone)
    if zone in {"green", "yellow", "red"}:
        icon = {"green": ":material/check_circle:", "yellow": ":material/warning:", "red": ":material/error:"}[zone]
        st.badge(
            f"Basel zone (latest year): {zone}, {int(row.latest_250d_exceptions)} exceptions",
            icon=icon,
            color=zone if zone != "yellow" else "orange",
        )

    if series is None:
        st.info(NO_SERIES)
    else:
        vf = var_frame(returns, forecasts, cfg, ticker, model, level, method)
        options = sorted({250, 500, 1000, len(vf)})
        window = st.select_slider(
            "History shown (trading days)", options, value=min(500, options[-1]), key="bt_window"
        )
        vf = vf.tail(window)
        viol = vf[vf.violation]
        st.markdown("**Daily return against the VaR threshold** (percent)")
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=vf.index, y=vf.ret, mode="markers", name="Daily return",
                marker=dict(size=4, color=MUTED, opacity=0.7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=vf.index, y=-vf["var"], mode="lines", name=f"-{level:.0%} VaR",
                line=dict(color=COLORS[model], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=viol.index, y=viol.ret, mode="markers", name=f"Violation ({len(viol)})",
                marker=dict(size=9, color=CRITICAL, symbol="x", line=dict(width=2, color=CRITICAL)),
            )
        )
        st.plotly_chart(fig_layout(fig, 400), width="stretch")
        with st.expander("Table view (violations in the window)"):
            st.dataframe(
                viol[["ret", "var", "es"]].round(3).reset_index().rename(columns={"ret": "return_pct"}),
                hide_index=True,
                width="stretch",
            )


    st.markdown("**All models for this ticker**")
    tbl = store.var_backtest[
        (store.var_backtest.ticker == ticker)
        & (store.var_backtest.method == method)
        & ((store.var_backtest.level - level).abs() < 1e-9)
    ].copy()
    tbl["model"] = tbl["model"].map(MODEL_LABELS)
    st.dataframe(
        tbl[["model", "violations", "violation_rate", "kupiec_p", "cc_p", "es_ratio", "passes"]].round(4),
        hide_index=True,
        width="stretch",
    )

# -- assistant ---------------------------------------------------------------------------------
with tab_ai:
    live = os.environ.get("VOLVAR_ASSISTANT_BACKEND", "rule").lower() == "anthropic"
    assistant = load_assistant(results_dir, live)
    st.subheader("Ask about the results or the methodology")
    st.caption(
        f"Backend: {assistant.backend.name}. Numbers come from the tools and are checked against "
        "their output. Try: 'What is the 99% VaR for " + store.tickers[0] + "?', "
        "'Which model has the lowest QLIKE?' or 'What is Expected Shortfall?'"
    )
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
    if question := st.chat_input("Your question"):
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            reply = assistant.ask(question)
            meta = (
                f"Tools: {', '.join(c['name'] for c in reply.tool_calls) or 'none'}"
                f" | Sources: {', '.join(reply.sources) or 'none'}"
                f" | Numbers verified: {'yes' if reply.grounded else 'NO'}"
            )
            content = reply.answer
        except (ValueError, RuntimeError) as exc:
            content, meta = f"Sorry: {exc}", ""
        st.session_state.chat.append({"role": "assistant", "content": content, "meta": meta})
        with st.chat_message("assistant"):
            st.markdown(content)
            st.caption(meta)

with st.sidebar:
    st.divider()
    st.caption(
        "Risk estimates for education and research. Not investment advice. "
        f"Dataset: {store.dataset}."
    )
