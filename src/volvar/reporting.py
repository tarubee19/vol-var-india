"""Static figures for the README (matplotlib, validated categorical palette).

Colours follow the entity: each model keeps one hue across every figure. The six model
colours are the first six slots of a CVD-validated categorical palette; the heatmap uses a
blue <-> gray <-> red diverging scale. Every figure has a CSV twin in ``results/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from volvar.config import BENCHMARK_MODEL, MODEL_LABELS, MODEL_ORDER  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
MODEL_COLORS = dict(
    zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"], strict=True)
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def _title(ax, title: str, subtitle: str | None = None, label: str = "real") -> None:
    """Title, optional subtitle and (for simulated data) a red banner, stacked above the axes."""
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", fontsize=9, color=INK_2, va="bottom")
    if label == "simulated":
        ax.annotate("SIMULATED DATA - pipeline validation only, not market performance",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 46 if subtitle else 30),
                    textcoords="offset points", fontsize=8, color=CRITICAL, va="bottom")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _example_ticker(panels: dict[str, pd.DataFrame]) -> str:
    return "NIFTY50" if "NIFTY50" in panels else sorted(panels)[0]


def fig_model_comparison(results_dir: Path, label: str) -> None:
    comp = pd.read_csv(results_dir / "model_comparison.csv", index_col="model")
    comp = comp.drop(index=BENCHMARK_MODEL, errors="ignore").sort_values("qlike_gain_vs_ewma")
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    colors = [MODEL_COLORS[m] for m in comp.index]
    bars = ax.barh(
        [MODEL_LABELS.get(m, m) for m in comp.index],
        comp["qlike_gain_vs_ewma"],
        color=colors,
        height=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.axvline(0, color=AXIS, linewidth=1)
    span = max(float(comp["qlike_gain_vs_ewma"].abs().max()), 1e-6)
    for bar, (_, row) in zip(bars, comp.iterrows(), strict=True):
        v = row["qlike_gain_vs_ewma"]
        star = "*" if row["dm_p"] < 0.05 else ""
        ax.text(
            v + (0.02 * span if v >= 0 else -0.02 * span),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.4f}{star}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xlim(-1.35 * span, 1.35 * span)
    ax.set_xlabel("Mean QLIKE gain over EWMA  (higher = better;  * Diebold-Mariano p < 0.05)")
    ax.grid(axis="y", visible=False)
    _title(ax, "Out-of-sample variance forecast accuracy vs the EWMA benchmark",
           "Pooled over all tickers, one-day-ahead, walk-forward", label)
    _save(fig, results_dir / "figures" / "fig_model_comparison.png")


def fig_regime_gain(results_dir: Path, label: str) -> None:
    reg = pd.read_csv(results_dir / "loss_by_regime.csv", index_col="regime")
    order = [r for r in ("calm", "normal", "turbulent") if r in reg.index]
    models = [m for m in MODEL_ORDER if m in reg.columns and m != BENCHMARK_MODEL]
    gains = {m: [reg.loc[r, BENCHMARK_MODEL] - reg.loc[r, m] for r in order] for m in models}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(order))
    width = 0.15
    for i, m in enumerate(models):
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            gains[m],
            width=width * 0.9,
            color=MODEL_COLORS[m],
            edgecolor=SURFACE,
            linewidth=2,
            label=MODEL_LABELS.get(m, m),
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, [f"{r.capitalize()} markets" for r in order])
    ax.set_ylabel("QLIKE gain over EWMA (higher = better)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    _title(ax, "Does the ranking hold across volatility regimes?",
           "Regime = tercile of trailing 22-day market variance (known the day before)", label)
    _save(fig, results_dir / "figures" / "fig_regime_gain.png")


def fig_vol_forecast(results_dir: Path, wide_fc: pd.DataFrame, ret: pd.Series, ticker: str,
                     best: str, label: str, last_n: int = 500) -> None:
    models = [best] if best == BENCHMARK_MODEL else [BENCHMARK_MODEL, best]
    df = wide_fc[models].dropna().tail(last_n)
    r = ret.reindex(df.index).abs()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df.index, r, color=AXIS, linewidth=0.8, label="|daily return|")
    ends = {m: float(np.sqrt(df[m].iloc[-1])) for m in models}
    for m in models:
        ax.plot(df.index, np.sqrt(df[m]), color=MODEL_COLORS[m], linewidth=2, label=MODEL_LABELS.get(m, m))
    ymax = float(max(r.max(), max(np.sqrt(df[m]).max() for m in models)))
    gap = 0.07 * ymax  # minimum vertical distance between direct labels
    placed: list[float] = []
    for m, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_label = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_label)
        ax.text(df.index[-1] + pd.Timedelta(days=6), y_label, MODEL_LABELS.get(m, m),
                color=INK_2, fontsize=9, va="center")
    ax.set_ylabel("Daily volatility (%)")
    ax.set_xlim(df.index[0], df.index[-1] + (df.index[-1] - df.index[0]) * 0.22)
    ticks = pd.date_range(df.index[0], df.index[-1], freq="QS")
    ax.set_xticks(ticks, [t.strftime("%b %Y") for t in ticks])
    ax.legend(loc="upper left", fontsize=9)
    _title(ax, f"{ticker}: one-day-ahead volatility forecast vs realised move",
           f"Last {len(df)} out-of-sample trading days", label)
    _save(fig, results_dir / "figures" / "fig_vol_forecast.png")


def fig_var_backtest(results_dir: Path, ret: pd.Series, var: pd.Series, ticker: str, model: str,
                     level: float, label: str, last_n: int = 750) -> None:
    v = var.dropna().tail(last_n)
    r = ret.reindex(v.index)
    breach = r < -v
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(r.index, r, color=AXIS, linewidth=0.8, label="Daily return")
    ax.plot(v.index, -v, color=MODEL_COLORS[model], linewidth=1.8,
            label=f"-{int(level * 100)}% VaR ({MODEL_LABELS.get(model, model)}, filtered historical simulation)")
    ax.scatter(r.index[breach], r[breach], marker="x", s=36, color=CRITICAL, linewidths=1.6, zorder=3,
               label=f"VaR breach ({int(breach.sum())} of {len(v)} days; expected {(1 - level) * len(v):.1f})")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylabel("Return (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=9)
    _title(ax, f"{ticker}: {int(level * 100)}% one-day Value-at-Risk and breaches",
           f"Last {len(v)} out-of-sample days; VaR is a loss threshold expected to be breached on "
           f"{100 * (1 - level):.0f}% of days", label)
    _save(fig, results_dir / "figures" / "fig_var_backtest.png")


def fig_violation_heatmap(results_dir: Path, label: str, level: float = 0.99, method: str = "fhs") -> None:
    t = pd.read_csv(results_dir / "var_backtest.csv")
    t = t[(t["method"] == method) & np.isclose(t["level"], level)]
    if t.empty:
        return
    models = [m for m in MODEL_ORDER if m in set(t["model"])]
    ratio = t.pivot(index="ticker", columns="model", values="violation_rate")[models] / (1 - level)
    counts = t.pivot(index="ticker", columns="model", values="violations")[models]
    fails = ~t.pivot(index="ticker", columns="model", values="passes")[models].astype(bool)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(ratio) + 2.2))
    im = ax.imshow(ratio.to_numpy(), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(models)), [MODEL_LABELS.get(m, m).split(" (")[0] for m in models],
                  rotation=20, ha="right")
    ax.set_yticks(range(len(ratio)), ratio.index)
    ax.grid(False)
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            txt = f"{int(counts.iloc[i, j])}" + ("*" if fails.iloc[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=INK,
                    fontweight="bold" if fails.iloc[i, j] else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("breach rate / expected rate  (1 = well calibrated)", color=INK_2)
    cb.outline.set_visible(False)
    _title(ax, f"{int(level * 100)}% VaR breach counts by ticker and model",
           "Cell = breach count; * = fails conditional-coverage test (5%). Red = too many breaches", label)
    _save(fig, results_dir / "figures" / "fig_violation_heatmap.png")


def make_figures(results_dir, result, panels, var_series, cfg, label: str) -> None:
    """Render every README figure into ``results_dir/figures``."""
    results_dir = Path(results_dir)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    _style()
    comp = pd.read_csv(results_dir / "model_comparison.csv", index_col="model")
    best = str(comp["qlike"].idxmin())
    ticker = _example_ticker(panels)
    fc = result.forecasts[result.forecasts["ticker"] == ticker].pivot(
        index="date", columns="model", values="var"
    ).sort_index()

    fig_model_comparison(results_dir, label)
    fig_regime_gain(results_dir, label)
    fig_vol_forecast(results_dir, fc, panels[ticker]["ret"], ticker, best, label)
    key = (ticker, best, "fhs", 0.99)
    if key in var_series:
        fig_var_backtest(results_dir, panels[ticker]["ret"], var_series[key][0], ticker, best, 0.99, label)
    fig_violation_heatmap(results_dir, label)
