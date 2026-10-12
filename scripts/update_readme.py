#!/usr/bin/env python
"""Write the README results block from the files in results/.

Every number in the block is read from results/*.csv|json, never typed by hand. After a new run
(for example on real data), run:

    python scripts/update_readme.py

The block sits between the RESULTS markers in README.md; everything outside it is left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volvar.config import BENCHMARK_MODEL, MODEL_LABELS  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def accuracy_section(res: Path, summary: dict) -> str:
    comp = pd.read_csv(res / "model_comparison.csv")
    won = summary.get("tickers_won_by_model", {})
    n_t = summary["n_tickers"]
    rows = []
    for _, r in comp.sort_values("qlike").iterrows():
        bench = r["model"] == BENCHMARK_MODEL
        rows.append(
            {
                "Model": r["label"],
                "QLIKE": f"{r['qlike']:.4f}",
                "Gain vs EWMA": "benchmark" if bench else f"{r['qlike_gain_vs_ewma']:+.4f}",
                "DM p-value": "-" if bench else pval(r["dm_p"]),
                "MSE": f"{r['mse']:.2f}",
                "Tickers won": f"{won.get(r['model'], 0)} of {n_t}",
            }
        )
    better = comp[(comp.model != BENCHMARK_MODEL) & (comp.qlike_gain_vs_ewma > 0) & (comp.dm_p < 0.05)]
    worse = comp[(comp.model != BENCHMARK_MODEL) & (comp.qlike_gain_vs_ewma < 0) & (comp.dm_p < 0.05)]
    names = lambda d: ", ".join(d.sort_values("qlike")["label"]) or "none"  # noqa: E731
    best = comp.sort_values("qlike").iloc[0]
    text = [
        f"**Variance forecast accuracy** - {int(comp['n'].iloc[0]):,} out-of-sample forecasts per "
        f"model on a common sample ({summary['oos_start']} to {summary['oos_end']}). QLIKE: lower is "
        "better. The gain is the QLIKE reduction relative to EWMA; DM p is a Newey-West "
        "Diebold-Mariano test pooled over tickers.",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        f"Lowest QLIKE: {best['label']}. Significantly better than EWMA at the 5% level: "
        f"{names(better)}. Significantly worse: {names(worse)}.",
    ]
    return "\n".join(text)


def regime_section(res: Path) -> str:
    reg = pd.read_csv(res / "loss_by_regime.csv").set_index("regime")
    order = [m for m in MODEL_LABELS if m in reg.columns]
    table = reg.loc[[r for r in ("calm", "normal", "turbulent") if r in reg.index], order]
    table = table.map(lambda v: f"{v:.4f}").rename(columns=MODEL_LABELS)
    table.insert(0, "Regime", [r.capitalize() for r in table.index])
    best = {r: MODEL_LABELS[reg.loc[r, order].idxmin()] for r in table["Regime"].str.lower()}
    note = "; ".join(f"{r}: {m}" for r, m in best.items())
    return (
        "**QLIKE by volatility regime** (terciles of trailing 22-day market variance, known the "
        "day before).\n\n"
        + md_table(table.reset_index(drop=True))
        + f"\n\nLowest QLIKE per regime - {note}."
    )


def covid_section(res: Path) -> str:
    path = res / "loss_covid_window.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    return "**COVID window (Feb-Jun 2020)**\n\n" + md_table(df.round(4))


def var_table(vs: pd.DataFrame, level: float) -> str:
    rows = []
    for model in [m for m in MODEL_LABELS if m in set(vs.model)]:
        row = {"Model": MODEL_LABELS[model]}
        for label, method in (("FHS", "fhs"), ("Normal", "normal")):
            r = vs[(vs.model == model) & (vs.method == method) & ((vs.level - level).abs() < 1e-9)]
            if r.empty:
                row[f"{label}: violation rate"] = row[f"{label}: tickers passing"] = "-"
            else:
                r = r.iloc[0]
                row[f"{label}: violation rate"] = pct(r["mean_violation_rate"])
                row[f"{label}: tickers passing"] = pct(r["pass_rate"], 0)
                row[f"{label}: ES ratio"] = f"{r['mean_es_ratio']:.2f}"
        rows.append(row)
    return md_table(pd.DataFrame(rows)[["Model", *[c for c in rows[0] if c != "Model"]]])


def var_section(res: Path) -> str:
    vs = pd.read_csv(res / "var_summary.csv")
    n_t = int(vs["n_tickers"].iloc[0])
    return (
        f"**Value-at-Risk backtests** - mean one-day violation rate across {n_t} tickers, the share "
        "of tickers passing the conditional-coverage test at the 5% level, and the mean ES ratio "
        "(realised / predicted Expected Shortfall on violation days; 1 is ideal). FHS = filtered "
        "historical simulation.\n\n"
        "99% level (expected violation rate 1.00%):\n\n"
        + var_table(vs, 0.99)
        + "\n\n95% level (expected violation rate 5.00%):\n\n"
        + var_table(vs, 0.95)
    )


def assistant_section(res: Path) -> str:
    path = res / "assistant_eval.json"
    if not path.exists():
        return ""
    e = json.loads(path.read_text())
    ra = e["retrieval_test"]
    ts, ee = e["tool_selection"], e["end_to_end"]
    n_kb = ra["bm25"]["n"]
    rows = [
        {
            "Check": "Retrieval hit@1 / hit@3 / MRR, BM25, test split",
            "Result": f"{pct(ra['bm25']['hit@1'], 0)} / {pct(ra['bm25']['hit@3'], 0)} / {ra['bm25']['mrr']:.2f}",
            "n": n_kb,
        },
        {
            "Check": "Retrieval hit@1 / hit@3 / MRR, TF-IDF baseline, test split",
            "Result": f"{pct(ra['tfidf']['hit@1'], 0)} / {pct(ra['tfidf']['hit@3'], 0)} / {ra['tfidf']['mrr']:.2f}",
            "n": n_kb,
        },
        {
            "Check": "Tool selection, held-out test split",
            "Result": f"{pct(ts['test']['tool_args_accuracy'], 0)}",
            "n": ts["test"]["n"],
        },
        {
            "Check": "Tool selection, paraphrase stress split",
            "Result": f"{pct(ts['stress']['tool_args_accuracy'], 0)}",
            "n": ts["stress"]["n"],
        },
        {
            "Check": "Refuses investment advice, test split",
            "Result": pct(ee["test"]["advice_refused"], 0),
            "n": ee["test"]["n_advice"],
        },
        {
            "Check": "Abstains on out-of-scope questions, test split",
            "Result": pct(ee["test"]["oos_abstained"], 0),
            "n": ee["test"]["n_oos"],
        },
        {
            "Check": "Answers whose numbers all trace to tool output",
            "Result": f"{pct(ee['test']['grounded_rate'], 0)} (dev {pct(ee['dev']['grounded_rate'], 0)})",
            "n": ts["test"]["n"],
        },
    ]
    sweep = next(s for s in e["abstention_sweep"] if abs(s["min_coverage"] - e["min_coverage"]) < 1e-9)
    wrong = round(sweep["test_false_abstain"] * sweep["test_n_in_scope"])
    return (
        f"**Assistant evaluation** ({e['n_questions']} author-written questions, rule-based backend; "
        f"results dataset: {e['dataset']}). Small and in-domain: a regression test, not a benchmark.\n\n"
        + md_table(pd.DataFrame(rows))
        + "\n\nCaveats. The test split was written by the same author as the router, so its 100% "
        "tool-selection score is optimistic; the stress split (naturally worded paraphrases, never "
        "tuned against) is the honest number. One test question exposed a missing term in a "
        "methodology note (look-ahead bias) that was added afterwards, so test retrieval is slightly "
        f"optimistic too. At the abstention threshold chosen on the dev split ({e['min_coverage']}), "
        f"{wrong} of {sweep['test_n_in_scope']} valid in-scope test questions are wrongly declined."
    )


def build_block(res: Path) -> str:
    summary = json.loads((res / "summary.json").read_text())
    simulated = summary["dataset"] == "simulated"
    head = (
        "### Results on SIMULATED data (pipeline validation, not market performance)"
        if simulated
        else "### Results on real Nifty 50 data"
    )
    note = (
        "> The numbers below come from a simulated GJR-GARCH universe (tickers `SIM01`...). "
        "GARCH-family models winning is therefore expected by construction. Real-data results "
        "come from `make fetch && make run`, then `python scripts/update_readme.py`."
        if simulated
        else f"> {summary['n_tickers']} series, "
        f"{summary['oos_start']} to {summary['oos_end']} out of sample."
    )
    parts = [
        head,
        "",
        note,
        "",
        accuracy_section(res, summary),
        "",
        regime_section(res),
        "",
        covid_section(res),
        var_section(res),
        "",
        assistant_section(res),
    ]
    text = "\n".join(p for p in parts)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--print", action="store_true", help="print the block instead of writing")
    args = ap.parse_args()

    block = build_block(Path(args.results_dir))
    if args.print:
        print(block)
        return
    readme = Path(args.readme)
    text = readme.read_text()
    if START not in text or END not in text:
        sys.exit(f"{readme} has no {START} ... {END} block to fill.")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{block}\n{END}{after}")
    print(f"Updated {readme}")


if __name__ == "__main__":
    main()
