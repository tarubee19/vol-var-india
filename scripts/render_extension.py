#!/usr/bin/env python
"""Write the extension-study tables (results/extension/*) into README.md between the EXTENSION markers."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "extension"
LABELS = {"hist": "Historical (250d)", "ewma": "EWMA (benchmark)", "garch": "GARCH(1,1)-t", "gjr": "GJR-GARCH(1,1,1)-t",
          "har": "HAR (Garman-Klass)", "lgbm": "LightGBM (pooled, shipped)", "lgbm_log": "LightGBM on log target",
          "ridge": "Ridge, same features", "ridge_log": "Ridge, log features + log target (HAR-X style)",
          "xgb": "XGBoost", "rf": "Random forest", "mlp": "MLP (32-16)", "egarch": "EGARCH(1,1)-t"}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    comp = pd.read_csv(EXT / "extended_model_comparison.csv", index_col=0)
    comp = comp.reset_index().rename(columns={"index": "model"})
    t1 = pd.DataFrame({
        "Model": comp["model"].map(lambda m: LABELS.get(m, m)),
        "QLIKE": comp["qlike"].map("{:.4f}".format),
        "Gain vs EWMA": comp["gain_vs_ewma_pct"].map("{:+.2f}%".format),
        "MCS p-value": comp["mcs_pvalue"].map("{:.3f}".format),
        "In 90% MCS": comp["in_90pct_mcs"].map(lambda b: "**yes**" if b else "no"),
    })
    sens = pd.read_csv(EXT / "lgbm_sensitivity.csv")
    t2 = pd.DataFrame({"LightGBM setting": sens["setting"],
                       "QLIKE, 1st half (selection)": sens["qlike_first_half"].map("{:.4f}".format),
                       "QLIKE, 2nd half (check)": sens["qlike_second_half"].map("{:.4f}".format)})
    abl = pd.read_csv(EXT / "feature_group_ablation.csv")
    t3 = pd.DataFrame({"Feature group removed": abl["dropped"], "QLIKE": abl["qlike"].map("{:.4f}".format),
                       "Change": abl["qlike_increase_pct"].map("{:+.2f}%".format)})
    gain = pd.read_csv(EXT / "lgbm_gain_importance.csv", index_col=0).iloc[:, 0].head(8)
    t4 = pd.DataFrame({"Feature": gain.index, "Share of gain": gain.map("{:.1%}".format).to_numpy()})
    block = "\n\n".join([
        "**All models on the same out-of-sample forecasts** (QLIKE, lower is better; Model Confidence Set of Hansen, "
        "Lunde and Nason 2011 at 90%, stationary bootstrap, 2,000 replications):", md(t1),
        "**LightGBM sensitivity** (settings ranked on the first half of the out-of-sample period, checked on the second):", md(t2),
        "**What the model uses:** drop-a-feature-group ablation (re-trained walk-forward) and average gain share:", md(t3), md(t4)])
    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = text.index("<!-- EXTENSION:START -->"), text.index("<!-- EXTENSION:END -->")
    readme.write_text(text[: a + len("<!-- EXTENSION:START -->")] + "\n" + block + "\n" + text[b:])
    print("README extension block updated")


if __name__ == "__main__":
    main()
