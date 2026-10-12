# Methodology

This document specifies exactly what the pipeline computes, so a reader can check the code against
it. Short explanatory notes for each topic also live in [`docs/kb/`](kb/), which is what the
assistant retrieves from.

## 1. Data and conventions

| Item | Convention |
|---|---|
| Universe | 15 long-listed Nifty 50 stocks plus the Nifty 50 index (`^NSEI`), listed in `scripts/fetch_data.py` |
| Source | Yahoo Finance via `yfinance`, `auto_adjust=True` (split and dividend adjusted) |
| Fields | Open, High, Low, Close, Volume, validated by `data.validate_ohlcv`: required columns, de-duplicated sorted dates, finite positive prices and consistent OHLC (invalid rows are dropped; it raises if more than 2% of rows are invalid or fewer than 300 remain) |
| Returns | daily log returns in **percent**: `r_t = 100 * ln(P_t / P_{t-1})` |
| Variance | in percent squared; volatility = `sqrt(variance)` per day; annualised = daily x `sqrt(252)` |
| Target | `r_t^2`, the squared close-to-close return (zero-mean proxy for daily variance) |
| Indexing | a forecast is stored under its **target date** `t` and uses information up to the close of `t-1` |

**Simulated data.** `volvar.synthetic` generates `SIM01...` series from a common market factor plus
idiosyncratic GJR-GARCH(1,1,1) processes with Student-t innovations. It exists so the code, tests
and CI run without network access. Because the data-generating process is GJR-GARCH, GARCH-family
models winning on it is expected by construction: it validates the code, not the models. Every
output derived from it is labelled simulated (files, figures, API header, dashboard banner).

## 2. Forecasting models

All models forecast the next-day variance `h_t` for each ticker. The walk-forward settings are in
`ExperimentConfig`: first forecast after `min_train = 1000` days, parameters re-estimated every
`refit_every = 63` days on an expanding window.

| Key | Model | Definition |
|---|---|---|
| `hist` | Historical | mean of the previous 250 squared returns |
| `ewma` | EWMA (benchmark) | `h_t = 0.94 h_{t-1} + 0.06 r_{t-1}^2`, seeded with the mean of the first 30 squared returns |
| `garch` | GARCH(1,1)-t | zero-mean, Student-t innovations, maximum likelihood via the `arch` library |
| `gjr` | GJR-GARCH(1,1,1)-t | adds `gamma r_{t-1}^2 1[r_{t-1}<0]` for the leverage effect |
| `har` | HAR regression | OLS of `r_t^2` on lagged Garman-Klass variance averaged over 1, 5 and 22 days plus lagged `r^2`; floored at 10% of the historical variance |
| `lgbm` | Pooled LightGBM | predicts the ratio `r_t^2 / EWMA_t` from 15 scale-free features pooled across tickers; ratio clipped to [0.2, 4]; training target capped at 20 |

**GARCH filtering between refits.** Parameters are estimated on data up to the refit date. For the
following days the variance recursion is run forward with those parameters *fixed*
(`arch_model(...).fix(params)`), using only returns already observed. This was checked against a
manual recursion.

**Next-day forecast.** The forecast for the first unseen day is obtained by appending a placeholder
row to the panel. A test proves the placeholder cannot influence its own forecast.

### No look-ahead

The evaluation is walk-forward and the code asserts the property in tests:

* features for target day `t` are shifted by one day;
* training rows for a refit are those whose targets are known before the first day they forecast;
* appending future data never changes earlier forecasts;
* the FHS quantile for day `t` is computed only from standardised returns before `t`.

## 3. Forecast evaluation

* **QLIKE** `ln(h) + r^2 / h` and **MSE** `(r^2 - h)^2`, both robust losses in the sense of Patton
  (2011), so they rank forecasts correctly even though `r^2` is a noisy proxy for true variance.
* **Common sample.** All models are scored on exactly the dates where every model has a forecast.
* **Diebold-Mariano.** For each model the loss differential `L(EWMA) - L(model)` is averaged across
  tickers on each date and tested for a zero mean with Newey-West (HAC) standard errors, 5 lags. A
  positive mean difference means the model has lower loss than EWMA.
* **Per-ticker wins.** The number of tickers on which each model has the lowest mean loss.
* **Regimes.** Dates are split into calm, normal and turbulent by terciles of the trailing 22-day
  equal-weighted market variance known the day before. Cut-offs use the whole sample because the
  regimes only group results afterwards; they never feed a forecast.
* **Real data only:** a separate COVID-window table (Feb-Jun 2020) is produced.

## 4. Value-at-Risk and Expected Shortfall

For each ticker, model, level (95%, 99%) and method:

* **Normal:** `VaR_t = z_level * sqrt(h_t)`, `ES_t` from the normal tail formula.
* **Filtered historical simulation (FHS):** standardise past returns by the model's own volatility,
  `z_s = r_s / sqrt(h_s)`; over the previous 500 days (at least 250 observations) take the lower
  empirical quantile `q` and tail mean `m` of `z`; then `VaR_t = -q sqrt(h_t)` and
  `ES_t = -m sqrt(h_t)`.

VaR and ES are reported as positive percentages of position value, for one day. Backtests use the
common sample across models so they are comparable.

## 5. Backtests

A **violation** is `r_t < -VaR_t`.

* **Kupiec POF** (unconditional coverage): likelihood ratio, chi-square with 1 degree of freedom.
* **Christoffersen independence** (violations should not cluster) and **conditional coverage**
  (sum of the two statistics), chi-square with 2 degrees of freedom. A ticker "passes" when the
  conditional-coverage p-value exceeds 0.05 (Kupiec p-value if independence cannot be tested).
* **Basel traffic light** at 99%: exceptions in the latest 250 trading days, green 0-4, yellow
  5-9, red 10 or more; also the share of all rolling 250-day windows that are green or red.
* **ES ratio:** mean realised loss on violation days divided by mean predicted ES on those days; 1
  is ideal, above 1 means the tail is understated.

At 99% a sample of about 1,350 days holds only about 13 expected violations, so the tests have
limited power: a pass is weak evidence, a clear failure is strong evidence.

## 6. Assistant

See the README. In short: six tools read the files in `results/` and one searches the knowledge base; a BM25 index over `docs/kb`
supports definitions; every number in an answer must match tool output or the question
(`assistant/grounding.py`); and the question set in `eval/assistant_eval.jsonl` is scored by
`scripts/eval_assistant.py`.

## 7. Reproducibility

* All randomness is seeded (`ExperimentConfig.seed`, simulator seed).
* `results/meta.json` records the configuration and library versions.
* Two consecutive runs on the same data produced byte-identical result CSVs (checked for model comparison, DM tests, VaR backtests and next-day VaR).
* `scripts/update_readme.py` rebuilds the README results block from `results/`, so reported
  numbers cannot drift from the files.

## References

* Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
* Diebold, F. and Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
* Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
* Glosten, L., Jagannathan, R. and Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*.
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*.
* Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*.
* Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*.
