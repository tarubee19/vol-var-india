# Evaluation methodology

## Walk-forward evaluation
Models are fitted on an expanding window of past data and used to forecast the next day, then re-estimated roughly every 63 trading days. Between re-estimations the variance is filtered forward with fixed parameters using only observed returns. This prevents look-ahead bias: no forecast ever uses data from its own day or the future, and the code tests this property directly.

## Loss functions
QLIKE is the loss ln(h) + y/h, where h is the variance forecast and y the squared return; lower is better. MSE is the mean squared error between the forecast and the squared return. Both are robust losses in the sense of Patton (2011): they rank forecasts correctly even though the squared return is only a noisy proxy for true variance.

## Diebold-Mariano test
To test whether a model beats the EWMA benchmark, the loss difference is averaged across tickers on each date and tested for a zero mean using Newey-West standard errors with 5 lags. A positive mean difference means the model has lower loss than EWMA. A p-value below 0.05 is called significant here.

## Common sample and regimes
All models are scored on exactly the same observations, where every model has a forecast. Results are also split into calm, normal and turbulent regimes, using terciles of the trailing 22-day market variance known the day before, to check whether a ranking is stable. The tercile cut-offs are computed over the whole sample because regimes are only used to group results afterwards, never to produce a forecast.

## Caveats
Pooled tests treat tickers as exchangeable and ignore multiple-comparison adjustments. Differences between good models are often small, so statistical significance matters more than the ranking of point estimates.
