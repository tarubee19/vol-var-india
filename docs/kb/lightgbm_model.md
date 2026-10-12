# Pooled LightGBM correction model

## What it predicts
Instead of predicting raw variance, which is very noisy and heavy-tailed, the gradient-boosted model predicts the ratio of the next-day squared return to the EWMA forecast. The final forecast is the EWMA forecast multiplied by the predicted ratio, which keeps the forecast positive and anchored to a sensible baseline.

## Features
All features use only information available at the previous close and are scaled by the EWMA forecast, so they are comparable across stocks. They are: average squared returns over 1, 5, 22 and 66 days; Garman-Klass range variance over 1, 5 and 22 days; the previous return, its negative-only part and the 5-day return, each divided by the EWMA volatility; the log of the EWMA variance; a volume surprise (log of yesterday's volume over its 22-day median); market-wide squared-return ratios over 1 and 22 days from an equal-weighted index of the tickers; and the day of the week.

## Pooling and retraining
One model is trained on all tickers together, which gives far more training rows than any single stock. It is retrained about every 63 trading days using only rows whose targets were already known, so there is no look-ahead. The training target is capped at 20 times the EWMA forecast to limit the influence of extreme days, and the predicted ratio is clipped to the range 0.2 to 4.

## Regularisation
The model is deliberately small: 200 trees, 8 leaves, a minimum of 200 rows per leaf and L2 regularisation. Volatility targets are noisy, so a larger model mostly learns noise.
