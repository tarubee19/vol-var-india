# HAR regression with range-based predictors

## The HAR idea
The Heterogeneous AutoRegressive model of Corsi (2009) predicts future variance from averages of past variance over different horizons: the previous day, the previous week and the previous month. Traders with different horizons react to different information, so volatility at several horizons all matter.

## Predictors used here
This project builds the daily, weekly (5 days) and monthly (22 days) predictors from the Garman-Klass range-based variance estimate, which uses the open, high, low and close prices and is more efficient than a squared return. The lagged squared return is added as a fourth predictor. The target is the next-day squared return so that HAR is scored exactly like every other model.

## Estimation
Coefficients are estimated by ordinary least squares and re-estimated about every 63 trading days using only targets that were already known. Forecasts are floored at 10 percent of the rolling historical variance so they stay positive.

## Limitation
The Garman-Klass estimator ignores the overnight gap between the previous close and the next open, so it understates total daily variance. The regression calibrates this, but HAR can still lag a gap-driven shock.
