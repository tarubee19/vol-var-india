# Value-at-Risk

## Definition
One-day 99 percent Value-at-Risk is the loss that is expected to be exceeded on only about 1 percent of days, which is roughly 1 day in 100. In this project VaR is reported as a positive percentage of position value. A 99 percent VaR of 3 percent means a one-day loss worse than 3 percent should happen on roughly 1 trading day in 100.

## What VaR is not
VaR is a threshold, not a worst case. It says nothing about how bad losses are on the days that exceed it. That is why Expected Shortfall is reported alongside it.

## From a volatility forecast to VaR
VaR for tomorrow equals tomorrow's forecast volatility times a quantile of the standardised return distribution. With a normal assumption the 99 percent multiplier is about 2.33 and the 95 percent multiplier about 1.645. Real returns are fat-tailed, so the project also uses filtered historical simulation to estimate the multiplier from data.

## Confidence levels
Two levels are produced: 95 percent, which is expected to be breached about 5 percent of days, and 99 percent, expected to be breached about 1 percent of days. The 99 percent level is the one used in the Basel traffic-light backtest.

## Limitations
VaR for a single stock is not portfolio VaR. It ignores intraday moves, liquidity, and transaction costs, and a model that passes a backtest can still fail in an unprecedented crisis.
