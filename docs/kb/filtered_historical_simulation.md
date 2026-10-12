# Filtered historical simulation

## The idea
Filtered historical simulation combines a volatility model with the empirical return distribution. Each past return is divided by the model's own volatility forecast for that day, giving a standardised return that should be roughly identically distributed. The VaR quantile is then read from these standardised returns instead of assuming a normal distribution.

## Procedure used here
For each forecast day the project takes the standardised returns from the previous 500 trading days, requiring at least 250 observations. VaR is the forecast volatility times minus the lower empirical quantile, and Expected Shortfall is the forecast volatility times minus the mean of the returns beyond that quantile. Only returns before the forecast day are used.

## Why not just assume normality
Equity returns have fatter tails than a normal distribution, so a normal 99 percent VaR is often too small, while the empirical quantile adapts to the actual tail. The project reports both methods, labelled normal and fhs, so the difference is visible in the backtests.

## Trade-off
Filtered historical simulation needs a warm-up window, so VaR starts later in the sample than a normal VaR. The 99 percent quantile of 500 observations rests on only five tail points, which makes it noisy.
