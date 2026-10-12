# Volatility basics

## What volatility is
Volatility is the standard deviation of an asset's returns; variance is its square. In this project returns are daily log returns in percent, so a daily volatility of 1.5 means a typical one-day move of about 1.5 percent. Annualised volatility is the daily figure multiplied by the square root of 252 trading days.

## Why forecast volatility instead of prices
Price direction is very hard to predict, but volatility is persistent: calm days tend to follow calm days and turbulent days follow turbulent days. This is called volatility clustering, and it makes tomorrow's variance forecastable from recent data. Risk limits, position sizing and Value-at-Risk all depend on a variance forecast rather than a price forecast.

## Stylised facts of equity returns
Equity returns show four well-known features: volatility clustering, fat tails (more extreme days than a normal distribution allows, measured by excess kurtosis), the leverage effect (negative returns raise future volatility more than positive returns of the same size) and mean reversion of volatility toward a long-run level.

## The evaluation proxy
True daily variance is never observed directly. The squared close-to-close return is an unbiased but noisy proxy for it, and it is the target used to score every forecast in this project. Robust loss functions such as QLIKE rank forecasts correctly even with this noisy proxy.
