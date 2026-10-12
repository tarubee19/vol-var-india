# Expected Shortfall

## Definition
Expected Shortfall, also called conditional VaR, is the average loss on the days when the loss exceeds the VaR. For a 99 percent confidence level it is the average of the worst 1 percent of outcomes. It is always at least as large as the VaR at the same level.

## Why it matters
Expected Shortfall answers the question VaR cannot: how bad is it when the threshold is breached? Unlike VaR it is a coherent risk measure. The Basel market-risk framework known as the Fundamental Review of the Trading Book moves internal-model capital from 99 percent VaR to 97.5 percent Expected Shortfall for this reason.

## Backtesting it here
This project checks Expected Shortfall with an ES ratio: the mean realised loss on breach days divided by the mean predicted Expected Shortfall on those days. A ratio near 1 means the tail is well calibrated, and a ratio above 1 means the model understates tail losses.

## Estimation
Under filtered historical simulation, Expected Shortfall is the forecast volatility times the average of the standardised returns that fall below the empirical quantile. Because it averages the tail, it needs more data than VaR for a stable estimate, so the 99 percent figure is noisier than the 95 percent one.
