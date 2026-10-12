# GARCH-family models

## GARCH(1,1)
The GARCH(1,1) model writes tomorrow's variance as a weighted sum of a constant, yesterday's squared return and yesterday's variance: sigma2(t) = omega + alpha * r(t-1)^2 + beta * sigma2(t-1). Alpha controls how strongly new shocks move volatility and beta controls how long the effect lasts. The parameters are estimated by maximum likelihood.

## Persistence and half-life
Persistence is alpha + beta (plus gamma/2 for the GJR variant). Values close to 1, typically 0.95 to 0.99 for daily equity data, mean shocks to volatility decay slowly. The half-life of a shock is ln(0.5) / ln(persistence) days. The long-run unconditional variance is omega / (1 - persistence), which exists only when persistence is below 1.

## GJR-GARCH and the leverage effect
GJR-GARCH adds a term gamma * r(t-1)^2 * I(r(t-1) < 0), so a negative return raises next-day variance by more than a positive return of equal size. A positive gamma captures the leverage effect seen in equities.

## Innovation distribution
The models here use Student-t innovations instead of normal ones because daily returns have fat tails. The estimated degrees of freedom summarise how heavy the tails are; lower values mean heavier tails. A zero-mean specification is used so the variance forecast targets the squared return directly.
