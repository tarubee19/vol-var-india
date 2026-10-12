# EWMA (RiskMetrics) volatility

## The EWMA recursion
The exponentially weighted moving average forecast is sigma2(t) = lambda * sigma2(t-1) + (1 - lambda) * r(t-1)^2. Each new squared return gets weight 1 - lambda and older observations decay geometrically. There is nothing to estimate except lambda.

## The 0.94 convention
The RiskMetrics methodology popularised lambda = 0.94 for daily data, which this project uses as its benchmark. A smaller lambda reacts faster to new shocks; a larger lambda gives a smoother estimate.

## Relation to GARCH
EWMA is a special case of GARCH(1,1) with omega equal to zero, alpha equal to 1 - lambda and beta equal to lambda, so persistence is exactly one. That means EWMA never reverts to a long-run level, which is why a mean-reverting GARCH can beat it after volatility spikes.

## Why it is the benchmark
EWMA is simple, needs no estimation and is hard to beat by much. Every other model in this project is compared against it using the Diebold-Mariano test, so a reported gain is a gain over a strong, practical baseline rather than over a naive rolling window.
