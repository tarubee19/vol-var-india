# VaR backtesting

## Violations
A violation, also called a breach or exception, occurs when the realised return is worse than minus the VaR. A good 99 percent model should be breached on about 1 percent of days and the breaches should not cluster.

## Kupiec proportion-of-failures test
The Kupiec test checks unconditional coverage: whether the observed breach rate equals the expected rate. It is a likelihood-ratio test compared with a chi-square distribution with 1 degree of freedom. A small p-value means the breach rate is statistically different from the target, either too high (risk understated) or too low (overly conservative).

## Christoffersen tests
The Christoffersen independence test checks whether a breach today makes a breach tomorrow more likely, which signals a model that reacts too slowly. Adding it to the Kupiec statistic gives the conditional coverage test, compared with a chi-square with 2 degrees of freedom. This project calls a model passing when the conditional coverage p-value exceeds 0.05, falling back to the Kupiec test when independence cannot be tested.

## Basel traffic light
The Basel backtesting rule counts breaches of the 99 percent VaR over the latest 250 trading days. Zero to 4 breaches is the green zone, 5 to 9 is yellow and 10 or more is red. The project reports the zone of the latest window and the share of all rolling windows in each zone.

## Statistical power
At the 99 percent level a sample of about 1,300 days contains only about 13 expected breaches, so the tests have limited power. A pass is weak evidence of a good model, while a clear failure is strong evidence of a bad one.
