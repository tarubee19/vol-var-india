# Data and limitations

## Data source
Real-data runs use daily open, high, low, close and volume downloaded from Yahoo Finance through the yfinance library, with prices adjusted for splits and dividends. The universe is 15 large, long-listed Nifty 50 constituents plus the Nifty 50 index. The Nifty 50 is the National Stock Exchange's benchmark index of 50 large Indian companies.

## Simulated data mode
For development, tests and continuous integration the project can generate simulated series named SIM01 and so on from a GJR-GARCH process. Results from simulated data validate the code and show that the models recover known structure, but they say nothing about real market performance. Every figure and summary produced from simulated data is labelled as simulated.

## Known limitations
Only daily data is used, so intraday volatility is not modelled. The tickers are current large constituents, which introduces survivorship bias. Yahoo's adjustment for mergers and demergers may be imperfect. VaR is computed per instrument, not for a portfolio. Backtests at the 99 percent level have low statistical power.

## Not investment advice
The outputs are risk estimates for education and research. They are not predictions of price direction and are not recommendations to buy or sell any security.
