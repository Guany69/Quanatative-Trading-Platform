# Limitations

Read this before drawing any conclusion from this platform's output. Everything below is a
real constraint, not a disclaimer formality.

## The demo uses synthetic data

`run-demo` operates on **randomly generated fixture securities**. Prices come from a factor
model plus noise, with a momentum effect deliberately planted so the pipeline has a known
ground truth to detect. Demo results measure whether the *software* works. They say nothing
about markets, and the "strategy" has no investment meaning whatsoever.

## Data limitations with open providers

| Concern | Status with free data |
|---|---|
| Historical index constituents | **Not available.** No free source provides point-in-time S&P 500 membership. Requires a vendor file via `LocalConstituentSource`, or results are explicitly marked survivorship-biased. |
| Survivorship bias | **Present with yfinance.** Only currently-listed tickers are queryable; bankrupt and acquired companies simply cannot be retrieved. |
| Delisting returns | **Absent with free data.** A delisted security stops having bars; the terminal loss is never booked, which systematically overstates returns. |
| Point-in-time fundamentals | **Available and wired in.** SEC EDGAR companyfacts supplies genuine `filed` dates, so value and quality factors are computed without look-ahead. Caveats: XBRL tag coverage varies by filer (tag fallbacks are used); foreign issuers and ETFs have no CIK; market cap uses the share count from the latest filing, so buybacks since then are not reflected. |
| Macro revision vintages | **Not available via standard FRED.** The API serves latest-revised values. Using them is a genuine look-ahead leak; ALFRED vintages would be required to fix this properly. |
| Corporate actions | yfinance supplies adjusted prices, but the methodology is undocumented and occasionally revised. |
| Ticker reuse | Not handled by free sources: a recycled symbol may splice two unrelated companies into one price history. |

The fixture provider has none of these problems *by construction* — which is exactly why
fixture results must never be read as evidence about real markets.

## Model and cost approximations

- **Market impact** uses a square-root approximation. It is a widely used functional form,
  not a calibrated measurement, and it is not a universal law. Real impact depends on order
  flow, venue, time of day, and your own footprint.
- **Slippage and spread** are modelled parametrically, not from quote data.
- **Benchmark sector weights** are approximated by equal-weighting the eligible universe,
  because true index constituent weights are not freely available.
- **Beta = 1.0** is assumed when no benchmark history is available for a security.
- **Trailing-twelve-month figures** prefer a sum of the four most recent quarters, falling
  back to the latest annual report when quarterly coverage is incomplete. A TTM figure will
  therefore often differ from the headline fiscal-year number, and is usually more current.
- **Market capitalization** = latest close x shares outstanding from the most recent filing.
  Share counts are stale between filings.
- The **correlation regime feature** is a cheap proxy (dispersion ratio), not a true average
  pairwise correlation.

## Backtesting limitations

- Fills assume the close price of the delayed session, adjusted by the cost model. No
  intraday microstructure, no queue position, no partial-day liquidity.
- The trading calendar covers weekends and standard NYSE holidays. **One-off historical
  closures (9/11, hurricanes, days of mourning) are not encoded.**
- Dividends are credited on the ex-date rather than the payment date.
- No securities lending, no borrow costs, no shorting (the MVP is long-only).
- Taxes are not modelled at all.

## Paper trading is not live trading

Paper fills are simulated with the same cost model as the backtest. Real execution differs in
price, timing, and available liquidity. **No broker integration exists**: `NautilusBrokerAdapter`
is an interface stub that raises on instantiation, so there is no path to routing a live order
from this codebase.

## Statistical limitations

- The Deflated Sharpe Ratio needs an honest count of configurations tried. It is only as
  correct as the experiment registry is complete — trials run outside the registry make it
  optimistic.
- PBO requires enough independent data blocks; it returns NaN rather than a fabricated number
  when the sample is too short.
- **A locked holdout can only be evaluated once.** Every subsequent look turns it into another
  validation set and the reported figure becomes selected rather than unbiased.

## Optional dependencies not installed

`vectorbt`, `alphalens-reloaded`, `skfolio`, `riskfolio-lib`, `nautilus_trader`, and
`quantstats` are not part of the environment (Python 3.13 compatibility, see
`docs/adr/0011-optional-vs-required-dependencies.md`). All critical metrics — IC, Sharpe,
Sortino, PSR, DSR, PBO, factor diagnostics — are implemented internally and tested, so nothing
essential depends on them.

## Platform constraints

- **PyTorch is pinned `<2.12`**: versions from 2.12 ship macOS arm64 wheels only for
  macOS 14+, and this project was developed on macOS 13.
- **LightGBM needs `libomp`**, normally supplied by Homebrew. `scripts/fix_macos_libomp.py`
  repairs the linkage using the copy already vendored by scikit-learn/PyTorch, so no
  system package manager is required.

## What this platform does NOT claim

It does not claim any strategy will be profitable, will outperform the S&P 500, or is ready
for live capital. Historical results are **simulated**. Simulated performance does not
guarantee future results.
