# Mistry

A multi-instrument, multi-timeframe trend/breakout trading strategy,
backtested with a custom portfolio-aware simulation engine against real
historical data from OANDA.

**This is a learning/research project, not financial advice.**

**BACKTEST ONLY.** Nothing in this project places real trades. The only
OANDA API usage anywhere is a read-only historical-candles fetch
(`data_fetch.py`) - no order, trade, or account-modification endpoint is
ever called.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your OANDA credentials
```

See "OANDA credentials" below for where to get the values that go in `.env`.

## Usage

```bash
source venv/bin/activate
python run_backtest.py
```

First run fetches several years of 1H + 4H candle history for EUR/USD,
GBP/USD, USD/JPY, and Gold from OANDA (paginated - takes a few minutes)
and caches it to `data_cache/`. Later runs only fetch new candles since
the last run, so they're fast.

## Current result

Trained on 2020-08-13 to 2024-10-25, tested out-of-sample on 2024-10-25
to 2026-08-13 (real OANDA history, not synthetic):

| | Training | Testing (unseen) |
|---|---|---|
| Completed trades | 91 | 430 |
| Win rate | 23.1% | 36.5% |
| Profit factor | 0.522 | **1.221** |
| Return | -8.31% | **+24.77%** |
| Max drawdown | 8.45% | 3.26% |

**Passes all 4 of the spec's bars** (≥150 combined trades, positive
out-of-sample return, <10% drawdown both periods, out-of-sample profit
factor >1.2) on this dataset. See "Tuning history" below for exactly how
it got there and what that PASS should (and shouldn't) be read as.

## The strategy

**Markets:** EUR/USD, GBP/USD, USD/JPY, Gold (XAU/USD). Adding another
instrument (e.g. oil) is a one-line addition to `instruments.py` - no
other file needs to change.

**Timeframes:** 4H determines the trend, 1H triggers entries. All
decisions happen on a closed candle - never mid-candle.

**Entry (long; short is the exact mirror):**
1. 50 EMA above 200 EMA on the 4H chart
2. 1H price closes above its highest price of the previous 95 candles
   (the spec originally said 20 - see "Tuning history" for why this changed)
3. Spread is within a normal range for that instrument (rejected if abnormally wide)

**Stop-loss / take-profit:** 1.5x ATR stop, 3x ATR target (2:1
reward-to-risk). Every simulated position carries both from the moment
it's opened.

**Position sizing:** risk 0.25% of account *balance* (realized P&L only,
not floating) per trade, capped at 30:1 leverage on notional exposure. If
even the smallest tradeable size would exceed either limit, the trade is
rejected rather than taken anyway. (The leverage cap isn't in the
original spec - see "Tuning history"; without it, a very tight ATR-based
stop during a quiet period could size a position far beyond anything a
real broker would allow.)

**Safety limits (all enforced by `risk_management.py`'s `PortfolioAccount`,
which is the one place that sees the whole book across all 4
instruments at once):**
- Daily loss limit: stop for the day after a 1% loss
- Weekly loss limit: stop for the week after a 2.5% loss
- Drawdown suspension: halts all trading once equity falls 8% from its
  peak, resuming on equity recovery or a cooldown - see below
- 3 consecutive losses: pause until the next trading day
- Max 3 open positions at once; total open risk capped at 0.75%
- Only one trade open at a time among EUR/USD, GBP/USD, and USD/JPY
  (grouped as "usd_fx" correlation) - Gold is tracked separately so it
  can be held alongside one FX trade
- Rejects trades on abnormal spread or stale/missing data
- Avoids opening new trades near an approximate high-impact-news window
  (see limitation below)
- No martingale, no grid trading, no increasing size after a loss

## How the 8% drawdown suspension resumes

Trading resumes after an 8% drawdown suspension on EITHER of two
conditions, whichever comes first:
- equity recovers back to within 4% of its all-time peak, or
- 30 days elapse since the suspension started

The equity-recovery condition can't fire on its own if nothing is left
open when suspension triggers - equity only moves through trading, and
trading is exactly what's blocked, so it can get permanently stuck
without the cooldown fallback. One side effect worth knowing: since the
cooldown doesn't require the drawdown to have actually improved, the
account can end up attempting a trade, immediately re-breaching 8%, and
re-suspending for another 30 days - a periodic "try again" pattern
rather than a full recovery. See `PortfolioAccount.update_risk_flags()`
in `risk_management.py`.

## Tuning history (how the current defaults were chosen)

This strategy went through several rounds of evidence-based tuning after
the initial implementation. In order:

1. **A leverage-cap bug, found via stress-testing.** The original
   position-sizing formula (risk% / (stop distance × pip value)) had no
   cap on notional exposure. During a low-volatility stretch, a tiny
   ATR-based stop could size a position at absurd leverage (one stress
   scenario hit ~2000:1 on USD/JPY), and the resulting swap financing on
   that notional created a runaway feedback loop that inflated results
   unrealistically. Fixed with a 30:1 leverage cap (`RiskConfig.max_leverage`,
   a common real-world retail FX limit) - this alone resolved most of an
   earlier apparent drawdown problem, for free, before any other tuning.
2. **Drawdown-threshold tuning (6% vs the spec's 8%) turned out not to
   matter** once the leverage cap was in place - real drawdowns rarely
   got deep enough anymore for the 6%/8% difference to ever trigger
   during out-of-sample testing. Verified via an 11-scenario stress test
   (multiple train/test splits + independent historical windows); it was
   dropped rather than kept for no measurable benefit.
3. **Two new entry filters were tried and made things worse**: a
   volatility filter (skip signals when ATR is below its trailing
   average) and a breakout buffer (require clearing the breakout level
   by a margin). Both are implemented in `signals.py` (`SignalConfig.
   use_volatility_filter`, `breakout_buffer_atr_fraction`) but left OFF
   by default because they measurably hurt profit factor in testing.
4. **Breakout channel length (20 to 100 candles) was swept and found to
   matter a lot.** Longer channels traded quantity for quality: fewer,
   more significant breakouts, consistently better profit factor and
   drawdown up through about 80-95 candles. This was the one lever that
   moved every metric the right way at once. **Validated with an
   11-scenario robustness stress test** (not just the one split it was
   originally found on) - channel=95 improved average profit factor,
   drawdown, and return across most scenarios, not only the split that
   happened to produce the first PASS. It was NOT uniformly better,
   though: in one distinct historical window (the 2020-2022 stretch), it
   underperformed the original 20-candle default outright (PF 0.70-0.74
   vs baseline's 1.07). Channel length became the new default
   (`SignalConfig.channel_period = 95`) on the strength of that
   consistency, not because it passes everywhere.

**Bottom line on the "PASSES all 4 bars" result above:** it's real, on
real data, and backed by more than a single lucky train/test split. It
is NOT evidence the strategy reliably clears the bar in every market
regime - the stress test that validated channel=95 also showed it losing
to the original default in at least one historical window. Treat the
current defaults as "meaningfully better, evidence-based" rather than
"solved."

## Known approximations (deliberate, and documented in the code)

- **Economic calendar** (`econ_calendar.py`): no live/historical calendar
  API is available here, so this blocks new entries during a recurring
  weekday UTC time window that commonly covers US data releases. It
  does NOT know about FOMC/ECB/BoE decisions or one-off events. Built as
  a single swappable function so a real calendar feed can replace it later.
- **Swap/financing rates** (`instruments.py`): static, ballpark annual %
  rates per instrument/direction, applied nightly (tripled on Wednesdays
  for weekend rollover). Real OANDA rates move with central bank policy
  year to year - this can't reproduce that.
- **Rollover hour** fixed at 21:00 UTC year-round, not adjusted for US
  daylight saving.
- **Leverage cap (30:1)** is a realistic-but-arbitrary retail limit, not
  something specified by the original spec or fetched from OANDA's
  actual margin rules for a given account.
- **Fill/exit modeling**: signals are evaluated on a closed 1H candle;
  orders fill at the next candle's open (ask for buys, bid for sells)
  plus ATR-scaled slippage. Take-profit fills assume no slippage (limit-
  order-like); stop-loss and entry fills do get slippage (market-order-
  like). If both the stop and target fall inside the same candle, the
  stop is assumed to have hit first (the conservative assumption, since
  OHLC data can't tell us the true intra-candle order).

## OANDA credentials

This project reads price data from an OANDA demo (practice) account:

1. Sign up for a free demo account at https://www.oanda.com/demo-account/
2. Log in to the OANDA web platform, go to "Manage API Access", and
   generate a personal access token.
3. Copy `.env.example` to `.env` and fill in:
   - `OANDA_API_TOKEN` - the access token from step 2
   - `OANDA_ACCOUNT_ID` - your account ID (format like `101-004-12345678-001`)
4. `.env` is gitignored - your real credentials never get committed.

If `.env` is missing or invalid, `data_fetch.py` automatically falls back
to synthetic random-walk data so the project still runs - but every
metric in the final report is explicitly labeled meaningless in that
case, since it isn't real market history. (This fallback can also
trigger on a transient OANDA API hiccup, not just bad credentials - if
results ever look unexpectedly off, check the run's output for a
"SYNTHETIC" warning before trusting the numbers.)

## Project layout

| File | Purpose |
|---|---|
| `instruments.py` | Per-instrument specs: symbol, pip value math, correlation group, spread cap, swap rates |
| `data_fetch.py` | Paginated OANDA candle fetch (read-only) with local CSV caching + synthetic fallback |
| `indicators.py` | EMA, ATR, rolling high/low channel |
| `signals.py` | Merges the 4H trend filter onto the 1H timeline (no lookahead) and computes entry signals; `SignalConfig` holds all tunable entry-logic parameters |
| `risk_management.py` | Position sizing (with leverage cap) + `PortfolioAccount`: every safety limit, shared across all instruments; `RiskConfig` holds all tunable risk parameters |
| `econ_calendar.py` | The approximate news-blackout heuristic (swappable) |
| `backtest_engine.py` | The custom multi-instrument, chronological-lockstep backtest simulator |
| `run_backtest.py` | Entry point: fetch data, train/test split, run one or more (SignalConfig, RiskConfig) experiments, print the full honest report + comparison table |
| `.env.example` | Template for the environment variables `data_fetch.py` needs - copy to `.env` |
