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

## The strategy

**Markets:** EUR/USD, GBP/USD, USD/JPY, Gold (XAU/USD). Adding another
instrument (e.g. oil) is a one-line addition to `instruments.py` - no
other file needs to change.

**Timeframes:** 4H determines the trend, 1H triggers entries. All
decisions happen on a closed candle - never mid-candle.

**Entry (long; short is the exact mirror):**
1. 50 EMA above 200 EMA on the 4H chart
2. 1H price closes above its highest price of the previous 20 candles
3. Spread is within a normal range for that instrument (rejected if abnormally wide)

**Stop-loss / take-profit:** 1.5x ATR stop, 3x ATR target (2:1
reward-to-risk). Every simulated position carries both from the moment
it's opened.

**Position sizing:** risk 0.25% of account *balance* (realized P&L only,
not floating) per trade. If even the smallest tradeable size would risk
more than that, the trade is rejected rather than taken anyway.

**Safety limits (all enforced by `risk_management.py`'s `PortfolioAccount`,
which is the one place that sees the whole book across all 4
instruments at once):**
- Daily loss limit: stop for the day after a 1% loss
- Weekly loss limit: stop for the week after a 2.5% loss
- Drawdown suspension: **halts all trading for the rest of the
  backtest** once equity falls 8% from its peak - see the important
  caveat about this below
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

The equity-recovery condition turned out to be unable to fire on its own
in practice: once suspended, no *new* trades can open, and if nothing is
left open at that point, equity has no way to move at all (it isn't
earning interest or anything else passively) - a genuine deadlock, not a
bug. The 30-day cooldown is the fallback that actually lets the account
try again. One side effect worth knowing: because the cooldown doesn't
require the drawdown to have actually improved, the account can end up
attempting a trade, immediately re-breaching 8%, and re-suspending for
another 30 days - a periodic "try again" pattern rather than a full
recovery. See `PortfolioAccount.update_risk_flags()` in
`risk_management.py`.

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
case, since it isn't real market history.

## Project layout

| File | Purpose |
|---|---|
| `instruments.py` | Per-instrument specs: symbol, pip value math, correlation group, spread cap, swap rates |
| `data_fetch.py` | Paginated OANDA candle fetch (read-only) with local CSV caching + synthetic fallback |
| `indicators.py` | EMA, ATR, rolling high/low channel |
| `signals.py` | Merges the 4H trend filter onto the 1H timeline (no lookahead) and computes entry signals |
| `risk_management.py` | Position sizing + `PortfolioAccount`: every safety limit, shared across all instruments |
| `econ_calendar.py` | The approximate news-blackout heuristic (swappable) |
| `backtest_engine.py` | The custom multi-instrument, chronological-lockstep backtest simulator |
| `run_backtest.py` | Entry point: fetch data, train/test split, run both, print the full honest report |
| `.env.example` | Template for the environment variables `data_fetch.py` needs - copy to `.env` |
