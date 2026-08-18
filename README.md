# Mistry

A multi-instrument, multi-timeframe trend/breakout trading strategy,
backtested with a custom portfolio-aware simulation engine against real
historical data from OANDA.

**This is a learning/research project, not financial advice.**

**BACKTEST ONLY.** Nothing in this project places real trades yet. The
only OANDA API usage in the committed backtest is a read-only historical-
candles fetch (`data_fetch.py`) - no order, trade, or account-
modification endpoint is called there. A separate, **incomplete and
unfinished** live-demo-trading connector lives in `live/` - see "Live
trading (paused, incomplete)" below before assuming it does anything.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your OANDA credentials
```

See "OANDA credentials" below for where to get the values that go in `.env`.

To run the test suite (position-sizing math, currency conversion):
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Current, honest status

Trained on 2020-08-13 to 2024-10-25, tested out-of-sample on 2024-10-25
to 2026-08-13 (real OANDA history):

| | Training | Testing (unseen) |
|---|---|---|
| Completed trades | ~30 | ~184 |
| Profit factor | ~0.76 | **1.24** |
| Return (trading only, financing excluded) | ~-1% | **+5.0%** |
| Max drawdown | ~4% | **2.4%** |

This **passes all 4 of the spec's bars** on this one split. It is **not
robust** - an 11-scenario stress test (different train/test splits plus
independent historical windows) shows the current default configuration
only clears every bar in about 2 of 11 scenarios, with genuinely bad
tail-risk drawdowns (worst observed: 28% test, 44% train) in adverse
windows. Treat the single-split PASS above as "true on this data, not
evidence of a robust edge" - see "Tuning history" for the full,
unflattering story of how this was found out.

### Systematic strategy search: on hold

Beyond the original 1H strategy above, this project tried a wide sweep
of alternatives - a from-scratch mean-reversion strategy, single-
timeframe 4H versions of both trend-following and mean-reversion, a
combined 4H portfolio running both at once, and single-timeframe daily
versions of both. **None of them cleared the validation bars either.**
Summary (see "Tuning history" items 13+ for the full detail):

| Timeframe / strategy | Outcome |
|---|---|
| 1H trend-following (original spec, above) | Fails robustness: 2/11 stress-test pass rate |
| 1H mean-reversion | Failed at every reward:risk / Bollinger-width tried |
| 4H trend-following (single timeframe) | Near-breakeven, tight drawdown, 0/11 stress-test pass rate |
| 4H mean-reversion (single timeframe) | Near-breakeven, tight drawdown, never cleared a single split cleanly |
| 4H combined portfolio (both at once) | Worse than either alone - shared-slot crowding, not diversification |
| Daily trend-following (single timeframe) | Genuinely losing (test PF 0.65) and short of the trade-count minimum |
| Daily mean-reversion (single timeframe) | Far short of the trade-count minimum; its "good" PF is a 13-trade sample, not signal |

The systematic search across this entry-logic family (EMA-trend/
channel-breakout and Bollinger/RSI mean-reversion, at every timeframe
tried) is **on hold** as of this point - not because any one bug was
found, but because moving to a lower-noise timeframe (4H, then daily)
consistently traded away edge without producing robustness in return,
and combining the two surviving 4H strategies made things worse rather
than better despite genuinely uncorrelated trades (see item 17). Picking
this back up would mean either a genuinely different signal class, or
accepting the original 1H strategy's documented lack of robustness.

## The strategy

**Markets:** EUR/USD, GBP/USD, USD/JPY, Gold (XAU/USD). Adding another
instrument (e.g. oil) is a one-line addition to `instruments.py` - no
other file needs to change.

**Timeframes:** 4H determines the trend, 1H triggers entries. All
decisions happen on a closed candle - never mid-candle.

**Entry (long; short is the exact mirror), current defaults:**
1. 50 EMA above 200 EMA on the 4H chart
2. 1H price closes above its highest price of the previous 20 candles
   (the original spec value - see "Tuning history": a 95-candle variant
   was adopted for a while based on evidence that later turned out to be
   built on two bugs, and was reverted once they were fixed)
3. Spread is within a normal range for that instrument (rejected if abnormally wide)

Two additional entry filters are implemented in `signals.py` but **OFF
by default** - `use_efficiency_filter` (backed by a real diagnosed
pattern, a genuine partial improvement, not yet adopted as default) and
`use_volatility_filter`/`breakout_buffer_atr_fraction` (tried, made
things worse). See "Tuning history".

**Stop-loss / take-profit:** 1.5x ATR stop, 3x ATR target (2:1
reward-to-risk). A stop-to-breakeven trade-management option is
implemented in `risk_management.py` (`use_breakeven_stop`) but **OFF by
default** - it helped the one bad episode it was designed around, but
made average stress-test performance worse (see "Tuning history").

**Position sizing:** risk 0.25% of account *balance* (realized P&L only,
not floating) per trade, capped at 30:1 leverage on notional exposure
against a **fixed reference balance** (never a balance inflated by
running P&L or financing - see "Tuning history" for why that distinction
matters). If even the smallest tradeable size would exceed either limit,
the trade is rejected rather than taken anyway.

**Account currency:** fully supports non-USD accounts (this project's
own demo account is GBP-denominated). `instruments.value_per_price_unit`
and `instruments.notional_value_per_unit` handle direct, base-matching,
and true cross-currency cases (e.g. a GBP account trading EUR/USD, where
neither currency is GBP) via triangulated conversion rates - see
`tests/test_gbp_position_sizing.py` for the validation.

**Safety limits (all enforced by `risk_management.py`'s `PortfolioAccount`,
which is the one place that sees the whole book across all 4
instruments at once):**
- Daily loss limit: stop for the day after a 1% loss
- Weekly loss limit: stop for the week after a 2.5% loss
- Drawdown suspension: halts all trading once equity falls 8% from its
  peak, resuming on equity recovery or a 30-day cooldown
- 3 consecutive losses: pause until the next trading day
- Max 3 open positions at once; total open risk capped at 0.75%
- Only one trade open at a time among EUR/USD, GBP/USD, and USD/JPY
  (grouped as "usd_fx" correlation) - Gold is tracked separately
- Rejects trades on abnormal spread or stale/missing data
- Avoids opening new trades near an approximate high-impact-news window
  (see limitation below)
- No martingale, no grid trading, no increasing size after a loss

## Tuning history (the honest, unflattering version)

This strategy went through many rounds of evidence-based tuning. Rather
than only keeping the current conclusion, this section keeps the real
order of events - including the parts that were wrong at the time -
because several "validated" results later turned out to rest on bugs,
and the process of finding that out is itself the most useful part of
this history to preserve.

1. **A leverage-cap bug** (no cap on notional exposure at all) caused a
   runaway swap-financing feedback loop during stress-testing. Fixed
   with a 30:1 leverage cap.
2. **Drawdown-threshold tuning (6% vs 8%) turned out not to matter**
   once the leverage cap was in place - real drawdowns rarely got deep
   enough anymore for the difference to ever trigger during testing.
3. **Two new entry filters (volatility filter, breakout buffer) were
   tried and made profit factor worse.** Left implemented, OFF by default.
4. **Breakout channel length (20-100 candles) was swept and 95
   appeared to win**, validated (at the time) by an 11-scenario
   robustness stress test showing consistent average improvement.
   Adopted as the new default.
5. **A second, more serious bug was then found**: the leverage cap's
   notional math was wrong for pairs where the account currency matches
   the BASE currency (e.g. USD_JPY on a USD account) - it divided by the
   raw price when it should not have divided at all, under-capping
   USD_JPY's real leverage by ~150x throughout the *entire* backtest
   history, including everything "validated" in steps 1-4.
6. **Fixing that bug exposed a third issue**: the leverage cap was
   scaling with *current* (financing-inflated) balance rather than a
   fixed reference, letting the static/approximate swap-financing model
   compound into results dominated by an untrustworthy assumption (one
   test showed financing at ~30x the strategy's actual trading P&L).
   Fixed by (a) pinning the leverage cap to a fixed reference balance,
   and (b) excluding financing from the scored return/drawdown metrics
   entirely (still reported separately - see `compute_metrics`).
7. **Re-running the full validation suite under corrected math reversed
   the earlier conclusion**: channel=95 no longer outperformed, and
   NEITHER channel=20 nor channel=95 reliably cleared the profit-factor
   bar across the 11-scenario stress test. Channel-length tuning was
   retired as a lever. Default reverted to channel=20 (the original spec
   value) as the least-worse, most literal choice - explicitly NOT
   because it was shown robust (it isn't).
8. **A trade-level diagnostic** (comparing winners vs losers on an
   INDEPENDENT trend-strength measure - Kaufman's Efficiency Ratio, not
   the strategy's own EMA filter, to avoid circular reasoning) found a
   real, consistent pattern across two different channel-length trade
   sets: entries during choppy/inefficient price action lose badly (win
   rate as low as 8%), entries during genuinely efficient directional
   moves do much better (37-39%, near the ~33% breakeven for a 2:1 R:R).
9. **An Efficiency Ratio entry filter was built and stress-tested**:
   genuine, cross-validated improvement (pass rate 18%->36%, better
   average drawdown/PF/return across all 11 scenarios) but still fails
   most scenarios and doesn't eliminate tail risk. NOT adopted as
   default pending further work - implemented and available
   (`use_efficiency_filter`) but off.
10. **A deep dive into the worst stress-test episode** (2023-12-28 to
    2024-07-23, a broad-but-choppy uptrend - ~200 uniform ~-1R losses,
    a 13-trade losing streak) found the efficiency filter does NOT fix
    this specific regime (win rate unchanged with/without it: 23.3% vs
    23.6%) - a real structural blind spot, not a threshold-tuning problem.
11. **A stop-to-breakeven trade-management mechanism was built and
    stress-tested** as a structural response (targeting the exact
    failure pattern from #10 rather than another entry filter): it
    helped the diagnosed episode substantially, but made AVERAGE
    stress-test performance worse (pass rate 18%->9%, worse average PF/
    return, and a WORSE worst-case drawdown in two scenarios). Likely
    mechanism: this strategy's thin edge depends on a minority of full
    2R winners, and a breakeven stop disproportionately clips those on
    their way to target in trending regimes, costing more than it saves
    in choppy ones. NOT adopted - implemented and available
    (`use_breakeven_stop`) but off.
12. **The structural redesign is currently paused** after two of three
    ideas (channel retuning, breakeven stop) failed stress-testing and
    one (efficiency filter) showed only partial improvement. Current
    defaults are the plain original spec entry/risk logic.

13. **A from-scratch mean-reversion strategy was designed and built**
    (`mean_reversion_signals.py`): Bollinger Bands + RSI entry on 1H,
    with a 4H Efficiency-Ratio trend-avoidance filter (reject entries
    when the market is trending efficiently rather than range-bound).
    Round 1 (untuned, literal proposal numbers) validated the core
    premise - a genuinely high win rate (53-56%) - but profit factor
    stayed below 1 (small average wins, R:R 1.5x/1.0x too tight to
    convert that win rate into net profit).
14. **A stop/target R:R sweep on the mean-reversion strategy** found
    wider stops (up to 2.5x) pushed win rate as high as 65-75% and
    produced one single-split PASS (PF 1.202) - but the 11-scenario
    stress test showed only 2/11 scenarios passing (avg PF 0.852, still
    net-losing on average). A follow-up Bollinger-width sweep found a
    noisy, non-monotonic PF curve with one promising-looking point
    (2.5std, PF 1.312) that stress-testing confirmed was statistically
    indistinguishable from the 2.0std baseline (both 2/11) - a null
    result, and a second confirmation (after item 4/7) that a noisy
    single-parameter sweep curve is a real overfitting signature worth
    distrusting on sight in this project.
15. **Moved both strategies to a single 4H timeframe**
    (`signals_4h.py`, `mean_reversion_signals_4h.py`), removing the 1H/
    4H mismatch diagnosed as a structural blind spot in item 10 -
    trend filter and entry trigger (or Bollinger/RSI and its trend-
    avoidance filter) now both computed from the same series, no
    cross-timeframe merge. ATR stop/target multiples reset to the
    literal spec numbers rather than reusing 1H-tuned values. Round-1
    results for BOTH: dramatically tighter, more consistent drawdown
    control than any 1H configuration (4H trend-following: max test
    drawdown never exceeded 8.02% across all 11 stress-test scenarios,
    vs 20-28% at 1H) - but neither produced a real edge above roughly
    breakeven (4H trend: 0/11 stress-test pass rate, avg test PF 0.997;
    4H mean-reversion: never cleared even the single split cleanly).
    Fine stop-multiple sweeps on both found tight, noisy bands with no
    clean winner - not stress-tested, since nothing cleared the single
    split.
16. **Two granularity-correctness bugs were found and fixed while
    building the 4H versions** (not just re-tuning - genuine
    correctness issues that would have silently produced wrong
    results): the overnight-financing rollover check (`ts.hour == 21`)
    never fires on 4H bars, which land on hours 0/4/8/12/16/20 -
    fixed via a new `bar_duration_hours` parameter to `run_backtest()`.
    And the stale-data gap threshold (3h default) was tighter than the
    NORMAL 4-hour gap between 4H bars, which would have flagged every
    single bar as stale - fixed by widening it to 6h for 4H runs.
17. **Tested combining the two 4H strategies into one portfolio**
    (`combined_signals_4h.py`), motivated by both being near-breakeven
    individually but structurally close to mutually exclusive by
    construction. Checked diversification BEFORE building anything
    further: daily P&L correlation between the two strategies' trades
    was -0.02 (essentially zero), with only ~2.7% of trades landing on
    the same instrument on the same day - genuinely uncorrelated, not
    duplicating the same edge. Built the combination sharing one
    `PortfolioAccount` and its existing risk/position limits (an
    instrument fires whichever strategy signals; same-direction
    agreement is tagged and taken once, opposite-direction conflicts
    are skipped). Result: **worse than either strategy alone** (test PF
    0.765, vs 1.004 and 0.919 standalone) - not a diversification
    failure but a resource-contention one: trend-following fires far
    more often than mean-reversion (~4:1 in the combination, vs ~1.5:1
    standalone), so sharing one "single open position per instrument"
    slot let it crowd out mean-reversion's opportunities more than half
    the time, rather than the two return profiles genuinely averaging.
    Uncorrelated trades turned out to be necessary but not sufficient -
    the specific sharing mechanism mattered and this one actively
    suppressed the benefit. Not stress-tested (failed the single split).
18. **Moved to a single daily timeframe as the documented fallback**
    (`signals_daily.py`, `mean_reversion_signals_daily.py`), carrying
    the same bar-count parameters forward (50/200 EMA, 20-bar channel,
    Bollinger 20/2std, RSI 14 - arguably more natural on daily than 4H,
    since 50/200-day EMA is the standard "golden/death cross"
    convention). Found and fixed a THIRD granularity-correctness bug
    while checking data availability: `data_is_stale()`'s weekend-gap
    detection checked `previous_bar.weekday() == 4 (Friday)`, correct
    for 1H/4H bars (whose last pre-weekend candle opens Friday) but
    wrong for daily bars, whose OANDA candle boundary (21:00 UTC) means
    the pre-weekend candle opens THURSDAY - every weekly weekend gap on
    daily data was being judged against the normal (not weekend)
    threshold and would have been wrongly flagged stale, every week.
    Fixed by checking both weekdays (safe for 1H/4H too).
19. **Daily results were worse than 4H, not better**: trend-following
    was genuinely losing on both train and test (test PF 0.653, test
    return -1.61%) as well as short of the 150-trade minimum (103
    total); mean-reversion fell far short of the trade-count minimum
    (42 total, vs a pre-build check that correctly flagged its raw
    signal frequency as thin) - its apparently-strong test PF (2.320)
    is a 13-trade sample, not a meaningful result. Lower sampling
    frequency didn't filter out noise the way 4H partly did; it just
    removed too much of the opportunity set, and trend-following's
    edge went negative rather than staying flat. Neither cleared the
    single split, so neither was stress-tested.
20. **The systematic search is now on hold** - see "Systematic strategy
    search: on hold" above for the full comparison table. Every
    timeframe tried (1H, 4H, daily) on both entry-logic families
    (trend-following, mean-reversion), individually and combined,
    failed to clear the validation bars robustly.

**Bottom line:** every "this fixes it" moment in this history except the
efficiency-ratio filter (item 9) was later found to be wrong or to not
generalize, and the broader search across timeframes and a combined
portfolio (items 13-19) didn't find anything that did generalize either.
The efficiency-ratio filter remains the one piece of real, partially-
validated signal found so far - not strong enough on its own to call any
version of this strategy solved. Treat everything in this project as "as
honest as we know how to make it," not "proven to work."

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

## Live trading (paused, incomplete)

A `live/` directory exists with the start of a live-demo-trading
connector: `account_safety.py` (a hard, mandatory, 3-layer check that
the connected account is genuinely a PRACTICE/demo account, never live -
tested and working) and `oanda_live_client.py` (the only place beyond
`data_fetch.py` that talks to OANDA - account state, open trades, and
order placement with an attached stop-loss). **This is unfinished**:
`live_state.py`, `live_account_sync.py`, `order_execution.py`,
`live_logging.py`, and the main `run_live.py` loop were never built -
work paused to investigate the sizing bugs and strategy weaknesses
documented above instead. Do not attempt to run anything from `live/` -
most of it doesn't exist yet, and what does exist has never placed an
order.

## Known approximations (deliberate, and documented in the code)

- **Economic calendar** (`econ_calendar.py`): no live/historical calendar
  API is available here, so this blocks new entries during a recurring
  weekday UTC time window that commonly covers US data releases. It
  does NOT know about FOMC/ECB/BoE decisions or one-off events. Built as
  a single swappable function so a real calendar feed can replace it
  later. **At daily granularity this is a documented no-op**: OANDA's
  daily candles are timestamped 21:00 UTC, which never falls inside the
  12:00-14:00 UTC blackout window, so the filter never rejects anything
  in `run_backtest_daily.py`/`run_mean_reversion_backtest_daily.py`.
  Making it apply meaningfully at daily resolution would need a
  different design (blocking whole announcement days, not a timestamp
  match) - not implemented.
- **Swap/financing rates** (`instruments.py`): static, ballpark annual %
  rates per instrument/direction, applied nightly (tripled on Wednesdays
  for weekend rollover). Real OANDA rates move with central bank policy
  year to year - this can't reproduce that, and is excluded from the
  scored backtest metrics for exactly this reason (see "Tuning history").
- **Rollover hour** fixed at 21:00 UTC year-round, not adjusted for US
  daylight saving.
- **Leverage cap (30:1)** is a realistic-but-arbitrary retail limit for
  scoring purposes; not fetched from OANDA's actual margin rules for a
  given account (real per-instrument margin rates were checked once
  live for the `live/` connector - Gold's real cap is stricter, 20:1).
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
| `instruments.py` | Per-instrument specs, pip/notional value math (incl. cross-currency), correlation group, spread cap, swap rates |
| `data_fetch.py` | Paginated OANDA candle fetch (read-only) with local CSV caching + synthetic fallback; supports H1/H4/D granularities |
| `indicators.py` | EMA, ATR, rolling high/low channel, SMA, rolling std, RSI |
| `signals.py` | Original 4H-trend/1H-entry trend-following strategy (merges the 4H trend filter onto the 1H timeline, no lookahead); `SignalConfig` holds all tunable entry-logic parameters |
| `signals_4h.py` | Single-timeframe 4H trend-following - trend filter and entry trigger both computed from the same 4H series; `Signal4HConfig` |
| `signals_daily.py` | Single-timeframe daily trend-following, same structure as `signals_4h.py`; `SignalDailyConfig` |
| `mean_reversion_signals.py` | 1H Bollinger/RSI mean-reversion entry with a 4H Efficiency-Ratio trend-avoidance filter; `MeanReversionConfig` |
| `mean_reversion_signals_4h.py` | Single-timeframe 4H mean-reversion - entry and trend-avoidance filter both on the same 4H series; `MeanReversion4HConfig` |
| `mean_reversion_signals_daily.py` | Single-timeframe daily mean-reversion, same structure as the 4H version; `MeanReversionDailyConfig` |
| `combined_signals_4h.py` | Merges the 4H trend-following and mean-reversion signal frames per instrument into one combined frame, for running both out of one shared account (see "Tuning history" item 17) |
| `risk_management.py` | Position sizing (with leverage cap) + `PortfolioAccount`: every safety limit, breakeven-stop trade management, shared across all instruments; `RiskConfig` holds all tunable risk parameters |
| `econ_calendar.py` | The approximate news-blackout heuristic (swappable) - a documented no-op at daily granularity, see "Known approximations" |
| `backtest_engine.py` | The custom multi-instrument, chronological-lockstep backtest simulator; `bar_duration_hours` parameter makes the financing-rollover check correct at any granularity, and a frame may optionally carry per-row `stop_distance_override`/`target_distance_override`/`signal_source` columns (used by `combined_signals_4h.py`) |
| `run_backtest.py` | Entry point for `signals.py`; also the shared helper module (`split_train_test`, `compute_metrics`, `evaluate_requirements`, report/comparison printers) every other `run_*.py` script imports from |
| `run_backtest_4h.py` / `run_backtest_daily.py` | Entry points for the single-timeframe 4H / daily trend-following strategies |
| `run_mean_reversion_backtest.py` / `run_mean_reversion_backtest_4h.py` / `run_mean_reversion_backtest_daily.py` | Entry points for the 1H / 4H / daily mean-reversion strategies |
| `run_combined_backtest_4h.py` | Entry point for the combined 4H trend-following + mean-reversion portfolio |
| `tests/` | pytest suite - currently covers currency-conversion/position-sizing math (`requirements-dev.txt`) |
| `live/` | Incomplete, paused live-demo-trading connector - see "Live trading" above |
| `.env.example` | Template for the environment variables `data_fetch.py`/`live/` need - copy to `.env` |
