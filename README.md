# Mistry

A multi-instrument systematic FX/gold trading research project: a
custom portfolio-aware backtesting engine, a rigorous validation
process, and a demo-account execution connector - built out to test
whether any of several standard technical trading signals hold up to
real scrutiny on real historical data from OANDA. **The systematic
search concluded without finding a robust edge** - see "Current, honest
status" immediately below before reading anything else here.

**This is a learning/research project, not financial advice.**

**No real (non-demo) money is or has ever been involved.** The backtest
never places real trades - the only OANDA API usage there is a read-only
historical-candles fetch (`data_fetch.py`). A separate live-demo-trading
connector in `live/` **has placed real orders on an OANDA PRACTICE
(demo) account** - multiple times, verified end-to-end, including
finding and fixing three real bugs along the way (see "Live trading"
below) - but it has only ever run an explicitly unvalidated placeholder
strategy, purely to prove the execution plumbing works, never a strategy
this project actually trusts.

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

**The systematic search for a robust trading edge is CONCLUDED (not
just paused) as of this point.** Nine independent axes were tried -
3 signal classes (trend-following, mean-reversion, time-series
momentum), 3 timeframes (1H, 4H, daily), 5 instruments, a portfolio
combination, a full reward:risk sweep, and 2 independent confirmation-
filter variants - and every one of them converged on the same result:
win rate landing within a few points of whatever breakeven the payout
structure implied, profit factor capped around 0.9-1.1, never robustly
clearing the 1.2 bar. See "Systematic strategy search: concluded" below
for the full comparison table and the reasoning behind stopping here
rather than continuing to test variations.

What remains from this project: a well-tested backtest engine, a
genuinely rigorous validation process (documented at length below,
including every case where that process caught a promising-looking
result that didn't hold up), and a live demo-execution connector that's
been verified end-to-end against a real OANDA practice account. What
does NOT remain: a strategy this project would actually recommend
trading. Read everything below as "the honest record of a real search
that came up empty," not as a working system.

### Standing validation criteria - STRENGTHENED after Candidate 2

Every strategy through Candidate 2 (17 independently tried structural
ideas total) was checked against the original spec's bar: 150+
completed trades, profit factor > 1.2 on out-of-sample data, positive
OOS return, max drawdown < 10%. Candidate 2 failed that bar decisively
(train PF 0.912, test PF 0.546) - and after that result, the bar itself
was raised for any FUTURE strategy round, not applied retroactively to
change any already-concluded verdict (nothing that already failed the
easier bar would have passed the stricter one either). The **current**
standing criteria (`run_backtest.py`):

- **≥200 completed trades** (train+test combined) - up from 150; 300 is
  treated as a "comfortable" level worth noting in reports, not an
  additional hard gate.
- **Profit factor > 1.3** - up from 1.2 - required independently in
  BOTH train and test, not just out-of-sample.
- **Positive expectancy after all costs** - unchanged in spirit (every
  backtest in this project already uses realistic Bid/Ask fills and
  slippage, never a zero-cost approximation for the pass/fail check
  itself), now stated as an explicit, named criterion rather than an
  implicit property of the engine.
- **In-sample/out-of-sample similarity** (new): test PF must not fall
  more than 35% below train PF (`IS_OOS_MAX_RELATIVE_PF_DROP`) - added
  specifically because Candidate 2's failure mode was exactly this: a
  test period that collapsed relative to train, not just an
  independently-bad test result.
- **No single trade or month dominating the result** (new): no single
  trade may account for more than 20% of gross profit
  (`MAX_SINGLE_TRADE_PROFIT_SHARE`), no single calendar month more than
  40% of net profit (`MAX_SINGLE_MONTH_PROFIT_SHARE`) - guards against a
  strategy that looks profitable only because of one lucky trade or
  regime.
- **Positive results across more than one market regime** (new): at
  least 60% of calendar years present in the combined sample must be
  individually net-profitable (`MIN_POSITIVE_REGIME_YEARS_FRACTION`),
  using calendar year as a pragmatic regime proxy - the same per-year
  breakdown already used throughout this project's reports (H4,
  Candidate 2), not a sophisticated volatility/trend-regime classifier.
  Explicitly documented as a first-pass convention, not a precisely
  derived number.

Implemented as `evaluate_extended_requirements()` /
`print_extended_verdict()` in `run_backtest.py`, additive to the
existing `evaluate_requirements()` (whose two thresholds were bumped in
place) so every already-written entry-point script keeps working
unchanged - any future strategy's entry point should call both
functions. Not applied to any of the 17 already-concluded experiments;
their results and verdicts stand exactly as documented, under the
criteria that were in effect when they were run.

### The original 1H strategy, for reference

The ORIGINAL spec strategy (4H trend filter / 1H entry, both described
under "The strategy" below) is the one config that ever passed a single
train/test split, trained on 2020-08-13 to 2024-10-25 and tested
out-of-sample on 2024-10-25 to 2026-08-13 (real OANDA history):

| | Training | Testing (unseen) |
|---|---|---|
| Completed trades | ~30 | ~184 |
| Profit factor | ~0.76 | **1.24** |
| Return (trading only, financing excluded) | ~-1% | **+5.0%** |
| Max drawdown | ~4% | **2.4%** |

This **passes all 4 of the spec's bars** on this one split. It is **not
robust** - an 11-scenario stress test (different train/test splits plus
independent historical windows) shows this configuration only clears
every bar in about 2 of 11 scenarios, with genuinely bad tail-risk
drawdowns (worst observed: 28% test, 44% train) in adverse windows.
Treat the single-split PASS above as "true on this data, not evidence
of a robust edge" - see "Tuning history" for the full, unflattering
story of how this was found out, and everything tried afterward to
improve on it.

### Systematic strategy search: concluded

Beyond the original 1H strategy above, this project tried a wide sweep
of alternatives - a from-scratch mean-reversion strategy, single-
timeframe 4H versions of both trend-following and mean-reversion, a
combined 4H portfolio running both at once, single-timeframe daily
versions of both, and (a genuinely different signal class, not a
timeframe variant of the first two) time-series momentum on 4H, swept
across 8 lookback windows. **None of them cleared the validation bars
either.** Summary (see "Tuning history" items 13+ for the full detail):

| Timeframe / strategy | Outcome |
|---|---|
| 1H trend-following (original spec, above) | Fails robustness: 2/11 stress-test pass rate |
| 1H mean-reversion | Failed at every reward:risk / Bollinger-width tried |
| 4H trend-following (single timeframe) | Near-breakeven, tight drawdown, 0/11 stress-test pass rate |
| 4H mean-reversion (single timeframe) | Near-breakeven, tight drawdown, never cleared a single split cleanly |
| 4H combined portfolio (both at once) | Worse than either alone - shared-slot crowding, not diversification |
| Daily trend-following (single timeframe) | Genuinely losing (test PF 0.65) and short of the trade-count minimum |
| Daily mean-reversion (single timeframe) | Far short of the trade-count minimum; its "good" PF is a 13-trade sample, not signal |
| 4H time-series momentum (8-point lookback sweep) | Every lookback failed outright - best test PF across the whole sweep was 0.966, still below breakeven |
| Relative-value/pairs (EUR_USD vs GBP_USD spread, approximate check) | Weak structural evidence to begin with (spread barely more mean-reverting than either leg); phase-1 cheap check showed train/test in opposite directions with zero transaction costs modeled - not built out further |
| AUD_USD on the existing 4H trend-following strategy | Clean fail, and worse than the existing 4-instrument basket (win rate 26-28%, below the ~33% breakeven the R:R needs) |
| 4H trend-following R:R sweep (2:1 down to 1:1, stop fixed) | Win rate climbs cleanly as hypothesized (34.8%->52.5%) but profit factor stays flat (0.965-1.094) regardless - the higher win rate is paid for by proportionally smaller wins; best PF still well short of 1.2 |
| Dual-confirmation (trend-following AND momentum must agree), momentum lookback=20 | Byte-for-byte IDENTICAL to the unfiltered trend baseline - proven algebraically to be a tautology when both signals share the same 20-bar window, not an empirical finding |
| Dual-confirmation, momentum lookback swept 25-150 (genuinely independent of the 20-bar channel) | No improvement - 6 of 7 configs WORSE than the unfiltered baseline (requiring agreement filtered out profitable trades along with bad ones); best case merely matched baseline, never beat it |
| Volatility-expansion confirmation (require ATR > its own 20-bar average) | Decisively worse on every dimension - trade count collapsed to 74 combined (below the 150 minimum), win rate dropped below the R:R breakeven (30-31% vs baseline 34-39%), PF fell below 1.0 in both periods. A genuine re-test of an idea tried once before on the original 1H strategy (also made things worse there) - landed in the same place on a different construction |

The systematic search across every entry-logic family tried
(EMA-trend/channel-breakout, Bollinger/RSI mean-reversion, and raw
time-series momentum - three mechanically distinct signal classes, not
variations on one idea) moved through a lower-noise timeframe (4H, then
daily), a combined portfolio of the two surviving 4H strategies (worse
than either alone, despite genuinely uncorrelated trades - item 17), a
completely different signal construction (momentum, item 21, landing in
the same territory as channel breakout across a wide range of
lookbacks), a relative-value/pairs approach that checked out weak even
at the cheap-approximation stage (item 24), a fifth instrument that
failed more clearly than the existing basket (item 25), a direct test
of the reward:risk structure itself (item 26, win rate climbing exactly
as hypothesized while profit factor stayed flat regardless), and two
independent confirmation-filter variants (items 27-28, both making
things worse, not just failing to help). Nine independent axes, all
converging on the same outcome: win rate landing within a few points of
whatever breakeven the payout structure implies, profit factor capped
around 0.9-1.1, never robustly clearing 1.2.

**This is now a decision, not just an observation: the search is
CONCLUDED, not paused.** When one idea fails, that's a fact about the
idea. When trend-following, mean-reversion, and momentum all land in
the same narrow band, and independent confirmation filters make things
worse rather than better, and changing the payout structure just trades
win rate for reward size with no net gain - that stops being a fact
about any one idea and becomes evidence about the market itself: major
FX pairs and gold are among the most liquid, heavily-arbitraged
instruments that exist, and simple, well-known technical patterns are
exactly the signals well-capitalized participants compete away fastest.
This project's validation discipline (distrust noisy sweep curves,
never trust a single split, stress-test before believing anything) did
its job throughout - it prevented false confidence at several points
(see items 4-7's channel-length saga, item 14's noisy Bollinger-width
curve) - but preventing false confidence is not the same as guaranteeing
an edge exists to eventually find.

What was considered and explicitly NOT pursued, with reasons:
- **The full (non-approximate) relative-value/pairs engine** - the
  cheap check specifically built to gate this decision came back weak
  (item 23-24); building the expensive real engine anyway would mean
  ignoring the result of the check designed to inform exactly this call.
- **Exit-logic changes** (trailing stops, time-based exits) - never
  tested on the current infrastructure, but the diagnostic pattern
  throughout (win rate consistently at the R:R-implied breakeven, not a
  skewed win/loss shape a smarter exit could capture) points at entries
  lacking real directional edge, not at good entries being exited
  poorly. Low-promise by that reasoning, not tried.
- **A fundamentally different data source** (calendar/seasonality,
  order-flow) - the most genuine remaining idea, but this project has
  no infrastructure for it and no adequate data (OANDA's "volume" here
  is a tick-count proxy, not real volume; no real economic calendar
  feed - see "Known approximations" below) to backtest it honestly.
- **A broader instrument universe** - the one new instrument tried
  (AUD_USD) failed more clearly than the existing basket, weak evidence
  against this being fruitful.

## London Liquidity Sweep Reversal V1 - REJECTED after development testing

A genuinely different, structural idea, not a variant of anything in
the concluded search above. Built, tested, run on development data, and
**rejected** - it does not show a robust edge. Full diagnostic evidence
below; raw results permanently preserved in `results/` so this exact
experiment never needs re-running.

Before writing any strategy code, this project's existing infrastructure
was audited (which parts are strategy-agnostic and safe to reuse vs.
which parts belong to the concluded, failed strategies and must not be),
and a stricter data discipline was put in place first: a three-way
chronological split (`dataset_split.py`), not the two-way train/test
split used for every strategy tried so far.

M15 candle data for EUR_USD/GBP_USD was fetched and its actual
availability confirmed directly (2020-08-24 to 2026-08-21 - eleven days
later at the start than the H1/H4/D series). A sample-sufficiency check
- one fixed, generic, deliberately non-tuned Asian-range-sweep pattern,
counted only on dates before the reserved period, never evaluated for
performance - found ample raw candidate density in both non-reserved
windows (~750-765 distinct days/instrument with a qualifying event in
development, ~175-190 in validation), supporting these as workable
window sizes before any real strategy exists yet to confirm it directly.

```
DEVELOPMENT:    2020-08-24 -> 2024-07-15   (~1,421 days, ~65%)
VALIDATION:     2024-07-15 -> 2025-07-15   (~365 days, ~17%)
FINAL RESERVED: 2025-07-15 -> 2026-08-21   (~402 days, ~18%)  - LOCKED
```

`dataset_split.py`'s `split_for_iteration()` is the only function
strategy-development scripts should import - it returns development and
validation only; the final reserved period isn't reachable through it
at all. Getting the reserved period at all requires calling
`get_final_reserved_period()` with an explicit,
spelled-out confirmation flag, which prints a loud banner every time
it's used - by construction, not just by convention, so leaking that
period into ongoing iteration takes a deliberate, visible act, not a
copy-pasted date range. See that module's docstring for the full
reasoning, and `tests/test_dataset_split.py` for tests locking in that
the guard rail actually works, not just documenting an intention.

Also fixed while wiring this up: `data_fetch.py`'s synthetic-data
fallback path only recognized H1/H4/D granularities (`KeyError` on
M15/M5, though the REAL fetch path already handled any granularity
generically) - extended additively, no existing caller's behavior
changed.

Nothing else changed - `risk_management.py`, `backtest_engine.py`,
`instruments.py`, the GBP-account currency-conversion logic, and the
entire `live/` connector are untouched throughout everything below.

### V1 design (round 1)

M15, EUR_USD/GBP_USD only. Asian range (00:00-07:00 Europe/London LOCAL
time, DST-aware via `zoneinfo`) built from completed bars only, blanked
out and frozen with the same no-lookahead discipline `signals.py`'s
4H/1H merge already established. During the London entry window
(07:00-10:00 local), a sweep beyond the Asian high/low is never itself
an entry - the first candle (same or later) that closes back inside the
range fires the signal. At most one trade per instrument per day.
No RSI/MACD/EMA confirmation - the close-back-inside structure is the
confirmation. Stop-loss structural (beyond the sweep's own extreme +
0.1x ATR buffer, not a flat ATR multiple like every prior strategy);
target a fixed 1:1 R:R. Both flow through the unchanged
`calculate_position_size()` via the same `stop_distance_override`
mechanism `combined_signals_4h.py` established. Sweep penetration
recorded (price/pips/ATR) on every setup, never used as a filter.
See `signals_london_sweep_m15.py` for the full implementation and
`tests/test_london_sweep_signals.py` (20 tests) for coverage of BST/GMT
transitions, Asian-range boundaries, no-lookahead, sweep/confirmation
logic, one-trade-per-day, stop/target math, and the validation/reserved-
period access guard.

### V1 round 1 result on DEVELOPMENT data - REJECTED

161 completed trades (EUR_USD 80, GBP_USD 81) over the full development
window. **Net losing, consistently across both instruments:**

| | Trades | Win rate | PF | Net P&L (USD) | Avg R |
|---|---|---|---|---|---|
| EUR_USD | 80 | 40.0% | 0.668 | -$383.13 | -0.200 |
| GBP_USD | 81 | 39.5% | 0.650 | -$413.71 | -0.210 |
| Combined | 161 | 39.8% | **0.659** | -$796.84 | -0.205 |

**Currency note, found while running this**: `backtest_engine.py`/
`risk_management.py` never pass `account_currency` to
`value_per_price_unit()`/`notional_value_per_unit()` anywhere, so the
backtest has always defaulted to USD - for every strategy ever tested in
this project, including this one - despite the live account being
GBP-denominated. The live connector and the GBP position-sizing tests
are correctly wired; the backtest engine itself simply never was. Not
changed (out of scope, and would affect every historical backtest result
in this project, not just this one) - flagged honestly instead.

**Diagnostic findings** (full trade-level data: `results/london_sweep_v1_round1_development_trades.csv`;
full structured summary: `results/london_sweep_v1_round1_summary.json`):

- **Severe long/short asymmetry, consistent across both instruments**:
  long PF 0.977 (near breakeven), short PF 0.455 (badly losing) -
  EUR_USD short 0.446, GBP_USD short 0.462. Nearly the entire loss is
  carried by the short side. This is the single strongest pattern found.
- **Losing trades reverse almost immediately, not near-misses**: for the
  97 stopped-out trades, median MFE (maximum favorable excursion before
  exit) was 0.112R - 61% never even reached 0.25R favorably before
  reversing. This rules out the 1:1 target being "too far away" as the
  cause; a tighter target would only have rescued a small minority.
- **Transaction costs are not the driver**: total entry-side cost (the
  only cost this engine models on this strategy - exits use precomputed
  levels with no further cost) was $441.68 across all 161 trades
  ($2.74/trade average). Zero trades were losses that would have been
  wins without that cost. Removing it entirely still leaves PF 0.826 and
  a net loss.
- **Sweep penetration size shows a real but incomplete effect**: `<3
  pips` (n=71) PF 0.508 vs `>=3 pips` (n=90) PF 0.801 - tiny sweeps are
  genuinely worse, but the larger-sweep bucket is *still losing*, ruling
  out "just filter out small/noisy sweeps" as a fix on its own. Finer
  pip/ATR buckets showed a non-monotonic shape (best bucket sits in the
  middle, on ~31 trades) - the kind of curve this project has
  consistently learned to distrust rather than chase.
- **No clean time-of-day effect** (07:00-08:00 PF 0.629, 08:00-09:00 PF
  0.817, 09:00-10:00 PF 0.407 on only 14 trades - too thin to read).

**Verdict**: the underlying reversal hypothesis itself, not the R:R,
stop placement, or transaction costs, appears to be the primary source
of failure - and it fails asymmetrically by direction. The long/short
split is real and worth carrying forward as a lead, but wasn't predicted
in advance (it was found by inspecting these results), so it isn't
treated as validated evidence of a working strategy on its own -
consistent with this project's standing discipline against acting on
patterns discovered by looking at results rather than predicted
beforehand. **V1 does not proceed to validation.**

Not deleted, per explicit decision: `signals_london_sweep_m15.py`,
`run_london_sweep_backtest.py`, and their tests remain in the repository
as a complete, working, honestly-labeled failed experiment - the same
philosophy already applied to the concluded systematic search above.

## London Liquidity Sweep Reversal V2 - Higher-Timeframe Trend-Aligned - REJECTED after development testing

Treated as a genuinely new hypothesis, not a V1 parameter tweak - built
as a separate module that imports and reuses V1's sweep+confirmation
logic completely unchanged, rather than modifying V1's config. V1 stays
independently re-runnable exactly as concluded.

**Research question**: does a London liquidity sweep have positive
expectancy when the reversal direction is aligned with the established
higher-timeframe market trend?

**Trend definition**: daily EMA(50) vs EMA(200) - the same measure and
periods already used everywhere else in this project (`signals.py`,
`signals_4h.py`, `signals_daily.py`), computed on daily closes. One
fixed definition, not searched or swept across period combinations.

**Rule**: V1's sweep+confirmation fires exactly as before. A long
reversal is kept only if the daily trend is up; a short reversal is
kept only if the daily trend is down. A signal against the trend is
dropped entirely, never flipped into a countertrend trade.

**Rationale**: a sweep with the higher-timeframe trend looks more like
a genuine stop-hunt that resumes the dominant order flow; a sweep
against it has to argue price reverses against the larger prevailing
flow on local, session-scale evidence alone. This is a direct,
falsifiable attempt to explain V1's own strongest diagnostic finding -
the severe long/short asymmetry - rather than a new, unmotivated guess.

**No lookahead**: daily EMA50/200 computed on daily closes, index
shifted forward by exactly one full day (a daily candle here is indexed
by its 21:00 UTC OPEN but isn't knowable until it closes 24 hours
later), then `merge_asof(direction="backward")` onto the M15 timeline -
the exact "shift by this candle's own bar duration" mechanism
`signals.py`'s 4H/1H merge already established, reused rather than
reinvented for the daily/M15 case. See
`tests/test_london_sweep_trend_aligned_signals.py` (7 tests) for
coverage of the trend-alignment gate (kept/dropped in each direction,
overrides cleaned up on drop) and the no-lookahead shift specifically.

**Pre-committed rejection criteria** (decided before running anything):
fails this project's standing bar (150+ trades, PF > 1.2, positive OOS
return, max drawdown < 10% on a single development split); or performs
no better than - or worse than - V1's already-rejected baseline; or the
trend filter leaves too few trades to read at all (a real risk, since it
will likely roughly halve V1's already-modest 161-trade sample).

**Status**: built and tested (59/59 project tests passing, 7 new), then
run on development data. **Rejected.** `risk_management.py`,
`backtest_engine.py`, `instruments.py`, the GBP-account currency-
conversion logic, and the entire `live/` connector remain untouched
throughout. Validation and final-reserved data were never accessed.

### V2 round 1 result on DEVELOPMENT data - REJECTED

82 completed trades (EUR_USD 42, GBP_USD 40) - roughly half of V1's 161,
as expected from the trend filter. **Worse than V1 on every dimension,
not just fewer trades:**

| | Trades | Win rate | PF | Net P&L (USD) | Expectancy/trade |
|---|---|---|---|---|---|
| V1 (unfiltered) | 161 | 39.8% | 0.659 | -$796.84 | -$4.95 |
| **V2 (trend-aligned)** | 82 | **29.3%** | **0.414** | -$817.17 | **-$9.97** |

Per-trade loss nearly doubled. Total dollar loss is *larger* than V1's
despite half as many trades.

**The central hypothesis was directly refuted, not just unconfirmed.**
V1's long side alone was PF 0.977 (near breakeven) - V2 existed
specifically to test whether trend alignment explained and could
preserve that. Instead, "trend-aligned longs" came in at **PF 0.393** -
dramatically worse than V1's unfiltered longs, the opposite of what the
hypothesis predicted. If trend alignment were the real explanation for
V1's asymmetry, filtering to alignment should have held or improved that
0.977; it collapsed instead.

**Against the pre-committed rejection criteria** (decided before this
run - see above): fails the standing bar (82<150 trades, PF 0.414<1.2,
negative return) - yes. Performs no better than, or worse than, V1's
baseline - yes, worse on every metric. Too few trades to read - a
contributing factor, not the deciding one given the other two. **All
three criteria met. V2 does not proceed to validation.**

Full trade-level data: `results/london_sweep_v2_round1_development_trades.csv`.
Full structured summary: `results/london_sweep_v2_round1_summary.json`.

Not deleted, per the same decision as V1:
`signals_london_sweep_trend_aligned_m15.py`,
`run_london_sweep_trend_aligned_backtest.py`, and their tests remain in
the repository as a complete, honestly-labeled failed experiment.

## V3 candidate search - cheap falsification before building anything

With V1 and V2 both rejected, before writing any more strategy code
three genuinely different structural hypotheses (not variations of
trend-following, mean-reversion, momentum, or the liquidity-sweep
family) were proposed for a possible V3, ranked, and screened with the
*cheapest possible* empirical test each - pure statistics on
DEVELOPMENT data only, one pre-registered specification per hypothesis,
no thresholds swept or variants searched to manufacture significance -
before any entry/exit code, risk-management change, or backtest engine
involvement. This is deliberately cheaper and faster to reject on than
building a full strategy (as V1/V2 both required) - the point is to
stop paying that cost for hypotheses that don't survive five minutes of
statistics.

### Hypothesis 1: Cross-Instrument Lead-Lag (Relative Strength Rotation) - REJECTED

Mechanism under test: EUR_USD and GBP_USD share substantial common USD-
direction exposure; if EUR_USD (the more liquid pair) makes an unusually
large move and GBP_USD hasn't yet moved a proportional amount, GBP_USD
might be "catching up" - a statistical-arbitrage-style, cross-instrument
mechanism, structurally different from any single-instrument price
pattern already tried.

Pre-registered test (M15, DEVELOPMENT only, n=96,813 bars): does
EUR_USD's just-closed bar return predict GBP_USD's *next* bar return,
one fixed 1-bar lag, unconditional on magnitude?

**Result: the wrong sign.** r = -0.0095 (p=0.003) and directional
accuracy 49.42% - significantly *below* the 50% no-signal baseline
(p=0.0004), not above it. Critically, this negative, tiny effect is
statistically indistinguishable from GBP_USD's own already-rejected
lag-1 autocorrelation (r=-0.030) and from the reverse direction,
GBP_USD leading EUR_USD (r=-0.0145) - the same pattern shows up
regardless of which instrument "leads," which is the signature of
generic microstructure-level negative serial correlation (bid-ask
bounce), not a real cross-instrument information-flow effect. The
correlation's sign also flips between years (2021 is the only positive
year of five). Realistic Bid/Ask P&L trading every bar toward the
signal: -0.0175%/trade (t=-86.8) - decisively negative, and the
zero-cost hypothetical version is also (weakly) negative, so there was
no positive raw edge to begin with.

Full pre-registered spec, all numbers, and the verdict:
`results/hypothesis1_leadlag_falsification_summary.json`. Re-runnable,
unmodified script preserved at `hypothesis_tests/leadlag_falsification.py`
(not imported by anything, kept purely so this exact test never needs
repeating).

**Does not proceed to V3.** Rejected at the cheap-falsification stage,
before any strategy code was written.

### Hypothesis 2: Weekend Gap Fade - REJECTED

Mechanism under test: FX closes for ~48h over the weekend; news
accumulating during that closure gets discovered all at once at Sunday
reopen, in a thin market prone to overshoot - a partial reversion
("fade") of that gap during the following, more liquid session would be
a distinct, liquidity-driven edge (not a continuous-trading statistical
extreme like the already-rejected mean-reversion family).

**Round 1** (fixed 24h-hold fade trade, all 4 instruments, H1, n=795
pooled): correlation between gap size and subsequent 24h return was
essentially zero (r=-0.005, p=0.885) and the reversion rate (50.64%)
was statistically indistinguishable from the 50% no-signal baseline
(p=0.721). The correlation's sign also flipped from weakly negative
(2020-2022) to significantly *positive* - continuation, the opposite of
the hypothesis - in 2023-2024.

**Round 2** (closure-tracking, the more rigorous test, EUR_USD/GBP_USD
only, n=400): does the gap tend to partially/fully close, and how fast?
Gaps *do* close at a high rate (92-97% reach 25/50/100% closure within
5 days) - but a **pre-registered baseline** (the identical test applied
to ordinary, non-weekend H1 moves) closes at a statistically
indistinguishable rate, and at full closure, the ordinary-move baseline
actually closes *more* often (94.4% vs 92.0%, p=0.040). High closure is
a generic feature of price wandering over a 5-day window relative to
any small recent price level - not something specific to weekends.
Closure also happens fast (median 0-2 hours after reopen), too fast to
represent a real multi-day drift-back trade. The raw, cost-free edge
was not statistically significant at any closure threshold
(p=0.21-0.26), and realistic Bid/Ask spread cost (~0.07%/trade, 3-4x
the size of the already-insignificant raw edge) turned every threshold
decisively negative (p=0.0002-0.069).

Full pre-registered specs, all numbers from both rounds, and the
verdict: `results/hypothesis2_gap_fade_falsification_summary.json`.
Re-runnable, unmodified scripts preserved at
`hypothesis_tests/gap_fade_falsification_round1_24h_hold.py` and
`hypothesis_tests/gap_fade_falsification_round2_closure.py` (neither
imported by anything - kept purely so this exact test never needs
repeating).

**Does not proceed to V3.** Rejected at the cheap-falsification stage,
before any strategy code was written.

### Hypothesis 3: Month-End Flow Bias - REJECTED

Mechanism under test: institutional portfolio rebalancing (pension
funds, currency-hedged mandates, corporate treasury flows) is
documented to concentrate around month-end, creating a mechanically-
driven - not "views"-driven - directional flow. Research question:
does the final trading day / final trading hours of each month show a
repeatable directional OR volatility bias, materially different from
ordinary days? (Ranked weakest of the three V3 candidates from the
start, on both direction-sourcing and trade-count grounds.)

EUR_USD/GBP_USD, H1, n=96 combined month-end days (48/instrument,
already below the 150-event convention, acknowledged before testing).
Both close-to-close and intraday returns, plus a final-4-hours window,
were tested against an ordinary-day control, alongside realized
volatility.

**No significant directional edge anywhere.** Pooled close-to-close
mean -0.052% (p=0.21), intraday -0.050% (p=0.22) - neither significant,
and no hit rate anywhere (40-46%) is statistically distinguishable from
50%. Direction is negative in 4 of 5 years but flips positive in 2022,
undermining the "consistent sign" requirement set before testing.
Cost-adjusted P&L, using the one direction implied by the pooled
sample's own sign (decided once, not cherry-picked): +0.026%/trade
combined, **not significant (p=0.52)** - and GBP_USD alone would have
lost money trading it.

**One genuine, separate finding**: realized volatility in the final 4
hours of the month is ~23% higher than ordinary days (p=0.012,
nominally significant) - consistent with well-documented real month-end
fixing-window activity. Per the pre-registered protocol this is
necessary but not sufficient to validate a *directional* flow bias -
it's a volatility fact, not a tradeable edge as this hypothesis was
framed, and per this project's standing discipline against acting on a
pattern spotted mid-analysis without independent advance justification,
it was not treated as grounds to loosen the verdict. Left as a possible
seed for a genuinely different future idea (volatility-timing), not
pursued here.

Full pre-registered protocol, all numbers, and the verdict:
`results/hypothesis3_monthend_falsification_summary.json`. Re-runnable,
unmodified script preserved at `hypothesis_tests/monthend_falsification.py`
(not imported by anything, kept purely so this exact test never needs
repeating).

**Does not proceed to V3.** Rejected at the cheap-falsification stage,
before any strategy code was written. **All three original V3
candidates (Cross-Instrument Lead-Lag, Weekend Gap Fade, Month-End Flow
Bias) are now rejected.** One further hypothesis, outside the original
three, was investigated afterward - see "Hypothesis 4" below.

## Hypothesis 4: Economic Calendar Event Volatility - REJECTED (decisively, after round 3)

Introduces a genuinely new data source (a scheduled macro-event
calendar) alongside price data - the first hypothesis in this entire
search (15 attempts now, across the original systematic search, V1/V2
London Sweep, and Hypotheses 1-3) to condition on something other than
OANDA's own OHLC/Bid-Ask series. Research question: does price action
around scheduled high-impact US releases show a statistically
meaningful, *tradeable* volatility expansion that persists after
realistic costs, regardless of direction?

**Data source**: not the project's own `econ_calendar.py` (explicitly a
rough recurring-weekday-window heuristic, doesn't know specific event
types or FOMC/ECB/BoE dates - unsuitable for this and left untouched).
Instead, a hand-curated, 125-event table built from official read-only
archives: federalreserve.gov's FOMC calendar (31 events, 2pm ET
decisions) and bls.gov's year-by-year release schedules (47 NFP + 47
CPI, both 8:30am ET) - `hypothesis_tests/data/economic_events_development.csv`,
DEVELOPMENT window only.

**Round 1 (pure statistics)**: realized volatility in the 2h after
these releases is **2.67x** the ordinary-day control level, stable
3.5-6.2x every year 2020-2024, present in all three event types
(FOMC 8.0x, CPI 4.96x, NFP 4.75x) - by a wide margin the strongest raw
statistical finding across the entire search. A cost-adjusted P&L using
hindsight direction (+0.085%/trade, p<0.0001) hinted the magnitude
clears realistic cost, but couldn't establish real-time tradeability.

**Spread-resolution check**: directly tested whether M15-resolution
cached data can even see the true cost at the instant of a release
(all 125 events land exactly on M15 boundaries). The bar's own
*Open*-based spread shows **no widening at all** (ratio 1.01-1.02 vs.
normal quiet-market spread), while that same bar's realized range is
**10-15x larger** (50-61 pips vs. 4-8 pips of "spread"). A genuine
spike lasting seconds is averaged away between 15-minute checkpoints -
**M15 data cannot reliably capture true execution cost at the moment
of a news-driven break.**

**Round 2 (genuine non-lookahead breakout-trigger simulation)**: a
standalone, mechanical, dual-sided OCO trigger (0.1xATR entry buffer,
structural stop at the opposite range side, 1:1 R:R target - same
buffer style and R:R as V1/V2, nothing new invented or tuned), walking
forward with no hindsight on direction. Not routed through
`backtest_engine.run_backtest()` - its signal-at-T/fill-at-T+1-Open
convention doesn't match "enter the instant broken"; `backtest_engine.py`
and `risk_management.py` are both untouched by this test. 184/250
events triggered single-sided (96% on the very first bar after the
event - the highest-risk window from the check above), 66 excluded as
ambiguous (both sides breached the same bar).

Reported under two pre-committed execution-cost scenarios, not one
falsely-precise number: **optimistic** (fill at trigger + that bar's
own average spread) showed a large, highly significant edge (mean
R=+0.352, p<0.0001, 67.6% win rate) - the strongest result in the
search. **Conservative** (+0.25x that bar's own realized range, applied
only to first-bar triggers) collapsed it to **statistically
indistinguishable from zero** (mean R=-0.025, p=0.73, 48.0% win rate),
with FOMC and NFP individually flipping negative and the year-by-year
pattern turning genuinely inconsistent (2020-2022 negative-leaning,
only 2023-2024 holding up) - the pre-committed "inconsistent across
years" rejection trigger, firing under the scenario that matters.

**Round 2 conclusion at the time**: not "no effect found" - the
volatility expansion was real and never in doubt, but tradeability
couldn't be established because M15 data was shown unable to measure
execution cost at the moment of the break. Left open, not rejected
outright, pending finer-grained data.

**Round 3 (finest available resolution, closes the question)**:
S5 (5-second candles) - confirmed empirically the finest granularity
OANDA's API offers, and available all the way back to the start of
DEVELOPMENT for both instruments via the exact same read-only
connection already used everywhere in this project. Fetched as ~310
targeted event windows (not a bulk download). Reran round 2's *exact*
fixed design unchanged (0.1xATR entry buffer, same structural stop,
1:1 R:R, 24h max hold) - only the resolution and cost-realism changed:
real S5 Ask/Bid fills (no more optimistic/conservative bracketing -
S5 lets the real spread be seen directly) plus `SLIPPAGE_ATR_FRACTION
= 0.02`, reused verbatim from `backtest_engine.py`. The
ambiguous-same-bar rate dropped from 26.4% (M15) to 5.2% (S5),
confirming finer resolution resolves almost all of round 2's entry
ambiguity.

Used the project's existing three-way split unchanged: DEVELOPMENT as
train (125 events, walk-forward by year) and, for the first time for
any strategy in this project, a genuine **VALIDATION** out-of-sample
check (31 new events for 2024-07-15 to 2025-07-15, hand-curated from
the same official sources) - checked exactly once. **FINAL_RESERVED
was not accessed.**

| | n | win rate | expectancy (mean R) | profit factor | max drawdown |
|---|---|---|---|---|---|
| DEVELOPMENT (train) | 221 | 39.4% | -0.213R (p=0.0012) | 0.649 | 53R |
| **VALIDATION (OOS, checked once)** | 53 | 20.8% | **-0.585R (p<0.0001)** | 0.262 | 30R |

Negative every year in DEVELOPMENT (trending toward breakeven late,
never positive), negative for both instruments individually in
VALIDATION, negative for all three event types, flat-to-negative in
every session bucket - and **more** unprofitable out-of-sample than in
training, not less. This directly confirms round 2's core concern: the
earlier "optimistic" M15 result (+0.352R) was an artifact of
underestimated execution cost. Real S5 bid/ask plus realistic slippage
doesn't just erase the edge - it flips it solidly negative and keeps
it negative out-of-sample.

**Verdict, per the pre-committed rule ("if H4 does not remain
profitable out-of-sample after costs, reject it")**: REJECTED,
decisively. The underlying volatility-expansion phenomenon (round 1)
remains a genuine, confirmed market fact - what round 3 resolves is
that it is **not** capturable as a tradeable edge once realistic
execution is modeled at the finest resolution available. This closes
the question round 2 left open; no further work on this specific
hypothesis is warranted absent a fundamentally different execution
model or a different instrument.

Full numeric record for all four pieces (round 1, spread-resolution
check, round 2, round 3):
`results/hypothesis4_econ_event_volatility_summary.json`. Re-runnable,
unmodified scripts:
`hypothesis_tests/econ_event_volatility_round1_statistical.py`,
`hypothesis_tests/econ_event_volatility_spread_resolution_check.py`,
`hypothesis_tests/econ_event_volatility_round2_breakout_trigger.py`,
`hypothesis_tests/econ_event_volatility_round3_s5_fetch.py`,
`hypothesis_tests/econ_event_volatility_round3_s5_simulation.py`
(none imported by anything, kept purely so this exact experiment never
needs repeating). The VALIDATION event table
(`hypothesis_tests/data/economic_events_validation.csv`) is committed;
the raw S5 candle cache (~60MB, regenerable via the fetch script) is
gitignored, same treatment as `data_cache/`.

**Does not proceed to a strategy build or demo-bot validation stage**
- the phenomenon is real but not tradeable at any resolution this
project has been able to test, including the finest OANDA offers.

## Post-H4 structural search: all 3 candidates resolved, all REJECTED

With all H1-H4 avenues closed, a fresh read-only project review proposed
three genuinely different structural hypotheses - not variations of
trend-following, mean-reversion, momentum, or the sweep/H1-H4 families -
each with entry/exit rules, an economic rationale, expected trade
frequency, an overfitting-prevention plan, and pre-committed pass/fail
criteria stated before any code was written:

1. **Asian/London Range Breakout - Continuation** (built and tested,
   see below) - the deliberate mirror opposite of the already-rejected
   London Sweep reversal: trade WITH a confirmed Asian-range breakout,
   not against it.
2. **London Liquidity Sweep, re-tried with S5-resolution exhaustion
   filter** - not built. Two prior direct tests of this exact family
   (V1, V2) already REJECTED it, including a diagnostic that identified
   the reversal hypothesis itself, not confirmation logic, as the
   primary failure - re-testing would need a genuinely new angle to be
   worth the cost, rated weakest of the three candidates for that reason.
3. **Round-Number Level Consolidation** (cheap pre-check run, see
   below) - a genuinely different, price-level-anchored (not
   session-time-anchored) mechanism; REJECTED at the pre-check stage,
   before any entry logic was written.

### Asian/London Range Breakout - Continuation - REJECTED after development testing

Research question: does London-session order flow tend to EXTEND a
genuine overnight Asian-session breakout (momentum transfer, new desks
entering aligned with the move) rather than reject it - the exact
opposite economic mechanism from V1's already-rejected stop-hunt/
reversal thesis, not a parameter variation of it?

**Design** (`signals_london_breakout_continuation_m15.py`): reuses V1's
Asian-range/session-window infrastructure completely unchanged
(imported, not copied - same 00:00-07:00 Asian session, same
07:00-10:00 London entry window, both Europe/London local, DST-aware).
Entry: a CONFIRMED Close beyond the Asian high/low by
`breakout_buffer_atr_fraction` (0.1xATR, same value as V1's stop
buffer, reused not retuned) - the breakout itself is the signal, no
reversal/reject-then-confirm step. Stop: structural, at the OPPOSITE
side of the Asian range plus a buffer - a deliberately wide, structural
choice (if price reverses all the way through the far boundary, the
"range held and flow is continuing" thesis has fully failed, not just
pulled back). Target: fixed 1:1 R:R, same as V1/V2. One trade per
instrument per day. Because signal-on-Close/fill-at-next-Open matches
`backtest_engine.py`'s standard convention exactly, this strategy runs
through the same unmodified `run_backtest()` every other strategy in
this project uses - no standalone simulator needed, unlike H4.

15 new tests (`tests/test_london_breakout_continuation_signals.py`) -
Asian-range no-lookahead reuse, breakout detection incl. buffer
precision and intrabar-vs-confirmed-close discipline, one-trade-per-day,
structural stop/target formula incl. R:R scaling, validation/reserved-
access prevention. Full suite: 74/74 passing (59 pre-existing + 15 new).

**Round 1 result on DEVELOPMENT data (train/test split within
development, unchanged from every other strategy's methodology)**:

| | Trades | Win rate | PF | Net P&L (USD) |
|---|---|---|---|---|
| TRAIN (2020-08-24 to 2023-05-15) | 488 | 47.8% | **0.912** | -$549.73 |
| TEST (2023-05-15 to 2024-07-14) | 116 | 35.3% | **0.546** | -$818.89 |

TRAIN already fails the 1.2 profit-factor bar; TEST is materially
*worse*, not a marginal miss around a real edge. Both directions (long
PF 0.893/0.526, short PF 0.933/0.573 across train/test) lose money in
both periods; both instruments lose money combined (EUR_USD net
-$385.62, GBP_USD net -$983.00); TRAIN's PF is stable year-over-year
(0.907-0.921, 2020-2022) but stably *below* the bar, not above it. The
strategy's own drawdown-suspension safety limit fired 464 times in
TRAIN and 363 times in TEST - an independent signal of how badly it was
losing, not just a narrow P&L miss.

**Verdict against the pre-committed criteria** (PF < 1.2, expectancy
not significantly positive, inconsistent sign across instruments/years,
doesn't survive realistic costs): **every criterion triggers.** The
momentum-transfer continuation hypothesis does not hold - confirmed
Asian-range breakouts do not extend reliably enough to overcome even a
1:1 R:R structure, on either instrument, either direction, or any year
tested. **REJECTED.**

This closes out all three structurally distinct ways of trading the
Asian/London range boundary tried in this project: reversal (V1,
REJECTED), trend-filtered reversal (V2, REJECTED), and continuation
(this candidate, REJECTED).

Full trade-level data:
`results/london_breakout_continuation_round1_development_trades.csv`.
Full structured summary:
`results/london_breakout_continuation_round1_summary.json`. Not
deleted, per the same decision as V1/V2:
`signals_london_breakout_continuation_m15.py`,
`run_london_breakout_continuation_backtest.py`, and their tests remain
in the repository as a complete, honestly-labeled failed experiment.

### Round-Number Level Consolidation - REJECTED at the pre-check stage

Research question: does realized volatility measurably *compress* when
price is near a round number (institutional order-clustering /
psychological-level effect - a genuinely different, price-level-
anchored mechanism, not session-time-anchored like every other
candidate in this search)? Per the approved design, this required a
cheap statistical pre-check *before* any entry/exit logic - exactly
the H1-H3 discipline, applied here for the first time to a post-H4
candidate.

**Frozen protocol**: round level = multiple of 0.0050 (the standard FX
big-figure + half-figure convention, not tuned); "near" = Close within
0.1xATR(14) of the nearest round level (the same buffer fraction
reused throughout this whole project); volatility = realized vol over
the following 2h window (the same measure already used for H2/H3/H4,
for direct comparability); binary near/far split, no separate control-
window definition needed. **Pre-committed survival bar**: near-level
volatility must be significantly *lower* than far - a null or wrong-
signed result rejects the idea outright, before any code is written.

**Result: wrong-signed, decisively.** Volatility near round levels was
**~28% HIGHER**, not lower, than far from them - combined ratio 1.282
(p<0.0001), consistent in both instruments (EUR_USD 1.252, GBP_USD
1.266) and, notably, in **every single year 2020-2024** (ratios
1.14-1.27, all p<0.0001) - not a fluke of one regime. A plausible,
not-further-investigated explanation: price is likely classified
"near" a round level in the first place because it's actively moving
*through* the area, not because it's stalling there - the
classification may be capturing momentum bars, not consolidation ones.
Order clustering at round numbers may still be a real microstructure
phenomenon; it simply doesn't produce the volatility-dampening
signature this candidate's entry logic needed.

**REJECTED at the pre-check stage** - per the candidate's own approved
design, this does not proceed to any entry/exit design or backtest.
Full pre-registered protocol, all numbers, and the verdict:
`results/candidate3_roundnumber_falsification_summary.json`.
Re-runnable, unmodified script preserved at
`hypothesis_tests/roundnumber_falsification.py` (not imported by
anything, kept purely so this exact test never needs repeating).

**All three post-H4 candidates are now resolved, all REJECTED.**
Combined with the original 9-approach systematic search, V1/V2, and
Hypotheses 1-4, this brings the project's total to **17 independently
tried structural ideas, none surviving.**

## Mean-Reversion V2: Short-Horizon Return-Extreme Reversal - REJECTED at the statistical-premise stage

The first hypothesis tested against the strengthened standing criteria
(see above), and an explicit re-test of mean-reversion **as a signal
type**, not a reuse of the old, already-rejected 4H mean-reversion
baseline (`mean_reversion_signals_4h.py`: Bollinger(20,2std) band +
RSI(14,30/70) + Kaufman trend-efficiency filter - a "stretched from a
short-term range, in a non-trending market" story, rejected near-
breakeven at 1H/4H/daily). This hypothesis is mechanistically
different: a z-scored **multi-bar return extreme** (no bands, no RSI,
no trend filter) - a classical "sharp move overreacts and partially
reverses" story. Also explicitly distinguished from **Hypothesis 1's
own Baseline A** (GBP_USD's unconditional 1-bar M15 lag-1
autocorrelation, r=-0.030, p<0.0001 - reversal-signed but attributed to
generic bid-ask-bounce microstructure noise, not a real effect, since
it appeared symmetrically regardless of which instrument "led"): this
test uses 4H (far coarser, much less bid-ask-bounce-prone) and
conditions on a genuine 2-sigma multi-bar extreme, not routine
bar-to-bar noise.

**Scope**: same instruments/timeframe as the old baseline being
re-tested - `PORTFOLIO_SYMBOLS` (EUR_USD/GBP_USD/USD_JPY/XAU_USD), 4H,
DEVELOPMENT only. **Frozen protocol**: 3-bar (12h) cumulative return,
z-scored against its own trailing 60-bar std dev; |z|>=2.0 (standard,
fixed threshold, not swept); forward window matches the trigger window
(3 bars). **Pre-committed rejection rule**, stated before running: if
the forward return isn't significantly reversal-signed in *both*
directions, stop before any mechanical trade simulation - the cheapest
possible kill point, one stage earlier than every prior cheap check in
this search.

**Result**: neither condition cleared. Combined extreme-up forward
return was correctly signed (-0.033%) but only borderline (p=0.057,
not significant); combined extreme-down was **wrong-signed** (-0.024%,
should be positive for reversal) and not significant (p=0.230).
Per-instrument, this is inconsistent, not just weak: EUR_USD showed a
real, significant reversal on the up side (p=0.001, 63% hit rate) - but
its down side wasn't significant. USD_JPY's extreme-up group and
XAU_USD's extreme-down group each showed the **wrong sign**
(continuation, not reversal). GBP_USD showed nothing in either
direction. Per-year: only 2020 was significant (p=0.009); 2021 and
2024 were wrong-signed; 2022-2023 near zero. With eight
instrument-x-direction cells tested, one nominally-significant,
correctly-signed result (EUR_USD extreme-up) is close to what chance
alone predicts, and it wasn't replicated anywhere else.

**REJECTED at the statistical-premise stage** - the mechanical trade
simulation (realistic costs, 70/30 split, evaluation against the new
standing bar) was designed but never executed, per the pre-committed
stop rule. This differs in kind from Candidates 1-3 (also premise-stage
rejections) and from Hypothesis 4 (which survived the premise and
failed on execution cost): a clean, cheap premise-stage kill, the
earliest possible stopping point used in this search so far.

Full pre-registered protocol, all numbers, and the verdict:
`results/meanrev_v2_feasibility_summary.json`. Re-runnable, unmodified
script preserved at `hypothesis_tests/meanrev_v2_feasibility.py` (not
imported by anything, kept purely so this exact test never needs
repeating).

**Does not proceed to a build.** Mean-reversion as a signal type,
retested fresh, still does not hold up in this project's data. This
brings the project's total to **18 independently tried structural
ideas, none surviving.**

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
20. **The systematic search was put on hold** - every timeframe tried
    (1H, 4H, daily) on both entry-logic families (trend-following,
    mean-reversion), individually and combined, had failed to clear the
    validation bars robustly.
21. **Resumed with a genuinely different signal class**: time-series
    momentum (`signals_momentum_4h.py`) - long/short off the SIGN of the
    trailing N-bar return, not any specific price level being crossed
    (unlike channel breakout or an EMA crossover) and not a distance-
    from-a-band measure (unlike mean-reversion). Round 1 (lookback=20
    bars, matching the channel strategy's window for comparability,
    spec-literal 1.5x/3.0x ATR stop/target) failed outright: test PF
    0.771, test return -4.57%, though it comfortably cleared the trade-
    count minimum (433 trades) since a signal defined by sign-of-return
    is active almost continuously, unlike the sparser breakout/reversion
    signals. An 8-point lookback sweep (5 to 100 bars) found no
    survivors - every single lookback failed, best test PF across the
    whole sweep only 0.966 (still below breakeven), with a noisy,
    non-monotonic drawdown pattern across lookbacks (3.67% to 20.15%)
    showing no coherent structure to chase. Not stress-tested - nothing
    cleared the single split. Notably, this mechanically unrelated
    signal construction converged on the SAME rough shape as channel
    breakout (win rate hugging ~28-35%, near the ~33% breakeven this
    2:1 R:R needs) - three now-independent signal classes landing in
    the same place is itself evidence of a structural limit somewhere
    in this instrument set/timeframe/R:R combination, not a specific
    idea worth re-trying with different parameters.
22. **The systematic search was paused again** - three mechanically
    distinct signal classes (trend/breakout, mean-reversion, time-series
    momentum), each tried across multiple timeframes, none robust. The
    one structurally different idea not yet attempted was relative-value/
    pairs trading between correlated instruments.
23. **Empirical check on the pairs idea, before building anything**:
    EUR_USD/GBP_USD return correlation is 0.744 - clearly the best pair
    available (USD_JPY: -0.46/-0.41 anti-correlated; XAU_USD: ~0.37
    weak) - but the SPREAD's own mean-reversion structure checked out
    weak on two independent measures: its 20-bar Efficiency Ratio (0.231)
    was no lower than either leg's own (0.236/0.233 - the classic pairs-
    trading rationale is that common-mode noise cancels, leaving a spread
    that's MORE mean-reverting than the legs; that wasn't found here),
    and an AR(1) half-life estimate came out to ~43 days - too slow to
    obviously exploit at a 4H holding period.
24. **A cheap phase-1 check before committing to the full two-leg engine**
    (an approximate, close-only, zero-transaction-cost simulation of the
    z-score/spread rule) confirmed the discouraging read: train was net
    LOSING (PF 0.933) while test "passed" (PF 1.255) - train and test
    pointing in opposite directions, a pattern this project has learned
    to distrust regardless of which side looks good, and this ignored
    real per-leg bid/ask costs entirely, which would likely erode the
    apparent test edge further given how weak the underlying structural
    evidence already was. Recommended NOT proceeding to the full,
    realistic two-leg backtest engine on this basis - not built.
25. **A fifth instrument, AUD_USD, tried on the existing (unmodified) 4H
    trend-following strategy** - reusing 100% of the existing
    infrastructure, just a new instrument. Failed more clearly than the
    existing 4-instrument basket: win rate 26.4%/28.4% (train/test),
    meaningfully BELOW the ~33% breakeven this 2:1 R:R needs, not just
    short of it (vs. 39.3%/34.3% for the original basket's round-1
    baseline). Data quality wasn't the issue (9,330 clean real candles).
    Added `AUD_USD` to `instruments.py`'s registry permanently for this
    check; introduced `PORTFOLIO_SYMBOLS` as an explicit separate
    constant so the registry can hold instruments available for
    standalone testing without silently pulling them into every existing
    multi-instrument script the moment they're added (every `run_*.py`
    entry point was looping over the full registry directly before this).
26. **A direct test of the R:R hypothesis**: every variant of this
    strategy had landed win rates around 26-52%, while the 2:1 R:R
    baseline needs ~33%+ to break even - tested whether a less greedy
    target lets the win rate actually achieved clear the profit-factor
    bar. Swept R:R from 2:1 down to 1:1 (target multiple only, stop
    fixed at the spec-literal 1.5x) on the 4-instrument 4H portfolio.
    Win rate climbed exactly as hypothesized - 34.8% to 52.5%, clean and
    monotonic - but profit factor did NOT follow, staying flat in a
    0.965-1.094 band across the entire range: the higher win rate was
    being paid for by proportionally smaller wins, netting out to
    roughly the same edge (or lack of one) regardless of R:R. Best PF
    across the whole sweep (1.094, at 1.25:1) still well short of 1.2.
    Also flagged honestly: trade counts swung wildly and non-
    monotonically across configs (27 to 220, no relationship to R:R) -
    the same chaotic drawdown-suspension path-dependence documented in
    item 10, meaning even the modest PF differences between configs
    shouldn't be read as a real ranking. Nothing cleared the single
    split, so nothing was stress-tested.
27. **Dual-confirmation (require trend-following AND a second signal to
    agree) was tried next** - trend-following (signals_4h.py) AND time-
    series momentum (signals_momentum_4h.py) both had to point the same
    direction (`signals_confirmed_4h.py`). Round 1 (momentum lookback=20,
    matching the breakout channel's own window) produced results BYTE-
    FOR-BYTE IDENTICAL to the unfiltered trend baseline - proven
    algebraically, not just observed empirically, to be a tautology: with
    matching windows, a 20-bar breakout mathematically guarantees 20-bar
    momentum agreement (every price in the breakout window, including the
    one 20 bars back, is bounded by the same rolling max), so the
    "confirmation" filters out zero trades by construction. A follow-up
    sweep of momentum lookbacks that genuinely break the tautology
    (25-150 bars, strictly longer than the 20-bar channel) found NO
    improvement - 6 of 7 configs were WORSE than the unfiltered baseline,
    the best case merely matched it. Requiring independent agreement
    filtered out profitable trades along with bad ones, net negative.
28. **A second confirmation-filter variant was tried and also failed**:
    volatility-expansion (require ATR to genuinely exceed its own 20-bar
    rolling average, `Signal4HConfig.use_volatility_filter`), tested the
    same cheap-check-first way. Decisively worse on every dimension -
    trade count collapsed to 74 combined (below the 150 minimum), win
    rate dropped BELOW the R:R breakeven (30-31% vs baseline's 34-39%),
    profit factor fell below 1.0 in both periods. A volatility filter was
    tried once before on the original 1H strategy and also made things
    worse there (item 3) - this was a genuine re-test on a different,
    later construction (single-timeframe 4H, after several unrelated
    fixes), not just repeating the same experiment, but it landed in the
    same place.
29. **Stepped back from confirmation-filter variants as a family**
    (two tried, both failed clearly) rather than continuing to test more
    of them, and took stock of the full search across every axis tried.
30. **The systematic search is CONCLUDED**, not paused - a deliberate
    decision, not just an observation. Nine independent axes (signal
    class, timeframe, instrument, portfolio combination, payout
    structure, confirmation filters) all converged on the same outcome;
    see "Systematic strategy search: concluded" above for the full
    comparison table, the reasoning behind concluding rather than
    continuing, and the specific remaining options considered and
    explicitly not pursued (the full pairs engine, exit-logic changes, a
    fundamentally different data source, a broader instrument universe)
    with reasons for each. What remains from this project: a well-tested
    backtest engine, a genuinely rigorous validation process, and a live
    demo-execution connector verified end-to-end against a real OANDA
    practice account - not a strategy this project would recommend
    trading.

**Bottom line:** every "this fixes it" moment in this history except the
efficiency-ratio filter (item 9) was later found to be wrong or to not
generalize, and the broader search across timeframes, a combined
portfolio, a genuinely different signal class, a new instrument, a
direct test of the payout structure itself, and two confirmation-filter
variants (items 13-30) didn't find anything that did generalize either.
The efficiency-ratio filter
remains the one piece of real, partially-
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

## Live trading (execution plumbing built AND verified against real OANDA, plugged into an UNVALIDATED strategy)

The `live/` directory has a complete demo-trading connector, built
specifically to prove the execution plumbing (fetch price -> check
signal -> size position -> place order -> log -> stop mechanism) works
end-to-end, before any strategy has cleared this project's own
validation bars - so that plumbing doesn't have to be built from scratch
later. **This has now actually been run against the real OANDA practice
server, including a real order** - see "What actually happened" below.

- `account_safety.py` - hard, mandatory, 3-layer check that the
  connected account is genuinely PRACTICE/demo, never live (tested and
  working, no bypass).
- `oanda_live_client.py` - the only place beyond `data_fetch.py` that
  talks to OANDA: account state, real per-instrument margin rate/
  precision, open trades, recent closed trades, order placement with an
  atomically-attached stop-loss/take-profit, force-close. The stop-loss
  is placed as a DISTANCE from the fill price, not an absolute price
  computed in advance - a market order's exact fill price isn't known
  until after it fills, so an advance price estimate could end up stale
  and produce a nonsensical or rejected stop (take-profit stays
  price-based - OANDA's API has no distance option for take-profit).
  The distance value is built as a raw request dict by hand, NOT via
  `oandapyV20`'s `StopLossDetails` convenience class - that class's
  constructor only accepts `price`, not `distance` (found by actually
  running this - see below), even though OANDA's real REST API accepts
  a hand-built `distance` field just fine.
- `live_state.py` - `LiveRiskState`, persisted to `live_data/state.json`
  so a restart doesn't lose the day's starting equity or the all-time
  peak. Shares its day/week-rollover, drawdown-suspension, and
  consecutive-loss-cooldown LOGIC with the backtest's `PortfolioAccount`
  via functions extracted into `risk_management.py`
  (`advance_day_week_rollover`, `advance_risk_flags`, `maybe_start_cooldown`)
  - one implementation of the circuit-breaker state machine, not two
  copies that could drift apart between backtest and live. Never tracks
  balance/NAV itself - OANDA's own numbers are always the source of truth.
- `live_account_sync.py` - reconciles local expectations against
  OANDA's real open trades/balance/NAV each cycle; a mismatch (e.g. an
  untracked open trade, a missing stop-loss field) is a loud logged
  warning, never a silent "correction." Consecutive-loss count is
  derived fresh from real closed-trade history each cycle rather than
  tracked as fragile local state.
- `order_execution.py` - sizes a trade using the exact same
  `calculate_stop_and_target`/`calculate_position_size` functions the
  backtest uses (no reimplementation), capped at the STRICTER of this
  project's 30:1 policy leverage or OANDA's real per-instrument margin
  rate (Gold's real cap, 20:1, is stricter - flagged when the client was
  first built, now actually wired in). Verifies the stop-loss actually
  attached after every fill and force-closes immediately if not.
- `live_logging.py` - one append-only JSONL line per cycle event to
  `live_data/run_live.jsonl` (gitignored) - the audit trail that actually
  confirmed the plumbing worked end-to-end (see below - stdout itself
  turned out not to be a reliable way to watch a redirected/background run).
- `run_live.py` - the main loop. Plugs in `signals_4h.py` as a
  deliberately swappable PLACEHOLDER strategy - it failed this
  project's validation bars (see "Systematic strategy search: concluded"
  above) and every startup banner/log line says so; this run exists to
  prove the plumbing, not to trade profitably. Polls every 5 minutes,
  only acting on a genuinely new closed 4H candle. Three independent
  stop mechanisms: Ctrl+C (finishes the current cycle cleanly), a
  `live_data/STOP` kill-switch file (checked every cycle and every
  second while sleeping), and a default 24-hour max-runtime cap.
- `manual_test_trade.py` - a standalone diagnostic (not part of the main
  loop) that calls `order_execution.execute_signal()` directly - the
  EXACT function `run_live.py` calls on a real signal - to force one
  real order on demand, rather than waiting for the placeholder
  strategy to happen to fire. Opens, verifies, and closes one trade.
  Meant to be re-run after any future change to `order_execution.py`.

### What actually happened (first real run against the practice account)

1. **`run_live.py` ran for 3 clean cycles** against the real account
   (101-004-40015068-001, GBP, ~£100,000 balance) with no errors, then
   was stopped deliberately via the `live_data/STOP` kill-switch -
   confirming account verification, real candle fetch, and the
   no-signal/reject-and-log path all work. Found along the way: Python
   buffers `print()` output when stdout isn't an interactive terminal
   (true for a background/redirected run), so the startup banner and
   cycle logs weren't visible in real time - not a bug, but it meant the
   JSONL audit log (which flushes on every event) turned out to be the
   actually-reliable way to watch a run, not stdout.
2. **A forced manual test trade surfaced a real bug on the first try**:
   `oandapyV20.contrib.requests.StopLossDetails`'s constructor only
   accepts `price`, not `distance`, and raised a plain `TypeError`
   before the request ever reached OANDA - meaning the distance-based
   stop-loss fix from the first build (see `oanda_live_client.py` above)
   had never actually been reachable code. Fixed by building that one
   field's request dict by hand instead of going through the convenience
   wrapper.
3. **A second test trade then succeeded completely**: EUR_USD long,
   sized off the real account balance (141,276 units), stop-loss
   attached at exactly `fill_price - stop_distance` (1.15513),
   take-profit at exactly `fill_price + target_distance` (1.16306) -
   independently reconfirmed via a second, separate `get_open_trades()`
   call (not just `execute_signal()`'s own internal check) - then closed
   immediately for a realized P&L of -£8.43 (spread cost of an instant
   round-trip, as expected). Final account check: 0 open trades
   remaining, balance down only that spread cost.
4. **Current status: stopped, account clean, nothing trading
   unattended.** `run_live.py` was stopped BEFORE the manual test ran
   (not after), specifically to avoid any race between the bot's own
   loop and the manual test both touching the same account at once.

This confirms the execution path itself - account verification,
real-time data fetch, real-balance-based sizing, order placement, and
atomic stop-loss/take-profit attachment - genuinely works against
OANDA's real practice server. It does NOT mean there's a strategy worth
running with it: `signals_4h.py` is still the unvalidated placeholder
throughout all of the above.

A few OANDA response field-name assumptions (`stopLossOrder`, `NAV`,
`marginUsed`, trade `id`) were confirmed correct by this real run;
anything not exercised by these 3 cycles + 1 forced trade (e.g. a
genuine consecutive-loss cooldown triggering, a drawdown suspension, a
real signal firing on its own) remains unverified against the live API
and should still be watched closely, not left unattended, the first
time it happens.

### Second real run - organic signals, and a genuine safety-limit bug found live

Run again later, this time letting it trade organically (no forced test
trade) for an extended period. Handled two transient OANDA/network
errors correctly (a connection reset, and a one-off "insufficient
authorization" response) - both caught by the per-cycle error handling,
logged, and recovered on the next cycle without intervention; verified
independently at the time that credentials were still valid and no
position was left unprotected by either.

Then a real signal fired organically - and surfaced a genuine bug: **3
instruments in the SAME correlation group (`EUR_USD`, `GBP_USD`,
`USD_JPY` - all `usd_fx`) opened positions in the SAME cycle**, directly
violating the "only one open trade per correlation group at a time"
rule this project has enforced since the original spec. Root cause:
`run_cycle()` fetched `open_positions`/`correlation_groups_open` ONCE at
the top of the cycle and checked every instrument's gate against that
same static snapshot - so if EUR_USD opened first, GBP_USD's gate check
moments later in the same cycle's loop still saw the pre-cycle snapshot
with no EUR_USD position in it. The backtest engine never had this bug
(its per-symbol loop shares one live, mutable `PortfolioAccount` object,
updated in real time as each symbol is processed within a bar) - this
was specific to `run_live.py` computing a local snapshot once per cycle
and never updating it as orders were placed within that same cycle.

Stopped the bot immediately on discovering this (via the kill-switch)
rather than let it keep running with the gate not doing its job. Fixed
by updating the local `open_positions`/`correlation_groups_open`
immediately after each successful order within the cycle's loop, not
just on the next cycle's fresh sync - verified offline with a direct
test that a same-group second entry is now correctly rejected within
the same cycle. Checked the account afterward: no lingering exposure -
all 4 positions (the 3 correlated ones plus an independent XAU_USD
trade) had already closed via their own broker-held stops/targets by
the time this was investigated, 0 open trades, no unrealized P&L. No
real financial consequence either way (demo account), but the safety
mechanism itself failed to do its job as designed, which mattered
regardless of the outcome this time.

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
- **Leverage cap (30:1)** in the BACKTEST is a realistic-but-arbitrary
  retail limit for scoring purposes; not fetched from OANDA's actual
  margin rules (backtesting 4 fixed instruments doesn't warrant a live
  API call per run). The LIVE connector (`live/order_execution.py`)
  does use OANDA's real per-instrument margin rate, taking the stricter
  of it and this 30:1 policy cap - Gold's real cap is stricter, 20:1.
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
| `instruments.py` | Per-instrument specs (now including `AUD_USD`, added for a standalone check - see "Tuning history" item 25), pip/notional value math (incl. cross-currency), correlation group, spread cap, swap rates; `PORTFOLIO_SYMBOLS` is the explicit 4-instrument list every multi-instrument script actually trades, kept separate from the full registry |
| `dataset_split.py` | Three-way chronological DEVELOPMENT/VALIDATION/FINAL RESERVED split for the London Liquidity Sweep work - see "London Liquidity Sweep Reversal V1" above. Separate from `run_backtest.py`'s own two-way split, which is unchanged |
| `signals_london_sweep_m15.py` | London Liquidity Sweep Reversal V1 - M15, Europe/London session-anchored, structural stop/target. **Rejected after development testing** - see "London Liquidity Sweep Reversal V1" above. Not deleted - kept as a complete, honestly-labeled failed experiment |
| `run_london_sweep_backtest.py` | Entry point for V1 - EUR_USD/GBP_USD only, `dataset_split.split_for_iteration()` exclusively |
| `signals_london_sweep_trend_aligned_m15.py` | V2 - imports V1's sweep+confirmation logic unchanged, adds a daily EMA50/200 trend-alignment gate. **Rejected after development testing** - see "London Liquidity Sweep Reversal V2" above |
| `run_london_sweep_trend_aligned_backtest.py` | Entry point for V2 - same scope as V1's entry point, plus daily candle data for the trend gate |
| `signals_london_breakout_continuation_m15.py` | Post-H4 Candidate 2 - reuses V1's Asian-range infrastructure unchanged, trades WITH a confirmed breakout instead of against it (opposite mechanism from V1/V2). **Rejected after development testing** - see "Post-H4 structural search" above |
| `run_london_breakout_continuation_backtest.py` | Entry point for Candidate 2 - runs through the standard, unmodified `run_backtest()` (unlike H4, this signal's timing matches the engine's own fill convention) |
| `hypothesis_tests/` | Cheap, pre-registered, single-shot statistical falsification tests for candidate hypotheses - deliberately not strategy code, nothing here is imported by any strategy or the backtest engine. Currently: `leadlag_falsification.py` (H1, rejected), `gap_fade_falsification_round1_24h_hold.py` + `gap_fade_falsification_round2_closure.py` (H2, rejected), `monthend_falsification.py` (H3, rejected), `econ_event_volatility_round1_statistical.py` + `..._spread_resolution_check.py` + `..._round2_breakout_trigger.py` + `..._round3_s5_fetch.py` + `..._round3_s5_simulation.py` + `data/economic_events_development.csv` + `data/economic_events_validation.csv` (H4, REJECTED decisively after round 3's finest-resolution, out-of-sample-confirmed test - `data/s5_cache/` is the large, regenerable, gitignored raw S5 candle cache behind it), `roundnumber_falsification.py` (post-H4 Candidate 3, rejected at the pre-check stage), `meanrev_v2_feasibility.py` (Mean-Reversion V2, rejected at the statistical-premise stage). See "V3 candidate search", "Hypothesis 4", "Post-H4 structural search", and "Mean-Reversion V2" above |
| `results/` | Permanently preserved raw results from concluded experiments (V1/V2 full trade-level data + structured summaries, V3 candidate falsification summaries), so an experiment never needs re-running just to recall what happened |
| `data_fetch.py` | Paginated OANDA candle fetch (read-only) with local CSV caching + synthetic fallback; supports M5/M15/H1/H4/D granularities |
| `indicators.py` | EMA, ATR, rolling high/low channel, SMA, rolling std, RSI |
| `signals.py` | Original 4H-trend/1H-entry trend-following strategy (merges the 4H trend filter onto the 1H timeline, no lookahead); `SignalConfig` holds all tunable entry-logic parameters |
| `signals_4h.py` | Single-timeframe 4H trend-following - trend filter and entry trigger both computed from the same 4H series; `Signal4HConfig` |
| `signals_daily.py` | Single-timeframe daily trend-following, same structure as `signals_4h.py`; `SignalDailyConfig` |
| `mean_reversion_signals.py` | 1H Bollinger/RSI mean-reversion entry with a 4H Efficiency-Ratio trend-avoidance filter; `MeanReversionConfig` |
| `mean_reversion_signals_4h.py` | Single-timeframe 4H mean-reversion - entry and trend-avoidance filter both on the same 4H series; `MeanReversion4HConfig` |
| `mean_reversion_signals_daily.py` | Single-timeframe daily mean-reversion, same structure as the 4H version; `MeanReversionDailyConfig` |
| `combined_signals_4h.py` | Merges the 4H trend-following and mean-reversion signal frames per instrument into one combined frame, for running both out of one shared account (see "Tuning history" item 17) |
| `signals_momentum_4h.py` | Single-timeframe 4H time-series momentum - long/short off the sign of the trailing N-bar return, a genuinely different signal class from breakout/mean-reversion (see "Tuning history" item 21); `MomentumConfig` |
| `risk_management.py` | Position sizing (with leverage cap) + `PortfolioAccount`: every safety limit, breakeven-stop trade management, shared across all instruments; `RiskConfig` holds all tunable risk parameters |
| `econ_calendar.py` | The approximate news-blackout heuristic (swappable) - a documented no-op at daily granularity, see "Known approximations" |
| `backtest_engine.py` | The custom multi-instrument, chronological-lockstep backtest simulator; `bar_duration_hours` parameter makes the financing-rollover check correct at any granularity, and a frame may optionally carry per-row `stop_distance_override`/`target_distance_override`/`signal_source` columns (used by `combined_signals_4h.py`) |
| `run_backtest.py` | Entry point for `signals.py`; also the shared helper module (`split_train_test`, `compute_metrics`, `evaluate_requirements`, report/comparison printers) every other `run_*.py` script imports from |
| `run_backtest_4h.py` / `run_backtest_daily.py` | Entry points for the single-timeframe 4H / daily trend-following strategies |
| `run_mean_reversion_backtest.py` / `run_mean_reversion_backtest_4h.py` / `run_mean_reversion_backtest_daily.py` | Entry points for the 1H / 4H / daily mean-reversion strategies |
| `run_combined_backtest_4h.py` | Entry point for the combined 4H trend-following + mean-reversion portfolio |
| `run_momentum_backtest_4h.py` | Entry point for the 4H time-series momentum strategy, including its lookback sweep |
| `tests/` | pytest suite (`requirements-dev.txt`) - currency-conversion/position-sizing math, the dataset-split guard rails, and London Sweep signal logic (BST/GMT, no-lookahead, sweep/confirmation, one-trade-per-day, stop/target math) |
| `live/` | Demo-trading execution connector - built AND verified against the real OANDA practice server (a real order placed, sized, protected, and closed successfully) - see "Live trading" above |
| `.env.example` | Template for the environment variables `data_fetch.py`/`live/` need - copy to `.env` |
