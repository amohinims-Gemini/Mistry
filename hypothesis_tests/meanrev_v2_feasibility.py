"""
hypothesis_tests/meanrev_v2_feasibility.py
--------------------------------------------
Cheap feasibility check for a FRESH mean-reversion hypothesis
("Short-Horizon Return-Extreme Reversal") - the standing bar for any
future strategy round, applied here for the first time. Explicitly NOT
a re-run of the old, already-rejected 4H mean-reversion baseline.

RESULT: REJECTED at the statistical-premise stage. Neither pre-
committed condition was cleared (extreme-down group wrong-signed and
non-significant; extreme-up group borderline, p=0.057, not
significant at the combined level). Inconsistent across instruments
(EUR_USD showed a real, significant up-side reversal; USD_JPY and
XAU_USD each showed a WRONG-signed continuation on one side; GBP_USD
showed nothing) and across years (only 2020 significant; 2021 and 2024
wrong-signed). Per the pre-committed stop rule, the mechanical trade
simulation (Part 2 of the script) was never run - killed at the
cheapest possible stage. See
results/meanrev_v2_feasibility_summary.json for the full numeric
record and README.md's "Mean-Reversion V2" section for the narrative
writeup. Preserved unmodified, purely so this exact experiment never
needs re-running - not imported by anything in the project.

Cheap feasibility check for a FRESH mean-reversion hypothesis
("Short-Horizon Return-Extreme Reversal"), explicitly NOT a re-run of
the old, already-rejected 4H mean-reversion baseline
(mean_reversion_signals_4h.py: Bollinger(20,2std) + RSI(14,30/70) +
Kaufman trend-efficiency filter). DEVELOPMENT data only, 4H, the same
4-instrument PORTFOLIO_SYMBOLS scope the old baseline used
(EUR_USD/GBP_USD/USD_JPY/XAU_USD).

===========================================================================
WHY THIS IS A GENUINELY DIFFERENT HYPOTHESIS, NOT A RETUNE
===========================================================================
1. Old baseline: price relative to a 20-bar Bollinger Band (mean +/-
   std-dev band), gated by an RSI(14) extreme AND a trend-efficiency
   filter - a "stretched from a short-term range, in a non-trending
   market" story. REJECTED (near-breakeven at best, across 1H/4H/daily).
2. This hypothesis: a z-scored MULTI-BAR RETURN extreme (not a price
   band), no RSI, no trend filter - a "sharp, fast move overreacted and
   partially reverses" story (classical short-term reversal /
   liquidity-provision-correction mechanism), mechanistically distinct
   from "stretched from a range."
3. Also explicitly distinct from Hypothesis 1's own Baseline A
   (GBP_USD's UNCONDITIONAL lag-1 M15 autocorrelation: r=-0.030,
   p<0.0001 - negative/reversal-SIGNED, but attributed to generic
   bid-ask-bounce microstructure noise, not a real economic effect,
   since it appeared symmetrically regardless of which instrument
   "led"). This test uses 4H (far coarser than M15, much less prone to
   bid-ask bounce) and CONDITIONS on a genuine multi-bar statistical
   EXTREME (|z|>=2), not the unconditional bar-to-bar autocorrelation
   H1 already found and explained away. If this test's own result
   turns out to be indistinguishable from H1's Baseline A in
   character, that itself would be a reason for skepticism - reported
   honestly either way.

===========================================================================
FROZEN DEFINITIONS (fixed BEFORE any numbers were looked at)
===========================================================================
1. Return window: 3 bars (12h) cumulative log return at 4H.
2. Z-score: that 3-bar return divided by the trailing 60-bar (~10
   trading days) standard deviation of 3-bar returns - a fixed,
   standard normalization, not tuned.
3. Extreme threshold: |z| >= 2.0 - the standard "2-sigma" convention,
   fixed, not swept.
4. Forward window: the FOLLOWING 3-bar (12h) cumulative log return -
   same horizon as the trigger window, for direct comparability.
5. No lookahead: the z-score and classification use only data up to
   and including the trigger bar's own close; the forward window is
   strictly after.
6. Mechanical trade simulation (only run if the statistical premise
   below survives): direction is COUNTER to the extreme (short after
   an extreme-up, long after an extreme-down) - mechanical, not
   hindsight. Entry at the next bar's Open (matches
   backtest_engine.py's own fill convention), realistic Ask/Bid fill +
   SLIPPAGE_ATR_FRACTION=0.02xATR (reused verbatim from
   backtest_engine.py). Stop: structural, beyond the extreme run's own
   most-adverse extreme (the multi-bar High/Low reached during the
   3-bar trigger window) + 0.1xATR buffer (reused buffer style from
   V1/V2/H4/Candidate 2 - not independently chosen). Target: fixed 1:1
   R:R (same value as every other structural candidate in this
   project). Chronological 70/30 train/test split, matching
   run_backtest.py's TRAIN_FRACTION exactly.

===========================================================================
PRE-COMMITTED REJECTION CRITERIA (stated before running)
===========================================================================
- If the forward return is NOT significantly reversal-signed (extreme-
  up group's forward return not significantly negative, OR extreme-
  down group's not significantly positive) -> REJECTED, no further
  build.
- If the statistical premise survives but the mechanical trade
  simulation's profit factor does not clear the NEW standing bar
  (>1.3) in BOTH train AND test -> REJECTED.
- If train/test profit factor diverges beyond the new standing
  tolerance (test PF < 0.65 x train PF) -> REJECTED.
- Only if ALL of the above are cleared does this proceed to a full
  build (signals module + tests + entry point) - not decided here.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from dataset_split import split_for_iteration, describe_split
from indicators import atr as atr_indicator
from run_backtest import (
    TRAIN_FRACTION, MIN_REQUIRED_TRADES, MIN_REQUIRED_PROFIT_FACTOR,
    IS_OOS_MAX_RELATIVE_PF_DROP, MAX_SINGLE_TRADE_PROFIT_SHARE,
    MAX_SINGLE_MONTH_PROFIT_SHARE, MIN_POSITIVE_REGIME_YEARS_FRACTION,
)

RETURN_WINDOW_BARS = 3        # 12h at 4H, fixed, not swept
ZSCORE_LOOKBACK_BARS = 60     # ~10 trading days of 4H bars, fixed
Z_THRESHOLD = 2.0             # standard 2-sigma convention, fixed
FORWARD_WINDOW_BARS = 3       # matches trigger window
ENTRY_STOP_BUFFER_ATR_FRACTION = 0.1   # reused from V1/V2/H4/Candidate 2
TARGET_RR = 1.0                        # reused from every structural candidate
SLIPPAGE_ATR_FRACTION = 0.02           # reused verbatim from backtest_engine.py


def _slip(price, direction, amount):
    return price + amount if direction == "long" else price - amount


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def one_sample_t(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), len(x)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    t_stat = m / se if se > 0 else 0.0
    return m, t_stat, two_sided_p_from_z(t_stat), len(x)


def hit_rate_and_p(x, expected_sign):
    x = np.asarray(x, dtype=float)
    x = x[x != 0]
    n = len(x)
    hits = int(((x * expected_sign) > 0).sum())
    rate = hits / n if n else float("nan")
    z = (hits - 0.5 * n) / math.sqrt(0.25 * n) if n else float("nan")
    return rate, n, hits, z, two_sided_p_from_z(z) if n else float("nan")


print("=" * 78)
print("MEAN-REVERSION V2 FEASIBILITY CHECK: Short-Horizon Return-Extreme Reversal")
print("4H, PORTFOLIO_SYMBOLS (EUR_USD/GBP_USD/USD_JPY/XAU_USD), DEVELOPMENT ONLY")
print("=" * 78)
print(describe_split())
print(f"Return window={RETURN_WINDOW_BARS} bars, z lookback={ZSCORE_LOOKBACK_BARS} bars, "
      f"threshold=|z|>={Z_THRESHOLD}, forward window={FORWARD_WINDOW_BARS} bars")
print()

frames = {}
for symbol in PORTFOLIO_SYMBOLS:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "H4")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    frames[symbol] = df
    print(f"{symbol}: {len(df)} H4 bars, {df.index.min()} -> {df.index.max()}")
print()

# ===========================================================================
# PART 1: pure statistical premise - does the reversal actually exist?
# ===========================================================================
print("=" * 78)
print("PART 1: STATISTICAL PREMISE - forward return after a statistical extreme")
print("=" * 78)

event_frames = {}
for symbol in PORTFOLIO_SYMBOLS:
    df = frames[symbol].copy()
    log_close = np.log(df["Close"])
    trigger_ret = log_close.diff(RETURN_WINDOW_BARS)
    z = (trigger_ret - trigger_ret.rolling(ZSCORE_LOOKBACK_BARS).mean()) / \
        trigger_ret.rolling(ZSCORE_LOOKBACK_BARS).std()
    forward_ret = log_close.shift(-FORWARD_WINDOW_BARS) - log_close

    df["trigger_ret"] = trigger_ret
    df["z"] = z
    df["forward_ret"] = forward_ret
    df["is_extreme_up"] = z >= Z_THRESHOLD
    df["is_extreme_down"] = z <= -Z_THRESHOLD
    event_frames[symbol] = df

    up = df[df["is_extreme_up"]].dropna(subset=["forward_ret"])
    down = df[df["is_extreme_down"]].dropna(subset=["forward_ret"])

    print(f"\n{symbol}: {len(up)} extreme-up events, {len(down)} extreme-down events")
    if len(up) >= 10:
        m, t, p, n = one_sample_t(up["forward_ret"].values)
        rate, n_h, hits, zh, ph = hit_rate_and_p(up["forward_ret"].values, expected_sign=-1)
        print(f"  extreme-up  -> forward return: mean={m*100:+.4f}% (t={t:.2f}, p={p:.4f})  "
              f"reversal(down) rate={rate*100:.1f}% (p={ph:.4f})")
    if len(down) >= 10:
        m, t, p, n = one_sample_t(down["forward_ret"].values)
        rate, n_h, hits, zh, ph = hit_rate_and_p(down["forward_ret"].values, expected_sign=1)
        print(f"  extreme-down-> forward return: mean={m*100:+.4f}% (t={t:.2f}, p={p:.4f})  "
              f"reversal(up) rate={rate*100:.1f}% (p={ph:.4f})")

# combined
all_up = pd.concat([event_frames[s][event_frames[s]["is_extreme_up"]].dropna(subset=["forward_ret"])
                     for s in PORTFOLIO_SYMBOLS])
all_down = pd.concat([event_frames[s][event_frames[s]["is_extreme_down"]].dropna(subset=["forward_ret"])
                       for s in PORTFOLIO_SYMBOLS])
print(f"\nCOMBINED: {len(all_up)} extreme-up, {len(all_down)} extreme-down (all 4 instruments)")
m_up, t_up, p_up, n_up = one_sample_t(all_up["forward_ret"].values)
rate_up, _, _, _, ph_up = hit_rate_and_p(all_up["forward_ret"].values, expected_sign=-1)
print(f"  extreme-up  -> forward return: mean={m_up*100:+.4f}% (t={t_up:.2f}, p={p_up:.4f})  "
      f"reversal rate={rate_up*100:.1f}% (p={ph_up:.4f})")
m_down, t_down, p_down, n_down = one_sample_t(all_down["forward_ret"].values)
rate_down, _, _, _, ph_down = hit_rate_and_p(all_down["forward_ret"].values, expected_sign=1)
print(f"  extreme-down-> forward return: mean={m_down*100:+.4f}% (t={t_down:.2f}, p={p_down:.4f})  "
      f"reversal rate={rate_down*100:.1f}% (p={ph_down:.4f})")

print(f"\n{'='*78}\nSTABILITY BY YEAR (combined, both directions pooled as 'reversal magnitude')\n{'='*78}")
all_events = pd.concat([
    all_up.assign(expected_sign=-1),
    all_down.assign(expected_sign=1),
])
all_events["reversal_ret"] = all_events["forward_ret"] * all_events["expected_sign"]  # positive = reversal happened
all_events["year"] = all_events.index.year
for yr, g in all_events.groupby("year"):
    if len(g) < 15:
        print(f"  {yr}: n={len(g)} (too few - skipping)")
        continue
    m, t, p, n = one_sample_t(g["reversal_ret"].values)
    print(f"  {yr}: n={n:4d}  mean reversal-signed return={m*100:+.4f}% (t={t:.2f}, p={p:.4f})")

premise_survives = (
    p_up < 0.05 and m_up < 0 and
    p_down < 0.05 and m_down > 0
)
print(f"\n{'='*78}\nSTATISTICAL PREMISE VERDICT: {'SURVIVES' if premise_survives else 'REJECTED'}\n{'='*78}")

if not premise_survives:
    print("\nStopping here per pre-committed criteria - the statistical premise itself")
    print("was not confirmed, so no mechanical trade simulation is run.")
    sys.exit(0)

# ===========================================================================
# PART 2: mechanical trade simulation, evaluated against the NEW standing bar
# ===========================================================================
print("\n" + "=" * 78)
print("PART 2: MECHANICAL TRADE SIMULATION (only run because Part 1 survived)")
print("=" * 78)

all_trades = []
for symbol in PORTFOLIO_SYMBOLS:
    df = event_frames[symbol]
    atr_series = atr_indicator(df["High"], df["Low"], df["Close"], period=14)
    idx = df.index
    pip = INSTRUMENTS[symbol].pip_size

    for i in range(ZSCORE_LOOKBACK_BARS, len(df) - FORWARD_WINDOW_BARS - 1):
        row = df.iloc[i]
        if not (row["is_extreme_up"] or row["is_extreme_down"]):
            continue
        atr_now = atr_series.iloc[i]
        if pd.isna(atr_now) or atr_now <= 0:
            continue

        direction = "short" if row["is_extreme_up"] else "long"
        trigger_window = df.iloc[i - RETURN_WINDOW_BARS + 1:i + 1]
        if direction == "short":
            adverse_extreme = trigger_window["High"].max()
            stop_price = adverse_extreme + ENTRY_STOP_BUFFER_ATR_FRACTION * atr_now
        else:
            adverse_extreme = trigger_window["Low"].min()
            stop_price = adverse_extreme - ENTRY_STOP_BUFFER_ATR_FRACTION * atr_now

        entry_bar = df.iloc[i + 1]
        if direction == "long":
            raw_fill = entry_bar["Ask_Open"]
            fill_price = _slip(raw_fill, "long", SLIPPAGE_ATR_FRACTION * atr_now)
            stop_distance = fill_price - stop_price
        else:
            raw_fill = entry_bar["Bid_Open"]
            fill_price = _slip(raw_fill, "short", SLIPPAGE_ATR_FRACTION * atr_now)
            stop_distance = stop_price - fill_price
        if stop_distance <= 0:
            continue
        target_distance = stop_distance * TARGET_RR
        target_price = fill_price + target_distance if direction == "long" else fill_price - target_distance

        exit_price, exit_reason = None, None
        hold_bars = df.iloc[i + 1:i + 1 + 40]  # generous max hold, ~6.7 days at 4H
        for _, hb in hold_bars.iterrows():
            if direction == "long":
                stop_hit = hb["Bid_Low"] <= stop_price
                tp_hit = hb["Bid_High"] >= target_price
            else:
                stop_hit = hb["Ask_High"] >= stop_price
                tp_hit = hb["Ask_Low"] <= target_price
            if stop_hit or tp_hit:
                if stop_hit:
                    exit_price, exit_reason = stop_price, "stop_loss"
                else:
                    exit_price, exit_reason = target_price, "take_profit"
                break
        if exit_price is None:
            continue  # forced/unresolved - excluded from stats, same convention as every other strategy here

        r_multiple = ((exit_price - fill_price) if direction == "long" else (fill_price - exit_price)) / stop_distance
        pnl_proxy = r_multiple  # R-multiple used directly as the P&L unit (no position sizing needed for this check)
        all_trades.append({
            "symbol": symbol, "direction": direction, "entry_time": entry_bar.name,
            "exit_reason": exit_reason, "r_multiple": r_multiple, "pnl": pnl_proxy,
        })

trades_df = pd.DataFrame(all_trades)
print(f"Total mechanical trades generated: {len(trades_df)}")

if len(trades_df) < 20:
    print("Too few trades for meaningful PF evaluation - REJECTED on sample size.")
    sys.exit(0)

trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
split_idx = int(len(trades_df) * TRAIN_FRACTION)
train_trades = trades_df.iloc[:split_idx]
test_trades = trades_df.iloc[split_idx:]


def perf(g):
    n = len(g)
    if n == 0:
        return {"n": 0, "win_rate_pct": float("nan"), "pf": float("nan"), "net_r": float("nan")}
    wins = g[g["pnl"] > 0]
    losses = g[g["pnl"] <= 0]
    gp, gl = wins["pnl"].sum(), -losses["pnl"].sum()
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else float("nan"))
    return {"n": n, "win_rate_pct": round(len(wins) / n * 100, 1), "pf": pf, "net_r": round(g["pnl"].sum(), 2)}


train_perf = perf(train_trades)
test_perf = perf(test_trades)
print(f"\nTRAIN: {train_perf}")
print(f"TEST:  {test_perf}")

pf_ratio = (test_perf["pf"] / train_perf["pf"]) if train_perf["pf"] not in (0, float("nan"), float("inf")) else float("nan")

print(f"\n{'='*78}\nEVALUATION AGAINST THE NEW STANDING CRITERIA\n{'='*78}")
trades_pass = len(trades_df) >= MIN_REQUIRED_TRADES
train_pf_pass = train_perf["pf"] > MIN_REQUIRED_PROFIT_FACTOR
test_pf_pass = test_perf["pf"] > MIN_REQUIRED_PROFIT_FACTOR
is_oos_pass = (train_pf_pass and test_pf_pass and not pd.isna(pf_ratio)
               and pf_ratio >= (1 - IS_OOS_MAX_RELATIVE_PF_DROP))

print(f"[{'PASS' if trades_pass else 'FAIL'}] >= {MIN_REQUIRED_TRADES} trades total ({len(trades_df)} observed)")
print(f"[{'PASS' if train_pf_pass else 'FAIL'}] Train PF > {MIN_REQUIRED_PROFIT_FACTOR} ({train_perf['pf']:.3f})")
print(f"[{'PASS' if test_pf_pass else 'FAIL'}] Test PF > {MIN_REQUIRED_PROFIT_FACTOR} ({test_perf['pf']:.3f})")
print(f"[{'PASS' if is_oos_pass else 'FAIL'}] IS/OOS similarity (ratio={pf_ratio:.3f}, "
      f"need >= {1-IS_OOS_MAX_RELATIVE_PF_DROP:.2f})")

# concentration + multi-regime, combined sample
gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
max_trade = trades_df["pnl"].max()
trade_share = max_trade / gross_profit if gross_profit > 0 else float("nan")
no_trade_dominance_pass = not pd.isna(trade_share) and trade_share <= MAX_SINGLE_TRADE_PROFIT_SHARE
print(f"[{'PASS' if no_trade_dominance_pass else 'FAIL'}] No single trade > "
      f"{MAX_SINGLE_TRADE_PROFIT_SHARE*100:.0f}% of gross profit "
      f"({'n/a' if pd.isna(trade_share) else f'{trade_share*100:.1f}%'})")

trades_df["month"] = pd.to_datetime(trades_df["entry_time"]).dt.tz_localize(None).dt.to_period("M")
monthly_pnl = trades_df.groupby("month")["pnl"].sum()
total_net = monthly_pnl.sum()
max_month = monthly_pnl.max() if len(monthly_pnl) else 0.0
month_share = max_month / total_net if total_net > 0 else float("nan")
no_month_dominance_pass = not pd.isna(month_share) and 0 <= month_share <= MAX_SINGLE_MONTH_PROFIT_SHARE
print(f"[{'PASS' if no_month_dominance_pass else 'FAIL'}] No single month > "
      f"{MAX_SINGLE_MONTH_PROFIT_SHARE*100:.0f}% of net profit "
      f"({'n/a' if pd.isna(month_share) else f'{month_share*100:.1f}%'})")

trades_df["year"] = pd.to_datetime(trades_df["entry_time"]).dt.year
yearly_pnl = trades_df.groupby("year")["pnl"].sum()
n_years = len(yearly_pnl)
n_pos_years = (yearly_pnl > 0).sum()
pos_year_frac = n_pos_years / n_years if n_years else 0.0
multi_regime_pass = n_years >= 2 and pos_year_frac >= MIN_POSITIVE_REGIME_YEARS_FRACTION
print(f"[{'PASS' if multi_regime_pass else 'FAIL'}] >= {MIN_POSITIVE_REGIME_YEARS_FRACTION*100:.0f}% of years "
      f"profitable ({n_pos_years}/{n_years})")

overall_pass = all([trades_pass, train_pf_pass, test_pf_pass, is_oos_pass,
                     no_trade_dominance_pass, no_month_dominance_pass, multi_regime_pass])
print(f"\n{'='*78}\nOVERALL: {'SURVIVES the cheap feasibility check' if overall_pass else 'REJECTED'}\n{'='*78}")

print("\nDONE. No signals module/entry point/tests built yet - this is a")
print("standalone feasibility script only. No validation/reserved data")
print("accessed, no live-connector changes.")
