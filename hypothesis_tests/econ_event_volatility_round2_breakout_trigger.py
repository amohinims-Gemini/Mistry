"""
hypothesis_tests/econ_event_volatility_round2_breakout_trigger.py
--------------------------------------------------------------------
RESULT: REJECTED, but not a clean rejection - see
results/hypothesis4_econ_event_volatility_summary.json and README.md's
"Hypothesis 4" section for the full honest framing (the raw/optimistic
result is the strongest in the whole search, but collapses to
statistically indistinguishable from zero and becomes unstable across
years/event types under a modest, formulaic, non-tuned conservative
execution-cost adjustment - meaning this project's data resolution
cannot determine tradeability, not that no edge exists). Preserved
unmodified, purely so this exact experiment never needs re-running -
not imported by anything in the project.

Genuine non-lookahead breakout-trigger simulation for the Economic
Calendar Event Volatility hypothesis - round 2 (round 1 was the pure
statistical volatility-expansion check, econ_event_volatility_round1_statistical.py).

Standalone simulation, NOT routed through backtest_engine.run_backtest()
- that engine fills a signal one full bar after detection (signal at
bar T's close -> fill at bar T+1's Open), which doesn't match "enter
the instant the range is broken". backtest_engine.py and
risk_management.py are both untouched by this file.

FROZEN DESIGN (approved before running):
  - Dual-sided OCO trigger: long_trigger = pre-event range High + 0.1xATR;
    short_trigger = pre-event range Low - 0.1xATR. Buffer reused from
    signals_london_sweep_m15.py's stop_buffer_atr_fraction, not new.
  - Walk forward mechanically from the event bar through the same 2h
    post-event window used in round 1. First bar whose High/Low crosses
    a trigger fires that side. Both sides crossed in the SAME bar ->
    excluded (ambiguous, no tick data to resolve order - conservative,
    not guessed).
  - Execution cost, TWO scenarios, both reported (M15 cannot reliably
    capture true cost at the instant of the break - see round 2 design
    proposal's spread-open check):
      optimistic:   fill = trigger +/- that bar's own (Ask_Close-Bid_Close)
      conservative: optimistic +/- 0.25 x that bar's own (Ask_High-Bid_Low),
                     applied ONLY when the trigger fires on the very
                     first bar after the event (the only window the
                     spread-open check showed elevated risk for).
  - Stop: opposite side of the pre-event range, same 0.1xATR buffer
    style (reusing V1/V2's structural-stop PATTERN, not a new idea).
  - Target: fixed 1:1 R:R - the same ratio already used in V1/V2, not
    picked for this test.
  - Stop/target touch detection: same convention backtest_engine.py
    already uses (Bid_Low/Bid_High for longs, Ask_High/Ask_Low for
    shorts; same-bar-both-hit -> stop wins, its own documented
    conservative assumption, reused unchanged).
  - Max hold: 24h from trigger fill; unresolved -> forced exit,
    excluded from win/loss stats (same treatment the engine gives
    period-end forced closes).
  - Outcome measure: R-multiple, not dollar P&L - risk_management.py's
    position-sizing math is never invoked.
  - Nothing swept: buffer, R:R, max hold, conservative-overlay fraction
    all fixed in the approved design, none tuned against these results.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration, describe_split
from indicators import atr as atr_indicator

INSTRUMENTS_TESTED = ["EUR_USD", "GBP_USD"]
PRE_MINUTES = 60
TRIGGER_WINDOW_MINUTES = 120     # reused from round 1's post-event window
ENTRY_BUFFER_ATR_FRACTION = 0.1  # reused from signals_london_sweep_m15.py
STOP_BUFFER_ATR_FRACTION = 0.1   # same style
TARGET_RR = 1.0                  # same as V1/V2
CONSERVATIVE_OVERLAY_FRACTION = 0.25
MAX_HOLD_HOURS = 24
NY_TZ = ZoneInfo("America/New_York")


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def one_sample_t(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(x)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    t_stat = m / se if se > 0 else 0.0
    return m, se, t_stat, two_sided_p_from_z(t_stat), len(x)


print("=" * 78)
print("HYPOTHESIS 4 ROUND 2: Economic Event Breakout-Trigger Simulation")
print("Non-lookahead, mechanical, dual-sided OCO trigger - EUR_USD & GBP_USD, M15")
print("DEVELOPMENT DATA ONLY")
print("=" * 78)
print(describe_split())
print(f"Entry buffer={ENTRY_BUFFER_ATR_FRACTION}xATR, stop buffer={STOP_BUFFER_ATR_FRACTION}xATR, "
      f"R:R={TARGET_RR}:1, trigger window={TRIGGER_WINDOW_MINUTES}min, max hold={MAX_HOLD_HOURS}h, "
      f"conservative overlay={CONSERVATIVE_OVERLAY_FRACTION}x(first-bar Ask_High-Bid_Low)")
print()

events_raw = pd.read_csv("/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_development.csv")
events_raw["event_time_utc"] = events_raw.apply(
    lambda r: pd.Timestamp(f"{r['date']} {r['local_time']}").tz_localize(NY_TZ).tz_convert("UTC"), axis=1)
print(f"Event table: {len(events_raw)} events (31 FOMC, 47 NFP, 47 CPI)")
print()


def simulate_instrument(symbol):
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "M15")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    idx = df.index
    atr_series = atr_indicator(df["High"], df["Low"], df["Close"], period=14)
    pip = INSTRUMENTS[symbol].pip_size

    rows = []
    for _, ev in events_raw.iterrows():
        t0 = ev["event_time_utc"]
        pre_mask = (idx >= t0 - pd.Timedelta(minutes=PRE_MINUTES)) & (idx < t0)
        pre_g = df.loc[pre_mask]
        if len(pre_g) == 0:
            continue
        pre_high, pre_low = pre_g["High"].max(), pre_g["Low"].min()

        prior_bars = idx[idx <= t0]
        if len(prior_bars) == 0:
            continue
        atr_val = atr_series.get(prior_bars[-1], np.nan)
        if pd.isna(atr_val) or atr_val == 0:
            continue

        long_trigger = pre_high + ENTRY_BUFFER_ATR_FRACTION * atr_val
        short_trigger = pre_low - ENTRY_BUFFER_ATR_FRACTION * atr_val

        window_mask = (idx >= t0) & (idx < t0 + pd.Timedelta(minutes=TRIGGER_WINDOW_MINUTES))
        window_bars = idx[window_mask]
        if len(window_bars) == 0:
            continue

        outcome = {"event_time": t0, "event_type": ev["event_type"], "year": t0.year, "symbol": symbol}
        triggered = False
        for bar_i, bts in enumerate(window_bars):
            bar = df.loc[bts]
            long_hit = bar["High"] >= long_trigger
            short_hit = bar["Low"] <= short_trigger
            if long_hit and short_hit:
                outcome["status"] = "ambiguous_same_bar"
                triggered = True
                break
            if long_hit or short_hit:
                direction = "long" if long_hit else "short"
                is_first_bar = (bar_i == 0)
                bar_spread = bar["Ask_Close"] - bar["Bid_Close"]
                bar_range_proxy = bar["Ask_High"] - bar["Bid_Low"]

                if direction == "long":
                    fill_opt = long_trigger + bar_spread
                    fill_cons = fill_opt + (CONSERVATIVE_OVERLAY_FRACTION * bar_range_proxy if is_first_bar else 0.0)
                    stop_price = pre_low - STOP_BUFFER_ATR_FRACTION * atr_val
                else:
                    fill_opt = short_trigger - bar_spread
                    fill_cons = fill_opt - (CONSERVATIVE_OVERLAY_FRACTION * bar_range_proxy if is_first_bar else 0.0)
                    stop_price = pre_high + STOP_BUFFER_ATR_FRACTION * atr_val

                for scenario, fill_price in [("optimistic", fill_opt), ("conservative", fill_cons)]:
                    if direction == "long":
                        stop_distance = fill_price - stop_price
                    else:
                        stop_distance = stop_price - fill_price
                    if stop_distance <= 0:
                        outcome[f"status_{scenario}"] = "invalid_stop_distance"
                        continue
                    target_distance = stop_distance * TARGET_RR
                    if direction == "long":
                        target_price = fill_price + target_distance
                    else:
                        target_price = fill_price - target_distance

                    exit_price, exit_reason = None, None
                    hold_end = bts + pd.Timedelta(hours=MAX_HOLD_HOURS)
                    hold_bars = idx[(idx >= bts) & (idx < hold_end)]
                    for hb_ts in hold_bars:
                        hb = df.loc[hb_ts]
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
                        exit_price = df.loc[hold_bars[-1]]["Close"] if len(hold_bars) else fill_price
                        exit_reason = "forced_exit_24h"

                    if direction == "long":
                        r_multiple = (exit_price - fill_price) / stop_distance
                    else:
                        r_multiple = (fill_price - exit_price) / stop_distance

                    outcome[f"direction"] = direction
                    outcome[f"trigger_bar_index"] = bar_i
                    outcome[f"fill_{scenario}"] = fill_price
                    outcome[f"exit_reason_{scenario}"] = exit_reason
                    outcome[f"r_multiple_{scenario}"] = r_multiple
                outcome["status"] = "triggered"
                triggered = True
                break
        if not triggered:
            outcome["status"] = "no_trigger"
        rows.append(outcome)
    return pd.DataFrame(rows)


all_rows = {}
for symbol in INSTRUMENTS_TESTED:
    df_out = simulate_instrument(symbol)
    all_rows[symbol] = df_out
    counts = df_out["status"].value_counts().to_dict()
    print(f"{symbol}: {len(df_out)} events processed -> {counts}")

combined = pd.concat(all_rows.values(), ignore_index=True)
print(f"\nCombined status counts: {combined['status'].value_counts().to_dict()}")
print()

traded = combined[combined["status"] == "triggered"].copy()
traded_valid = traded.dropna(subset=["r_multiple_optimistic", "r_multiple_conservative"])
print(f"Valid triggered trades (both scenarios computed): {len(traded_valid)}")
print(f"  by trigger_bar_index: {traded_valid['trigger_bar_index'].value_counts().sort_index().to_dict()}")
print(f"  first-bar triggers (conservative overlay applies): "
      f"{(traded_valid['trigger_bar_index']==0).sum()}")
print(f"  by exit reason (optimistic): {traded_valid['exit_reason_optimistic'].value_counts().to_dict()}")
print(f"  by direction: {traded_valid['direction'].value_counts().to_dict()}")
print()

non_forced = traded_valid[traded_valid["exit_reason_optimistic"] != "forced_exit_24h"]
print(f"Non-forced-exit trades (used for primary stats): {len(non_forced)}")
print()


def report(label, g):
    print(f"\n{'-'*78}\n{label}  (n={len(g)})\n{'-'*78}")
    if len(g) < 5:
        print("  Too few trades for meaningful stats.")
        return
    for scenario in ["optimistic", "conservative"]:
        col = f"r_multiple_{scenario}"
        m, se, t_stat, p, n = one_sample_t(g[col].values)
        win_rate = (g[col] > 0).mean() * 100
        print(f"  {scenario:12s}: n={n:3d}  mean R={m:+.3f}  (t={t_stat:.2f}, p={p:.4f})  win rate={win_rate:.1f}%")


report("COMBINED (EUR_USD + GBP_USD), non-forced exits", non_forced)
for symbol in INSTRUMENTS_TESTED:
    report(f"{symbol}, non-forced exits", non_forced[non_forced["symbol"] == symbol])

print(f"\n{'='*78}\nBY EVENT TYPE (combined, non-forced exits)\n{'='*78}")
for etype, g in non_forced.groupby("event_type"):
    report(f"{etype}", g)

print(f"\n{'='*78}\nSTABILITY BY YEAR (combined, non-forced exits)\n{'='*78}")
for yr, g in non_forced.groupby("year"):
    if len(g) < 5:
        print(f"  {yr}: n={len(g)} (too few - skipping)")
        continue
    m_o, se_o, t_o, p_o, n_o = one_sample_t(g["r_multiple_optimistic"].values)
    m_c, se_c, t_c, p_c, n_c = one_sample_t(g["r_multiple_conservative"].values)
    print(f"  {yr}: n={n_o:3d}  optimistic mean R={m_o:+.3f} (p={p_o:.4f})   "
          f"conservative mean R={m_c:+.3f} (p={p_c:.4f})")

print(f"\n{'='*78}\nSAMPLE SIZE / DISPOSITION SUMMARY\n{'='*78}")
n_events_total = len(combined)
n_no_trigger = (combined["status"] == "no_trigger").sum()
n_ambiguous = (combined["status"] == "ambiguous_same_bar").sum()
n_triggered = (combined["status"] == "triggered").sum()
n_forced = (traded_valid["exit_reason_optimistic"] == "forced_exit_24h").sum()
print(f"  Total event-instrument pairs: {n_events_total}")
print(f"  No trigger within {TRIGGER_WINDOW_MINUTES}min: {n_no_trigger} ({n_no_trigger/n_events_total*100:.1f}%)")
print(f"  Ambiguous (both sides same bar), excluded: {n_ambiguous} ({n_ambiguous/n_events_total*100:.1f}%)")
print(f"  Triggered (single-sided): {n_triggered} ({n_triggered/n_events_total*100:.1f}%)")
print(f"  Of triggered: forced exit at {MAX_HOLD_HOURS}h (excluded from primary stats): {n_forced}")
print(f"  Used for primary stats (non-forced): {len(non_forced)} "
      f"(project MIN_REQUIRED_TRADES convention: 150)")

print(f"\n{'='*78}\nDONE. Standalone simulation only - backtest_engine.py, risk_management.py,\n"
      f"live connector untouched. No validation/reserved data accessed.\n{'='*78}")
