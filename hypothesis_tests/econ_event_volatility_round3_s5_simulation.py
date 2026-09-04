"""
hypothesis_tests/econ_event_volatility_round3_s5_simulation.py
--------------------------------------------------------------------
H4 round 3: genuine finest-available-resolution (S5, 5-second)
breakout-trigger simulation, replacing round 2's M15-based optimistic/
conservative bracketing now that real bid/ask dynamics at the moment
of the break can actually be observed.

RESULT: REJECTED, decisively and cleanly - resolves round 2's "data
resolution insufficient to determine tradeability" finding into a
confident answer. DEVELOPMENT: n=221, expectancy -0.213R (p=0.0012),
PF=0.649. VALIDATION (genuine out-of-sample, checked once): n=53,
expectancy -0.585R (p<0.0001), PF=0.262 - MORE unprofitable
out-of-sample, not less. The round-2 "optimistic" scenario (+0.352R)
is confirmed to have been an artifact of underestimated execution
cost: with real S5 bid/ask and realistic slippage, the edge doesn't
just vanish, it flips solidly negative and stays negative OOS. See
results/hypothesis4_econ_event_volatility_summary.json ("round_3") and
README.md's "Hypothesis 4" section for the full record. Preserved
unmodified, not imported by anything in the project. FINAL_RESERVED
was not accessed.

DESIGN (unchanged from round 2 where round 2 already got it right; only
the DATA RESOLUTION and cost-realism improve):
  - Dual-sided OCO trigger: long = pre-event M15 range High + 0.1xATR;
    short = pre-event M15 range Low - 0.1xATR (buffer/ATR source
    UNCHANGED - still M15, since establishing the structural range
    doesn't need finer resolution; only post-event EXECUTION does).
  - Walk forward S5 bars (T -> T+2h), no hindsight on direction. First
    bar whose mid High/Low crosses a trigger fires that side. Both
    sides in the same S5 bar (5 seconds) -> excluded, pre-committed
    rule, expected to be far rarer than round 2's 26.4% M15 rate.
  - Fill = realistic Ask_High (long)/Bid_Low (short) of the triggering
    S5 bar, worsened by SLIPPAGE_ATR_FRACTION(0.02)xATR via the EXACT
    _slip() convention already used in backtest_engine.py - reused,
    not reinvented. No more optimistic/conservative bracketing - S5
    lets us see the real spread directly.
  - Stop: opposite side of the pre-event M15 range, same 0.1xATR
    buffer style (V1/V2 precedent, unchanged). Target: fixed 1:1 R:R
    (unchanged).
  - Stop/target monitoring: S5 bars (Bid_Low/Bid_High long,
    Ask_High/Ask_Low short, same-bar-both -> stop wins, same engine
    convention) through T+2h; if still open, falls back to the
    EXISTING cached M15 data (unchanged infrastructure) from T+2h to
    24h max hold - the spread-resolution check showed risk concentrates
    in the first minutes, not hours later.
  - Nothing swept: every parameter above is identical to round 2's
    frozen design, none tuned against these results.

Run on DEVELOPMENT (with year-by-year walk-forward reporting) and,
ONCE, on VALIDATION (genuine out-of-sample - the first time ANY
strategy in this project has looked at this period). FINAL_RESERVED is
NOT accessed.
"""
import sys, os
sys.path.insert(0, "/Users/user/Projects/Mistry")
import math
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration, DEVELOPMENT_END, VALIDATION_END
from indicators import atr as atr_indicator

INSTRUMENTS_TESTED = ["EUR_USD", "GBP_USD"]
PRE_MINUTES = 60
TRIGGER_WINDOW_MINUTES = 120
ENTRY_BUFFER_ATR_FRACTION = 0.1
STOP_BUFFER_ATR_FRACTION = 0.1
TARGET_RR = 1.0
SLIPPAGE_ATR_FRACTION = 0.02   # reused verbatim from backtest_engine.py
MAX_HOLD_HOURS = 24
NY_TZ = ZoneInfo("America/New_York")
S5_CACHE_DIR = "/Users/user/Projects/Mistry/hypothesis_tests/data/s5_cache"


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


def load_events(path):
    events = pd.read_csv(path)
    events["event_time_utc"] = events.apply(
        lambda r: pd.Timestamp(f"{r['date']} {r['local_time']}").tz_localize(NY_TZ).tz_convert("UTC"), axis=1)
    return events


def load_s5_window(symbol, date, event_type):
    path = os.path.join(S5_CACHE_DIR, f"{symbol}_{date}_{event_type}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Timestamp", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def simulate(events, m15_frames, m15_atr, symbol):
    m15 = m15_frames[symbol]
    m15_idx = m15.index
    atr_series = m15_atr[symbol]
    rows = []
    for _, ev in events.iterrows():
        t0 = ev["event_time_utc"]
        s5 = load_s5_window(symbol, ev["date"], ev["event_type"])
        if s5 is None or len(s5) == 0:
            continue

        pre_mask = (m15_idx >= t0 - pd.Timedelta(minutes=PRE_MINUTES)) & (m15_idx < t0)
        pre_g = m15.loc[pre_mask]
        if len(pre_g) == 0:
            continue
        pre_high, pre_low = pre_g["High"].max(), pre_g["Low"].min()

        prior_bars = m15_idx[m15_idx <= t0]
        if len(prior_bars) == 0:
            continue
        atr_val = atr_series.get(prior_bars[-1], np.nan)
        if pd.isna(atr_val) or atr_val == 0:
            continue

        long_trigger = pre_high + ENTRY_BUFFER_ATR_FRACTION * atr_val
        short_trigger = pre_low - ENTRY_BUFFER_ATR_FRACTION * atr_val

        window = s5[(s5.index >= t0) & (s5.index < t0 + pd.Timedelta(minutes=TRIGGER_WINDOW_MINUTES))]
        if len(window) == 0:
            continue

        outcome = {"event_time": t0, "event_type": ev["event_type"], "year": t0.year, "symbol": symbol}
        direction, fill_price, stop_price, target_price, trigger_ts = None, None, None, None, None
        for ts, bar in window.iterrows():
            long_hit = bar["High"] >= long_trigger
            short_hit = bar["Low"] <= short_trigger
            if long_hit and short_hit:
                outcome["status"] = "ambiguous_same_bar"
                break
            if long_hit or short_hit:
                direction = "long" if long_hit else "short"
                if direction == "long":
                    raw_fill = bar["Ask_High"]
                    fill_price = _slip(raw_fill, "long", SLIPPAGE_ATR_FRACTION * atr_val)
                    stop_price = pre_low - STOP_BUFFER_ATR_FRACTION * atr_val
                    stop_distance = fill_price - stop_price
                else:
                    raw_fill = bar["Bid_Low"]
                    fill_price = _slip(raw_fill, "short", SLIPPAGE_ATR_FRACTION * atr_val)
                    stop_price = pre_high + STOP_BUFFER_ATR_FRACTION * atr_val
                    stop_distance = stop_price - fill_price
                if stop_distance <= 0:
                    outcome["status"] = "invalid_stop_distance"
                    break
                target_distance = stop_distance * TARGET_RR
                target_price = fill_price + target_distance if direction == "long" else fill_price - target_distance
                trigger_ts = ts
                outcome["status"] = "triggered"
                outcome["trigger_lag_seconds"] = (ts - t0).total_seconds()
                break
        if outcome.get("status") is None:
            outcome["status"] = "no_trigger"
            rows.append(outcome)
            continue
        if outcome["status"] != "triggered":
            rows.append(outcome)
            continue

        # --- walk forward for exit: S5 through window end, then M15 fallback ---
        exit_price, exit_reason = None, None
        s5_rest = s5[(s5.index >= trigger_ts) & (s5.index < t0 + pd.Timedelta(minutes=TRIGGER_WINDOW_MINUTES))]
        for ts, bar in s5_rest.iterrows():
            if direction == "long":
                stop_hit = bar["Bid_Low"] <= stop_price
                tp_hit = bar["Bid_High"] >= target_price
            else:
                stop_hit = bar["Ask_High"] >= stop_price
                tp_hit = bar["Ask_Low"] <= target_price
            if stop_hit or tp_hit:
                if stop_hit:
                    exit_price, exit_reason = stop_price, "stop_loss"
                else:
                    exit_price, exit_reason = target_price, "take_profit"
                break

        if exit_price is None:
            # fall back to existing cached M15 data from window end to 24h max hold
            hold_end = trigger_ts + pd.Timedelta(hours=MAX_HOLD_HOURS)
            m15_rest = m15[(m15_idx >= t0 + pd.Timedelta(minutes=TRIGGER_WINDOW_MINUTES)) & (m15_idx < hold_end)]
            for ts, bar in m15_rest.iterrows():
                if direction == "long":
                    stop_hit = bar["Bid_Low"] <= stop_price
                    tp_hit = bar["Bid_High"] >= target_price
                else:
                    stop_hit = bar["Ask_High"] >= stop_price
                    tp_hit = bar["Ask_Low"] <= target_price
                if stop_hit or tp_hit:
                    if stop_hit:
                        exit_price, exit_reason = stop_price, "stop_loss"
                    else:
                        exit_price, exit_reason = target_price, "take_profit"
                    break
            if exit_price is None:
                exit_price = m15_rest["Close"].iloc[-1] if len(m15_rest) else fill_price
                exit_reason = "forced_exit_24h"

        r_multiple = ((exit_price - fill_price) if direction == "long" else (fill_price - exit_price)) / stop_distance
        outcome.update({"direction": direction, "fill_price": fill_price, "exit_price": exit_price,
                         "exit_reason": exit_reason, "r_multiple": r_multiple})
        rows.append(outcome)
    return pd.DataFrame(rows)


def max_drawdown_r(r_series_in_time_order):
    cum = np.cumsum(r_series_in_time_order)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return dd.max() if len(dd) else 0.0


def report(label, g):
    print(f"\n{'-'*78}\n{label}\n{'-'*78}")
    n = len(g)
    if n == 0:
        print("  0 trades.")
        return
    wins = g[g["r_multiple"] > 0]
    losses = g[g["r_multiple"] <= 0]
    win_rate = len(wins) / n * 100
    gross_win = wins["r_multiple"].sum()
    gross_loss = -losses["r_multiple"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else float("nan"))
    m, t_stat, p, n_stat = one_sample_t(g["r_multiple"].values)
    g_sorted = g.sort_values("event_time")
    mdd = max_drawdown_r(g_sorted["r_multiple"].values)
    print(f"  n={n}  win_rate={win_rate:.1f}%  expectancy(mean R)={m:+.3f} (t={t_stat:.2f}, p={p:.4f})  "
          f"PF={pf:.3f}  max_drawdown={mdd:.2f}R  gross_win={gross_win:.2f}R  gross_loss={gross_loss:.2f}R")


print("=" * 78)
print("H4 ROUND 3: S5 (5-second) resolution breakout-trigger simulation")
print("=" * 78)

m15_frames_dev, m15_atr_dev = {}, {}
m15_frames_val, m15_atr_val = {}, {}
for symbol in INSTRUMENTS_TESTED:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "M15")
    dev, validation = split_for_iteration({symbol: raw})
    dev, validation = dev[symbol].sort_index(), validation[symbol].sort_index()
    # validation window per dataset_split.py: [DEVELOPMENT_END, VALIDATION_END)
    validation = validation[(validation.index >= DEVELOPMENT_END) & (validation.index < VALIDATION_END)]
    m15_frames_dev[symbol] = dev
    m15_frames_val[symbol] = validation
    m15_atr_dev[symbol] = atr_indicator(dev["High"], dev["Low"], dev["Close"], period=14)
    m15_atr_val[symbol] = atr_indicator(validation["High"], validation["Low"], validation["Close"], period=14)
    print(f"{symbol}: DEVELOPMENT {len(dev)} M15 bars, VALIDATION {len(validation)} M15 bars")

dev_events = load_events("/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_development.csv")
val_events = load_events("/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_validation.csv")
print(f"\nDEVELOPMENT events: {len(dev_events)}   VALIDATION events: {len(val_events)}")

print("\n" + "=" * 78)
print("PART A: DEVELOPMENT (train) - same role as round 2, now at S5 resolution")
print("=" * 78)
dev_all = []
for symbol in INSTRUMENTS_TESTED:
    out = simulate(dev_events, m15_frames_dev, m15_atr_dev, symbol)
    dev_all.append(out)
    print(f"{symbol}: {out['status'].value_counts().to_dict()}")
dev_combined = pd.concat(dev_all, ignore_index=True)
dev_traded = dev_combined[dev_combined["status"] == "triggered"]
dev_non_forced = dev_traded[dev_traded["exit_reason"] != "forced_exit_24h"]

print(f"\nDisposition (combined): {dev_combined['status'].value_counts().to_dict()}")
print(f"Triggered: {len(dev_traded)}  Non-forced (primary stats): {len(dev_non_forced)}")
if len(dev_traded):
    print(f"Median trigger lag from event time: {dev_traded['trigger_lag_seconds'].median():.1f}s  "
          f"(mean {dev_traded['trigger_lag_seconds'].mean():.1f}s)")

report("DEVELOPMENT COMBINED", dev_non_forced)
for symbol in INSTRUMENTS_TESTED:
    report(f"DEVELOPMENT {symbol}", dev_non_forced[dev_non_forced["symbol"] == symbol])
for etype, g in dev_non_forced.groupby("event_type"):
    report(f"DEVELOPMENT by event type: {etype}", g)

print(f"\n{'='*78}\nWALK-FORWARD: DEVELOPMENT by year\n{'='*78}")
for yr, g in dev_non_forced.groupby("year"):
    report(f"  {yr}", g)

print("\n" + "=" * 78)
print("PART B: VALIDATION (genuine out-of-sample, checked ONCE)")
print("=" * 78)
print("*** ACCESSING VALIDATION DATA - first time any strategy in this project has ***")
print("*** looked at this period. Rules are 100% frozen from DEVELOPMENT - nothing  ***")
print("*** below will be tuned regardless of outcome.                              ***")
val_all = []
for symbol in INSTRUMENTS_TESTED:
    out = simulate(val_events, m15_frames_val, m15_atr_val, symbol)
    val_all.append(out)
    print(f"{symbol}: {out['status'].value_counts().to_dict()}")
val_combined = pd.concat(val_all, ignore_index=True)
val_traded = val_combined[val_combined["status"] == "triggered"]
val_non_forced = val_traded[val_traded["exit_reason"] != "forced_exit_24h"]

print(f"\nDisposition (combined): {val_combined['status'].value_counts().to_dict()}")
report("VALIDATION COMBINED", val_non_forced)
for symbol in INSTRUMENTS_TESTED:
    report(f"VALIDATION {symbol}", val_non_forced[val_non_forced["symbol"] == symbol])
for etype, g in val_non_forced.groupby("event_type"):
    report(f"VALIDATION by event type: {etype}", g)

print(f"\n{'='*78}\nBY SESSION (UTC hour of event - all events cluster at 2 fixed times)\n{'='*78}")
for combo_label, g in [("DEVELOPMENT", dev_non_forced), ("VALIDATION", val_non_forced)]:
    g2 = g.copy()
    g2["utc_hour"] = g2["event_time"].apply(lambda t: t.hour)
    print(f"\n  -- {combo_label} --")
    for hr, gg in g2.groupby("utc_hour"):
        session_label = "London/NY overlap (NFP/CPI)" if hr in (12, 13) else "NY afternoon, London closed (FOMC)"
        report(f"    UTC hour {hr} - {session_label}", gg)

print(f"\n{'='*78}\nSAMPLE SIZE\n{'='*78}")
print(f"  DEVELOPMENT non-forced trades: {len(dev_non_forced)} (MIN_REQUIRED_TRADES convention: 150)")
print(f"  VALIDATION non-forced trades: {len(val_non_forced)}")

print(f"\n{'='*78}\nDONE. backtest_engine.py, risk_management.py, live connector untouched.\n"
      f"FINAL_RESERVED not accessed.\n{'='*78}")
