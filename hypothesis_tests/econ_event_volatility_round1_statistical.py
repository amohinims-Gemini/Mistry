"""
hypothesis_tests/econ_event_volatility_round1_statistical.py
--------------------------------------------------------------
Round 1 of the Economic Calendar Event Volatility hypothesis: pure
statistical falsification test (volatility-expansion vs. control,
breakout incidence, sample cost-adjusted proxy). Round 2
(econ_event_volatility_round2_breakout_trigger.py) is the genuine
non-lookahead breakout-trigger simulation that actually decided
tradeability.

RESULT: the core volatility-expansion finding here is real and the
strongest raw signal in the whole V3/V4 search - but round 2 and the
spread-resolution check (econ_event_volatility_spread_resolution_check.py)
found the data resolution available to this project cannot determine
whether that expansion is tradeable. See
results/hypothesis4_econ_event_volatility_summary.json for the full
numeric record (all three files) and README.md's "Hypothesis 4"
section for the narrative writeup. Preserved unmodified, purely so
this exact experiment never needs re-running - not imported by
anything in the project.

Cheap empirical falsification test for the Economic Calendar Event
Volatility hypothesis. Pure statistics on DEVELOPMENT data only - no
strategy code, no thresholds swept. EUR_USD and GBP_USD, M15.

Event table: hypothesis_tests/data/economic_events_development.csv,
hand-curated from federalreserve.gov (FOMC) and bls.gov (NFP/CPI)
official archive pages, DEVELOPMENT window only (2020-08-24 to
2024-07-15). 125 events: 31 FOMC, 47 NFP, 47 CPI.

FROZEN PROTOCOL (as proposed and approved):
  - Pre-event window: 1h before scheduled release (T-60min -> T).
  - Post-event windows: 30min (T -> T+30min) AND 2h (T -> T+120min),
    both reported.
  - Realized vol = sqrt(sum of squared M15 log returns) in a window.
  - Breakout = post-2h max High/min Low beyond pre-event [Low,High]
    range, by >= 0.5x ATR(14) - tradeable threshold, fixed, not swept.
  - Control: systematic (every 15th M15 bar) non-event anchor times,
    >=24h from any of the 125 events - same measurement.
  - No lookahead: window anchored strictly to the pre-announced
    official release time (America/New_York -> UTC via zoneinfo,
    DST-aware - the exact pattern already used for Europe/London in
    signals_london_sweep_m15.py), never to when price actually moved.
  - Direction-agnostic: breakout traded in whichever direction the
    range actually breaks (mechanical, not chosen with hindsight).
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
POST_MINUTES_SHORT = 30
POST_MINUTES_LONG = 120
BREAKOUT_ATR_MULTIPLE = 0.5     # fixed, not swept
CONTROL_STRIDE = 15             # systematic subsample, not random
CONTROL_MIN_HOURS_FROM_EVENT = 24
NY_TZ = ZoneInfo("America/New_York")


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def one_sample_t(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    t_stat = m / se if se > 0 else 0.0
    return m, se, t_stat, two_sided_p_from_z(t_stat)


def two_sample_t(x, y):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (x.mean() - y.mean()) / se if se > 0 else 0.0
    return t_stat, two_sided_p_from_z(t_stat)


def prop_test_2sample(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, two_sided_p_from_z(z)


print("=" * 78)
print("HYPOTHESIS 4: Economic Calendar Event Volatility - EUR_USD & GBP_USD, M15")
print("DEVELOPMENT DATA ONLY")
print("=" * 78)
print(describe_split())
print(f"Pre={PRE_MINUTES}min, Post_short={POST_MINUTES_SHORT}min, Post_long={POST_MINUTES_LONG}min, "
      f"breakout>={BREAKOUT_ATR_MULTIPLE}xATR(14)")
print()

# --- load event table, convert to UTC (DST-aware) ---
events_raw = pd.read_csv("/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_development.csv")


def to_utc(row):
    local_dt = pd.Timestamp(f"{row['date']} {row['local_time']}").tz_localize(NY_TZ)
    return local_dt.tz_convert("UTC")


events_raw["event_time_utc"] = events_raw.apply(to_utc, axis=1)
print(f"Event table loaded: {len(events_raw)} events "
      f"({(events_raw['event_type']=='FOMC').sum()} FOMC, "
      f"{(events_raw['event_type']=='NFP').sum()} NFP, "
      f"{(events_raw['event_type']=='CPI').sum()} CPI)")
print(f"Date range: {events_raw['event_time_utc'].min()} -> {events_raw['event_time_utc'].max()}")
print()

event_times = events_raw["event_time_utc"].sort_values().values


def is_far_from_any_event(ts, event_times_arr, min_hours):
    ts64 = np.datetime64(ts)
    diffs_hours = np.abs((event_times_arr - ts64).astype("timedelta64[m]").astype(float)) / 60.0
    return diffs_hours.min() >= min_hours


def measure_window(df, close_col, high_col, low_col, start_ts, end_ts):
    mask = (df.index >= start_ts) & (df.index < end_ts)
    g = df.loc[mask]
    if len(g) == 0:
        return None
    closes = g[close_col].values
    log_rets = np.diff(np.log(closes)) if len(closes) > 1 else np.array([])
    return {
        "n_bars": len(g),
        "high": g[high_col].max(),
        "low": g[low_col].min(),
        "close": g[close_col].iloc[-1],
        "open": g[close_col].iloc[0],  # first Close as an M15-consistent "open of window" proxy
        "realized_vol": math.sqrt(np.sum(log_rets ** 2)) if len(log_rets) else np.nan,
        "first_bar_bid_close": g["Bid_Close"].iloc[0], "first_bar_ask_close": g["Ask_Close"].iloc[0],
        "last_bar_bid_close": g["Bid_Close"].iloc[-1], "last_bar_ask_close": g["Ask_Close"].iloc[-1],
    }


def build_event_rows(df, atr_series, event_times_local):
    rows = []
    idx = df.index
    for _, ev in event_times_local.iterrows():
        t0 = ev["event_time_utc"]
        pre = measure_window(df, "Close", "High", "Low", t0 - pd.Timedelta(minutes=PRE_MINUTES), t0)
        post_s = measure_window(df, "Close", "High", "Low", t0, t0 + pd.Timedelta(minutes=POST_MINUTES_SHORT))
        post_l = measure_window(df, "Close", "High", "Low", t0, t0 + pd.Timedelta(minutes=POST_MINUTES_LONG))
        if pre is None or post_s is None or post_l is None:
            continue
        # ATR as of the last bar at/before t0 (no lookahead)
        prior_bars = idx[idx <= t0]
        if len(prior_bars) == 0 or pd.isna(atr_series.get(prior_bars[-1], np.nan)):
            continue
        atr_val = atr_series[prior_bars[-1]]
        if atr_val == 0 or pd.isna(atr_val):
            continue

        breakout_up = max(0.0, post_l["high"] - pre["high"])
        breakout_down = max(0.0, pre["low"] - post_l["low"])
        breakout_distance = max(breakout_up, breakout_down)
        breakout_atr = breakout_distance / atr_val
        tradeable = breakout_atr >= BREAKOUT_ATR_MULTIPLE
        direction = 1 if breakout_up >= breakout_down else -1  # mechanical, whichever side broke further

        rows.append({
            "event_time": t0, "event_type": ev["event_type"], "year": t0.year,
            "pre_vol": pre["realized_vol"], "post_vol_30m": post_s["realized_vol"],
            "post_vol_2h": post_l["realized_vol"],
            "vol_ratio_30m": post_s["realized_vol"] / pre["realized_vol"] if pre["realized_vol"] else np.nan,
            "vol_ratio_2h": post_l["realized_vol"] / pre["realized_vol"] if pre["realized_vol"] else np.nan,
            "atr": atr_val, "breakout_atr": breakout_atr, "tradeable": tradeable, "direction": direction,
            "pre_high": pre["high"], "pre_low": pre["low"],
            "post_close_2h": post_l["close"],
            "entry_bid": post_s["first_bar_bid_close"], "entry_ask": post_s["first_bar_ask_close"],
            "exit_bid": post_l["last_bar_bid_close"], "exit_ask": post_l["last_bar_ask_close"],
        })
    return pd.DataFrame(rows)


def build_control_rows(df, atr_series, event_times_arr):
    rows = []
    idx = df.index
    for i in range(0, len(idx), CONTROL_STRIDE):
        t0 = idx[i]
        if not is_far_from_any_event(t0, event_times_arr, CONTROL_MIN_HOURS_FROM_EVENT):
            continue
        pre = measure_window(df, "Close", "High", "Low", t0 - pd.Timedelta(minutes=PRE_MINUTES), t0)
        post_s = measure_window(df, "Close", "High", "Low", t0, t0 + pd.Timedelta(minutes=POST_MINUTES_SHORT))
        post_l = measure_window(df, "Close", "High", "Low", t0, t0 + pd.Timedelta(minutes=POST_MINUTES_LONG))
        if pre is None or post_s is None or post_l is None:
            continue
        atr_val = atr_series.get(t0, np.nan)
        if pd.isna(atr_val) or atr_val == 0:
            continue
        breakout_up = max(0.0, post_l["high"] - pre["high"])
        breakout_down = max(0.0, pre["low"] - post_l["low"])
        breakout_distance = max(breakout_up, breakout_down)
        breakout_atr = breakout_distance / atr_val
        rows.append({
            "event_time": t0, "year": t0.year,
            "pre_vol": pre["realized_vol"], "post_vol_30m": post_s["realized_vol"],
            "post_vol_2h": post_l["realized_vol"],
            "vol_ratio_30m": post_s["realized_vol"] / pre["realized_vol"] if pre["realized_vol"] else np.nan,
            "vol_ratio_2h": post_l["realized_vol"] / pre["realized_vol"] if pre["realized_vol"] else np.nan,
            "breakout_atr": breakout_atr, "tradeable": breakout_atr >= BREAKOUT_ATR_MULTIPLE,
        })
    return pd.DataFrame(rows)


event_dfs, control_dfs, raw_frames = {}, {}, {}
for symbol in INSTRUMENTS_TESTED:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "M15")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    raw_frames[symbol] = df
    atr_series = atr_indicator(df["High"], df["Low"], df["Close"], period=14)
    ev_df = build_event_rows(df, atr_series, events_raw)
    ev_df["symbol"] = symbol
    event_dfs[symbol] = ev_df
    ctrl_df = build_control_rows(df, atr_series, event_times)
    ctrl_df["symbol"] = symbol
    control_dfs[symbol] = ctrl_df
    print(f"{symbol}: {len(df)} M15 bars -> {len(ev_df)}/{len(events_raw)} events usable, "
          f"{len(ctrl_df)} control (ordinary-time) samples")

combined_ev = pd.concat(event_dfs.values(), ignore_index=True)
combined_ctrl = pd.concat(control_dfs.values(), ignore_index=True)
print(f"\nCombined: {len(combined_ev)} event observations, {len(combined_ctrl)} control observations")
print()


# --- SPREAD-WIDENING SANITY CHECK (does cached Bid/Ask actually widen around these events?) ---
print("=" * 78)
print("SANITY CHECK: does cached Bid/Ask spread actually widen around these events?")
print("=" * 78)
for symbol in INSTRUMENTS_TESTED:
    df = raw_frames[symbol]
    pip = INSTRUMENTS[symbol].pip_size
    pre_spreads, post_spreads = [], []
    for _, ev in events_raw.iterrows():
        t0 = ev["event_time_utc"]
        pre_mask = (df.index >= t0 - pd.Timedelta(minutes=PRE_MINUTES)) & (df.index < t0)
        post_mask = (df.index >= t0) & (df.index < t0 + pd.Timedelta(minutes=POST_MINUTES_SHORT))
        pre_g, post_g = df.loc[pre_mask], df.loc[post_mask]
        if len(pre_g) and len(post_g):
            pre_spreads.append(((pre_g["Ask_Close"] - pre_g["Bid_Close"]) / pip).mean())
            post_spreads.append(((post_g["Ask_Close"] - post_g["Bid_Close"]) / pip).mean())
    pre_spreads, post_spreads = np.array(pre_spreads), np.array(post_spreads)
    t_stat, p = two_sample_t(post_spreads, pre_spreads)
    print(f"  {symbol}: pre-event avg spread={pre_spreads.mean():.2f} pips, "
          f"post-event(30m) avg spread={post_spreads.mean():.2f} pips, "
          f"ratio={post_spreads.mean()/pre_spreads.mean():.2f}  (t={t_stat:.2f}, p={p:.4f})")
print()

# --- realized vol: event vs control ---
print("=" * 78)
print("REALIZED VOLATILITY: event windows vs control (ordinary-time) windows")
print("=" * 78)
for label, ev_df, ctrl_df in [("EUR_USD", event_dfs["EUR_USD"], control_dfs["EUR_USD"]),
                                ("GBP_USD", event_dfs["GBP_USD"], control_dfs["GBP_USD"]),
                                ("COMBINED", combined_ev, combined_ctrl)]:
    print(f"\n  -- {label} --  (n_event={len(ev_df)}, n_control={len(ctrl_df)})")
    for col, name in [("post_vol_30m", "post 30min"), ("post_vol_2h", "post 2h")]:
        m_ev, m_ctrl = ev_df[col].mean(), ctrl_df[col].mean()
        t_stat, p = two_sample_t(ev_df[col].values, ctrl_df[col].values)
        print(f"    {name:10s}: event mean vol={m_ev*100:.4f}%  control mean vol={m_ctrl*100:.4f}%  "
              f"ratio={m_ev/m_ctrl:.3f}  (t={t_stat:.2f}, p={p:.4f})")
    # vol_ratio (post/pre) itself, one-sample vs 1.0 (no expansion) and two-sample vs control's own ratio
    for col, name in [("vol_ratio_30m", "vol ratio post30/pre"), ("vol_ratio_2h", "vol ratio post2h/pre")]:
        vals = ev_df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        ctrl_vals = ctrl_df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        m, se, t_stat_1, p_1 = one_sample_t(vals - 1.0)  # vs "no expansion" (ratio=1)
        t_2, p_2 = two_sample_t(vals, ctrl_vals)
        print(f"    {name:22s}: mean ratio={np.mean(vals):.3f} (vs 1.0: t={t_stat_1:.2f}, p={p_1:.4f})   "
              f"vs control ratio (mean={np.mean(ctrl_vals):.3f}): t={t_2:.2f}, p={p_2:.4f}")

# --- breakout incidence: event vs control ---
print(f"\n{'=' * 78}\nBREAKOUT INCIDENCE (>= {BREAKOUT_ATR_MULTIPLE}x ATR beyond pre-event range)\n{'=' * 78}")
for label, ev_df, ctrl_df in [("EUR_USD", event_dfs["EUR_USD"], control_dfs["EUR_USD"]),
                                ("GBP_USD", event_dfs["GBP_USD"], control_dfs["GBP_USD"]),
                                ("COMBINED", combined_ev, combined_ctrl)]:
    n_ev, n_ctrl = len(ev_df), len(ctrl_df)
    x_ev, x_ctrl = int(ev_df["tradeable"].sum()), int(ctrl_df["tradeable"].sum())
    z, p = prop_test_2sample(x_ev, n_ev, x_ctrl, n_ctrl)
    print(f"  {label:10s}: event rate={x_ev}/{n_ev} ({x_ev/n_ev*100:.2f}%)   "
          f"control rate={x_ctrl}/{n_ctrl} ({x_ctrl/n_ctrl*100:.2f}%)   z={z:.2f}, p={p:.4f}")

# --- by event type ---
print(f"\n{'=' * 78}\nBY EVENT TYPE (combined instruments)\n{'=' * 78}")
for etype, g in combined_ev.groupby("event_type"):
    n = len(g)
    x = int(g["tradeable"].sum())
    print(f"  {etype:5s}: n={n:3d}  breakout rate={x}/{n} ({x/n*100:.2f}%)  "
          f"mean vol_ratio_2h={g['vol_ratio_2h'].replace([np.inf,-np.inf],np.nan).mean():.3f}  "
          f"mean vol_ratio_30m={g['vol_ratio_30m'].replace([np.inf,-np.inf],np.nan).mean():.3f}")

# --- cost-adjusted P&L for tradeable breakouts ---
# CAVEAT (reported, not hidden): direction here is the EVENTUAL/known breakout side (ex-post),
# not a real-time detection rule - a genuine strategy would need a live breakout trigger, not
# built here per instruction. This is a best-case/upper-bound estimate of capturable edge, not
# a claim that this exact P&L is achievable by a real-time system.
print(f"\n{'=' * 78}\nCOST-ADJUSTED EDGE (tradeable breakouts only, direction = eventual/known breakout\n"
      f"side [ex-post - see caveat above], enter at first post-event bar's realistic fill,\n"
      f"exit at 2h-window-end's realistic fill)\n{'=' * 78}")
for symbol in INSTRUMENTS_TESTED + ["COMBINED"]:
    ev_df = event_dfs[symbol] if symbol != "COMBINED" else combined_ev
    tradeable = ev_df[ev_df["tradeable"]]
    if len(tradeable) < 5:
        print(f"  {symbol:10s}: n={len(tradeable)} (too few tradeable breakouts for stats)")
        continue
    raw_ret = tradeable.apply(
        lambda r: r["direction"] * (r["post_close_2h"] - r["pre_high"] if r["direction"] > 0
                                     else r["pre_low"] - r["post_close_2h"]) / r["pre_high"], axis=1)
    m, se, t_stat, p = one_sample_t(raw_ret.values)
    realistic_ret = tradeable.apply(
        lambda r: ((r["exit_bid"] - r["entry_ask"]) / r["entry_ask"] if r["direction"] > 0
                   else (r["entry_bid"] - r["exit_ask"]) / r["entry_bid"]), axis=1)
    m_r, se_r, t_r, p_r = one_sample_t(realistic_ret.values)
    print(f"  {symbol:10s}: n={len(tradeable):3d}  raw(mid,range-anchored)={m*100:+.4f}% (p={p:.4f})   "
          f"realistic(Bid/Ask,near-immediate entry)={m_r*100:+.4f}% (t={t_r:.2f}, p={p_r:.4f})   "
          f"cost drag={(m-m_r)*100:.4f}%")

# --- stability by year ---
print(f"\n{'=' * 78}\nSTABILITY BY YEAR (combined, vol_ratio_2h and breakout rate)\n{'=' * 78}")
for yr, g in combined_ev.groupby("year"):
    n = len(g)
    if n < 15:
        print(f"  {yr}: n={n} (too few - skipping)")
        continue
    x = int(g["tradeable"].sum())
    print(f"  {yr}: n={n:3d}  mean vol_ratio_2h={g['vol_ratio_2h'].replace([np.inf,-np.inf],np.nan).mean():.3f}  "
          f"breakout rate={x}/{n} ({x/n*100:.2f}%)")

# --- sample size ---
print(f"\n{'=' * 78}\nSAMPLE SIZE ASSESSMENT\n{'=' * 78}")
print(f"  Combined event observations: {len(combined_ev)} (project MIN_REQUIRED_TRADES convention: 150)")
print(f"  Per instrument: EUR_USD={len(event_dfs['EUR_USD'])}, GBP_USD={len(event_dfs['GBP_USD'])}")
print(f"  By event type (combined): {combined_ev.groupby('event_type').size().to_dict()}")
print(f"  Per-year (combined): {combined_ev.groupby('year').size().to_dict()}")

print(f"\n{'=' * 78}\nDONE. No strategy/entry-exit code, no risk-management changes, no live\n"
      f"data feeds, no live-connector changes, no validation/reserved data access.\n{'=' * 78}")
