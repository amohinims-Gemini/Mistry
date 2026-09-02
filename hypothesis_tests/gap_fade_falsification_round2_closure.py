"""
hypothesis_tests/gap_fade_falsification_round2_closure.py
---------------------------------------------------------------
Round 2 of the Weekend Gap Fade falsification test - the more rigorous
of the two rounds, and the one whose evidence was decisive for the
final verdict (real weekend-gap closure rates are statistically
indistinguishable from an ordinary-move baseline over the same
forward window).

RESULT: REJECTED. See results/hypothesis2_gap_fade_falsification_summary.json
for the full numeric record (both rounds) and README.md's "V3 candidate
search" section for the narrative writeup. Preserved unmodified, purely
so this exact experiment never needs re-running - not imported by
anything in the project.

Cheap empirical falsification test, Hypothesis 2 (Weekend Gap Fade),
round 2: does the weekend opening gap show a statistically meaningful
tendency to PARTIALLY OR FULLY CLOSE, and how long does that take?
DEVELOPMENT data only. EUR_USD and GBP_USD only (per this round's
scope), H1, no strategy code, one frozen definition, no threshold
search.

===========================================================================
FROZEN DEFINITIONS (fixed BEFORE any numbers were looked at)
===========================================================================
1. Friday reference price: Close of the last H1 bar before the weekend
   closure (the bar whose own weekday, UTC, is Friday and which is
   immediately followed by a >40h timestamp gap - same detection rule
   used in the first gap-fade check).
2. "Monday"/week-opening price: Open of the first H1 bar after that
   gap. In practice this bar's timestamp is Sunday evening UTC (OANDA's
   actual weekly reopen), not literally Monday - called out explicitly
   so the number isn't misread as a Monday-morning price.
3. Minimum gap definition: NONE - every non-zero difference between (1)
   and (2) counts as a gap event. Deliberately no magnitude floor, so
   there is no threshold to tune or optimise. (A bar-count/detection
   rule - the >40h closure gap - still applies, but that is a data-
   integrity rule for finding weekends at all, not a size filter on the
   price gap itself.)
4. Gap closure %, defined symmetrically for up and down gaps:
     gap_signed    = reopen_open - friday_close
     remaining(t)  = close(t) - friday_close
     closure_frac(t) = 1 - remaining(t) / gap_signed
   closure_frac = 0 at the moment of reopen, 1 when price has fully
   returned to the Friday reference price, >1 on overshoot beyond it,
   <0 if the gap widens further before closing at all. Measured using
   each subsequent bar's CLOSE (not intrabar High/Low), a deliberately
   conservative choice so the test doesn't credit a threshold as "hit"
   from a wick that wasn't reliably tradeable.
   25/50/100% closure = the first bar (chronologically, within the
   observation window) at which closure_frac(t) >= 0.25 / 0.50 / 1.00.
5. Maximum observation window: 120 hours (5 days) from the reopen bar -
   approximately one trading week, ending before the following
   weekend's own closure. Chosen for that structural reason, not by
   checking which window maximises the closure rate.

BASELINE (pre-specified, not searched): the identical closure-tracking
procedure applied to ORDINARY, non-weekend H1 bar-to-bar moves - every
20th H1 bar in the development window (a systematic, not random or
cherry-picked, subsample - keeps the baseline sample large without
tracking all ~24,000 bars/instrument) is treated as its own "gap"
(open vs prior bar's close) and given the same 120h forward window and
25/50/100% closure test. This asks the cleanly different question:
is post-weekend closure behaviour different from ordinary post-move
reversion tendency in general, or is a weekend gap not special at all?
Note: unlike the real weekend-gap sample (independent, non-overlapping
weekly events), these baseline windows overlap heavily and are not
independent - so the baseline's own p-values are descriptive context
only; the real weekend-gap sample's own significance test is what is
load-bearing for the verdict.

No strategy code, no entry/exit engine, no risk-management changes, no
validation/reserved data access, no live-connector changes.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration, describe_split

GAP_DETECT_HOURS = 40     # data-integrity rule for finding weekend closures (unchanged from round 1)
WINDOW_HOURS = 120        # frozen observation window
THRESHOLDS = [0.25, 0.50, 1.00]
BASELINE_STRIDE = 20      # systematic (not random) subsample for the ordinary-move baseline

INSTRUMENTS_TESTED = ["EUR_USD", "GBP_USD"]


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def prop_test_vs(x, n, p0):
    """z-test of an observed proportion x/n against a fixed baseline p0."""
    if n == 0:
        return float("nan"), float("nan")
    rate = x / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (rate - p0) / se if se > 0 else 0.0
    return z, two_sided_p_from_z(z)


def mean_t_and_p(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    t_stat = m / se if se > 0 else 0.0
    return m, se, t_stat, two_sided_p_from_z(t_stat)


def find_weekend_gap_positions(ts):
    deltas = ts.to_series().diff().dt.total_seconds().values / 3600.0
    post_positions = np.where(deltas > GAP_DETECT_HOURS)[0]
    events = []
    for pos in post_positions:
        if ts[pos - 1].weekday() == 4:  # Friday, UTC
            events.append((pos - 1, pos))  # (pre_pos, post_pos)
    return events


def _track_closure_core(close, ts, bid_open, ask_open, bid_close, ask_close, pre_pos, post_pos, friday_close):
    reopen_open_mid = (bid_open[post_pos] + ask_open[post_pos]) / 2.0
    gap_signed = reopen_open_mid - friday_close
    if gap_signed == 0:
        return None

    window_end = ts[post_pos] + pd.Timedelta(hours=WINDOW_HOURS)
    end_pos = post_pos
    n = len(ts)
    while end_pos + 1 < n and ts[end_pos + 1] <= window_end:
        end_pos += 1

    path_close = close[post_pos:end_pos + 1]
    path_ts = ts[post_pos:end_pos + 1]
    closure_frac = 1.0 - (path_close - friday_close) / gap_signed

    result = {
        "gap_signed": gap_signed,
        "direction": "up" if gap_signed > 0 else "down",
        "n_bars_in_window": len(path_close),
    }
    fade_is_long = gap_signed < 0  # fade toward friday_close: if gap up, fade short; if gap down, fade long

    for thr in THRESHOLDS:
        hit_idx = np.where(closure_frac >= thr)[0]
        key = f"{int(thr*100)}"
        if len(hit_idx) > 0:
            i = hit_idx[0]
            result[f"reached_{key}"] = True
            result[f"hours_to_{key}"] = (path_ts[i] - ts[post_pos]).total_seconds() / 3600.0
            exit_pos = post_pos + i
        else:
            result[f"reached_{key}"] = False
            result[f"hours_to_{key}"] = float("nan")
            exit_pos = end_pos

        # realistic (Bid/Ask) and hypothetical (mid) fade-trade return to this exit
        if fade_is_long:
            realistic = (bid_close[exit_pos] - ask_open[post_pos]) / ask_open[post_pos]
        else:
            realistic = (bid_open[post_pos] - ask_close[exit_pos]) / bid_open[post_pos]
        mid_entry = reopen_open_mid
        mid_exit = close[exit_pos]
        hyp = (mid_exit - mid_entry) / mid_entry if fade_is_long else (mid_entry - mid_exit) / mid_entry
        result[f"realistic_ret_{key}"] = realistic
        result[f"hyp_ret_{key}"] = hyp

    result["final_closure_frac"] = closure_frac[-1]
    return result


def build_events(df, event_positions):
    close = df["Close"].values
    ts = df.index
    bid_open, ask_open = df["Bid_Open"].values, df["Ask_Open"].values
    bid_close, ask_close = df["Bid_Close"].values, df["Ask_Close"].values
    out = []
    for pre_pos, post_pos in event_positions:
        r = _track_closure_core(close, ts, bid_open, ask_open, bid_close, ask_close,
                                 pre_pos, post_pos, close[pre_pos])
        if r is not None:
            r["post_ts"] = ts[post_pos]
            r["year"] = ts[post_pos].year
            out.append(r)
    return pd.DataFrame(out)


def build_baseline_events(df, stride):
    close = df["Close"].values
    ts = df.index
    bid_open, ask_open = df["Bid_Open"].values, df["Ask_Open"].values
    bid_close, ask_close = df["Bid_Close"].values, df["Ask_Close"].values
    n = len(df)
    out = []
    for post_pos in range(stride, n - 1, stride):
        pre_pos = post_pos - 1
        r = _track_closure_core(close, ts, bid_open, ask_open, bid_close, ask_close,
                                 pre_pos, post_pos, close[pre_pos])
        if r is not None:
            out.append(r)
    return pd.DataFrame(out)


print("=" * 78)
print("HYPOTHESIS 2 (ROUND 2): Weekend Gap CLOSURE test - EUR_USD & GBP_USD, H1")
print("DEVELOPMENT DATA ONLY")
print("=" * 78)
print(describe_split())
print(f"Window={WINDOW_HOURS}h, thresholds={THRESHOLDS}, gap-detect>{GAP_DETECT_HOURS}h+Friday pre-bar, "
      f"no minimum gap size, closure measured via bar Close")
print()

per_instrument = {}
per_instrument_baseline = {}
for symbol in INSTRUMENTS_TESTED:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "H1")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    positions = find_weekend_gap_positions(df.index)
    ev = build_events(df, positions)
    ev["symbol"] = symbol
    per_instrument[symbol] = ev

    base = build_baseline_events(df, BASELINE_STRIDE)
    base["symbol"] = symbol
    per_instrument_baseline[symbol] = base

    print(f"{symbol}: {len(df)} H1 development bars -> {len(ev)} weekend-gap events, "
          f"{len(base)} baseline (ordinary-move) events")

combined = pd.concat(per_instrument.values(), ignore_index=True)
combined_baseline = pd.concat(per_instrument_baseline.values(), ignore_index=True)
print(f"\nCombined weekend-gap events: {len(combined)}   Combined baseline events: {len(combined_baseline)}")
print()


def pip_gap(row_df, symbol):
    pip = INSTRUMENTS[symbol].pip_size
    return row_df["gap_signed"] / pip


def report_group(label, g, baseline_g=None, min_for_stats=15):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    n = len(g)
    print(f"  n = {n}")
    if n < min_for_stats:
        print("  Too few events for meaningful stats - reporting counts only.")
        return
    for thr in THRESHOLDS:
        key = f"{int(thr*100)}"
        reached = g[f"reached_{key}"]
        rate = reached.mean()
        hours = g.loc[reached, f"hours_to_{key}"]
        m_r, se_r, t_r, p_r = mean_t_and_p(g[f"realistic_ret_{key}"].values)
        m_h, se_h, t_h, p_h = mean_t_and_p(g[f"hyp_ret_{key}"].values)
        base_note = ""
        if baseline_g is not None and len(baseline_g) >= min_for_stats:
            base_rate = baseline_g[f"reached_{key}"].mean()
            z, p_vs_base = prop_test_vs(int(reached.sum()), n, base_rate)
            base_note = (f"   | baseline(ordinary-move) rate={base_rate*100:5.2f}%  "
                         f"z(vs baseline)={z:+.2f} p={p_vs_base:.4f}")
        print(f"  {key:>3}% closure: reached {reached.sum():4d}/{n} ({rate*100:5.2f}%)  "
              f"avg time-to-close={hours.mean():6.1f}h (median {hours.median():6.1f}h){base_note}")
        print(f"           realistic fade P&L/trade = {m_r*100:+.4f}% (t={t_r:.2f}, p={p_r:.4f})   "
              f"hypothetical(zero-cost) = {m_h*100:+.4f}% (t={t_h:.2f}, p={p_h:.4f})   "
              f"cost drag = {(m_h-m_r)*100:.4f}%")


# --- gap-size distribution ---
print("=" * 78)
print("GAP-SIZE DISTRIBUTION (pips; no size filter applied)")
print("=" * 78)
for symbol in INSTRUMENTS_TESTED:
    g = per_instrument[symbol]
    pips = pip_gap(g, symbol)
    print(f"  {symbol}: n={len(g)}  mean|gap|={pips.abs().mean():.2f}  median|gap|={pips.abs().median():.2f}  "
          f"p25={pips.abs().quantile(.25):.2f}  p75={pips.abs().quantile(.75):.2f}  max={pips.abs().max():.2f}")
    print(f"           up gaps: {(pips>0).sum()}   down gaps: {(pips<0).sum()}")

# --- per instrument + combined ---
for symbol in INSTRUMENTS_TESTED:
    report_group(f"{symbol} - ALL WEEKEND GAPS", per_instrument[symbol], per_instrument_baseline[symbol])
report_group("COMBINED (EUR_USD + GBP_USD) - ALL WEEKEND GAPS", combined, combined_baseline)

# --- up vs down, combined ---
report_group("COMBINED - UP GAPS ONLY", combined[combined["direction"] == "up"],
              combined_baseline[combined_baseline["direction"] == "up"])
report_group("COMBINED - DOWN GAPS ONLY", combined[combined["direction"] == "down"],
              combined_baseline[combined_baseline["direction"] == "down"])

# --- stability by year, combined ---
print(f"\n{'=' * 78}\nSTABILITY BY YEAR (combined, 50% closure threshold as representative)\n{'=' * 78}")
for yr, g in combined.groupby("year"):
    n = len(g)
    if n < 15:
        print(f"  {yr}: n={n} (too few - skipping)")
        continue
    reached50 = g["reached_50"]
    m_r, se_r, t_r, p_r = mean_t_and_p(g["realistic_ret_50"].values)
    print(f"  {yr}: n={n:4d}  50%-closure rate={reached50.mean()*100:5.2f}%  "
          f"avg time={g.loc[reached50,'hours_to_50'].mean():6.1f}h  "
          f"realistic P&L/trade={m_r*100:+.4f}% (p={p_r:.4f})")

# --- sample size assessment ---
print(f"\n{'=' * 78}\nSAMPLE SIZE ASSESSMENT\n{'=' * 78}")
print(f"  Combined weekend-gap events: {len(combined)} (project MIN_REQUIRED_TRADES convention: 150)")
print(f"  Per instrument: EUR_USD={len(per_instrument['EUR_USD'])}, GBP_USD={len(per_instrument['GBP_USD'])}")
print(f"  Up gaps: {(combined['direction']=='up').sum()}   Down gaps: {(combined['direction']=='down').sum()}")
print(f"  Per-year counts (combined): {combined.groupby('year').size().to_dict()}")

print(f"\n{'=' * 78}\nDONE. No strategy/entry-exit code, no risk-management changes, no\n"
      f"validation/reserved data access, no live-connector changes.\n{'=' * 78}")
