"""
hypothesis_tests/gap_fade_falsification_round1_24h_hold.py
--------------------------------------------------------------
Round 1 of the Weekend Gap Fade falsification test: fixed 24h-hold fade
trade, all 4 instruments. Superseded in rigor (not in relevance - both
rounds' evidence fed the final verdict) by round 2's closure-tracking
test - see gap_fade_falsification_round2_closure.py.

RESULT: contributed to the overall REJECTED verdict. See
results/hypothesis2_gap_fade_falsification_summary.json for the full
numeric record (both rounds) and README.md's "V3 candidate search"
section for the narrative writeup. Preserved unmodified, purely so
this exact experiment never needs re-running - not imported by
anything in the project.

Cheapest-possible empirical falsification test for Hypothesis 2 (Weekend
Gap Fade). Pure statistics on DEVELOPMENT data only - no strategy code,
no thresholds, no parameter search. ONE pre-specified test, run once,
reported in full regardless of outcome.

PRE-REGISTERED SPECIFICATION (fixed before any numbers were looked at):
  - Instruments: all 4 (EUR_USD, GBP_USD, USD_JPY, XAU_USD) - reported
    per-instrument AND combined/pooled, since the mechanism (weekend
    liquidity vacuum) is instrument-agnostic in principle. Reporting
    all 4 rather than picking whichever looks best afterward.
  - Timeframe: H1 - fine enough to identify the exact reopen bar,
    coarse enough that ~4 years of weekly events is a small, fast
    computation (no new data fetch needed - already cached).
  - Weekend gap definition: consecutive H1 bars whose timestamp gap
    exceeds 40 hours (only genuine market closures - weekends, and
    incidentally some holidays - produce gaps this large at H1
    resolution) AND whose PRE-gap bar falls on a Friday (UTC) - this
    isolates genuine weekly weekend closures specifically (the
    hypothesis under test) from occasional mid-week data gaps or
    holiday closures starting on other weekdays, without needing any
    tuned threshold (40h is a generous margin above the ~48-55h a real
    weekend/holiday produces and far above normal 1h spacing - not
    swept or optimised).
  - Gap (signed): log(Open of first post-gap bar) - log(Close of last
    pre-gap bar). Positive = price opened higher after the weekend;
    negative = opened lower.
  - Target: subsequent 24h return, log(Close at reopen+24h, nearest
    available bar) - log(Open of the reopen bar) - i.e. the return
    over the first day of trading after reopen.
  - No lookahead: the fade trade is entered AT the reopen bar's own
    open (the instant the market reopens - the earliest realistic
    execution point) and only uses the gap size, which is fully known
    at that instant (both endpoints - Friday's close and Sunday's
    reopen open - have already happened). The subsequent-return window
    is strictly after entry.
  - No threshold: every qualifying weekend gap is traded, direction
    unconditional on magnitude (fade = bet gap partially reverts,
    direction = -sign(gap)) - nothing to tune.

Realistic P&L uses cached Bid/Ask columns exactly as the backtest
engine's own MAE/MFE convention does: buy at Ask, sell at Bid.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from dataset_split import split_for_iteration, describe_split

GAP_HOURS_THRESHOLD = 40   # fixed, generous margin - see module docstring; not tuned
HOLD_HOURS = 24            # fixed single holding period; not swept

INSTRUMENTS_TESTED = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]

print("=" * 78)
print("HYPOTHESIS 2 FALSIFICATION CHECK: Weekend Gap Fade")
print(f"All 4 instruments, H1, gap>{GAP_HOURS_THRESHOLD}h + pre-gap bar=Friday, "
      f"{HOLD_HOURS}h hold, DEVELOPMENT ONLY")
print("=" * 78)
print(describe_split())
print()


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def pearson_r_and_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_ = len(x)
    r = np.corrcoef(x, y)[0, 1]
    t_stat = r * math.sqrt(max(1, n_ - 2)) / math.sqrt(max(1e-12, 1 - r ** 2))
    p = two_sided_p_from_z(t_stat)
    return r, t_stat, p


def reversion_rate_and_p(gap, subsequent_ret):
    """Fraction of events where the subsequent move has the OPPOSITE
    sign to the gap (reversion) - vs 50% (equally likely to continue
    or revert under the null)."""
    gap = np.asarray(gap, dtype=float)
    sub = np.asarray(subsequent_ret, dtype=float)
    mask = (gap != 0) & (sub != 0)
    gap, sub = gap[mask], sub[mask]
    n_ = len(gap)
    reverted = int((np.sign(gap) != np.sign(sub)).sum())
    rate = reverted / n_
    z = (reverted - 0.5 * n_) / math.sqrt(0.25 * n_)
    p = two_sided_p_from_z(z)
    return rate, n_, reverted, z, p


def mean_t_and_p(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    t_stat = m / se if se and se > 0 else 0.0
    p = two_sided_p_from_z(t_stat)
    return m, se, t_stat, p


def find_weekend_gaps(df):
    """df: H1 frame, DatetimeIndex UTC, sorted. Returns a DataFrame, one
    row per qualifying weekend gap, with pre/post bar info."""
    idx = df.index
    deltas = idx.to_series().diff().dt.total_seconds() / 3600.0
    gap_positions = np.where(deltas.values > GAP_HOURS_THRESHOLD)[0]  # index of the POST-gap bar
    rows = []
    for pos in gap_positions:
        pre_ts = idx[pos - 1]
        post_ts = idx[pos]
        if pre_ts.weekday() != 4:  # 4 = Friday, UTC
            continue
        pre_close = df["Close"].iloc[pos - 1]
        post_open = df["Open"].iloc[pos]
        # subsequent 24h return: nearest bar at/after post_ts + HOLD_HOURS
        target_time = post_ts + pd.Timedelta(hours=HOLD_HOURS)
        future_idx = idx[idx >= target_time]
        if len(future_idx) == 0:
            continue
        exit_ts = future_idx[0]
        exit_pos = idx.get_loc(exit_ts)
        exit_close = df["Close"].iloc[exit_pos]
        rows.append({
            "pre_ts": pre_ts, "post_ts": post_ts, "exit_ts": exit_ts,
            "pre_close": pre_close, "post_open": post_open, "exit_close": exit_close,
            "post_bid_open": df["Bid_Open"].iloc[pos], "post_ask_open": df["Ask_Open"].iloc[pos],
            "exit_bid_close": df["Bid_Close"].iloc[exit_pos], "exit_ask_close": df["Ask_Close"].iloc[exit_pos],
        })
    return pd.DataFrame(rows)


all_events = []
for symbol in INSTRUMENTS_TESTED:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "H1")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    events = find_weekend_gaps(df)
    events["symbol"] = symbol
    all_events.append(events)
    print(f"{symbol}: {len(df)} H1 development bars, {len(events)} qualifying weekend-gap events found")

events = pd.concat(all_events, ignore_index=True)
print(f"\nTotal weekend-gap events (all 4 instruments, pooled): {len(events)}")
print()

events["gap"] = np.log(events["post_open"]) - np.log(events["pre_close"])
events["subsequent_ret"] = np.log(events["exit_close"]) - np.log(events["post_open"])
events["year"] = events["post_ts"].dt.year

# fade direction = -sign(gap); realistic fade P&L uses Bid/Ask
direction = -np.sign(events["gap"].values)  # +1 = fade-long, -1 = fade-short
long_mask = direction > 0
short_mask = direction < 0

realistic_ret = np.full(len(events), np.nan)
realistic_ret[long_mask] = (
    (events["exit_bid_close"].values[long_mask] - events["post_ask_open"].values[long_mask])
    / events["post_ask_open"].values[long_mask]
)
realistic_ret[short_mask] = (
    (events["post_bid_open"].values[short_mask] - events["exit_ask_close"].values[short_mask])
    / events["post_bid_open"].values[short_mask]
)

hyp_ret = -direction * events["subsequent_ret"].values  # mid-price, zero-cost

events["realistic_fade_ret"] = realistic_ret
events["hyp_fade_ret"] = hyp_ret

print("=" * 78)
print("SAMPLE SIZE")
print("=" * 78)
for symbol in INSTRUMENTS_TESTED:
    n_sym = (events["symbol"] == symbol).sum()
    print(f"  {symbol}: {n_sym} events")
print(f"  Combined (pooled): {len(events)}")
print(f"  Project's standing minimum-sample convention (MIN_REQUIRED_TRADES): 150")
print()

print("=" * 78)
print("PRIMARY TEST (combined, pooled across all 4 instruments):")
print("gap (signed) vs subsequent 24h return (signed) - reversion implies NEGATIVE correlation")
print("=" * 78)
r, t_stat, p = pearson_r_and_p(events["gap"].values, events["subsequent_ret"].values)
print(f"  n = {len(events)}")
print(f"  Pearson r = {r:.5f}   t = {t_stat:.3f}   p = {p:.4f}")
rate, n_rev, reverted, z, p_rev = reversion_rate_and_p(events["gap"].values, events["subsequent_ret"].values)
print(f"  Reversion rate = {rate*100:.2f}%  ({reverted}/{n_rev})  vs 50% baseline: z = {z:.3f}, p = {p_rev:.4f}")
print()

print("=" * 78)
print("PER-INSTRUMENT BREAKDOWN (same test, each instrument alone)")
print("=" * 78)
for symbol in INSTRUMENTS_TESTED:
    g = events[events["symbol"] == symbol]
    if len(g) < 20:
        print(f"  {symbol}: n={len(g)} (too few events for meaningful stats)")
        continue
    r_s, t_s, p_s = pearson_r_and_p(g["gap"].values, g["subsequent_ret"].values)
    rate_s, n_s, rev_s, z_s, p_rev_s = reversion_rate_and_p(g["gap"].values, g["subsequent_ret"].values)
    print(f"  {symbol}: n={len(g):4d}  r={r_s:+.4f} (p={p_s:.4f})   "
          f"reversion={rate_s*100:5.2f}% (p={p_rev_s:.4f})")
print()

print("=" * 78)
print("STABILITY ACROSS YEARS (combined, pooled)")
print("=" * 78)
for yr, g in events.groupby("year"):
    if len(g) < 20:
        print(f"  {yr}: n={len(g)} (too few events - skipping stats)")
        continue
    r_y, t_y, p_y = pearson_r_and_p(g["gap"].values, g["subsequent_ret"].values)
    rate_y, n_y, rev_y, z_y, p_rev_y = reversion_rate_and_p(g["gap"].values, g["subsequent_ret"].values)
    print(f"  {yr}: n={len(g):4d}  r={r_y:+.4f} (p={p_y:.4f})   "
          f"reversion={rate_y*100:5.2f}% (p={p_rev_y:.4f})")
print()

print("=" * 78)
print("COST-ADJUSTED EDGE: fade every qualifying gap, hold 24h")
print("(unconditional on magnitude - no threshold to tune)")
print("=" * 78)
valid = ~np.isnan(events["realistic_fade_ret"].values) & ~np.isnan(events["hyp_fade_ret"].values)
r_real = events["realistic_fade_ret"].values[valid]
r_hyp = events["hyp_fade_ret"].values[valid]
m_r, se_r, t_r, p_r = mean_t_and_p(r_real)
m_h, se_h, t_h, p_h = mean_t_and_p(r_hyp)
print(f"  n trades = {valid.sum()}")
print(f"  Realistic (Bid/Ask, spread-inclusive) avg return/trade = {m_r*100:.4f}%  (t={t_r:.3f}, p={p_r:.4f})")
print(f"  Hypothetical (mid-price, zero-cost)    avg return/trade = {m_h*100:.4f}%  (t={t_h:.3f}, p={p_h:.4f})")
print(f"  Cost drag (hypothetical - realistic)                   = {(m_h-m_r)*100:.4f}%")
print(f"  Total realistic P&L if traded every event, additive, no compounding/sizing: "
      f"{r_real.sum()*100:.2f}% over {len(r_real)} trades")
print()

print("=" * 78)
print("GAP SIZE DISTRIBUTION (context, not a threshold search)")
print("=" * 78)
for symbol in INSTRUMENTS_TESTED:
    g = events[events["symbol"] == symbol]
    pip = INSTRUMENTS[symbol].pip_size
    gap_pips = (np.exp(g["gap"]) - 1) * g["pre_close"] / pip
    print(f"  {symbol}: median |gap| = {gap_pips.abs().median():.2f} pips, "
          f"mean |gap| = {gap_pips.abs().mean():.2f} pips, max |gap| = {gap_pips.abs().max():.2f} pips")
print()

print("=" * 78)
print("DONE. No strategy code written, no thresholds tuned, no validation/")
print("reserved data accessed, no live connector touched.")
print("=" * 78)
