"""
hypothesis_tests/roundnumber_falsification.py
--------------------------------------------------
Cheap empirical falsification test for Post-H4 Candidate 3: Round-
Number Level Consolidation - the pre-check step the approved candidate
design required before any entry/exit logic could be written.

RESULT: REJECTED at the pre-check stage. Volatility near round levels
was significantly HIGHER (~28%), not lower, than far from them - the
opposite of the consolidation hypothesis - stable across both
instruments and every year 2020-2024 (all p<0.0001). Per the pre-
committed survival bar, this does not proceed to any entry/exit design.
See results/candidate3_roundnumber_falsification_summary.json for the
full numeric record and README.md's "Post-H4 structural search"
section for the narrative writeup. Preserved unmodified, purely so
this exact experiment never needs re-running - not imported by
anything in the project.

Pure statistics on DEVELOPMENT data only -
no strategy/entry-exit code, no thresholds swept, single frozen
definition, decided before any numbers were looked at. EUR_USD and
GBP_USD, M15.

Research question: does realized volatility measurably COMPRESS when
price is near a round number (institutional order-clustering /
psychological level), relative to an ordinary-time control? Per the
approved candidate-3 proposal, this is step 1 - if compression is not
confirmed, the idea is rejected before any entry/exit logic is written.

===========================================================================
FROZEN DEFINITIONS (fixed BEFORE any numbers were looked at)
===========================================================================
1. Round level: any price that is a multiple of 0.0050 (50 pips) for
   EUR_USD/GBP_USD - the standard FX "big figure" (multiples of 0.0100,
   e.g. 1.1000) AND "half figure" (the midpoint, e.g. 1.1050) convention
   combined into one level set. A fixed market convention, not tuned or
   swept.
2. "Near" a round level: a bar's Close is within
   NEAR_BUFFER_ATR_FRACTION x ATR(14) of the nearest round level (0.1,
   the SAME buffer fraction reused throughout every other module in
   this project - not independently chosen for this test). Binary
   classification - every bar is either "near" or "far", no ambiguous
   middle band, so there's no separate "control window" definition to
   second-guess: "far" is simply the complement of "near".
3. Volatility measure: realized vol = sqrt(sum of squared M15 log
   returns) over the FOLLOWING 2-hour window from that bar - the exact
   same measure and window already used for H2's closure check, H3's
   day-level check, and H4's event check, reused for direct
   comparability across this whole search rather than inventing a new
   one for this test.
4. No lookahead: classification (near/far) uses ONLY that bar's own
   Close and the ATR value known as of that bar - the round level
   itself is a fixed grid, not derived from any future data. The
   volatility measure is over the bar's own FOLLOWING window, never
   overlapping with the classification decision itself.
5. Survival bar (stated before running): realized volatility in the
   "near" group must be significantly LOWER than the "far" group
   (compression, as the consolidation hypothesis predicts), stable in
   sign across both instruments and at least most years, for the idea
   to proceed to any entry/exit design. A null or wrong-signed result
   (near-level volatility equal to or HIGHER than far) rejects the
   hypothesis outright, same standard as H1-H3.

Note on control-window independence (same caveat H2's baseline noted):
"far" bars vastly outnumber "near" bars and their forward windows
overlap heavily - the "far" aggregate is treated as descriptive
context, same as before; the "near" group's own large sample size is
what's statistically load-bearing.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration, describe_split
from indicators import atr as atr_indicator

INSTRUMENTS_TESTED = ["EUR_USD", "GBP_USD"]
ROUND_LEVEL_SPACING = 0.0050      # fixed FX convention (big figure + half figure), not tuned
NEAR_BUFFER_ATR_FRACTION = 0.1    # reused from every other module in this project
FORWARD_WINDOW_MINUTES = 120      # reused from H2/H3/H4's own 2h window


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


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


def distance_to_nearest_round_level(close):
    remainder = close % ROUND_LEVEL_SPACING
    return np.minimum(remainder, ROUND_LEVEL_SPACING - remainder)


def forward_realized_vol(close_values, window_bars):
    """Vectorized: for each index i, realized vol of log returns over
    the following `window_bars` M15 bars (i+1 .. i+window_bars)."""
    log_close = np.log(close_values)
    log_rets = np.diff(log_close, prepend=np.nan)  # log_rets[i] = ret from i-1 to i
    n = len(close_values)
    out = np.full(n, np.nan)
    sq = log_rets ** 2
    # cumulative sum of squared returns, so window sum = cumsum[i+window] - cumsum[i]
    cumsq = np.nancumsum(np.where(np.isnan(sq), 0, sq))
    valid_count = np.cumsum(~np.isnan(sq))
    for i in range(n - window_bars):
        start_cum = cumsq[i]
        end_cum = cumsq[i + window_bars]
        n_valid = valid_count[i + window_bars] - valid_count[i]
        if n_valid > 0:
            out[i] = math.sqrt(end_cum - start_cum)
    return out


print("=" * 78)
print("POST-H4 CANDIDATE 3 PRE-CHECK: Round-Number Level Consolidation")
print("EUR_USD & GBP_USD, M15, DEVELOPMENT ONLY")
print("=" * 78)
print(describe_split())
print(f"Round level spacing={ROUND_LEVEL_SPACING}, near-buffer={NEAR_BUFFER_ATR_FRACTION}xATR, "
      f"forward window={FORWARD_WINDOW_MINUTES}min")
print()

WINDOW_BARS = FORWARD_WINDOW_MINUTES // 15

all_results = {}
for symbol in INSTRUMENTS_TESTED:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "M15")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    pip = INSTRUMENTS[symbol].pip_size

    atr_series = atr_indicator(df["High"], df["Low"], df["Close"], period=14)
    dist = distance_to_nearest_round_level(df["Close"].values)
    near_buffer = NEAR_BUFFER_ATR_FRACTION * atr_series.values

    valid = ~np.isnan(near_buffer) & (near_buffer > 0)
    is_near = valid & (dist <= near_buffer)

    fwd_vol = forward_realized_vol(df["Close"].values, WINDOW_BARS)

    result = pd.DataFrame({
        "is_near": is_near, "fwd_vol": fwd_vol, "dist_pips": dist / pip, "atr_valid": valid,
    }, index=df.index)
    result["year"] = result.index.year
    # Keep only rows with BOTH a valid ATR-based classification AND a
    # computable forward volatility window - aligned filtering on the
    # DataFrame itself (not positional slicing after a length-changing
    # dropna), so indices never get mismatched.
    result = result[result["atr_valid"] & result["fwd_vol"].notna()].drop(columns=["atr_valid"])

    all_results[symbol] = result

    near = result[result["is_near"]]
    far = result[~result["is_near"]]
    print(f"{symbol}: {len(df)} M15 bars -> {len(near)} 'near round level' ({len(near)/len(result)*100:.2f}%), "
          f"{len(far)} 'far' bars")

combined = pd.concat(all_results.values(), ignore_index=True)
print(f"\nCombined: {combined['is_near'].sum()} near, {(~combined['is_near']).sum()} far")
print()


def report(label, near, far):
    print(f"\n{'-'*78}\n{label}\n{'-'*78}")
    n_near, n_far = len(near), len(far)
    if n_near < 15:
        print(f"  Too few 'near' observations (n={n_near}) for meaningful stats.")
        return
    m_near, m_far = near["fwd_vol"].mean(), far["fwd_vol"].mean()
    t_stat, p = two_sample_t(near["fwd_vol"].values, far["fwd_vol"].values)
    ratio = m_near / m_far if m_far else float("nan")
    print(f"  n_near={n_near}  n_far={n_far}")
    print(f"  mean fwd 2h realized vol: near={m_near*100:.4f}%  far={m_far*100:.4f}%  ratio(near/far)={ratio:.4f}")
    print(f"  t={t_stat:.3f}  p={p:.4f}  "
          f"{'COMPRESSION (near < far), as hypothesized' if m_near < m_far else 'NO compression / EXPANSION (near >= far)'}")


for symbol in INSTRUMENTS_TESTED:
    r = all_results[symbol]
    report(f"{symbol} - ALL", r[r["is_near"]], r[~r["is_near"]])

report("COMBINED", combined[combined["is_near"]], combined[~combined["is_near"]])

print(f"\n{'='*78}\nSTABILITY BY YEAR (combined)\n{'='*78}")
for yr, g in combined.groupby("year"):
    near_y, far_y = g[g["is_near"]], g[~g["is_near"]]
    if len(near_y) < 15:
        print(f"  {yr}: n_near={len(near_y)} (too few - skipping)")
        continue
    t_stat, p = two_sample_t(near_y["fwd_vol"].values, far_y["fwd_vol"].values)
    m_near, m_far = near_y["fwd_vol"].mean(), far_y["fwd_vol"].mean()
    print(f"  {yr}: n_near={len(near_y):5d}  near={m_near*100:.4f}%  far={m_far*100:.4f}%  "
          f"ratio={m_near/m_far:.4f}  (t={t_stat:.2f}, p={p:.4f})")

print(f"\n{'='*78}\nSAMPLE SIZE ASSESSMENT\n{'='*78}")
print(f"  Combined 'near round level' observations: {combined['is_near'].sum()} "
      f"(project MIN_REQUIRED_TRADES convention: 150 - this is a pre-check sample, not a trade count)")
print(f"  Per instrument: " +
      ", ".join(f"{s}={all_results[s]['is_near'].sum()}" for s in INSTRUMENTS_TESTED))

print(f"\n{'='*78}\nDONE. No entry/exit code, no thresholds swept, no validation/reserved\n"
      f"data access, no live-connector changes.\n{'='*78}")
