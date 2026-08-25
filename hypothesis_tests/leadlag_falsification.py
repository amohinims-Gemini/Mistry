"""
hypothesis_tests/leadlag_falsification.py
--------------------------------------------
Cheapest-possible empirical falsification test for V3 Candidate
Hypothesis 1 (Cross-Instrument Lead-Lag: EUR_USD leads GBP_USD). Pure
statistics on DEVELOPMENT data only - no strategy code, no thresholds,
no parameter search. ONE pre-specified test, run once, reported in
full regardless of outcome.

RESULT: REJECTED. See results/hypothesis1_leadlag_falsification_summary.json
for the full numeric record and README.md's "V3 candidate search" section
for the narrative writeup. Preserved here, unmodified, purely so this
exact experiment never needs to be re-run or re-derived - it is not
part of any strategy and nothing in this project imports it.

Re-runnable at any time (`python3 hypothesis_tests/leadlag_falsification.py`
from the project root, inside venv) - deterministic given the same
cached data, since dataset_split.split_for_iteration() only ever
returns the DEVELOPMENT slice.

PRE-REGISTERED SPECIFICATION (fixed before any numbers were looked at):
  - Instruments: EUR_USD (leader), GBP_USD (lagger). EUR_USD chosen as
    leader on liquidity grounds (it is the single most liquid FX pair
    globally; GBP_USD is comparatively thinner and more likely to be
    the one "catching up") - not chosen by testing both ways and
    keeping the one that worked. The reverse direction (GBP_USD ->
    EUR_USD) and GBP_USD's own lag-1 autocorrelation (the already-
    rejected single-instrument momentum mechanism) are BOTH computed
    and reported alongside the primary test as pre-committed baselines
    for comparison, not as alternative hypotheses to cherry-pick from.
  - Timeframe: M15 (matches cached data already used for the sweep
    work; both instruments trade continuously so M15 bars align
    exactly on the same clock times).
  - Lag: exactly 1 bar (15 minutes) ahead. A single fixed lag, not
    swept across many lags to find whichever looks best.
  - Leader's move: ret_EUR[T] = log(Close_EUR[T]) - log(Close_EUR[T-1])
    (log return over the bar that has JUST closed at T).
  - Target: ret_GBP[T+1] = log(Close_GBP[T+1]) - log(Close_GBP[T]) (the
    lagger's NEXT bar's return - not yet known at time T).
  - No lookahead: at decision time T (the moment bar T has just
    closed), ret_EUR[T] is fully known (T has closed) and ret_GBP[T+1]
    is NOT yet known (T+1 hasn't happened). A trade taken at T's close
    and exited at T+1's close is realistic and uses only information
    available at the moment of entry.
  - No threshold: every bar with a valid ret_EUR[T] is treated as a
    signal (direction = sign(ret_EUR[T])) - unconditional on magnitude,
    so there is nothing to tune here.

Realistic P&L uses the cached Bid/Ask columns exactly as the backtest
engine's own MAE/MFE convention does: buy at Ask, sell at Bid.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration, describe_split

print("=" * 78)
print("HYPOTHESIS 1 FALSIFICATION CHECK: Cross-Instrument Lead-Lag")
print("EUR_USD (leader) -> GBP_USD (lagger), 1-bar (15min) ahead, M15, DEVELOPMENT ONLY")
print("=" * 78)
print(describe_split())
print()

raw = {
    "EUR_USD": get_instrument_data(INSTRUMENTS["EUR_USD"].oanda_symbol, "M15"),
    "GBP_USD": get_instrument_data(INSTRUMENTS["GBP_USD"].oanda_symbol, "M15"),
}
dev, _validation_not_used = split_for_iteration(raw)
del _validation_not_used

eur = dev["EUR_USD"].sort_index()
gbp = dev["GBP_USD"].sort_index()
print(f"EUR_USD development bars: {len(eur)}  ({eur.index.min()} -> {eur.index.max()})")
print(f"GBP_USD development bars: {len(gbp)}  ({gbp.index.min()} -> {gbp.index.max()})")

# Align on common timestamps only (both instruments trade continuously so
# this should be almost total overlap; anything not shared is dropped).
common_idx = eur.index.intersection(gbp.index)
eur = eur.loc[common_idx]
gbp = gbp.loc[common_idx]
print(f"Common aligned bars: {len(common_idx)}")
print()

df = pd.DataFrame(index=common_idx)
df["eur_close"] = eur["Close"]
df["gbp_close"] = gbp["Close"]
df["eur_bid"] = eur["Bid_Close"]
df["eur_ask"] = eur["Ask_Close"]
df["gbp_bid"] = gbp["Bid_Close"]
df["gbp_ask"] = gbp["Ask_Close"]

df["ret_eur_T"] = np.log(df["eur_close"]) - np.log(df["eur_close"].shift(1))
df["ret_gbp_T"] = np.log(df["gbp_close"]) - np.log(df["gbp_close"].shift(1))
df["ret_gbp_Tplus1"] = np.log(df["gbp_close"].shift(-1)) - np.log(df["gbp_close"])
df["ret_eur_Tplus1"] = np.log(df["eur_close"].shift(-1)) - np.log(df["eur_close"])

df = df.dropna(subset=["ret_eur_T", "ret_gbp_T", "ret_gbp_Tplus1", "ret_eur_Tplus1"]).copy()
n = len(df)
print(f"Usable rows after dropping NaN warmup/tail bars: {n}")
print()


# ---------------------------------------------------------------------------
# Stats helpers (no scipy/statsmodels available in this venv - implemented
# directly; standard formulas, large-n normal approximations throughout).
# ---------------------------------------------------------------------------
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def pearson_r_and_p(x, y):
    r = np.corrcoef(x, y)[0, 1]
    n_ = len(x)
    # t-statistic for H0: rho=0; with n in the tens of thousands, t ~= z
    t_stat = r * math.sqrt(n_ - 2) / math.sqrt(max(1e-12, 1 - r ** 2))
    p = two_sided_p_from_z(t_stat)
    return r, t_stat, p


def sign_match_rate_and_p(x, y):
    mask = (x != 0) & (y != 0)
    x, y = x[mask], y[mask]
    n_ = len(x)
    matches = int((np.sign(x) == np.sign(y)).sum())
    rate = matches / n_
    z = (matches - 0.5 * n_) / math.sqrt(0.25 * n_)
    p = two_sided_p_from_z(z)
    return rate, n_, matches, z, p


def mean_t_and_p(x):
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(len(x))
    t_stat = m / se if se > 0 else 0.0
    p = two_sided_p_from_z(t_stat)
    return m, se, t_stat, p


# ---------------------------------------------------------------------------
# PRIMARY TEST: EUR_USD[T] -> GBP_USD[T+1]
# ---------------------------------------------------------------------------
print("=" * 78)
print("PRIMARY TEST: ret_EUR[T] predicting ret_GBP[T+1]")
print("=" * 78)
r, t_stat, p = pearson_r_and_p(df["ret_eur_T"].values, df["ret_gbp_Tplus1"].values)
print(f"  n = {n}")
print(f"  Pearson r = {r:.5f}   t = {t_stat:.3f}   p = {p:.4f}")
rate, n_sign, matches, z, p_sign = sign_match_rate_and_p(df["ret_eur_T"].values, df["ret_gbp_Tplus1"].values)
print(f"  Directional (sign-match) accuracy = {rate*100:.2f}%  ({matches}/{n_sign})  "
      f"vs 50% baseline: z = {z:.3f}, p = {p_sign:.4f}")
print()

# ---------------------------------------------------------------------------
# BASELINE A: GBP_USD's own lag-1 autocorrelation (the already-rejected
# single-instrument momentum mechanism) - same test shape, own history only.
# ---------------------------------------------------------------------------
print("=" * 78)
print("BASELINE A: GBP_USD's own ret[T] predicting its own ret[T+1] (single-instrument")
print("momentum - the already-rejected mechanism; included for direct comparison)")
print("=" * 78)
r_a, t_a, p_a = pearson_r_and_p(df["ret_gbp_T"].values, df["ret_gbp_Tplus1"].values)
rate_a, n_a, matches_a, z_a, p_sign_a = sign_match_rate_and_p(df["ret_gbp_T"].values, df["ret_gbp_Tplus1"].values)
print(f"  Pearson r = {r_a:.5f}   t = {t_a:.3f}   p = {p_a:.4f}")
print(f"  Directional accuracy = {rate_a*100:.2f}%  ({matches_a}/{n_a})  z = {z_a:.3f}, p = {p_sign_a:.4f}")
print()

# ---------------------------------------------------------------------------
# BASELINE B (symmetric check): reverse direction, GBP_USD[T] -> EUR_USD[T+1]
# ---------------------------------------------------------------------------
print("=" * 78)
print("BASELINE B (symmetric check): ret_GBP[T] predicting ret_EUR[T+1] (reverse direction)")
print("=" * 78)
r_b, t_b, p_b = pearson_r_and_p(df["ret_gbp_T"].values, df["ret_eur_Tplus1"].values)
rate_b, n_b, matches_b, z_b, p_sign_b = sign_match_rate_and_p(df["ret_gbp_T"].values, df["ret_eur_Tplus1"].values)
print(f"  Pearson r = {r_b:.5f}   t = {t_b:.3f}   p = {p_b:.4f}")
print(f"  Directional accuracy = {rate_b*100:.2f}%  ({matches_b}/{n_b})  z = {z_b:.3f}, p = {p_sign_b:.4f}")
print()

# ---------------------------------------------------------------------------
# STABILITY ACROSS YEARS
# ---------------------------------------------------------------------------
print("=" * 78)
print("STABILITY ACROSS YEARS (primary test, per calendar year)")
print("=" * 78)
df["year"] = df.index.year
for yr, g in df.groupby("year"):
    if len(g) < 200:
        print(f"  {yr}: n={len(g)} (too few bars, partial year at data start - skipping stats)")
        continue
    r_y, t_y, p_y = pearson_r_and_p(g["ret_eur_T"].values, g["ret_gbp_Tplus1"].values)
    rate_y, n_y, m_y, z_y, p_sign_y = sign_match_rate_and_p(g["ret_eur_T"].values, g["ret_gbp_Tplus1"].values)
    print(f"  {yr}: n={len(g):6d}  r={r_y:+.5f} (p={p_y:.4f})   "
          f"dir.acc={rate_y*100:5.2f}% (p={p_sign_y:.4f})")
print()

# ---------------------------------------------------------------------------
# COST-ADJUSTED EDGE: trade GBP_USD every bar in sign(ret_EUR[T]) direction,
# enter at T's close, exit at T+1's close. Realistic (Bid/Ask) vs
# hypothetical zero-cost (mid Close) - isolates cost drag exactly like the
# project's existing transaction-cost-isolation methodology.
# ---------------------------------------------------------------------------
print("=" * 78)
print("COST-ADJUSTED EDGE: always trade GBP_USD toward sign(ret_EUR[T]), 1 bar hold")
print("(unconditional on magnitude - no threshold to tune)")
print("=" * 78)

direction = np.sign(df["ret_eur_T"].values)  # +1 long GBP, -1 short GBP, 0 = no trade (rare)
long_mask = direction > 0
short_mask = direction < 0

gbp_bid = df["gbp_bid"].values
gbp_ask = df["gbp_ask"].values
gbp_bid_next = df["gbp_bid"].shift(-1).values
gbp_ask_next = df["gbp_ask"].shift(-1).values
gbp_close = df["gbp_close"].values
gbp_close_next = df["gbp_close"].shift(-1).values

# realistic: buy at Ask, sell at Bid (and reverse for shorts)
realistic_ret = np.full(n, np.nan)
realistic_ret[long_mask] = (gbp_bid_next[long_mask] - gbp_ask[long_mask]) / gbp_ask[long_mask]
realistic_ret[short_mask] = (gbp_bid[short_mask] - gbp_ask_next[short_mask]) / gbp_bid[short_mask]

# hypothetical zero-cost: mid Close to mid Close
hyp_ret = np.full(n, np.nan)
hyp_ret[long_mask] = (gbp_close_next[long_mask] - gbp_close[long_mask]) / gbp_close[long_mask]
hyp_ret[short_mask] = (gbp_close[short_mask] - gbp_close_next[short_mask]) / gbp_close[short_mask]

valid = ~np.isnan(realistic_ret) & ~np.isnan(hyp_ret)
realistic_ret = realistic_ret[valid]
hyp_ret = hyp_ret[valid]
n_trades = len(realistic_ret)

m_r, se_r, t_r, p_r = mean_t_and_p(realistic_ret)
m_h, se_h, t_h, p_h = mean_t_and_p(hyp_ret)
avg_gbp_spread_pips = ((df["gbp_ask"] - df["gbp_bid"]) / INSTRUMENTS["GBP_USD"].pip_size).mean()

print(f"  n trades = {n_trades}")
print(f"  Realistic (Bid/Ask, spread-inclusive) avg return/trade = {m_r*100:.5f}%  "
      f"(t={t_r:.3f}, p={p_r:.4f})")
print(f"  Hypothetical (mid Close, zero-cost)    avg return/trade = {m_h*100:.5f}%  "
      f"(t={t_h:.3f}, p={p_h:.4f})")
print(f"  Cost drag (hypothetical - realistic)                   = {(m_h-m_r)*100:.5f}%")
print(f"  Average GBP_USD spread over development window: {avg_gbp_spread_pips:.2f} pips")
print(f"  Annualised (approx, {n_trades} trades / ~3.9yr): "
      f"realistic total (if compounded additively, no sizing) = {m_r*n_trades*100:.2f}%")
print()

print("=" * 78)
print("DONE. No strategy code written, no thresholds tuned, no validation/")
print("reserved data accessed, no live connector touched.")
print("=" * 78)
