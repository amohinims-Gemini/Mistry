"""
hypothesis_tests/monthend_falsification.py
--------------------------------------------
Cheap empirical falsification test for V3 Candidate Hypothesis 3
(Month-End Flow Bias). Pure statistics on DEVELOPMENT data only - no
strategy code, no thresholds swept, no variant search. EUR_USD and
GBP_USD, H1.

RESULT: REJECTED. See results/hypothesis3_monthend_falsification_summary.json
for the full numeric record and README.md's "V3 candidate search" section
for the narrative writeup. Preserved here, unmodified, purely so this
exact experiment never needs to be re-run or re-derived - it is not
part of any strategy and nothing in this project imports it.

Re-runnable at any time (`python3 hypothesis_tests/monthend_falsification.py`
from the project root, inside venv) - deterministic given the same
cached data, since dataset_split.split_for_iteration() only ever
returns the DEVELOPMENT slice.

===========================================================================
FROZEN PROTOCOL (fixed BEFORE any numbers were looked at)
===========================================================================
Research question: does FX price behaviour during the final trading
day / final trading hours of each month show a repeatable directional
OR volatility bias that is materially different from ordinary trading
days?

1. Exact month-end observation window:
   - Window A ("final trading day"): the last UTC calendar date within
     each (year, month) that has any H1 bars present in the cached
     data - i.e. the actual last trading day of that month, robust to
     the calendar month-end falling on a weekend/holiday.
   - Window B ("final trading hours"): the last 4 H1 bars of that same
     final trading day (a fixed, non-tuned bar count - not searched).

2. Exact comparison/control window: every OTHER trading day in
   DEVELOPMENT that is NOT a Window-A day, using the identical
   measurement definitions (close-to-close, intraday, last-4-bars,
   realized vol). Window B's control is each ordinary day's own last 4
   H1 bars, computed the same way.

3. Direction is measured BOTH ways, both reported regardless of which
   (if either) turns out significant:
   - close-to-close: log(Close of final day's last bar) -
     log(Close of the immediately preceding trading day's last bar)
     (includes that day's own overnight/weekend-adjacent gap).
   - intraday: log(Close of final day's last bar) -
     log(Open of final day's first bar) (that day's own session only).
   Window B direction: log(Close of day's last bar) -
   log(Open of the 4th-from-last bar) (the final 4 hours' own return).

4. Volatility is measured as realized volatility: sqrt(sum of squared
   H1 log returns) within the window (whole final day, or final 4
   hours) - one fixed, standard measure, not swept across alternatives.

5. What would count as evidence strong enough to survive: a
   statistically significant (p<0.05) mean/median return AND a
   directional hit rate significantly different from 50%, in a
   CONSISTENT sign across at least 4 of the ~4 observed years (not
   just pooled), for at least one of the three return measures - AND
   the effect must survive realistic Bid/Ask transaction costs (stay
   positive and significant, not just the cost-free/mid version). A
   volatility difference alone (significant and stable across years,
   without a directional edge) would be necessary but not sufficient
   for a directional V3 - it would point toward a differently-shaped
   idea (volatility timing), not this hypothesis as stated.

6. What automatically rejects it: the sign of the mean/median return
   flips across years; the directional hit rate is statistically
   indistinguishable from 50%; the raw (cost-free) pooled effect is
   not statistically significant; realistic transaction costs erase or
   reverse a nominally significant raw edge; or there is no material,
   stable volatility difference either.

KNOWN, PRE-ACKNOWLEDGED LIMITATION (stated before running, not after):
month-end events occur ~12x/year - across the ~3.9-year DEVELOPMENT
window this gives roughly 47 events per instrument (~94 combined),
already below the project's 150-event minimum-sample convention BEFORE
any further stratification by year or direction. This was flagged as
the leading risk when Hypothesis 3 was originally proposed, ranked
weakest of the three V3 candidates for exactly this reason.

Cost-adjusted comparison uses the ONE directional choice implied by the
full POOLED combined-instrument sample's own observed mean sign (decided
once, from the whole sample, not cherry-picked per instrument/year/
window) - reported transparently as such, not presented as a
pre-committed a-priori direction.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")

import math
import numpy as np
import pandas as pd

from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration, describe_split

INSTRUMENTS_TESTED = ["EUR_USD", "GBP_USD"]
LAST_N_HOURS = 4  # fixed, not swept


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z):
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def one_sample_t(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    m = x.mean()
    med = np.median(x)
    se = x.std(ddof=1) / math.sqrt(len(x))
    t_stat = m / se if se > 0 else 0.0
    return m, med, se, t_stat, two_sided_p_from_z(t_stat)


def two_sample_t(x, y):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 0 else 0.0
    return t_stat, two_sided_p_from_z(t_stat)


def hit_rate_and_p(x, direction=1):
    x = np.asarray(x, dtype=float)
    x = x[x != 0]
    n = len(x)
    hits = int(((x * direction) > 0).sum())
    rate = hits / n if n else float("nan")
    z = (hits - 0.5 * n) / math.sqrt(0.25 * n) if n else float("nan")
    return rate, n, hits, z, two_sided_p_from_z(z) if n else float("nan")


def build_day_table(df):
    """One row per UTC calendar trading day: first Open, last Close,
    all H1 log returns (for realized vol), last-4-bar Open/Close/rets,
    and Bid/Ask at day-open/day-close for cost-adjusted estimates."""
    dates = df.index.tz_convert("UTC").date
    df = df.copy()
    df["date"] = dates
    rows = []
    for date, g in df.groupby("date"):
        g = g.sort_index()
        closes = g["Close"].values
        log_rets = np.diff(np.log(closes)) if len(closes) > 1 else np.array([])
        n_bars = len(g)
        last4 = g.iloc[-LAST_N_HOURS:] if n_bars >= LAST_N_HOURS else None
        rows.append({
            "date": date,
            "n_bars": n_bars,
            "day_open": g["Open"].iloc[0],
            "day_close": g["Close"].iloc[-1],
            "day_bid_open": g["Bid_Open"].iloc[0], "day_ask_open": g["Ask_Open"].iloc[0],
            "day_bid_close": g["Bid_Close"].iloc[-1], "day_ask_close": g["Ask_Close"].iloc[-1],
            "day_realized_vol": math.sqrt(np.sum(log_rets ** 2)) if len(log_rets) else np.nan,
            "last4_open": last4["Open"].iloc[0] if last4 is not None else np.nan,
            "last4_close": last4["Close"].iloc[-1] if last4 is not None else np.nan,
            "last4_bid_open": last4["Bid_Open"].iloc[0] if last4 is not None else np.nan,
            "last4_ask_open": last4["Ask_Open"].iloc[0] if last4 is not None else np.nan,
            "last4_bid_close": last4["Bid_Close"].iloc[-1] if last4 is not None else np.nan,
            "last4_ask_close": last4["Ask_Close"].iloc[-1] if last4 is not None else np.nan,
            "last4_realized_vol": (math.sqrt(np.sum(np.diff(np.log(last4["Close"].values)) ** 2))
                                    if last4 is not None and len(last4) > 1 else np.nan),
        })
    day_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    day_df["date"] = pd.to_datetime(day_df["date"])
    day_df["year"] = day_df["date"].dt.year
    day_df["month"] = day_df["date"].dt.month
    day_df["prev_close"] = day_df["day_close"].shift(1)
    day_df["prev_bid_close"] = day_df["day_bid_close"].shift(1)
    day_df["prev_ask_close"] = day_df["day_ask_close"].shift(1)

    is_month_end = day_df.groupby(["year", "month"])["date"].transform("max") == day_df["date"]
    day_df["is_month_end"] = is_month_end

    day_df["ret_close_to_close"] = np.log(day_df["day_close"]) - np.log(day_df["prev_close"])
    day_df["ret_intraday"] = np.log(day_df["day_close"]) - np.log(day_df["day_open"])
    day_df["ret_last4"] = np.log(day_df["last4_close"]) - np.log(day_df["last4_open"])
    return day_df.iloc[1:].reset_index(drop=True)  # drop first row (no prev_close)


print("=" * 78)
print("HYPOTHESIS 3: Month-End Flow Bias - EUR_USD & GBP_USD, H1, DEVELOPMENT ONLY")
print("=" * 78)
print(describe_split())
print(f"Window A = final trading day of month (close-to-close + intraday); "
      f"Window B = final {LAST_N_HOURS}h of that day. Control = all other days.")
print()

tables = {}
for symbol in INSTRUMENTS_TESTED:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "H1")
    dev, _validation_not_used = split_for_iteration({symbol: raw})
    del _validation_not_used
    df = dev[symbol].sort_index()
    day_df = build_day_table(df)
    day_df["symbol"] = symbol
    tables[symbol] = day_df
    n_me = day_df["is_month_end"].sum()
    print(f"{symbol}: {len(df)} H1 bars -> {len(day_df)} trading days, {n_me} month-end days, "
          f"{len(day_df)-n_me} control days")

combined = pd.concat(tables.values(), ignore_index=True)
print(f"\nCombined: {combined['is_month_end'].sum()} month-end days, "
      f"{(~combined['is_month_end']).sum()} control days")
print()


def report_returns(label, me, ctrl):
    print(f"\n{'-'*78}\n{label} - RETURNS\n{'-'*78}")
    for col, name in [("ret_close_to_close", "close-to-close"), ("ret_intraday", "intraday"),
                       ("ret_last4", f"last-{LAST_N_HOURS}h")]:
        me_vals = me[col].dropna().values
        ctrl_vals = ctrl[col].dropna().values
        m, med, se, t, p = one_sample_t(me_vals)
        t_vs_ctrl, p_vs_ctrl = two_sample_t(me_vals, ctrl_vals)
        hit_rate, n_hit, hits, z_hit, p_hit = hit_rate_and_p(me_vals, direction=1)
        ctrl_hit_rate = (ctrl_vals > 0).mean()
        print(f"  {name:16s}: n={len(me_vals):3d}  mean={m*100:+.4f}%  median={med*100:+.4f}%  "
              f"(t-vs-0={t:.2f}, p={p:.4f})")
        print(f"  {'':16s}  vs control (n={len(ctrl_vals)}, mean={ctrl_vals.mean()*100:+.4f}%): "
              f"t={t_vs_ctrl:.2f}, p={p_vs_ctrl:.4f}")
        print(f"  {'':16s}  hit rate (%>0) = {hit_rate*100:.2f}% (p vs 50%={p_hit:.4f})   "
              f"control hit rate = {ctrl_hit_rate*100:.2f}%")


def report_vol(label, me, ctrl):
    print(f"\n{'-'*78}\n{label} - VOLATILITY (realized vol, sqrt sum sq H1 log returns)\n{'-'*78}")
    for col, name in [("day_realized_vol", "full day"), ("last4_realized_vol", f"last-{LAST_N_HOURS}h")]:
        me_vals = me[col].dropna().values
        ctrl_vals = ctrl[col].dropna().values
        m_me, m_ctrl = me_vals.mean(), ctrl_vals.mean()
        t_stat, p = two_sample_t(me_vals, ctrl_vals)
        print(f"  {name:12s}: month-end mean vol={m_me*100:.4f}%  control mean vol={m_ctrl*100:.4f}%  "
              f"ratio={m_me/m_ctrl:.3f}  (t={t_stat:.2f}, p={p:.4f})")


for symbol in INSTRUMENTS_TESTED:
    d = tables[symbol]
    me, ctrl = d[d["is_month_end"]], d[~d["is_month_end"]]
    report_returns(f"{symbol}", me, ctrl)
    report_vol(f"{symbol}", me, ctrl)

me_c, ctrl_c = combined[combined["is_month_end"]], combined[~combined["is_month_end"]]
report_returns("COMBINED (EUR_USD + GBP_USD)", me_c, ctrl_c)
report_vol("COMBINED (EUR_USD + GBP_USD)", me_c, ctrl_c)

# --- stability by year ---
print(f"\n{'='*78}\nSTABILITY BY YEAR (combined, close-to-close return)\n{'='*78}")
for yr, g in me_c.groupby(me_c["date"].dt.year):
    vals = g["ret_close_to_close"].dropna().values
    if len(vals) < 3:
        print(f"  {yr}: n={len(vals)} (too few - skipping stats)")
        continue
    m, med, se, t, p = one_sample_t(vals)
    hit_rate, n_hit, hits, z_hit, p_hit = hit_rate_and_p(vals, direction=1)
    print(f"  {yr}: n={len(vals):2d}  mean={m*100:+.4f}%  median={med*100:+.4f}%  (p={p:.4f})   "
          f"hit rate={hit_rate*100:5.1f}% (p={p_hit:.4f})")

# --- cost-adjusted, using the ONE pooled-sample observed direction ---
print(f"\n{'='*78}\nCOST-ADJUSTED EDGE (close-to-close window, direction = pooled combined-sample\n"
      f"observed mean sign, chosen once from the full sample)\n{'='*78}")
pooled_sign = 1 if me_c["ret_close_to_close"].mean() > 0 else -1
print(f"  Pooled combined mean close-to-close return sign: {'positive (long)' if pooled_sign>0 else 'negative (short)'}")

for symbol in INSTRUMENTS_TESTED + ["COMBINED"]:
    d = tables[symbol] if symbol != "COMBINED" else combined
    me = d[d["is_month_end"]].dropna(subset=["prev_bid_close", "prev_ask_close", "day_bid_close", "day_ask_close"])
    if pooled_sign > 0:
        realistic = (me["day_bid_close"].values - me["prev_ask_close"].values) / me["prev_ask_close"].values
        hyp = (me["day_close"].values - me["prev_close"].values) / me["prev_close"].values
    else:
        realistic = (me["prev_bid_close"].values - me["day_ask_close"].values) / me["prev_bid_close"].values
        hyp = (me["prev_close"].values - me["day_close"].values) / me["prev_close"].values
    m_r, med_r, se_r, t_r, p_r = one_sample_t(realistic)
    m_h, med_h, se_h, t_h, p_h = one_sample_t(hyp)
    print(f"  {symbol:10s}: n={len(realistic):3d}  realistic={m_r*100:+.4f}% (p={p_r:.4f})   "
          f"hypothetical(zero-cost)={m_h*100:+.4f}% (p={p_h:.4f})   cost drag={((m_h-m_r))*100:.4f}%")

# --- sample size assessment ---
print(f"\n{'='*78}\nSAMPLE SIZE ASSESSMENT\n{'='*78}")
print(f"  Combined month-end days: {combined['is_month_end'].sum()} "
      f"(project MIN_REQUIRED_TRADES convention: 150)")
print(f"  Per instrument: " + ", ".join(f"{s}={tables[s]['is_month_end'].sum()}" for s in INSTRUMENTS_TESTED))
print(f"  Per-year counts (combined): "
      f"{me_c.groupby(me_c['date'].dt.year).size().to_dict()}")

print(f"\n{'='*78}\nDONE. No strategy/entry-exit code, no risk-management changes, no\n"
      f"validation/reserved data access, no live-connector changes.\n{'='*78}")
