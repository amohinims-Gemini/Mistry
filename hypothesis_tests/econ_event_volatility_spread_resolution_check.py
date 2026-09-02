"""
hypothesis_tests/econ_event_volatility_spread_resolution_check.py
--------------------------------------------------------------------
Small, targeted check run between round 1 and round 2 of the Economic
Calendar Event Volatility hypothesis, to answer directly: can M15-
resolution cached data capture the true execution cost at the instant
of a news release? Compares each event bar's own Bid/Ask *Open* (close
to the release instant, since all 125 events land exactly on M15
boundaries) against the prior bar's close-based spread, and against
that bar's own realized intrabar range as a rough upper bound.

RESULT: no - Open-based spread is statistically indistinguishable from
normal quiet-market spread (ratio ~1.0-1.02), while the bar's own
realized range is 10-15x larger. A genuine spread spike lasting
seconds is averaged away between 15-minute checkpoints. This finding
directly shaped round 2's "optimistic vs. conservative" dual-scenario
design. See results/hypothesis4_econ_event_volatility_summary.json and
README.md's "Hypothesis 4" section. Preserved unmodified, not imported
by anything in the project.
"""
import sys
sys.path.insert(0, "/Users/user/Projects/Mistry")
import math
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from data_fetch import get_instrument_data
from instruments import INSTRUMENTS
from dataset_split import split_for_iteration

NY_TZ = ZoneInfo("America/New_York")
events_raw = pd.read_csv("/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_development.csv")
events_raw["event_time_utc"] = events_raw.apply(
    lambda r: pd.Timestamp(f"{r['date']} {r['local_time']}").tz_localize(NY_TZ).tz_convert("UTC"), axis=1)

for symbol in ["EUR_USD", "GBP_USD"]:
    raw = get_instrument_data(INSTRUMENTS[symbol].oanda_symbol, "M15")
    dev, _ = split_for_iteration({symbol: raw})
    df = dev[symbol].sort_index()
    pip = INSTRUMENTS[symbol].pip_size
    aligned = 0
    open_spreads, prior_close_spreads, open_high_spreads = [], [], []
    for _, ev in events_raw.iterrows():
        t0 = ev["event_time_utc"]
        if t0 not in df.index:
            continue
        aligned += 1
        row = df.loc[t0]
        prior_idx = df.index[df.index < t0]
        if len(prior_idx) == 0:
            continue
        prior_row = df.loc[prior_idx[-1]]
        open_spreads.append((row["Ask_Open"] - row["Bid_Open"]) / pip)
        prior_close_spreads.append((prior_row["Ask_Close"] - prior_row["Bid_Close"]) / pip)
        # widest spread seen anywhere within the first bar (High side proxy: Ask_High - Bid_Low as an upper bound)
        open_high_spreads.append((row["Ask_High"] - row["Bid_Low"]) / pip)
    print(f"{symbol}: {aligned}/{len(events_raw)} events land exactly on an M15 bar boundary")
    print(f"  avg spread at event bar's OPEN:            {np.mean(open_spreads):.2f} pips")
    print(f"  avg spread at PRIOR bar's close (just before): {np.mean(prior_close_spreads):.2f} pips")
    print(f"  avg max intra-bar spread proxy (Ask_High-Bid_Low) of event bar: {np.mean(open_high_spreads):.2f} pips")
    print(f"  ratio open-spread / prior-close-spread: {np.mean(open_spreads)/np.mean(prior_close_spreads):.2f}")
    print(f"  max single-event open spread: {np.max(open_spreads):.2f} pips, median: {np.median(open_spreads):.2f}")
