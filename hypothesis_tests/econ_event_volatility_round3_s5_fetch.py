"""
hypothesis_tests/econ_event_volatility_round3_s5_fetch.py
--------------------------------------------------------------------
Round 3 of the Economic Calendar Event Volatility hypothesis: fetches
the finest-resolution historical data OANDA's API offers (S5, 5-second
candles - confirmed available back to the start of DEVELOPMENT for
both instruments) for a targeted window around every event, resolving
round 2's open question (M15 resolution couldn't measure true
execution cost at the moment of a break).

Caches each event's window to hypothesis_tests/data/s5_cache/ - NOT
committed (see .gitignore; ~60MB, regenerable by re-running this
script against the same read-only OANDA connection, same rationale as
data_cache/). The hand-curated event DATE tables themselves
(economic_events_development.csv, economic_events_validation.csv) ARE
committed - only this raw re-fetchable candle cache is excluded.

RESULT: see econ_event_volatility_round3_s5_simulation.py and
results/hypothesis4_econ_event_volatility_summary.json ("round_3") for
the decisive outcome - REJECTED, cleanly, confirmed out-of-sample.

Read-only. Caches each event's
window to disk so this never needs re-fetching. Pre-event range/ATR
still come from the existing, already-cached M15 data (unchanged) -
only the post-event execution window needs finer resolution, which is
exactly what round 2's spread-resolution check identified as the gap.
"""
import sys, os, time
sys.path.insert(0, "/Users/user/Projects/Mistry")
from zoneinfo import ZoneInfo
import pandas as pd
from data_fetch import get_client, _candles_to_dataframe
from instruments import INSTRUMENTS
import oandapyV20.endpoints.instruments as instruments_api

NY_TZ = ZoneInfo("America/New_York")
CACHE_DIR = "/Users/user/Projects/Mistry/hypothesis_tests/data/s5_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
PRE_PAD_MIN = 15
POST_PAD_MIN = 125  # 2h post-event window + 5min buffer

client = get_client()


def fetch_window(oanda_symbol, from_ts, to_ts):
    params = {"granularity": "S5", "price": "BAM", "count": 5000,
              "from": from_ts.strftime("%Y-%m-%dT%H:%M:%SZ")}
    all_candles = []
    for _ in range(10):
        req = instruments_api.InstrumentsCandles(instrument=oanda_symbol, params=params)
        for attempt in range(4):
            try:
                client.request(req)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        page = [c for c in req.response.get("candles", []) if c.get("complete", True)]
        if not page:
            break
        all_candles.extend(page)
        last_time = pd.to_datetime(page[-1]["time"])
        if last_time >= to_ts or len(page) < 5000:
            break
        params = {"granularity": "S5", "price": "BAM", "count": 5000,
                  "from": last_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "includeFirst": False}
        time.sleep(0.2)
    df = _candles_to_dataframe(all_candles)
    if df.empty:
        return df
    return df[(df.index >= from_ts) & (df.index <= to_ts)]


def load_event_table(path):
    events = pd.read_csv(path)
    events["event_time_utc"] = events.apply(
        lambda r: pd.Timestamp(f"{r['date']} {r['local_time']}").tz_localize(NY_TZ).tz_convert("UTC"), axis=1)
    return events


for table_name, path in [
    ("DEVELOPMENT", "/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_development.csv"),
    ("VALIDATION", "/Users/user/Projects/Mistry/hypothesis_tests/data/economic_events_validation.csv"),
]:
    events = load_event_table(path)
    print(f"{table_name}: {len(events)} events")
    for symbol in ["EUR_USD", "GBP_USD"]:
        oanda_symbol = INSTRUMENTS[symbol].oanda_symbol
        fetched, cached_hit = 0, 0
        for _, ev in events.iterrows():
            t0 = ev["event_time_utc"]
            cache_file = os.path.join(CACHE_DIR, f"{symbol}_{ev['date']}_{ev['event_type']}.csv")
            if os.path.exists(cache_file):
                cached_hit += 1
                continue
            from_ts = t0 - pd.Timedelta(minutes=PRE_PAD_MIN)
            to_ts = t0 + pd.Timedelta(minutes=POST_PAD_MIN)
            df = fetch_window(oanda_symbol, from_ts, to_ts)
            df.to_csv(cache_file)
            fetched += 1
        print(f"  {symbol}: {fetched} fetched, {cached_hit} already cached")

print("DONE fetching S5 windows.")
