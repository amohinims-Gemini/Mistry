"""
live_logging.py
-------------------
One append-only JSONL file under live_data/ (gitignored) - one
structured line per cycle/instrument/event, so a completed test run can
be read back afterward as the audit trail proving the plumbing worked
end-to-end (or showing exactly where it didn't).
"""

import os
import json
import datetime as dt

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_data")
LOG_PATH = os.path.join(LOG_DIR, "run_live.jsonl")


def log_event(event_type, **fields):
    """`event_type`: e.g. "run_started", "account_state", "warning",
    "signal_checked", "order_result", "error", "stop_requested",
    "run_stopped". Timestamps its own call in UTC - callers never pass
    one. `default=str` on the dump so any stray non-JSON-native value
    (a Timestamp, a Position object, etc.) degrades to its string form
    instead of crashing logging itself."""
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event_type,
        **fields,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
