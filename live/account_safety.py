"""
account_safety.py
-------------------
Hard, mandatory verification that we are connected to an OANDA PRACTICE
(demo) account - never live. Runs first, every single time run_live.py
starts, with no bypass flag, argument, or configuration option that can
skip it.

Three independent layers, ALL must pass:
  1. The environment is hardcoded to "practice" in oanda_live_client.py -
     not read from any configurable source, so it can't be silently
     flipped to "live" by an env var, typo, or future edit elsewhere.
  2. The account ID must start with "101-" - OANDA's documented prefix
     convention for practice accounts (live accounts start "001-").
  3. The account must actually respond when queried through the
     practice-server client - proving the credentials genuinely resolve
     to an account that exists on OANDA's practice infrastructure. This
     matters because practice and live are structurally separate OANDA
     systems (different servers, different account pools) - it is not
     just a flag on one shared system, so a live account literally
     cannot pass this check even if layers 1-2 were somehow bypassed.

If ANY layer fails, this raises and the whole program refuses to start.
"""

from oanda_live_client import get_live_client, OANDA_ENVIRONMENT, OANDA_ACCOUNT_ID
import oandapyV20.endpoints.accounts as accounts_api


class NotADemoAccountError(RuntimeError):
    """Raised when any layer of the demo-account verification fails.
    Never caught anywhere except main() printing it and exiting - this
    must always halt the program, never be silently handled."""


def verify_practice_account():
    """Run all three verification layers in order. Raises
    NotADemoAccountError if any of them fail. Call this before doing
    ANYTHING else, every single time run_live.py starts."""

    # --- Layer 1: hardcoded environment ---
    if OANDA_ENVIRONMENT != "practice":
        raise NotADemoAccountError(
            f"OANDA_ENVIRONMENT is '{OANDA_ENVIRONMENT}', not 'practice'. This is "
            "a hardcoded safety constant in oanda_live_client.py - if you see this "
            "error, someone edited it. Refusing to start."
        )

    # --- Layer 2: account ID prefix convention ---
    if not OANDA_ACCOUNT_ID:
        raise NotADemoAccountError(
            "OANDA_ACCOUNT_ID is not set. Add it to .env - see .env.example."
        )
    if not OANDA_ACCOUNT_ID.startswith("101-"):
        raise NotADemoAccountError(
            f"OANDA_ACCOUNT_ID '{OANDA_ACCOUNT_ID}' does not start with '101-', "
            "OANDA's documented prefix for practice accounts (live accounts start "
            "'001-'). Refusing to start - this could be a live account."
        )

    # --- Layer 3: the account must actually exist on the practice server ---
    client = get_live_client()
    request = accounts_api.AccountDetails(accountID=OANDA_ACCOUNT_ID)
    try:
        client.request(request)
    except Exception as e:
        raise NotADemoAccountError(
            f"Could not fetch account details from OANDA's PRACTICE server for "
            f"account {OANDA_ACCOUNT_ID}: {e}. If this were actually a live "
            f"account, it would fail exactly like this here, because live "
            f"accounts don't exist on the practice server. Refusing to start."
        )

    account_info = request.response.get("account", {})
    currency = account_info.get("currency", "?")
    balance = account_info.get("balance", "?")

    print("=" * 78)
    print("DEMO ACCOUNT VERIFIED (all 3 safety checks passed):")
    print(f"  Environment: {OANDA_ENVIRONMENT} (hardcoded, not configurable)")
    print(f"  Account ID:  {OANDA_ACCOUNT_ID} (starts with '101-')")
    print(f"  Account responds on OANDA's practice server")
    print(f"  Currency: {currency}, Balance: {balance}")
    print("=" * 78)
    return account_info
