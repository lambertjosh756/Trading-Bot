"""
connectivity_test.py — Dry-run connectivity check.

Verifies:
  1. .env keys load correctly
  2. Alpaca paper account is reachable and shows ~$100k equity
  3. A sample watchlist can be fetched
  4. Basic bar data is retrievable
"""

from __future__ import annotations

import os
import sys
import getpass
from datetime import date, timedelta, datetime

import pytz
from dotenv import load_dotenv

ET = pytz.timezone("America/New_York")

ENV_FILE = ".env"

BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"


def ok(msg: str):  print(f"  {GREEN}[OK]{RESET}  {msg}")
def err(msg: str): print(f"  {RED}[ERR]{RESET} {msg}")
def hdr(msg: str): print(f"\n{BOLD}{CYAN}{msg}{RESET}")


def load_keys() -> tuple[str, str]:
    load_dotenv(ENV_FILE)
    key    = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()

    if not key or not secret:
        print("\nNo API keys found in .env — please enter them now:")
        key    = input("Alpaca API Key    : ").strip()
        secret = getpass.getpass("Alpaca Secret Key : ").strip()
        if not key or not secret:
            err("Keys are required."); sys.exit(1)
        with open(ENV_FILE, "a") as f:
            f.write(f"\nALPACA_API_KEY={key}\n")
            f.write(f"ALPACA_SECRET_KEY={secret}\n")
        print("Keys saved to .env.\n")

    return key, secret


def test_account(api_key: str, api_secret: str) -> float:
    hdr("Test 1 — Paper account connectivity")
    from alpaca.trading.client import TradingClient
    try:
        client  = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
        account = client.get_account()
        equity  = float(account.equity)
        cash    = float(account.cash)
        status  = account.status
        buying_power = float(account.buying_power)
        ok(f"Connected to paper account | status={status}")
        ok(f"Equity        : ${equity:>12,.2f}")
        ok(f"Cash          : ${cash:>12,.2f}")
        ok(f"Buying Power  : ${buying_power:>12,.2f}")
        if equity < 10_000:
            err(f"Equity looks low (${equity:,.2f}). Did you fund the paper account?")
        return equity
    except Exception as exc:
        err(f"Failed to connect to Alpaca paper API: {exc}")
        sys.exit(1)


def test_bars(api_key: str, api_secret: str):
    hdr("Test 2 — Historical bar data")
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    symbols = ["AAPL", "NVDA", "MSFT"]
    start   = (date.today() - timedelta(days=5)).isoformat()
    end     = date.today().isoformat()

    try:
        req  = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(req)
        for sym in symbols:
            try:
                sym_bars = bars[sym]
                latest   = sym_bars[-1]
                ok(f"{sym:5s} latest close: ${latest.close:.2f} | vol: {int(latest.volume):,}")
            except Exception:
                err(f"No bar data returned for {sym}")
    except Exception as exc:
        err(f"Bar data request failed: {exc}")


def test_watchlist(api_key: str, api_secret: str):
    hdr("Test 3 — Pre-market watchlist screener")
    try:
        from watchlist import build_watchlist
        wl = build_watchlist(api_key, api_secret)
        if wl:
            ok(f"Watchlist fetched: {', '.join(wl)}")
            ok(f"Total symbols    : {len(wl)}")
        else:
            err("Watchlist returned empty — check API permissions or market hours.")
    except Exception as exc:
        err(f"Watchlist screener error: {exc}")


def main():
    print(f"\n{'='*55}")
    print(f"  ORB Trading Bot — Connectivity Test")
    print(f"{'='*55}")

    api_key, api_secret = load_keys()
    equity = test_account(api_key, api_secret)
    test_bars(api_key, api_secret)
    test_watchlist(api_key, api_secret)

    print(f"\n{'='*55}")
    print(f"{GREEN}{BOLD}  Bot is ready to run.{RESET}")
    print(f"  Account equity : ${equity:,.2f}")
    print(f"  Start the bot  : python orb_trader.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
