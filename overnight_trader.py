"""
overnight_trader.py — Overnight momentum strategy live paper trading bot.

Strategy:
  - Capital : $50,000 (tracked separately from ORB)
  - Universe : S&P 500 stocks, price > $10, avg daily vol > 2M
  - Selection: Top 5 by 20-day momentum, equal-weight ($10,000 each)
  - Entry    : Every trading day at 15:30 ET (NOT Friday)
  - Exit     : Next morning at 09:35 ET
  - Filters  : No earnings within 2 days | No weekend holds

Run alongside orb_trader.py — non-overlapping windows:
  ORB runs 09:15–13:30 | Overnight runs 15:30–09:35

Usage:
  python overnight_trader.py
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from logger import get_logger
from momentum_universe import (
    get_filtered_universe,
    rank_by_momentum,
    has_earnings_soon,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ET               = pytz.timezone("America/New_York")
ENV_FILE         = ".env"
STATE_FILE       = ".overnight_state.json"
CAPITAL          = 50_000.0      # total overnight allocation
POSITION_CAPITAL = 10_000.0      # per-position allocation
TOP_N            = 5             # max positions

logger = get_logger("overnight_trader", log_prefix="overnight")


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load persistent overnight state from JSON file."""
    if not os.path.exists(STATE_FILE):
        return {"date": None, "symbols": [], "entries": {}}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"STATE: Could not load state file — {exc}")
        return {"date": None, "symbols": [], "entries": {}}


def save_state(symbols: List[str], entries: Dict[str, dict]) -> None:
    """Persist current overnight positions to JSON file."""
    state = {
        "date":    date.today().isoformat(),
        "symbols": symbols,
        "entries": entries,
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        logger.warning(f"STATE: Could not save state — {exc}")


# ── Main Bot ──────────────────────────────────────────────────────────────────

class OvernightTrader:

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,
        )

        from alpaca.data.historical import StockHistoricalDataClient
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        # Running stats
        self.stats = {
            "days":       0,
            "trades":     0,
            "wins":       0,
            "losses":     0,
            "net_pnl":    0.0,
            "start_equity": 0.0,
        }

    # ── Account ───────────────────────────────────────────────────────────────

    def validate_account(self) -> float:
        account      = self.trading_client.get_account()
        equity       = float(account.equity)
        buying_power = float(account.buying_power)
        logger.info(
            f"ACCOUNT: Paper confirmed | Equity: ${equity:,.2f} | "
            f"Buying power: ${buying_power:,.2f}"
        )
        self.stats["start_equity"] = equity
        return equity

    def get_open_positions(self) -> Dict[str, dict]:
        """Return {symbol: {qty, entry_price, pnl}} for all open positions."""
        result: Dict[str, dict] = {}
        try:
            positions = self.trading_client.get_all_positions()
            for p in positions:
                result[p.symbol] = {
                    "qty":         abs(int(float(p.qty))),
                    "entry_price": float(p.avg_entry_price),
                    "pnl":         float(p.unrealized_pl),
                    "current":     float(p.current_price),
                }
        except Exception as exc:
            logger.error(f"POSITIONS: Could not fetch — {exc}")
        return result

    # ── Exit positions ─────────────────────────────────────────────────────────

    def sell_positions(self, symbols: List[str]) -> Dict[str, float]:
        """
        Market-sell each symbol in `symbols`.
        Returns {symbol: pnl} from position data before selling.
        """
        if not symbols:
            logger.info("EXIT: No overnight positions to close.")
            return {}

        # Snapshot P&L before we close
        open_pos = self.get_open_positions()
        pnl_map: Dict[str, float] = {}

        for sym in symbols:
            pos = open_pos.get(sym)
            if pos is None:
                logger.warning(f"EXIT: {sym} — no open position found, skipping.")
                continue
            try:
                self.trading_client.close_position(sym)
                pnl     = pos["pnl"]
                pnl_pct = (pos["current"] - pos["entry_price"]) / pos["entry_price"] * 100
                sign    = "+" if pnl >= 0 else ""
                logger.info(
                    f"SELL: {sym} {pos['qty']} shares @ ${pos['current']:.2f} | "
                    f"Entry: ${pos['entry_price']:.2f} | "
                    f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
                )
                pnl_map[sym] = pnl
                self.stats["net_pnl"] += pnl
                if pnl > 0:
                    self.stats["wins"] += 1
                else:
                    self.stats["losses"] += 1
            except Exception as exc:
                logger.error(f"EXIT: Failed to close {sym} — {exc}")

        return pnl_map

    def close_all_overnight_positions(self, state: dict) -> None:
        """Safety cleanup: close any position in the state file."""
        symbols = state.get("symbols", [])
        if not symbols:
            return
        logger.info(f"CLEANUP: Closing overnight positions: {', '.join(symbols)}")
        self.sell_positions(symbols)

    # ── Entry positions ────────────────────────────────────────────────────────

    def buy_positions(self, candidates: List[Tuple[str, float]]) -> Dict[str, dict]:
        """
        Market-buy top candidates. Equal-weight at $10,000 each.
        Returns {symbol: {qty, price_est}} for state file.
        """
        entries: Dict[str, dict] = {}

        if not candidates:
            logger.warning("ENTRY: No candidates passed filters — no positions taken.")
            return entries

        for sym, momentum_pct in candidates[:TOP_N]:
            try:
                # Estimate share price from latest bar for sizing
                from alpaca.data.requests import StockLatestBarRequest
                latest = self.data_client.get_stock_latest_bar(
                    StockLatestBarRequest(symbol_or_symbols=[sym], feed="iex")
                )
                price = latest[sym].close
                qty   = max(1, int(POSITION_CAPITAL / price))

                req = MarketOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                self.trading_client.submit_order(req)

                logger.info(
                    f"BUY: {sym} {qty} shares @ ~${price:.2f} | "
                    f"20d momentum: {momentum_pct:+.1f}%"
                )
                entries[sym] = {"qty": qty, "price_est": price}
                self.stats["trades"] += 1

            except Exception as exc:
                logger.error(f"ENTRY: Failed to buy {sym} — {exc}")

        return entries

    # ── Momentum selection ─────────────────────────────────────────────────────

    def select_candidates(
        self,
        ranked: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        From the momentum-ranked list, remove stocks with earnings soon.
        Return top 5 (or fewer if filtered).
        """
        candidates: List[Tuple[str, float]] = []
        skipped_earnings: List[str] = []

        for sym, ret in ranked:
            if len(candidates) >= TOP_N:
                break
            if has_earnings_soon(sym, days=2):
                skipped_earnings.append(sym)
                continue
            candidates.append((sym, ret))

        if skipped_earnings:
            logger.info(f"EARNINGS FILTER: Skipped {', '.join(skipped_earnings)} (earnings within 2 days)")

        return candidates

    # ── Scheduler helpers ──────────────────────────────────────────────────────

    async def wait_until(self, hour: int, minute: int) -> None:
        """Sleep until target ET time today (if already past, return immediately)."""
        while True:
            now    = datetime.now(ET)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                return
            wait_secs = (target - now).total_seconds()
            logger.info(f"SCHEDULER: Waiting {wait_secs:.0f}s until {hour:02d}:{minute:02d} ET…")
            await asyncio.sleep(min(wait_secs, 60))

    async def wait_until_tomorrow(self, hour: int, minute: int) -> None:
        """Sleep until a specific time on the next calendar day (ET)."""
        now    = datetime.now(ET)
        target = (now + timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        wait_secs = (target - now).total_seconds()
        logger.info(
            f"SCHEDULER: Waiting {wait_secs/3600:.1f}h until "
            f"tomorrow {hour:02d}:{minute:02d} ET…"
        )
        # Sleep in 60-second chunks so we can log progress
        while True:
            now       = datetime.now(ET)
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 60))

    # ── Morning exit ──────────────────────────────────────────────────────────

    async def morning_exit(self, state: dict) -> None:
        """09:35 ET — sell all overnight positions from prior day."""
        symbols = state.get("symbols", [])
        if not symbols:
            return

        now_et = datetime.now(ET)
        exit_time = now_et.replace(hour=9, minute=35, second=0, microsecond=0)

        if now_et < exit_time:
            logger.info(
                f"MORNING: Found {len(symbols)} overnight positions. "
                f"Waiting for 09:35 ET to exit."
            )
            await self.wait_until(9, 35)

        logger.info(f"MORNING EXIT: 09:35 ET — closing {len(symbols)} positions: {', '.join(symbols)}")
        pnl_map = self.sell_positions(symbols)

        total_pnl = sum(pnl_map.values())
        sign      = "+" if total_pnl >= 0 else ""
        logger.info(f"MORNING SUMMARY: Net overnight P&L: {sign}${total_pnl:.2f} | Closed: {len(pnl_map)}/{len(symbols)}")

        save_state([], {})
        self.stats["days"] += 1

    # ── EOD summary ───────────────────────────────────────────────────────────

    def write_eod_summary(self, entries: Dict[str, dict]) -> None:
        try:
            equity = float(self.trading_client.get_account().equity)
        except Exception:
            equity = 0.0

        s     = self.stats
        total = s["wins"] + s["losses"]
        wr    = (s["wins"] / total * 100) if total else 0.0

        logger.info(
            f"\n{'='*60}\n"
            f"OVERNIGHT ENTRY SUMMARY — {date.today()}\n"
            f"{'='*60}\n"
            f"  Positions entered : {len(entries)}\n"
            f"  Symbols           : {', '.join(entries.keys())}\n"
            f"  Running P&L       : ${s['net_pnl']:+.2f}\n"
            f"  Total days        : {s['days']}\n"
            f"  Win rate          : {wr:.1f}% ({s['wins']}W / {s['losses']}L)\n"
            f"  Account equity    : ${equity:,.2f}\n"
            f"{'='*60}"
        )

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        self.validate_account()

        now_et = datetime.now(ET)
        logger.info(f"BOT START: {now_et.strftime('%H:%M:%S ET on %A %Y-%m-%d')}")

        # ── Step 1: Check for prior-day positions and exit at morning ─────────
        state = load_state()
        if state.get("symbols"):
            logger.info(
                f"STATE: Found positions from {state.get('date')}: "
                f"{', '.join(state['symbols'])}"
            )
            await self.morning_exit(state)

        # ── Continuous daily loop ─────────────────────────────────────────────
        while True:
            now_et    = datetime.now(ET)
            weekday   = now_et.weekday()   # 0=Mon … 4=Fri … 6=Sun

            # Skip Friday entries (no weekend holds)
            if weekday == 4:  # Friday
                logger.info("SCHEDULER: Friday — skipping overnight entry. Resuming Monday morning.")
                # Sleep until Monday 09:30 ET
                days_until_monday = 3
                target = (now_et + timedelta(days=days_until_monday)).replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                wait_secs = (target - now_et).total_seconds()
                logger.info(f"SCHEDULER: Sleeping {wait_secs/3600:.1f}h until Monday 09:30 ET…")
                await asyncio.sleep(max(wait_secs, 1))
                continue

            # Skip weekends (shouldn't happen if we sleep correctly, but guard)
            if weekday >= 5:
                await asyncio.sleep(3600)
                continue

            # ── 15:25 ET: Fetch universe data ─────────────────────────────────
            await self.wait_until(15, 25)
            logger.info("UNIVERSE: Fetching 20-day price data…")
            universe, bars_data = get_filtered_universe(
                self.api_key, self.api_secret, logger
            )

            # ── 15:28 ET: Rank by momentum ────────────────────────────────────
            await self.wait_until(15, 28)
            ranked = rank_by_momentum(bars_data, universe)

            logger.info("MOMENTUM RANK (top 10):")
            for sym, ret in ranked[:10]:
                logger.info(f"  {sym:6s}  {ret:+.1f}%")

            candidates = self.select_candidates(ranked)

            if not candidates:
                logger.warning("ENTRY: All top candidates filtered out — no positions today.")
                # Still wait overnight and check morning
                await self.wait_until_tomorrow(9, 35)
                continue

            logger.info(
                f"SELECTED: {', '.join(f'{s} ({r:+.1f}%)' for s, r in candidates)}"
            )

            # ── 15:30 ET: Safety cleanup + buy ───────────────────────────────
            await self.wait_until(15, 30)

            # Safety: close any leftover overnight positions not yet cleared
            state = load_state()
            self.close_all_overnight_positions(state)

            entries = self.buy_positions(candidates)
            if entries:
                save_state(list(entries.keys()), entries)
                self.write_eod_summary(entries)
            else:
                logger.warning("ENTRY: No orders placed — see errors above.")

            # ── 09:35 ET next morning: exit ───────────────────────────────────
            await self.wait_until_tomorrow(9, 35)

            # Check if today (next morning) is Monday — positions carried from Friday
            now_et  = datetime.now(ET)
            state   = load_state()
            symbols = state.get("symbols", [])

            if symbols:
                logger.info(f"MORNING EXIT: 09:35 ET — closing {len(symbols)} positions: {', '.join(symbols)}")
                pnl_map   = self.sell_positions(symbols)
                total_pnl = sum(pnl_map.values())
                sign      = "+" if total_pnl >= 0 else ""
                logger.info(f"MORNING SUMMARY: Net overnight P&L: {sign}${total_pnl:.2f}")
                save_state([], {})
                self.stats["days"] += 1


# ── Key management ────────────────────────────────────────────────────────────

def load_or_prompt_keys() -> Tuple[str, str]:
    """Load API keys from .env; prompt user if missing."""
    load_dotenv(ENV_FILE)
    api_key    = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_SECRET_KEY", "").strip()

    if not api_key or not api_secret:
        print("\n── Alpaca Paper API Setup ──────────────────────────────────")
        print("Keys will be saved to .env for future runs.\n")
        api_key    = input("Enter your Alpaca API Key    : ").strip()
        api_secret = getpass.getpass("Enter your Alpaca Secret Key : ").strip()
        if not api_key or not api_secret:
            print("Error: both keys are required.")
            sys.exit(1)
        with open(ENV_FILE, "a") as f:
            f.write(f"\nALPACA_API_KEY={api_key}\n")
            f.write(f"ALPACA_SECRET_KEY={api_secret}\n")
        print(f"Keys saved to {ENV_FILE}.\n")

    return api_key, api_secret


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key, api_secret = load_or_prompt_keys()
    bot = OvernightTrader(api_key=api_key, api_secret=api_secret)
    asyncio.run(bot.run())
