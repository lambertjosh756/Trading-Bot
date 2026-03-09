"""
swing_trader.py -- Momentum-14 swing trading bot (paper account).

Strategy (backtest-confirmed best config):
  Capital   : $75,000
  Universe  : S&P 500 stocks, price > $20, avg vol > 3M daily
  Entry     : Top-20% 20d momentum + above SMA50 + 1.5x vol today + no earnings within 5d
  Sizing    : $5,000 per position, max 15 simultaneous
  Exit      : 14 calendar days (market order at 15:45) OR -5% stop (market order)
  Scan time : 15:45 ET daily (Mon-Fri)

Capital allocation (all bots, $100k paper account):
  ORB        $75k  -- intraday only (exits by 13:30)
  Swing      $75k  -- enters at 15:45, exits on close after 14 days  [shared window with ORB: safe, non-overlapping]
  Overnight  $50k  -- enters 15:30, exits 09:35 next morning
  ETF Rot.   $25k  -- Monday 09:35 weekly rebalance
  Worst-case simultaneous: Swing $75k + Overnight $50k + ETF $25k = $150k < $200k buying power

Usage:
  python swing_trader.py          # run continuously (schedules itself at 15:45 ET)
  python swing_trader.py --scan   # dry-run: show today's candidates, no orders
  python swing_trader.py --status # show current open positions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from logger import get_logger
from swing_universe import (
    SwingCandidate,
    get_swing_universe,
    has_earnings_soon,
    spy_return_since,
)

# -- Constants -----------------------------------------------------------------
ET              = pytz.timezone("America/New_York")
ENV_FILE        = ".env"
STATE_FILE      = ".swing_state.json"

CAPITAL         = 75_000.0
POS_SIZE        = 5_000.0
MAX_POSITIONS   = 15
HOLD_DAYS       = 14        # calendar days
STOP_PCT        = 0.05      # -5% hard stop
SCAN_HOUR       = 15
SCAN_MIN        = 45

logger = get_logger("swing_trader", log_prefix="swing")


# -- State management ----------------------------------------------------------

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "positions":       {},
            "stats":           {"net_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0},
            "start_date":      None,
            "start_spy_price": None,
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"STATE: Could not load -- {exc}")
        return {
            "positions":       {},
            "stats":           {"net_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0},
            "start_date":      None,
            "start_spy_price": None,
        }


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        logger.warning(f"STATE: Could not save -- {exc}")


# -- Main bot class -------------------------------------------------------------

class SwingTrader:

    def __init__(self, api_key: str, api_secret: str):
        self.trading_client = TradingClient(
            api_key=api_key, secret_key=api_secret, paper=True
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key, secret_key=api_secret
        )
        self.api_key    = api_key
        self.api_secret = api_secret

    # -- Account ---------------------------------------------------------------

    def validate_account(self) -> Tuple[float, float]:
        account      = self.trading_client.get_account()
        equity       = float(account.equity)
        buying_power = float(account.buying_power)
        logger.info(
            f"ACCOUNT: Paper confirmed | Equity: ${equity:,.2f} | "
            f"Buying power: ${buying_power:,.2f}"
        )
        return equity, buying_power

    def get_open_positions(self) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        try:
            for p in self.trading_client.get_all_positions():
                result[p.symbol] = {
                    "qty":          abs(float(p.qty)),
                    "entry_price":  float(p.avg_entry_price),
                    "current":      float(p.current_price),
                    "pnl":          float(p.unrealized_pl),
                    "market_value": float(p.market_value),
                }
        except Exception as exc:
            logger.error(f"POSITIONS: Could not fetch -- {exc}")
        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            req  = StockLatestBarRequest(symbol_or_symbols=[symbol], feed="iex")
            resp = self.data_client.get_stock_latest_bar(req)
            return float(resp[symbol].close)
        except Exception:
            return None

    # -- Exit logic ------------------------------------------------------------

    def check_exits(self, state: dict) -> Tuple[List[str], List[str]]:
        """
        Identify positions to exit:
          - time_exits : held >= HOLD_DAYS calendar days
          - stop_exits : current price <= entry * (1 - STOP_PCT)

        Returns (time_exits, stop_exits) as lists of symbols.
        """
        positions    = state.get("positions", {})
        today        = date.today()
        time_exits   = []
        stop_exits   = []

        alpaca_pos   = self.get_open_positions()

        for sym, pos in positions.items():
            entry_date = date.fromisoformat(pos["entry_date"])
            days_held  = (today - entry_date).days
            stop_price = pos["entry_price"] * (1 - STOP_PCT)

            # Get current price (Alpaca position if available, else latest bar)
            alpaca      = alpaca_pos.get(sym)
            current_px  = alpaca["current"] if alpaca else self.get_current_price(sym)

            if current_px is None:
                continue

            if days_held >= HOLD_DAYS:
                time_exits.append(sym)
            elif current_px <= stop_price:
                stop_exits.append(sym)

        return time_exits, stop_exits

    def execute_exit(
        self, sym: str, reason: str, state: dict
    ) -> Optional[float]:
        """
        Market-sell `sym`. Updates state stats. Returns net P&L or None on failure.
        """
        pos_data    = state["positions"].get(sym, {})
        entry_price = pos_data.get("entry_price", 0.0)
        shares      = pos_data.get("shares", 0)

        # Get current price for logging before the order
        alpaca_pos = self.get_open_positions()
        ap         = alpaca_pos.get(sym)
        current_px = ap["current"] if ap else (self.get_current_price(sym) or entry_price)

        try:
            self.trading_client.close_position(sym)
        except Exception as exc:
            logger.error(f"EXIT: Failed to close {sym} -- {exc}")
            return None

        pnl       = (current_px - entry_price) * shares
        pnl_pct   = (current_px - entry_price) / entry_price * 100 if entry_price else 0
        sign      = "+" if pnl >= 0 else ""

        if reason == "time":
            logger.info(
                f"EXIT: {sym} 14-day hold complete @ ${current_px:.2f} | "
                f"Entry: ${entry_price:.2f} | "
                f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
            )
        else:
            logger.info(
                f"STOP: {sym} {pnl_pct:.1f}% stop triggered | "
                f"Entry: ${entry_price:.2f} | "
                f"Exit: ${current_px:.2f} | "
                f"P&L: {sign}${pnl:.2f}"
            )

        # Update stats
        stats = state.setdefault("stats", {})
        stats["net_pnl"]  = stats.get("net_pnl", 0.0) + pnl
        stats["trades"]   = stats.get("trades", 0) + 1
        if pnl > 0:
            stats["wins"]   = stats.get("wins", 0) + 1
        else:
            stats["losses"] = stats.get("losses", 0) + 1

        # Remove from state
        state["positions"].pop(sym, None)
        return pnl

    # -- Entry logic -----------------------------------------------------------

    def select_entries(
        self,
        candidates: List[SwingCandidate],
        state: dict,
    ) -> List[SwingCandidate]:
        """
        From the ranked candidate list, remove:
          - Symbols already held
          - Symbols with earnings within 5 days
        Return up to (MAX_POSITIONS - current_count) candidates.
        """
        held        = set(state.get("positions", {}).keys())
        n_open      = len(held)
        slots       = MAX_POSITIONS - n_open

        if slots <= 0:
            logger.info(f"ENTRY: Portfolio full ({n_open}/{MAX_POSITIONS} positions) -- no new entries.")
            return []

        selected         = []
        earnings_skipped = []

        for c in candidates:
            if len(selected) >= slots:
                break
            if c.symbol in held:
                continue
            if has_earnings_soon(c.symbol, days=5):
                earnings_skipped.append(c.symbol)
                logger.info(f"EARNINGS: Skipping {c.symbol} -- earnings within 5 days")
                continue
            selected.append(c)

        if earnings_skipped:
            logger.info(f"EARNINGS FILTER: Excluded {', '.join(earnings_skipped)}")

        return selected

    def execute_entry(self, c: SwingCandidate, state: dict) -> bool:
        """
        Market-buy `c.symbol` at ~$5,000 notional. Records position in state.
        Returns True on success.
        """
        try:
            # Get live price for share-count calculation
            live_price = self.get_current_price(c.symbol) or c.price
            shares     = max(1, int(POS_SIZE / live_price))

            req = MarketOrderRequest(
                symbol=c.symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            self.trading_client.submit_order(req)

            entry_date = date.today().isoformat()
            exit_date  = (date.today() + timedelta(days=HOLD_DAYS)).strftime("%b %d")

            logger.info(
                f"ENTRY: {c.symbol} ${POS_SIZE:,.0f} @ ${live_price:.2f} | "
                f"20d momentum: {c.momentum_pct:+.1f}% | "
                f"Exit target: {exit_date}"
            )

            state["positions"][c.symbol] = {
                "entry_date":   entry_date,
                "entry_price":  live_price,
                "shares":       shares,
                "momentum_pct": round(c.momentum_pct, 2),
            }
            return True

        except Exception as exc:
            logger.error(f"ENTRY: Failed to buy {c.symbol} -- {exc}")
            return False

    # -- EOD summary -----------------------------------------------------------

    def eod_summary(
        self,
        state: dict,
        exits_today: List[str],
        entries_today: List[str],
        daily_pnl: float,
    ) -> None:
        try:
            account = self.trading_client.get_account()
            equity  = float(account.equity)
        except Exception:
            equity = 0.0

        stats     = state.get("stats", {})
        net_pnl   = stats.get("net_pnl", 0.0)
        start_dt  = state.get("start_date")
        start_spy = state.get("start_spy_price")

        # SPY comparison
        spy_line = ""
        if start_dt and start_spy:
            spy_ret = spy_return_since(start_dt, start_spy)
            if spy_ret is not None:
                strat_ret = net_pnl / CAPITAL * 100
                alpha     = strat_ret - spy_ret
                spy_line  = (
                    f"  Strategy vs SPY   : {strat_ret:+.2f}% vs {spy_ret:+.2f}% "
                    f"(alpha {alpha:+.2f}%)\n"
                )

        positions = state.get("positions", {})
        pos_lines = ""
        alpaca_pos = self.get_open_positions()
        for sym, pos in positions.items():
            ap       = alpaca_pos.get(sym)
            curr     = ap["current"] if ap else pos["entry_price"]
            pnl      = (curr - pos["entry_price"]) * pos["shares"]
            pnl_pct  = (curr - pos["entry_price"]) / pos["entry_price"] * 100
            days_held = (date.today() - date.fromisoformat(pos["entry_date"])).days
            sign     = "+" if pnl >= 0 else ""
            pos_lines += (
                f"    {sym:<6}  entry ${pos['entry_price']:.2f}  "
                f"current ${curr:.2f}  "
                f"held {days_held}d  "
                f"P&L {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)\n"
            )

        total  = stats.get("trades", 0)
        wins   = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        wr     = wins / total * 100 if total else 0.0
        sign_d = "+" if daily_pnl >= 0 else ""
        sign_n = "+" if net_pnl >= 0 else ""

        logger.info(
            f"\n{'='*60}\n"
            f"SWING SUMMARY -- {date.today()}\n"
            f"{'='*60}\n"
            f"  Open positions    : {len(positions)}/{MAX_POSITIONS}\n"
            f"{pos_lines}"
            f"  Exits today       : {', '.join(exits_today) or 'none'}\n"
            f"  Entries today     : {', '.join(entries_today) or 'none'}\n"
            f"  Daily P&L         : {sign_d}${daily_pnl:.2f}\n"
            f"  Total P&L         : {sign_n}${net_pnl:.2f}\n"
            f"{spy_line}"
            f"  Win rate          : {wr:.1f}% ({wins}W / {losses}L)\n"
            f"  Account equity    : ${equity:,.2f}\n"
            f"{'='*60}"
        )

    # -- Scheduler helpers -----------------------------------------------------

    async def wait_until(self, hour: int, minute: int) -> None:
        while True:
            now    = datetime.now(ET)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                return
            secs = (target - now).total_seconds()
            logger.info(f"SCHEDULER: Waiting {secs/60:.1f}m until {hour:02d}:{minute:02d} ET...")
            await asyncio.sleep(min(secs, 60))

    async def sleep_until_tomorrow(self, hour: int, minute: int) -> None:
        now    = datetime.now(ET)
        target = (now + timedelta(days=1)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        secs   = (target - now).total_seconds()
        logger.info(f"SCHEDULER: Next scan in {secs/3600:.1f}h ({hour:02d}:{minute:02d} ET tomorrow)...")
        while True:
            now       = datetime.now(ET)
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 60))

    # -- Main scan cycle -------------------------------------------------------

    async def run_scan(self, state: dict, dry_run: bool = False) -> None:
        """Execute one full 15:45 scan: exits -> entries -> summary."""
        now_et = datetime.now(ET)
        logger.info(f"SCAN START: {now_et.strftime('%H:%M:%S ET on %A %Y-%m-%d')}")

        exits_today   = []
        entries_today = []
        daily_pnl     = 0.0

        # -- 1. Check exits -------------------------------------------------
        time_exits, stop_exits = self.check_exits(state)

        for sym in time_exits:
            if not dry_run:
                pnl = self.execute_exit(sym, "time", state)
                if pnl is not None:
                    exits_today.append(sym)
                    daily_pnl += pnl
            else:
                pos = state["positions"][sym]
                logger.info(f"[DRY] EXIT: {sym} 14-day hold complete | Entry: ${pos['entry_price']:.2f}")
                exits_today.append(sym)

        for sym in stop_exits:
            if not dry_run:
                pnl = self.execute_exit(sym, "stop", state)
                if pnl is not None:
                    exits_today.append(sym)
                    daily_pnl += pnl
            else:
                pos = state["positions"][sym]
                logger.info(f"[DRY] STOP: {sym} | Entry: ${pos['entry_price']:.2f}")
                exits_today.append(sym)

        # -- 2. Fetch universe and scan candidates --------------------------
        candidates, _ = get_swing_universe(self.api_key, self.api_secret, logger)

        if not candidates:
            logger.warning("SCAN: No candidates passed all filters today.")
        else:
            selected = self.select_entries(candidates, state)

            for c in selected:
                if not dry_run:
                    ok = self.execute_entry(c, state)
                    if ok:
                        entries_today.append(c.symbol)
                else:
                    logger.info(
                        f"[DRY] ENTRY: {c.symbol} ${POS_SIZE:,.0f} @ ${c.price:.2f} | "
                        f"momentum {c.momentum_pct:+.1f}% | "
                        f"vol {c.vol_mult:.1f}x | "
                        f"SMA50 ${c.sma50:.2f}"
                    )
                    entries_today.append(c.symbol)

        # -- 3. Persist state and print summary -----------------------------
        if not dry_run:
            save_state(state)
        self.eod_summary(state, exits_today, entries_today, daily_pnl)

    # -- Main run loop ---------------------------------------------------------

    async def run(self) -> None:
        equity, buying_power = self.validate_account()
        state = load_state()

        # First run: record start date + SPY price for benchmark
        if state.get("start_date") is None:
            state["start_date"] = date.today().isoformat()
            # Fetch SPY starting price
            try:
                import yfinance as yf
                spy_hist = yf.Ticker("SPY").history(period="2d", auto_adjust=True)
                if not spy_hist.empty:
                    state["start_spy_price"] = float(spy_hist["Close"].iloc[-1])
                    logger.info(f"BENCHMARK: SPY reference price set at ${state['start_spy_price']:.2f}")
            except Exception:
                pass
            save_state(state)

        logger.info(
            f"BOT START: Swing Momentum-14 | "
            f"Capital: ${CAPITAL:,.0f} | "
            f"Max positions: {MAX_POSITIONS} | "
            f"Scan: {SCAN_HOUR:02d}:{SCAN_MIN:02d} ET daily"
        )

        # Print existing positions
        n_open = len(state.get("positions", {}))
        if n_open:
            logger.info(f"STATE: Loaded {n_open} existing positions: {', '.join(state['positions'].keys())}")
        else:
            logger.info("STATE: No existing positions -- starting fresh.")

        while True:
            now_et  = datetime.now(ET)
            weekday = now_et.weekday()  # 0=Mon, 4=Fri

            if weekday >= 5:
                await asyncio.sleep(3600)
                continue

            await self.wait_until(SCAN_HOUR, SCAN_MIN)
            await self.run_scan(state)
            await self.sleep_until_tomorrow(SCAN_HOUR, SCAN_MIN)


# -- Status display -------------------------------------------------------------

def show_status(trader: SwingTrader, state: dict) -> None:
    logger.info("=" * 60)
    logger.info("SWING TRADER STATUS")
    logger.info("=" * 60)

    positions = state.get("positions", {})
    if not positions:
        logger.info("POSITIONS: None open.")
    else:
        alpaca_pos = trader.get_open_positions()
        for sym, pos in positions.items():
            ap        = alpaca_pos.get(sym)
            curr      = ap["current"] if ap else pos["entry_price"]
            pnl       = (curr - pos["entry_price"]) * pos["shares"]
            pnl_pct   = (curr - pos["entry_price"]) / pos["entry_price"] * 100
            days_held = (date.today() - date.fromisoformat(pos["entry_date"])).days
            exit_date = (date.fromisoformat(pos["entry_date"]) + timedelta(days=HOLD_DAYS)).isoformat()
            stop_px   = pos["entry_price"] * (1 - STOP_PCT)
            sign      = "+" if pnl >= 0 else ""
            logger.info(
                f"  {sym:<6} entry ${pos['entry_price']:.2f} | "
                f"current ${curr:.2f} | "
                f"held {days_held}d/{HOLD_DAYS}d | "
                f"exit {exit_date} | "
                f"stop ${stop_px:.2f} | "
                f"P&L {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
            )

    stats   = state.get("stats", {})
    net_pnl = stats.get("net_pnl", 0.0)
    total   = stats.get("trades", 0)
    wins    = stats.get("wins", 0)
    wr      = wins / total * 100 if total else 0.0
    sign_n  = "+" if net_pnl >= 0 else ""
    logger.info("-" * 60)
    logger.info(f"Total P&L : {sign_n}${net_pnl:.2f} | Trades: {total} | Win rate: {wr:.1f}%")

    start_dt  = state.get("start_date")
    start_spy = state.get("start_spy_price")
    if start_dt and start_spy:
        spy_ret = spy_return_since(start_dt, start_spy)
        if spy_ret is not None:
            strat_ret = net_pnl / CAPITAL * 100
            alpha     = strat_ret - spy_ret
            logger.info(f"vs SPY    : strategy {strat_ret:+.2f}% | SPY {spy_ret:+.2f}% | alpha {alpha:+.2f}%")

    logger.info("=" * 60)


# -- Entry point ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Swing Momentum-14 Trading Bot")
    parser.add_argument("--scan",   action="store_true", help="Dry-run scan: show candidates, no orders")
    parser.add_argument("--status", action="store_true", help="Show current positions and stats")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not api_secret:
        logger.error("INIT: ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        sys.exit(1)

    trader = SwingTrader(api_key, api_secret)
    trader.validate_account()
    state  = load_state()

    if args.status:
        show_status(trader, state)
        return

    if args.scan:
        logger.info("DRY RUN: Scanning universe -- no orders will be placed.")
        asyncio.run(trader.run_scan(state, dry_run=True))
        return

    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
