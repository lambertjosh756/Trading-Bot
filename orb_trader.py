"""
orb_trader.py — 5-minute Opening Range Breakout (ORB) live paper trading bot.

Strategy:
  - Opening range = first 5-min candle after 9:30 ET
  - Entry: breakout above ORB high, volume > 1.5x 20-period avg, RSI(14) 40–60
  - Profit target : +2.5% above entry
  - Stop loss     : -0.3% below entry
  - Time exit     : 13:30 ET — force close all positions
  - Max positions : 8 (equal-weight sizing)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import getpass
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv, set_key

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from logger import get_logger, log_trade, log_exit, log_signal
from watchlist import build_watchlist, BLACKLIST

# ── Constants ────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
BASE_URL         = "https://paper-api.alpaca.markets"
DATA_URL         = "https://data.alpaca.markets"
MAX_POSITIONS    = 8
TARGET_CAPITAL   = 75_000.0    # Reduced from 100k — overnight bot holds
                                # up to $50k until 09:35, this keeps
                                # combined exposure under $125k cash balance
VOL_MULTIPLIER   = 1.5       # volume filter threshold
RSI_LOW          = 40.0
RSI_HIGH         = 60.0
PROFIT_PCT       = 0.025     # 2.5%
STOP_PCT         = 0.003     # 0.3%
RSI_PERIOD       = 14
VOL_AVG_PERIOD   = 20
ENV_FILE         = ".env"

logger = get_logger()


# ── RSI helper ───────────────────────────────────────────────────────────────

def calc_rsi(closes: list[float], period: int = 14) -> float:
    """Wilder RSI from a list of close prices (most recent last)."""
    if len(closes) < period + 1:
        return 50.0   # neutral default when insufficient data
    deltas = np.diff(closes[-period - 1:])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── Main Bot ─────────────────────────────────────────────────────────────────

class ORBTrader:

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

        # State
        self.watchlist:       List[str]            = []
        self.orb_high:        Dict[str, float]     = {}
        self.orb_low:         Dict[str, float]     = {}
        self.orb_recorded:    Dict[str, bool]      = {}
        self.orb_candles:     Dict[str, list]      = defaultdict(list)  # raw bars during 9:30-9:35

        # Per-symbol rolling bars for volume + RSI
        self.bar_closes:      Dict[str, deque]     = defaultdict(lambda: deque(maxlen=RSI_PERIOD + 5))
        self.bar_volumes:     Dict[str, deque]     = defaultdict(lambda: deque(maxlen=VOL_AVG_PERIOD + 5))

        # Active trades
        self.active_symbols:  Dict[str, dict]      = {}   # symbol -> trade info
        self.orders_placed:   set                  = set()

        # Daily stats
        self.stats = {
            "signals":    0,
            "trades":     0,
            "wins":       0,
            "losses":     0,
            "net_pnl":    0.0,
            "best_trade": None,
            "worst_trade": None,
            "start_equity": 0.0,
            "end_equity":   0.0,
        }

    # ── Account helpers ───────────────────────────────────────────────────────

    def validate_account(self) -> float:
        """Confirm paper account and return current equity."""
        account       = self.trading_client.get_account()
        equity        = float(account.equity)
        cash          = float(account.cash)
        buying_power  = float(account.buying_power)
        logger.info(
            f"ACCOUNT: Paper trading confirmed | "
            f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Buying Power: ${buying_power:,.2f}"
        )
        self.stats["start_equity"] = equity
        return equity

    def get_buying_power(self) -> float:
        return float(self.trading_client.get_account().buying_power)

    # ── Watchlist ─────────────────────────────────────────────────────────────

    async def run_watchlist_fetch(self):
        logger.info("WATCHLIST: Fetching pre-market screener at 9:15 ET…")
        self.watchlist = build_watchlist(self.api_key, self.api_secret)
        if not self.watchlist:
            logger.warning("WATCHLIST: No symbols selected — using fallback list.")
            fallback = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD",
                        "META", "AMZN", "GOOGL", "NFLX", "CRM",
                        "PLTR", "COIN", "SOFI", "MSTR"]
            blacklist_set = set(BLACKLIST)
            self.watchlist = [s for s in fallback if s not in blacklist_set]
            logger.info(f"WATCHLIST: Fallback list ({len(self.watchlist)} symbols, blacklist applied)")
        # Initialise ORB tracking
        for sym in self.watchlist:
            self.orb_recorded[sym] = False

    # ── Pre-load historical bars for RSI/volume baseline ─────────────────────

    def load_historical_bars(self):
        """Load ~25 1-min bars per symbol so RSI and vol avg are warm at open."""
        if not self.watchlist:
            return
        today = date.today()
        # go back far enough to get intraday data from yesterday
        start = (datetime.now(ET) - timedelta(days=2)).isoformat()
        end   = datetime.now(ET).isoformat()
        try:
            req = StockBarsRequest(
                symbol_or_symbols=self.watchlist,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = self.data_client.get_stock_bars(req)
            for sym in self.watchlist:
                try:
                    sym_bars = bars[sym]
                    for b in sym_bars[-VOL_AVG_PERIOD:]:
                        self.bar_closes[sym].append(b.close)
                        self.bar_volumes[sym].append(b.volume)
                except Exception:
                    pass
            logger.info("HISTORY: Pre-loaded historical bars for RSI/volume baseline.")
        except Exception as exc:
            logger.warning(f"HISTORY: Could not pre-load bars — {exc}")

    # ── ORB recording ─────────────────────────────────────────────────────────

    def record_orb_bar(self, symbol: str, bar) -> None:
        """Accumulate bars in the 9:30–9:35 window, then set ORB."""
        bar_time = bar.timestamp
        if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is None:
            bar_time = pytz.utc.localize(bar_time)
        bar_et = bar_time.astimezone(ET)

        market_open  = bar_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        orb_end      = bar_et.replace(hour=9,  minute=35, second=0, microsecond=0)

        if bar_et < market_open:
            return

        if market_open <= bar_et < orb_end:
            self.orb_candles[symbol].append(bar)
            return

        # At 9:35 — consolidate ORB from accumulated candles
        if bar_et >= orb_end and not self.orb_recorded.get(symbol, False):
            candles = self.orb_candles.get(symbol, [])
            if candles:
                highs = [c.high  for c in candles]
                lows  = [c.low   for c in candles]
                self.orb_high[symbol] = max(highs)
                self.orb_low[symbol]  = min(lows)
                self.orb_recorded[symbol] = True
                logger.info(
                    f"ORB: {symbol} range recorded | "
                    f"High: ${self.orb_high[symbol]:.2f} | Low: ${self.orb_low[symbol]:.2f}"
                )

    # ── Signal detection ──────────────────────────────────────────────────────

    def check_signal(self, symbol: str, bar) -> bool:
        """Return True if all entry conditions are met."""
        if not self.orb_recorded.get(symbol, False):
            return False
        if symbol in self.active_symbols or symbol in self.orders_placed:
            return False
        if len(self.active_symbols) >= MAX_POSITIONS:
            return False

        close  = bar.close
        volume = bar.volume

        # Breakout condition
        orb_h = self.orb_high.get(symbol)
        if orb_h is None or close <= orb_h:
            return False

        # Volume filter
        vols = list(self.bar_volumes[symbol])
        if len(vols) < 5:
            return False
        avg_vol = sum(vols[-VOL_AVG_PERIOD:]) / min(len(vols), VOL_AVG_PERIOD)
        if avg_vol == 0 or volume < VOL_MULTIPLIER * avg_vol:
            return False
        vol_mult = volume / avg_vol

        # RSI filter
        closes = list(self.bar_closes[symbol])
        closes.append(close)
        rsi = calc_rsi(closes)
        if not (RSI_LOW <= rsi <= RSI_HIGH):
            return False

        log_signal(logger, symbol, close, orb_h, vol_mult, rsi)
        self.stats["signals"] += 1
        return True

    # ── Order submission ──────────────────────────────────────────────────────

    def submit_bracket_order(self, symbol: str, entry_price: float) -> None:
        """Calculate position size and submit bracket order."""
        try:
            account        = self.trading_client.get_account()
            buying_power   = float(account.buying_power)
            # Cap at TARGET_CAPITAL so margin doesn't inflate position sizes.
            # Alpaca paper can show 2-4x buying power; we only want to deploy
            # $100k across 8 slots = $12,500 each.
            effective_bp   = min(buying_power, TARGET_CAPITAL)
            alloc = effective_bp / MAX_POSITIONS
            qty   = max(1, int(alloc / entry_price))

            target_price = round(entry_price * (1 + PROFIT_PCT), 2)
            stop_price   = round(entry_price * (1 - STOP_PCT),   2)

            # Market buy
            buy_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class="bracket",
                take_profit={"limit_price": target_price},
                stop_loss={"stop_price": stop_price},
            )
            order = self.trading_client.submit_order(buy_req)

            self.orders_placed.add(symbol)
            self.active_symbols[symbol] = {
                "qty":    qty,
                "entry":  entry_price,
                "target": target_price,
                "stop":   stop_price,
                "order_id": str(order.id),
            }

            log_trade(logger, symbol, "BUY", qty, entry_price, target_price, stop_price)
            self.stats["trades"] += 1

        except Exception as exc:
            logger.error(f"ORDER FAILED: {symbol} — {exc}")

    # ── Bar handler (called by websocket stream) ──────────────────────────────

    async def on_bar(self, bar) -> None:
        symbol = bar.symbol

        if symbol not in self.watchlist:
            return

        # Update rolling data
        self.bar_closes[symbol].append(bar.close)
        self.bar_volumes[symbol].append(bar.volume)

        # Record ORB if in window
        self.record_orb_bar(symbol, bar)

        # Check time (force exit at 13:30)
        bar_et = bar.timestamp.astimezone(ET) if bar.timestamp.tzinfo else \
                 pytz.utc.localize(bar.timestamp).astimezone(ET)
        now_et = datetime.now(ET)
        force_exit_time = now_et.replace(hour=13, minute=30, second=0, microsecond=0)

        if now_et >= force_exit_time:
            return   # handled by time_exit()

        # Check entry signal
        if self.check_signal(symbol, bar):
            self.submit_bracket_order(symbol, bar.close)

    # ── Time exit ─────────────────────────────────────────────────────────────

    def time_exit(self) -> None:
        """13:30 ET — cancel all open orders and close all positions."""
        logger.info("TIME EXIT: 13:30 ET reached — closing all positions…")

        # Cancel open orders
        try:
            self.trading_client.cancel_orders()
            logger.info("TIME EXIT: All open orders cancelled.")
        except Exception as exc:
            logger.error(f"TIME EXIT: Failed to cancel orders — {exc}")

        # Close all positions
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                sym = pos.symbol
                qty = abs(int(float(pos.qty)))
                try:
                    self.trading_client.close_position(sym)
                    current_price = float(pos.current_price)
                    avg_entry     = float(pos.avg_entry_price)
                    pnl           = float(pos.unrealized_pl)
                    pnl_pct       = (current_price - avg_entry) / avg_entry * 100
                    log_exit(logger, sym, "TIME EXIT", current_price, pnl, pnl_pct)
                    self._record_trade_result(sym, pnl)
                except Exception as exc:
                    logger.error(f"TIME EXIT: Failed to close {sym} — {exc}")
        except Exception as exc:
            logger.error(f"TIME EXIT: Could not fetch positions — {exc}")

    # ── Track P&L ─────────────────────────────────────────────────────────────

    def _record_trade_result(self, symbol: str, pnl: float) -> None:
        self.stats["net_pnl"] += pnl
        if pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        if self.stats["best_trade"] is None or pnl > self.stats["best_trade"][1]:
            self.stats["best_trade"] = (symbol, pnl)
        if self.stats["worst_trade"] is None or pnl < self.stats["worst_trade"][1]:
            self.stats["worst_trade"] = (symbol, pnl)

        self.active_symbols.pop(symbol, None)

    # ── Daily summary ─────────────────────────────────────────────────────────

    def write_daily_summary(self) -> None:
        try:
            self.stats["end_equity"] = float(self.trading_client.get_account().equity)
        except Exception:
            pass

        s      = self.stats
        total  = s["wins"] + s["losses"]
        wr     = (s["wins"] / total * 100) if total else 0
        best   = f"{s['best_trade'][0]} ${s['best_trade'][1]:+.2f}" if s["best_trade"] else "N/A"
        worst  = f"{s['worst_trade'][0]} ${s['worst_trade'][1]:+.2f}" if s["worst_trade"] else "N/A"

        summary = (
            f"\n{'='*60}\n"
            f"DAILY SUMMARY — {date.today()}\n"
            f"{'='*60}\n"
            f"  Watchlist count : {len(self.watchlist)}\n"
            f"  Signals         : {s['signals']}\n"
            f"  Trades taken    : {s['trades']}\n"
            f"  Wins / Losses   : {s['wins']} / {s['losses']}\n"
            f"  Win rate        : {wr:.1f}%\n"
            f"  Net P&L         : ${s['net_pnl']:+.2f}\n"
            f"  Best trade      : {best}\n"
            f"  Worst trade     : {worst}\n"
            f"  Starting equity : ${s['start_equity']:,.2f}\n"
            f"  Ending equity   : ${s['end_equity']:,.2f}\n"
            f"{'='*60}"
        )
        logger.info(summary)

    # ── Scheduler helpers ─────────────────────────────────────────────────────

    async def wait_until(self, hour: int, minute: int) -> None:
        while True:
            now = datetime.now(ET)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                return
            wait_secs = (target - now).total_seconds()
            logger.info(f"SCHEDULER: Waiting {wait_secs:.0f}s until {hour:02d}:{minute:02d} ET…")
            await asyncio.sleep(min(wait_secs, 60))

    async def wait_for_time_exit(self, stream: StockDataStream) -> None:
        """Background task: sleep until 13:30 ET, then trigger time exit."""
        await self.wait_until(13, 30)
        logger.info("TIME EXIT: 13:30 ET — stopping stream and closing positions.")
        self.time_exit()
        await stream.stop_ws()

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        # Validate account
        self.validate_account()

        now_et = datetime.now(ET)
        logger.info(f"BOT START: Current time {now_et.strftime('%H:%M:%S ET')}")

        # ── 9:15 ET — fetch watchlist ────────────────────────────────────────
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 15):
            await self.wait_until(9, 15)
        await self.run_watchlist_fetch()

        # Pre-load historical bars
        self.load_historical_bars()

        # ── 9:30 ET — wait for market open ──────────────────────────────────
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):
            await self.wait_until(9, 30)
        logger.info("MARKET OPEN: Starting ORB recording and live stream.")

        # ── Stream live 1-min bars ───────────────────────────────────────────
        stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )
        stream.subscribe_bars(self.on_bar, *self.watchlist)

        # Run time-exit watcher concurrently
        asyncio.create_task(self.wait_for_time_exit(stream))

        try:
            await stream._run_forever()
        except Exception as exc:
            logger.error(f"STREAM ERROR: {exc}")
        finally:
            self.write_daily_summary()


# ── Key management ────────────────────────────────────────────────────────────

def load_or_prompt_keys() -> tuple[str, str]:
    """Load API keys from .env; prompt user if missing and save for future runs."""
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
    bot = ORBTrader(api_key=api_key, api_secret=api_secret)
    asyncio.run(bot.run())
