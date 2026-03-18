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
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from logger import get_logger, log_trade, log_exit, log_signal
from watchlist import build_watchlist, BLACKLIST

# ── Constants ────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
BASE_URL         = "https://paper-api.alpaca.markets"
DATA_URL         = "https://data.alpaca.markets"
MAX_POSITIONS    = 8
TARGET_CAPITAL   = 75_000.0    # 75% of $100k portfolio (shared window with Swing)
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
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Wilder's smoothed moving average: seed with SMA, then smooth remainder
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
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
        self._bar_count:      int                  = 0
        self._logged_first:   set                  = set()
        self._exit_flag:      bool                 = False
        self._exit_event:     asyncio.Event        = asyncio.Event()
        self._time_exit_done: bool                 = False
        self._seen_bar_times: Dict[str, object]    = {}

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

    # ── Late-start ORB backfill ───────────────────────────────────────────────

    def backfill_orb(self) -> None:
        """If started after 9:35 ET, fetch today's 9:30–9:35 bars and set ORB."""
        if not self.watchlist:
            return
        now_et = datetime.now(ET)
        orb_end = now_et.replace(hour=9, minute=35, second=0, microsecond=0)
        if now_et <= orb_end:
            return  # still in or before ORB window, let live stream handle it

        today = now_et.date()
        start = ET.localize(datetime(today.year, today.month, today.day, 9, 30))
        end   = ET.localize(datetime(today.year, today.month, today.day, 9, 36))
        logger.info("LATE START: Backfilling ORB from historical data…")
        try:
            req = StockBarsRequest(
                symbol_or_symbols=self.watchlist,
                timeframe=TimeFrame.Minute,
                start=start.isoformat(),
                end=end.isoformat(),
                feed="iex",
            )
            bars = self.data_client.get_stock_bars(req)
            found = 0
            for sym in self.watchlist:
                try:
                    sym_bars = bars[sym]
                    if sym_bars:
                        highs = [b.high for b in sym_bars]
                        lows  = [b.low  for b in sym_bars]
                        self.orb_high[sym]     = max(highs)
                        self.orb_low[sym]      = min(lows)
                        self.orb_recorded[sym] = True
                        found += 1
                except Exception:
                    pass
            logger.info(f"LATE START: ORB backfilled for {found}/{len(self.watchlist)} symbols.")
        except Exception as exc:
            logger.warning(f"LATE START: Could not backfill ORB — {exc}")

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
                feed="iex",
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
            logger.debug(f"SIGNAL SKIP {symbol}: ORB not recorded")
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
            logger.info(f"SIGNAL SKIP {symbol}: not enough volume history ({len(vols)} bars)")
            return False
        avg_vol = sum(vols[-VOL_AVG_PERIOD:]) / min(len(vols), VOL_AVG_PERIOD)
        if avg_vol == 0 or volume < VOL_MULTIPLIER * avg_vol:
            logger.info(f"SIGNAL SKIP {symbol}: low volume ({volume:.0f} vs {VOL_MULTIPLIER}x avg {avg_vol:.0f})")
            return False
        vol_mult = volume / avg_vol

        # RSI filter (bar.close already appended to bar_closes in on_bar)
        closes = list(self.bar_closes[symbol])
        rsi = calc_rsi(closes)
        if not (RSI_LOW <= rsi <= RSI_HIGH):
            logger.info(f"SIGNAL SKIP {symbol}: RSI {rsi:.1f} out of range [{RSI_LOW},{RSI_HIGH}]")
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
            # Alpaca requires stop >= $0.01 below entry — enforce minimum gap
            stop_price   = min(stop_price, round(entry_price - 0.01, 2))

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

        # Diagnostic: log first bar per symbol + periodic total count
        if symbol not in self._logged_first:
            orb_h = self.orb_high.get(symbol, "N/A")
            logger.info(f"STREAM: First bar — {symbol} close=${bar.close:.2f} ORB_high={orb_h}")
            self._logged_first.add(symbol)
        self._bar_count += 1
        if self._bar_count % 50 == 0:
            logger.info(f"STREAM: {self._bar_count} bars received total, {len(self.active_symbols)} active positions.")

        # Record ORB if in window
        self.record_orb_bar(symbol, bar)

        # Check time gates
        bar_et = bar.timestamp.astimezone(ET) if bar.timestamp.tzinfo else \
                 pytz.utc.localize(bar.timestamp).astimezone(ET)
        now_et = datetime.now(ET)
        force_exit_time  = now_et.replace(hour=13, minute=30, second=0, microsecond=0)
        entry_cutoff     = now_et.replace(hour=12, minute=45, second=0, microsecond=0)

        if now_et >= force_exit_time:
            return   # handled by time_exit()

        # Check entry signal — no new entries after 12:45 ET
        if now_et < entry_cutoff and self.check_signal(symbol, bar):
            self.submit_bracket_order(symbol, bar.close)

    # ── Time exit ─────────────────────────────────────────────────────────────

    def time_exit(self) -> None:
        """13:30 ET — cancel this bot's open orders and close only ORB positions."""
        if self._time_exit_done:
            return
        self._time_exit_done = True

        # Only act on positions this bot opened — never touch other bots' positions
        orb_symbols = set(self.active_symbols.keys())
        if not orb_symbols:
            logger.info("TIME EXIT: No active ORB positions to close — skipping.")
            return

        logger.info(f"TIME EXIT: 13:30 ET — closing {len(orb_symbols)} ORB position(s): {', '.join(orb_symbols)}")

        # Cancel open orders for our symbols only
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = self.trading_client.get_orders(req)
            for order in open_orders:
                if order.symbol in orb_symbols:
                    try:
                        self.trading_client.cancel_order_by_id(order.id)
                    except Exception:
                        pass
            logger.info("TIME EXIT: ORB open orders cancelled.")
        except Exception as exc:
            logger.error(f"TIME EXIT: Failed to cancel orders — {exc}")

        # Close only ORB positions
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                sym = pos.symbol
                if sym not in orb_symbols:
                    continue
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

    # ── Poll bracket exits ────────────────────────────────────────────────────

    def _check_bracket_exits(self) -> None:
        """Detect bracket orders (TP/SL) that Alpaca closed while we weren't looking."""
        if not self.active_symbols:
            return
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=50)
            closed_orders = self.trading_client.get_orders(req)
            open_syms = set(self.active_symbols.keys())
            for order in closed_orders:
                sym = order.symbol
                if sym not in open_syms:
                    continue
                side = str(order.side)
                if "sell" not in side.lower():
                    continue
                filled_price = float(order.filled_avg_price or 0)
                if filled_price == 0:
                    continue
                trade = self.active_symbols[sym]
                entry  = trade["entry"]
                qty    = trade["qty"]
                pnl    = (filled_price - entry) * qty
                pnl_pct = (filled_price - entry) / entry * 100
                exit_type = "TP" if filled_price >= trade["target"] * 0.999 else "SL/BRACKET"
                log_exit(logger, sym, exit_type, filled_price, pnl, pnl_pct)
                self._record_trade_result(sym, pnl)
                open_syms.discard(sym)
        except Exception as exc:
            logger.debug(f"BRACKET CHECK: {exc}")

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

    async def wait_for_time_exit(self) -> None:
        """Background task: sleep until 13:30 ET, then signal poll loop to stop."""
        await self.wait_until(13, 30)
        logger.info("TIME EXIT: 13:30 ET reached — signalling exit.")
        self._exit_flag = True
        self._exit_event.set()

    # ── Polling loop (replaces websocket stream for IEX free tier) ───────────

    async def poll_bars(self) -> None:
        """Poll historical API every 15s for the latest 1-min bar per symbol."""
        logger.info("POLL: Starting 15-second bar polling loop (IEX feed).")
        while not self._exit_flag:
            try:
                await asyncio.wait_for(self._exit_event.wait(), timeout=15)
                break  # exit event fired
            except asyncio.TimeoutError:
                pass   # normal poll interval
            if self._exit_flag:
                break
            now_et = datetime.now(ET)
            start  = (now_et - timedelta(minutes=5)).isoformat()
            end    = now_et.isoformat()
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=self.watchlist,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    feed="iex",
                )
                bars = self.data_client.get_stock_bars(req)
                fetched = 0
                for sym in self.watchlist:
                    try:
                        sym_bars = bars[sym]
                        if not sym_bars:
                            continue
                        latest = sym_bars[-1]
                        # Skip bars we've already processed
                        if self._seen_bar_times.get(sym) == latest.timestamp:
                            continue
                        self._seen_bar_times[sym] = latest.timestamp
                        await self.on_bar(latest)
                        fetched += 1
                    except Exception:
                        pass
                if fetched:
                    logger.info(f"POLL: Fetched {fetched} new bars at {now_et.strftime('%H:%M:%S ET')}")
                # Detect any bracket exits (TP/SL) Alpaca closed between polls
                self._check_bracket_exits()
            except Exception as exc:
                logger.warning(f"POLL: Bar fetch failed — {exc}")

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        self.validate_account()

        while True:
            now_et  = datetime.now(ET)
            weekday = now_et.weekday()  # 0=Mon … 4=Fri

            # Skip weekends
            if weekday >= 5:
                await asyncio.sleep(3600)
                continue

            # If started after 13:30 ET, sleep until next trading day 09:15 ET
            cutoff = now_et.replace(hour=13, minute=30, second=0, microsecond=0)
            if now_et >= cutoff:
                # Sleep until tomorrow (or Monday if Friday) 09:15 ET
                days_ahead = 3 if weekday == 4 else 1
                next_start = (now_et + timedelta(days=days_ahead)).replace(
                    hour=9, minute=15, second=0, microsecond=0
                )
                wait_secs = (next_start - now_et).total_seconds()
                logger.info(f"SCHEDULER: Past 13:30 ET — sleeping {wait_secs/3600:.1f}h until next session 09:15 ET.")
                await asyncio.sleep(wait_secs)
                # Reset state for new day
                self.orb_high.clear(); self.orb_low.clear(); self.orb_recorded.clear()
                self.orb_candles.clear(); self.active_symbols.clear(); self.orders_placed.clear()
                self._bar_count = 0; self._logged_first.clear()
                self._exit_flag = False; self._exit_event.clear(); self._time_exit_done = False
                self._seen_bar_times.clear()
                self.stats = {"signals":0,"trades":0,"wins":0,"losses":0,"net_pnl":0.0,
                              "best_trade":None,"worst_trade":None,"start_equity":0.0,"end_equity":0.0}
                self.validate_account()
                continue

            logger.info(f"BOT START: Current time {now_et.strftime('%H:%M:%S ET')}")

            # ── 9:15 ET — fetch watchlist ──────────────────────────────────
            if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 15):
                await self.wait_until(9, 15)
            await self.run_watchlist_fetch()

            self.load_historical_bars()
            self.backfill_orb()

            # ── 9:30 ET — wait for market open ────────────────────────────
            if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):
                await self.wait_until(9, 30)
            logger.info("MARKET OPEN: Starting ORB recording and bar polling.")

            self._exit_flag = False
            self._exit_event.clear()
            self._time_exit_done = False

            asyncio.create_task(self.wait_for_time_exit())

            try:
                await self.poll_bars()
            finally:
                self.time_exit()
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
