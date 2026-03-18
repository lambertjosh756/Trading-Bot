"""
crypto_scalper.py — 15-min Momentum Burst Scalper (Alpaca Crypto)

Strategy: Momentum Burst
  - Entry  : bar closes up ≥0.6% in one 15-min bar  AND  volume ≥ 2× 20-bar avg
  - TP     : +0.8% fixed target
  - SL     : −0.3% hard floor; trailing stop follows 0.3% below highest bar high
  - Time   : exit after 16 bars (4 h) if neither TP nor SL hit
  - Symbols: BTC/USD, ETH/USD, SOL/USD, DOGE/USD, LTC/USD
  - Max    : 1 position per symbol (5 total)

Backtest (2024-2025, 5 symbols):
  Sharpe +34.9 | Avg Return +243%/sym | Win Rate 56.5% | 3.8 trades/day | MaxDD -1.7%

Run:
    python crypto_scalper.py
    python crypto_scalper.py --capital 10000
    python crypto_scalper.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from collections import deque
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytz
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TrailingStopOrderRequest,
    LimitOrderRequest, StopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ── Constants ──────────────────────────────────────────────────────────────────

SYMBOLS       = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD"]

BURST_PCT     = 0.006   # single bar must close up ≥0.6%
VOL_MIN       = 2.0     # volume must be ≥2.0× the 20-bar rolling average
TP_PCT        = 0.008   # +0.8% fixed take-profit
SL_PCT        = 0.003   # −0.3% hard SL floor; also trailing gap (default)
SL_PCT_BY_SYMBOL = {
    "DOGE/USD": 0.004,  # widened to 0.4%: high volatility causes stop slippage at 0.3%
}
MAX_BARS_HELD = 16      # time exit after 16 bars (4 h)
VOL_LOOKBACK  = 20      # bars for volume MA
BAR_BUFFER    = 60      # bars to keep per symbol
WARMUP_BARS   = 25      # minimum bars before we start signalling
MAX_POSITIONS = 2       # max simultaneous open positions (limit correlated losses)

API_KEY    = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

LOG_FILE          = Path("scalper_trades.log")
HTML_FILE         = Path("scalper_dashboard.html")
TRADE_HIST_FILE   = Path("scalper_trade_history.json")
START_EQUITY_FILE = Path("scalper_start_equity.json")
DASHBOARD_PORT    = int(os.getenv("SCALPER_PORT", "8081"))

UTC      = pytz.utc
TF_15MIN = TimeFrame(15, TimeFrameUnit.Minute)

# ── Logger ─────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

def log(msg: str, level: str = "INFO"):
    line = f"[{ts()}] [{level}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── Persistence ────────────────────────────────────────────────────────────────

def _load_trade_history() -> dict:
    if TRADE_HIST_FILE.exists():
        try:
            return json.loads(TRADE_HIST_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_trade_history(states: dict):
    data = {sym: st.trade_log for sym, st in states.items()}
    try:
        TRADE_HIST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"Failed to save trade history: {e}", "WARN")

# ── Timing ─────────────────────────────────────────────────────────────────────

def _next_15min_close(now: datetime) -> datetime:
    """Return the UTC datetime of the next 15-min bar close + 10 s buffer."""
    quarter = ((now.minute // 15) + 1) * 15
    if quarter >= 60:
        base = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        base = now.replace(minute=quarter, second=0, microsecond=0)
    return base + timedelta(seconds=10)

# ── Per-symbol state ───────────────────────────────────────────────────────────

class SymbolState:
    def __init__(self, symbol: str):
        self.symbol       = symbol
        self.closes       : deque = deque(maxlen=BAR_BUFFER)
        self.volumes      : deque = deque(maxlen=BAR_BUFFER)
        self.highs        : deque = deque(maxlen=BAR_BUFFER)
        self.lows         : deque = deque(maxlen=BAR_BUFFER)

        self.in_position  = False
        self.entry_price  = 0.0
        self.tp_price     = 0.0
        self.sl_price     = 0.0
        self.highest_price= 0.0
        self.bars_held    = 0
        self.order_id     = ""
        self.stop_order_id = ""   # live Alpaca trailing/stop order protecting downside
        self.tp_order_id   = ""   # live Alpaca limit sell order at take-profit
        self.entry_time   : Optional[datetime] = None
        self.trade_log    : list = []

    def push(self, close: float, high: float, low: float, volume: float):
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)

    def ready(self) -> bool:
        return len(self.closes) >= WARMUP_BARS

    def signal(self, close: float, volume: float) -> tuple[bool, float, float]:
        """
        Returns (entry_signal, bar_chg_pct, vol_ratio).
        bar_chg = % change of the just-closed bar vs the one before it.
        """
        if not self.ready() or len(self.closes) < 2:
            return False, 0.0, 0.0
        prev_close = list(self.closes)[-1]   # last pushed bar = previous bar
        bar_chg = (close - prev_close) / prev_close if prev_close > 0 else 0.0
        recent_vols = list(self.volumes)[-VOL_LOOKBACK:]  # last 20 bars for vol MA
        if len(recent_vols) < 5:
            return False, bar_chg, 0.0
        vol_ma = float(np.mean(recent_vols))
        if vol_ma <= 0:
            return False, bar_chg, 0.0
        vol_ratio = volume / vol_ma
        triggered = (bar_chg >= BURST_PCT) and (vol_ratio >= VOL_MIN)
        return triggered, bar_chg, vol_ratio


# ── Scalper Bot ────────────────────────────────────────────────────────────────

class ScalperBot:
    def __init__(self, symbols: List[str], capital: float, dry_run: bool = False):
        self.symbols  = symbols
        self.capital  = capital
        self.dry_run  = dry_run
        self.states   = {s: SymbolState(s) for s in symbols}
        self.trading  = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.hist     = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

        # Restore trade history
        saved = _load_trade_history()
        for sym, trades in saved.items():
            if sym in self.states:
                self.states[sym].trade_log = trades

        self._load_warmup_bars()
        self._sync_open_positions()

    # ── Warmup ────────────────────────────────────────────────────────────

    def _load_warmup_bars(self):
        end   = datetime.now(UTC).replace(tzinfo=None)
        start = end - timedelta(hours=int(BAR_BUFFER * 15 / 60) + 2)
        log(f"Loading 15-min warmup bars for {len(self.symbols)} symbols …")
        for sym in self.symbols:
            try:
                req  = CryptoBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=TF_15MIN,
                    start=start,
                    end=end,
                )
                resp = self.hist.get_crypto_bars(req)
                raw  = getattr(resp, "data", {})
                bars = raw.get(sym) or raw.get(sym.replace("/", "")) or []
                for b in bars[-BAR_BUFFER:]:
                    self.states[sym].push(
                        float(b.close), float(b.high), float(b.low), float(b.volume)
                    )
                log(f"  {sym}: {len(self.states[sym].closes)} warmup bars")
            except Exception as e:
                log(f"  {sym}: warmup failed — {e}", "WARN")

    # ── Sync open positions ────────────────────────────────────────────────

    def _sync_open_positions(self):
        """Re-attach any open positions on the Alpaca account that belong to us."""
        try:
            positions = self.trading.get_all_positions()
            # Build a map of open orders to restore stop/tp order IDs
            open_orders = {}
            try:
                orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
                for o in orders:
                    open_orders.setdefault(o.symbol, []).append(o)
            except Exception:
                pass

            for p in positions:
                sym_raw = p.symbol
                for s in self.symbols:
                    if s.replace("/", "") == sym_raw:
                        st = self.states[s]
                        sl_pct = self._get_sl(s)
                        if not st.in_position:
                            st.in_position   = True
                            st.entry_price   = float(p.avg_entry_price)
                            st.tp_price      = st.entry_price * (1 + TP_PCT)
                            st.sl_price      = st.entry_price * (1 - sl_pct)
                            st.highest_price = st.entry_price
                            st.bars_held     = 0
                            # Restore order IDs from open orders on Alpaca
                            for o in open_orders.get(sym_raw, []):
                                side = str(o.side).lower()
                                otype = str(o.type).lower()
                                if "sell" in side:
                                    if "limit" in otype:
                                        st.tp_order_id = str(o.id)
                                    elif "stop" in otype or "trailing" in otype:
                                        st.stop_order_id = str(o.id)
                            log(f"  Synced open position: {s} @ ${st.entry_price:.6f}"
                                f"  stop_id={st.stop_order_id or 'none'}"
                                f"  tp_id={st.tp_order_id or 'none'}")
                        break
        except Exception as e:
            log(f"Position sync failed: {e}", "WARN")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_sl(self, sym: str) -> float:
        return SL_PCT_BY_SYMBOL.get(sym, SL_PCT)

    def _open_count(self) -> int:
        return sum(1 for s in self.states.values() if s.in_position)

    def _slot_usd(self) -> float:
        try:
            acct = self.trading.get_account()
            bp   = float(acct.buying_power)
            slot = min(self.capital / MAX_POSITIONS, bp * 0.95 / max(1, MAX_POSITIONS))
            return max(slot, 0)
        except Exception:
            return self.capital / MAX_POSITIONS

    def _has_external_position(self, sym: str) -> bool:
        """True if Alpaca has an open position we didn't open (e.g. trend-follower)."""
        if self.states[sym].in_position:
            return False
        try:
            self.trading.get_open_position(sym.replace("/", ""))
            return True
        except Exception:
            return False

    # ── Entry ─────────────────────────────────────────────────────────────

    def _enter(self, sym: str, close: float, bar_chg: float, vol_ratio: float):
        st = self.states[sym]
        if st.in_position:
            return
        if self._open_count() >= MAX_POSITIONS:
            log(f"  {sym}: max positions reached ({MAX_POSITIONS}), skip")
            return
        if self._has_external_position(sym):
            log(f"  {sym}: external position open (trend-follower?), skip scalper entry")
            return

        sl_pct = self._get_sl(sym)
        usd = self._slot_usd()
        qty = round(usd / close, 6)
        tp  = round(close * (1 + TP_PCT), 6)
        sl  = round(close * (1 - sl_pct), 6)

        log(f"ENTRY  {sym}  qty={qty:.6f}  price=${close:.6f}"
            f"  bar={bar_chg*100:+.2f}%  vol={vol_ratio:.1f}x"
            f"  TP=${tp:.6f}  SL=${sl:.6f} (trail={sl_pct*100:.1f}%)"
            + ("  [DRY RUN]" if self.dry_run else ""))

        if not self.dry_run:
            try:
                sym_raw = sym.replace("/", "")
                # Submit market buy
                order = self.trading.submit_order(
                    MarketOrderRequest(
                        symbol=sym_raw,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                    )
                )
                st.order_id = order.id

                # Poll for fill (crypto fills in < 1s usually)
                import time as _time
                for _ in range(10):
                    _time.sleep(0.5)
                    order = self.trading.get_order_by_id(order.id)
                    if str(order.status) in ("filled", "OrderStatus.filled"):
                        break
                filled_price = float(order.filled_avg_price or close)
                filled_qty   = float(order.filled_qty or qty)

                # Recalculate SL/TP based on actual fill
                sl    = round(filled_price * (1 - sl_pct), 6)
                tp    = round(filled_price * (1 + TP_PCT), 6)
                close = filled_price

                # Place limit sell at TP so it fills at the exact target price
                try:
                    tp_ord = self.trading.submit_order(
                        LimitOrderRequest(
                            symbol=sym_raw,
                            qty=filled_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC,
                            limit_price=tp,
                        )
                    )
                    st.tp_order_id = str(tp_ord.id)
                    log(f"  {sym}: TP limit order placed @ ${tp:.6f}")
                except Exception as e:
                    log(f"  {sym}: TP limit order failed — {e}", "WARN")

                # Place trailing stop; fall back to plain stop-market if trailing fails
                try:
                    stop_ord = self.trading.submit_order(
                        TrailingStopOrderRequest(
                            symbol=sym_raw,
                            qty=filled_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC,
                            trail_percent=sl_pct * 100,
                        )
                    )
                    st.stop_order_id = str(stop_ord.id)
                    log(f"  {sym}: trailing stop placed (trail={sl_pct*100:.1f}%)")
                except Exception as e:
                    log(f"  {sym}: trailing stop failed ({e}) — trying plain stop", "WARN")
                    try:
                        stop_ord = self.trading.submit_order(
                            StopOrderRequest(
                                symbol=sym_raw,
                                qty=filled_qty,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.GTC,
                                stop_price=sl,
                            )
                        )
                        st.stop_order_id = str(stop_ord.id)
                        log(f"  {sym}: plain stop placed @ ${sl:.6f}")
                    except Exception as e2:
                        log(f"  {sym}: plain stop also failed — {e2}", "ERROR")

            except Exception as e:
                log(f"  {sym}: order failed — {e}", "ERROR")
                return

        st.in_position   = True
        st.entry_price   = close
        st.tp_price      = tp
        st.sl_price      = sl
        st.highest_price = close
        st.bars_held     = 0
        st.entry_time    = datetime.now(UTC)
        st.trade_log.append({
            "action": "BUY", "price": close, "tp": tp, "sl": sl,
            "bar_chg": round(bar_chg * 100, 3),
            "vol_ratio": round(vol_ratio, 2),
            "time": st.entry_time.isoformat(),
            "dry_run": self.dry_run,
        })
        self._write_dashboard()

    # ── Exit ──────────────────────────────────────────────────────────────

    def _exit(self, sym: str, exit_price: float, reason: str):
        st = self.states[sym]
        if not st.in_position:
            return

        pnl_pct = (exit_price - st.entry_price) / st.entry_price * 100
        log(f"EXIT   {sym}  reason={reason}  price=${exit_price:.6f}"
            f"  entry=${st.entry_price:.6f}  pnl={pnl_pct:+.2f}%"
            + ("  [DRY RUN]" if self.dry_run else ""))

        pnl_usd = None
        if not self.dry_run:
            try:
                sym_raw = sym.replace("/", "")
                # Cancel both standing orders so they don't double-sell
                for oid, label in [(st.stop_order_id, "stop"), (st.tp_order_id, "TP")]:
                    if oid:
                        try:
                            self.trading.cancel_order_by_id(oid)
                            log(f"  {sym}: {label} order cancelled")
                        except Exception:
                            pass
                st.stop_order_id = ""
                st.tp_order_id   = ""
                try:
                    pos = self.trading.get_open_position(sym_raw)
                    pnl_usd = float(pos.unrealized_pl)
                    log(f"  {sym}: actual P&L = ${pnl_usd:+.2f}")
                except Exception:
                    pass
                self.trading.close_position(sym_raw)
            except Exception as e:
                log(f"  {sym}: close failed — {e}", "ERROR")

        st.trade_log.append({
            "action": "SELL", "price": exit_price, "reason": reason,
            "pnl_pct": round(pnl_pct, 3),
            "pnl_usd": round(pnl_usd, 2) if pnl_usd is not None else None,
            "time": datetime.now(UTC).isoformat(),
            "dry_run": self.dry_run,
        })
        st.in_position    = False
        st.entry_price    = 0.0
        st.tp_price       = 0.0
        st.sl_price       = 0.0
        st.highest_price  = 0.0
        st.bars_held      = 0
        st.order_id       = ""
        st.stop_order_id  = ""
        st.tp_order_id    = ""
        _save_trade_history(self.states)
        self._write_dashboard()

    # ── Bar handler ───────────────────────────────────────────────────────

    def _on_bar(self, sym: str, close: float, high: float, low: float, volume: float):
        st = self.states[sym]
        triggered, bar_chg, vol_ratio = st.signal(close, volume)

        # Push bar AFTER computing signal (so current bar is new data)
        st.push(close, high, low, volume)

        log(f"  BAR {sym}  chg={bar_chg*100:+.2f}%  vol={vol_ratio:.1f}x  close={close:.4f}"
            + (" *** SIGNAL ***" if triggered and not st.in_position else ""))

        if not st.in_position:
            if triggered:
                log(f"SIGNAL {sym}  bar={bar_chg*100:+.2f}%  vol={vol_ratio:.1f}x  close={close:.6f}")
                self._enter(sym, close, bar_chg, vol_ratio)
        else:
            # Check if Alpaca already closed this position (TP or SL fired intrabar)
            if not self.dry_run:
                position_gone = False
                try:
                    self.trading.get_open_position(sym.replace("/", ""))
                except Exception as _pos_exc:
                    # Distinguish real 404 (closed) from transient API errors
                    if "404" in str(_pos_exc) or "position does not exist" in str(_pos_exc).lower():
                        position_gone = True
                    else:
                        log(f"  {sym}: position check error (non-404) — {_pos_exc}", "WARN")
                if position_gone:
                    # Determine which order fired by checking recently filled orders
                    filled_price = close  # fallback
                    reason = "ORDER(intrabar)"
                    try:
                        from alpaca.trading.requests import GetOrdersRequest
                        from alpaca.trading.enums import QueryOrderStatus
                        recent = self.trading.get_orders(
                            GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=5)
                        )
                        for o in recent:
                            if o.symbol == sym.replace("/", "") and o.filled_avg_price:
                                filled_price = float(o.filled_avg_price)
                                otype = str(o.type).lower()
                                if "limit" in otype:
                                    reason = f"TP_ORDER(intrabar,+{TP_PCT*100:.1f}%)"
                                else:
                                    sl_pct = self._get_sl(sym)
                                    reason = f"SL_ORDER(intrabar,-{sl_pct*100:.1f}%)"
                                break
                    except Exception:
                        pass
                    pnl_pct = (filled_price - st.entry_price) / st.entry_price * 100
                    log(f"ORDER_FIRED {sym}  reason={reason}  filled≈${filled_price:.6f}  pnl≈{pnl_pct:+.2f}%")
                    st.trade_log.append({
                        "action": "SELL", "price": filled_price, "reason": reason,
                        "pnl_pct": round(pnl_pct, 3),
                        "pnl_usd": None, "time": datetime.now(UTC).isoformat(), "dry_run": False,
                    })
                    st.in_position = False; st.entry_price = 0.0; st.tp_price = 0.0
                    st.sl_price = 0.0; st.highest_price = 0.0; st.bars_held = 0
                    st.order_id = ""; st.stop_order_id = ""; st.tp_order_id = ""
                    _save_trade_history(self.states)
                    self._write_dashboard()
                    return

            st.highest_price  = max(st.highest_price, high)
            st.bars_held     += 1

            # Trailing stop: 0.3% below highest bar high, floored at hard SL
            trail_stop = max(st.sl_price, st.highest_price * (1 - SL_PCT))

            reason     = None
            exit_price = close
            if high >= st.tp_price:
                reason     = f"TP(+{TP_PCT*100:.1f}%)"
                exit_price = st.tp_price
            elif low <= trail_stop:
                locked_pct = (trail_stop / st.entry_price - 1) * 100
                if trail_stop > st.sl_price * 1.001:
                    reason = f"TRAIL({locked_pct:+.1f}%)"
                else:
                    reason = f"SL(-{SL_PCT*100:.1f}%)"
                exit_price = trail_stop
            elif st.bars_held >= MAX_BARS_HELD:
                reason = f"TIME({st.bars_held}bars)"

            if reason:
                self._exit(sym, exit_price, reason)

        self._write_dashboard()

    # ── Dashboard ─────────────────────────────────────────────────────────

    def _fetch_live_pnl(self) -> dict:
        result = {"equity": 0.0, "today_pnl": 0.0, "net_pnl": 0.0,
                  "unrealized_total": 0.0, "pos_unrealized": {}}
        try:
            acct        = self.trading.get_account()
            equity      = float(acct.equity)
            last_equity = float(acct.last_equity)
            result["equity"]    = equity
            result["today_pnl"] = equity - last_equity

            if START_EQUITY_FILE.exists():
                start_eq = float(json.loads(START_EQUITY_FILE.read_text())["equity"])
            else:
                start_eq = equity
                START_EQUITY_FILE.write_text(json.dumps({"equity": equity}))
                log(f"Scalper starting equity: ${equity:,.2f}")
            result["net_pnl"] = equity - start_eq

            positions = self.trading.get_all_positions()
            unreal_total = 0.0
            for p in positions:
                for s in self.symbols:
                    if s.replace("/", "") == p.symbol:
                        upl = float(p.unrealized_pl)
                        result["pos_unrealized"][s] = upl
                        unreal_total += upl
                        break
            result["unrealized_total"] = unreal_total
        except Exception as e:
            log(f"Live P&L fetch failed: {e}", "WARN")
        return result

    def _write_dashboard(self):
        pnl_data        = self._fetch_live_pnl()
        today_pnl       = pnl_data["today_pnl"]
        net_pnl         = pnl_data["net_pnl"]
        unrealized_total = pnl_data["unrealized_total"]
        equity          = pnl_data["equity"]
        today_utc       = datetime.now(UTC).strftime("%Y-%m-%d")

        all_sells    = [t for st in self.states.values()
                        for t in st.trade_log if t["action"] == "SELL"]
        total_trades = len(all_sells)
        total_wins   = sum(1 for t in all_sells if t.get("pnl_pct", 0) > 0)
        total_losses = total_trades - total_wins

        net_col    = "#3fb950" if net_pnl >= 0 else "#f85149"
        today_col  = "#3fb950" if today_pnl >= 0 else "#f85149"
        unreal_col = "#3fb950" if unrealized_total >= 0 else "#f85149"

        rows = ""
        for sym, st in self.states.items():
            status     = "IN POSITION" if st.in_position else "FLAT"
            status_col = "#3fb950" if st.in_position else "#8b949e"
            entry_s    = f"${st.entry_price:.6f}" if st.in_position else "—"
            last_close = list(st.closes)[-1] if st.closes else 0.0

            if st.in_position and st.entry_price > 0:
                real_upl   = pnl_data["pos_unrealized"].get(sym)
                if real_upl is not None:
                    unreal_pct = real_upl / (self.capital / MAX_POSITIONS) * 100
                    unreal_s   = f"${real_upl:+.2f} ({unreal_pct:+.1f}%)"
                else:
                    unreal_pct = (last_close - st.entry_price) / st.entry_price * 100
                    unreal_s   = f"{unreal_pct:+.2f}%"
                unreal_c   = "pos" if unreal_pct > 0 else "neg"
                trail_stop = max(st.sl_price, st.highest_price * (1 - SL_PCT))
                locked_pct = (trail_stop / st.entry_price - 1) * 100
                bars_left  = MAX_BARS_HELD - st.bars_held
                tp_s = (f"TP ${st.tp_price:.4f} / Trail ${trail_stop:.4f}"
                        f" ({locked_pct:+.1f}%) | {bars_left}bars left")
            else:
                unreal_s  = "—"
                unreal_c  = "neu"
                tp_s      = "—"

            sells    = [t for t in st.trade_log if t["action"] == "SELL"]
            avg_pnl  = sum(t.get("pnl_pct", 0) for t in sells) / len(sells) if sells else 0
            wins     = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)
            win_rate = f"{wins}/{len(sells)}" if sells else "—"

            rows += f"""<tr>
              <td class="sym">{sym}</td>
              <td style="color:{status_col};font-weight:600">{status}</td>
              <td>${last_close:.6f}</td>
              <td>{entry_s}</td>
              <td class="{unreal_c}">{unreal_s}</td>
              <td style="font-size:12px">{tp_s}</td>
              <td class="{"pos" if avg_pnl>0 else "neg" if avg_pnl<0 else "neu"}">{avg_pnl:+.2f}%</td>
              <td>{win_rate}</td>
            </tr>"""

        # Closed trades, newest first
        closed = []
        for sym, st in self.states.items():
            for t in st.trade_log:
                if t["action"] == "SELL":
                    closed.append({**t, "symbol": sym})
        closed.sort(key=lambda x: x["time"], reverse=True)

        running_total = 0.0
        trade_rows = ""
        slot_usd = self.capital / MAX_POSITIONS
        for t in closed[:25]:
            pnl_usd = t.get("pnl_usd")
            pnl_pct = t.get("pnl_pct", 0)
            dollar_val = pnl_usd if pnl_usd is not None else pnl_pct * slot_usd / 100
            running_total += dollar_val
            pnl_c = "pos" if dollar_val > 0 else "neg"
            run_c = "pos" if running_total >= 0 else "neg"
            dry   = " [DRY]" if t.get("dry_run") else ""
            trade_rows += f"""<tr>
              <td>{t["time"][:19]}</td>
              <td class="sym">{t["symbol"]}</td>
              <td>${t["price"]:.6f}</td>
              <td>{t.get("reason","—")}</td>
              <td class="{pnl_c}" style="font-weight:600">${dollar_val:+.2f} ({pnl_pct:+.2f}%){dry}</td>
              <td class="{run_c}">${running_total:+.2f}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>Crypto Scalper Dashboard</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0d1117; color:#e6edf3; }}
  h1 {{ padding:28px 36px 6px; font-size:26px; color:#f0883e; }}
  h2 {{ padding:20px 36px 8px; font-size:18px; color:#ffa657; }}
  .subtitle {{ padding:0 36px 20px; color:#8b949e; font-size:13px; }}
  .section {{ padding:0 36px 32px; }}
  .pos {{ color:#3fb950; }} .neg {{ color:#f85149; }} .neu {{ color:#8b949e; }}
  .sym {{ color:#ffa657; font-weight:600; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th {{ background:#161b22; color:#8b949e; text-transform:uppercase; font-size:11px;
       letter-spacing:.05em; padding:10px 14px; text-align:left; }}
  td {{ padding:9px 14px; border-bottom:1px solid #21262d; }}
  tr:hover {{ background:#161b22; }}
  .config-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }}
  .config-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .config-label {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.05em; }}
  .config-val {{ font-size:20px; font-weight:700; color:#f0883e; margin-top:4px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
            background:#1f2937; font-size:11px; color:#ffa657; border:1px solid #374151; }}
</style>
</head>
<body>
<h1>Crypto Scalper — Live Dashboard</h1>
<p class="subtitle">
  Auto-refresh 15s &nbsp;|&nbsp; Last update: {ts()} &nbsp;|&nbsp;
  <span class="badge">15-min Momentum Burst &nbsp; +0.6% bar &nbsp; 2× vol &nbsp; TP +0.8% &nbsp; SL −0.3% trail</span>
  &nbsp;|&nbsp; {"DRY RUN MODE" if self.dry_run else "LIVE PAPER TRADING"}
</p>

<div class="section">
<div class="config-grid">
  <div class="config-card"><div class="config-label">Net P&L (vs Start)</div>
    <div class="config-val" style="color:{net_col}">${net_pnl:+.2f}</div></div>
  <div class="config-card"><div class="config-label">Today's P&L ({today_utc})</div>
    <div class="config-val" style="color:{today_col}">${today_pnl:+.2f}</div></div>
  <div class="config-card"><div class="config-label">Unrealized P&L</div>
    <div class="config-val" style="color:{unreal_col}">${unrealized_total:+.2f}</div></div>
  <div class="config-card"><div class="config-label">Portfolio Equity</div>
    <div class="config-val" style="color:#58a6ff">${equity:,.2f}</div></div>
</div>
<div class="config-grid" style="margin-top:0">
  <div class="config-card"><div class="config-label">Closed Trades</div>
    <div class="config-val">{total_trades}</div></div>
  <div class="config-card"><div class="config-label">Wins</div>
    <div class="config-val" style="color:#3fb950">{total_wins}</div></div>
  <div class="config-card"><div class="config-label">Losses</div>
    <div class="config-val" style="color:#f85149">{total_losses}</div></div>
  <div class="config-card"><div class="config-label">Win Rate</div>
    <div class="config-val">{"N/A" if total_trades==0 else f"{100*total_wins//total_trades}%"}</div></div>
</div>
</div>

<h2>Symbol Status</h2>
<div class="section">
<table>
  <thead><tr>
    <th>Symbol</th><th>Status</th><th>Last Price</th><th>Entry Price</th>
    <th>Unrealized P&L</th><th>TP / Trail Stop</th><th>Avg Closed P&L</th><th>W/L</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>

<h2>Closed Trades (last 25)</h2>
<div class="section">
<table>
  <thead><tr>
    <th>Time (UTC)</th><th>Symbol</th><th>Exit Price</th>
    <th>Reason</th><th>P&L</th><th>Running Total</th>
  </tr></thead>
  <tbody>{trade_rows if trade_rows else "<tr><td colspan='6' style='color:#8b949e;text-align:center'>No closed trades yet</td></tr>"}</tbody>
</table>
</div>
</body></html>"""

        HTML_FILE.write_text(html, encoding="utf-8")

    # ── Dashboard server ──────────────────────────────────────────────────

    def _start_dashboard_server(self):
        os.chdir(Path(__file__).parent)

        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass

        server = HTTPServer(("0.0.0.0", DASHBOARD_PORT), QuietHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        log(f"Scalper dashboard on port {DASHBOARD_PORT}")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self):
        log("=" * 60)
        log(f"Crypto Scalper starting — {len(self.symbols)} symbols")
        log(f"Strategy: Momentum Burst ≥{BURST_PCT*100:.1f}% + {VOL_MIN:.1f}× vol (15-min bars)")
        log(f"TP={TP_PCT*100:.1f}%  SL={SL_PCT*100:.1f}% (trailing)  MaxBars={MAX_BARS_HELD}  MaxPos={MAX_POSITIONS}")
        log(f"Capital=${self.capital:,.0f}  Slot=${self.capital/MAX_POSITIONS:,.0f}/symbol")
        log(f"Dry run: {self.dry_run}")
        log("=" * 60)

        self._write_dashboard()
        self._start_dashboard_server()

        # Dashboard refresh every 60 s (live P&L for open positions)
        async def _dash_loop():
            while True:
                await asyncio.sleep(60)
                try:
                    self._write_dashboard()
                except Exception as e:
                    log(f"Dashboard refresh error: {e}", "WARN")

        asyncio.ensure_future(_dash_loop())

        while True:
            now      = datetime.now(UTC).replace(tzinfo=None)
            next_run = _next_15min_close(now)
            wait     = (next_run - now).total_seconds()
            log(f"Next 15-min bar check in {wait/60:.1f} min at {next_run.strftime('%H:%M:%S UTC')}")
            await asyncio.sleep(wait)

            # Fetch the just-closed 15-min bar for every symbol
            bar_end   = next_run - timedelta(seconds=5)
            bar_start = bar_end - timedelta(minutes=30)  # fetch last 2 bars (need prev close)

            for sym in self.symbols:
                try:
                    req  = CryptoBarsRequest(
                        symbol_or_symbols=sym,
                        timeframe=TF_15MIN,
                        start=bar_start,
                        end=bar_end,
                    )
                    resp = self.hist.get_crypto_bars(req)
                    raw  = getattr(resp, "data", {})
                    bars = raw.get(sym) or raw.get(sym.replace("/", "")) or []
                    if not bars:
                        log(f"  {sym}: no bar returned", "WARN")
                        continue
                    b = bars[-1]
                    self._on_bar(
                        sym,
                        close=float(b.close),
                        high=float(b.high),
                        low=float(b.low),
                        volume=float(b.volume),
                    )
                except Exception as e:
                    log(f"  {sym}: bar fetch failed — {e}", "WARN")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="15-min Momentum Burst Scalper")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Total USD capital for scalper (default $10,000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print signals without submitting orders")
    args = parser.parse_args()

    bot = ScalperBot(
        symbols  = args.symbols,
        capital  = args.capital,
        dry_run  = args.dry_run,
    )
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log("Scalper stopped by user.")
        bot._write_dashboard()


if __name__ == "__main__":
    main()
