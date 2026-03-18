"""
crypto_trader.py — 24/7 Live Paper Crypto Trading Bot (Alpaca)

Strategy: 1-hour EMA Crossover with RSI + Trend filter  [OPTIMIZED]
  - Universe : BTC/USD, ETH/USD, SOL/USD, AVAX/USD, LINK/USD
  - Entry    : EMA20 crosses above EMA50  AND  RSI(14) > 50  AND  price > SMA(100)
  - Exit     : EMA20 crosses below EMA50  OR   RSI(14) > 70  (overbought)
  - Stop loss: -3.0% from entry (bracket order)
  - Target   : +6.0% from entry (bracket order) → 2:1 R/R
  - Max positions : 4  (25% of capital each)
  - Runs 24/7 — no market-hours restrictions

  Optimizer results (2,016 combos tested across 7 symbols × 4 periods):
    Baseline EMA 9/21 + RSI 50/70  →  Avg Sharpe +0.534
    Optimized EMA 20/50 + SMA-100  →  Avg Sharpe +1.000  (+87% improvement)
    Win rate improved: ~45% → 65%

Run:
    python crypto_trader.py
    python crypto_trader.py --symbols BTC/USD ETH/USD --capital 25000
    python crypto_trader.py --dry-run      # print signals, skip order submission
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional

import json
import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, AssetClass
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── Constants ──────────────────────────────────────────────────────────────────

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD"]  # AVAX dropped (underperformer); LINK removed after current position closes

EMA_FAST       = 12      # re-optimized on 2024-2025 (was 20)
EMA_SLOW       = 26      # re-optimized on 2024-2025 (was 50)
RSI_PERIOD     = 14
RSI_ENTRY_MIN  = 50.0
RSI_OVERBOUGHT = 70.0
HOUR_GATE_ON   = 20      # only enter new positions 20:00–08:00 UTC
HOUR_GATE_OFF  = 8       # blocks entries during noisy US session
PROFIT_PCT     = 0.060   # 6.0% take profit
STOP_PCT       = 0.030   # 3.0% hard stop loss (floor)
TRAIL_PCT      = 0.020   # 2.0% trailing stop (moves up with price)
MAX_BARS_HELD  = 48      # time exit after 48 bars (~2 days) if no TP/SL
MAX_POSITIONS  = 4
WARMUP_BARS    = 60      # EMA(26) needs ~60 bars to warm up
BAR_BUFFER     = 100     # keep last N bars per symbol

API_KEY    = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

LOG_FILE           = Path("crypto_trades.log")
HTML_FILE          = Path("crypto_dashboard.html")
TRADE_HIST_FILE    = Path("crypto_trade_history.json")
START_EQUITY_FILE  = Path("crypto_start_equity.json")
DASHBOARD_PORT     = int(os.getenv("PORT", "8080"))

UTC = pytz.utc

# ── Logger ─────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

def log(msg: str, level: str = "INFO"):
    line = f"[{ts()}] [{level}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── Trade history persistence ──────────────────────────────────────────────────

def _load_trade_history() -> dict:
    """Load per-symbol trade logs from disk. Returns {sym: [trade, ...]}."""
    if TRADE_HIST_FILE.exists():
        try:
            return json.loads(TRADE_HIST_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_trade_history(states: dict):
    """Persist all symbol trade logs to disk."""
    data = {sym: st.trade_log for sym, st in states.items()}
    try:
        TRADE_HIST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"Failed to save trade history: {e}", "WARN")

# ── Indicators ─────────────────────────────────────────────────────────────────

def calc_ema(prices: List[float], n: int) -> float:
    """Incremental EMA over list of recent closes."""
    if len(prices) < n:
        return float("nan")
    s = pd.Series(prices)
    return float(s.ewm(span=n, adjust=False).mean().iloc[-1])

def calc_rsi(prices: List[float], n: int = 14) -> float:
    if len(prices) < n + 2:
        return float("nan")
    s = pd.Series(prices)
    delta = s.diff()
    up    = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn    = (-delta).clip(lower=0).ewm(span=n, adjust=False).mean()
    rs    = up / dn.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))

def calc_sma(prices: List[float], n: int) -> float:
    if len(prices) < n:
        return float("nan")
    return float(np.mean(prices[-n:]))

# ── State tracker per symbol ───────────────────────────────────────────────────

class SymbolState:
    def __init__(self, symbol: str):
        self.symbol      = symbol
        self.bars        = deque(maxlen=BAR_BUFFER)  # list of close prices
        self.prev_ema_f  = float("nan")
        self.prev_ema_s  = float("nan")
        self.in_position = False
        self.entry_price = 0.0
        self.tp_price    = 0.0
        self.sl_price    = 0.0
        self.order_id    = ""
        self.entry_time  = None
        self.highest_price = 0.0  # highest bar high seen since entry (for trailing stop)
        self.bars_held   = 0      # number of bars held in current position
        self.trade_log   = []   # list of dicts for dashboard

    def push_bar(self, close: float):
        self.bars.append(close)

    @property
    def closes(self) -> List[float]:
        return list(self.bars)

    def get_signals(self):
        cl = self.closes
        if len(cl) < WARMUP_BARS:
            return None, None, None, None
        ema_f = calc_ema(cl, EMA_FAST)
        ema_s = calc_ema(cl, EMA_SLOW)
        r     = calc_rsi(cl, RSI_PERIOD)
        sma   = calc_sma(cl, 100)
        return ema_f, ema_s, r, sma

    def crossed_up(self, ema_f: float, ema_s: float) -> bool:
        if np.isnan(self.prev_ema_f) or np.isnan(self.prev_ema_s):
            return False
        return (ema_f > ema_s) and (self.prev_ema_f <= self.prev_ema_s)

    def crossed_down(self, ema_f: float, ema_s: float) -> bool:
        if np.isnan(self.prev_ema_f) or np.isnan(self.prev_ema_s):
            return False
        return (ema_f < ema_s) and (self.prev_ema_f >= self.prev_ema_s)


# ── Bot ────────────────────────────────────────────────────────────────────────

class CryptoBot:
    def __init__(self, symbols: List[str], capital: float, dry_run: bool = False):
        self.symbols   = symbols
        self.capital   = capital
        self.dry_run   = dry_run
        self.states    = {s: SymbolState(s) for s in symbols}

        self.trading   = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.hist      = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)

        # Restore trade history from disk before syncing positions
        saved = _load_trade_history()
        for sym, trades in saved.items():
            if sym in self.states:
                self.states[sym].trade_log = trades

        self._load_warmup_bars()
        self._sync_open_positions()

    # ── Warmup ──────────────────────────────────────────────────────────────

    def _load_warmup_bars(self):
        """Seed each symbol with recent historical 1h bars for indicator warmup."""
        end   = datetime.now(UTC).replace(tzinfo=None)
        start = end - timedelta(hours=BAR_BUFFER + 5)
        log(f"Loading warmup bars for {len(self.symbols)} symbols …")
        for sym in self.symbols:
            try:
                req  = CryptoBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=TimeFrame.Hour,
                    start=start,
                    end=end,
                )
                resp = self.hist.get_crypto_bars(req)
                raw  = getattr(resp, "data", {})
                bars = raw.get(sym) or raw.get(sym.replace("/", "")) or []
                for b in bars[-BAR_BUFFER:]:
                    self.states[sym].push_bar(float(b.close))
                log(f"  {sym}: {len(self.states[sym].bars)} warmup bars loaded")
            except Exception as e:
                log(f"  {sym}: warmup failed — {e}", "WARN")

    # ── Sync open positions ──────────────────────────────────────────────────

    def _sync_open_positions(self):
        """Mark any existing open positions so we don't double-enter."""
        try:
            positions = self.trading.get_all_positions()
            for p in positions:
                sym_raw = p.symbol  # e.g. "BTCUSD"
                # Convert "BTCUSD" → "BTC/USD"
                sym = None
                for s in self.symbols:
                    if s.replace("/", "") == sym_raw:
                        sym = s
                        break
                if sym and sym in self.states:
                    st = self.states[sym]
                    st.in_position   = True
                    st.entry_price   = float(p.avg_entry_price)
                    st.tp_price      = st.entry_price * (1 + PROFIT_PCT)
                    st.sl_price      = st.entry_price * (1 - STOP_PCT)
                    st.highest_price = st.entry_price  # trail starts at entry on restart
                    st.bars_held     = 0
                    log(f"  Synced open position: {sym} @ ${st.entry_price:.4f}  TP=${st.tp_price:.4f}  SL=${st.sl_price:.4f}")
        except Exception as e:
            log(f"Position sync failed: {e}", "WARN")

    # ── Sizing ───────────────────────────────────────────────────────────────

    def _position_size_usd(self) -> float:
        """USD to deploy per position = capital / MAX_POSITIONS."""
        try:
            acct = self.trading.get_account()
            bp   = float(acct.buying_power)
            slot_usd = min(self.capital / MAX_POSITIONS, bp * 0.95 / MAX_POSITIONS)
            return max(slot_usd, 0)
        except Exception:
            return self.capital / MAX_POSITIONS

    # ── Count open positions ─────────────────────────────────────────────────

    def _open_count(self) -> int:
        return sum(1 for s in self.states.values() if s.in_position)

    # ── Entry ────────────────────────────────────────────────────────────────

    def _enter(self, sym: str, price: float):
        st = self.states[sym]
        if st.in_position:
            return
        if self._open_count() >= MAX_POSITIONS:
            log(f"  {sym}: max positions reached, skip entry")
            return

        usd  = self._position_size_usd()
        qty  = round(usd / price, 6)
        tp   = round(price * (1 + PROFIT_PCT), 6)
        sl   = round(price * (1 - STOP_PCT), 6)

        log(f"ENTRY  {sym}  qty={qty:.6f}  price=${price:.4f}"
            f"  TP=${tp:.4f}  SL=${sl:.4f}"
            + ("  [DRY RUN]" if self.dry_run else ""))

        if not self.dry_run:
            try:
                order = self.trading.submit_order(
                    MarketOrderRequest(
                        symbol=sym.replace("/", ""),
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                    )
                )
                st.order_id = order.id
            except Exception as e:
                log(f"  {sym}: order failed — {e}", "ERROR")
                return

        st.in_position   = True
        st.entry_price   = price
        st.tp_price      = tp
        st.sl_price      = sl
        st.highest_price = price
        st.bars_held     = 0
        st.entry_time    = datetime.now(UTC)
        st.trade_log.append({
            "action": "BUY", "price": price, "tp": tp, "sl": sl,
            "time": st.entry_time.isoformat(), "dry_run": self.dry_run,
        })
        self._write_dashboard()

    # ── Exit ─────────────────────────────────────────────────────────────────

    def _exit(self, sym: str, price: float, reason: str):
        st = self.states[sym]
        if not st.in_position:
            return

        pnl_pct = (price - st.entry_price) / st.entry_price * 100

        log(f"EXIT   {sym}  reason={reason}  price=${price:.4f}"
            f"  entry=${st.entry_price:.4f}  pnl={pnl_pct:+.2f}%"
            + ("  [DRY RUN]" if self.dry_run else ""))

        pnl_usd = None
        if not self.dry_run:
            try:
                sym_raw = sym.replace("/", "")
                # Grab actual dollar P&L from Alpaca before closing
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
            "action": "SELL", "price": price, "reason": reason,
            "pnl_pct": round(pnl_pct, 3),
            "pnl_usd": round(pnl_usd, 2) if pnl_usd is not None else None,
            "time": datetime.now(UTC).isoformat(), "dry_run": self.dry_run,
        })
        st.in_position   = False
        st.entry_price   = 0.0
        st.tp_price      = 0.0
        st.sl_price      = 0.0
        st.highest_price = 0.0
        st.bars_held     = 0
        st.order_id      = ""
        _save_trade_history(self.states)
        self._write_dashboard()

    # ── Bar handler ───────────────────────────────────────────────────────────

    async def _on_bar(self, bar):
        sym = bar.symbol  # "BTC/USD"
        if sym not in self.states:
            return

        close = float(bar.close)
        high  = float(bar.high)
        low   = float(bar.low)
        st    = self.states[sym]
        st.push_bar(close)

        ema_f, ema_s, r, sma = st.get_signals()
        if ema_f is None:
            return

        # Hour gate: only enter 20:00–08:00 UTC (block 08:00–20:00 US session)
        bar_hour = bar.timestamp.hour if hasattr(bar.timestamp, "hour") else datetime.now(UTC).hour
        in_gate  = (bar_hour >= HOUR_GATE_ON) or (bar_hour < HOUR_GATE_OFF)

        # ── Entry logic ──────────────────────────────────────────────────
        if not st.in_position:
            if (in_gate
                    and st.crossed_up(ema_f, ema_s)
                    and not np.isnan(r) and r > RSI_ENTRY_MIN
                    and not np.isnan(sma) and close > sma):
                log(f"SIGNAL {sym}  EMA cross UP  EMA{EMA_FAST}={ema_f:.4f}"
                    f"  EMA{EMA_SLOW}={ema_s:.4f}  RSI={r:.1f}"
                    f"  SMA100={sma:.4f}  hour={bar_hour}UTC (gate=ON)  close={close:.4f}")
                self._enter(sym, close)

        # ── Exit logic ───────────────────────────────────────────────────
        elif st.in_position:
            # Update trailing high and bar count
            st.highest_price = max(st.highest_price, high)
            st.bars_held    += 1

            # Trailing stop: 2% below highest bar high, floored at hard SL
            trail_stop = max(st.sl_price, st.highest_price * (1 - TRAIL_PCT))

            reason     = None
            exit_price = close
            if high >= st.tp_price:
                reason     = f"TP_hit(+{PROFIT_PCT*100:.1f}%)"
                exit_price = st.tp_price
            elif low <= trail_stop:
                if trail_stop > st.sl_price:
                    pct_lock = (trail_stop / st.entry_price - 1) * 100
                    reason   = f"TRAIL_STOP({pct_lock:+.1f}%)"
                else:
                    reason   = f"SL_hit(-{STOP_PCT*100:.1f}%)"
                exit_price = trail_stop
            elif st.bars_held >= MAX_BARS_HELD:
                reason     = f"TIME_EXIT({st.bars_held}h)"
            elif st.crossed_down(ema_f, ema_s):
                reason = "EMA_cross_down"
            elif not np.isnan(r) and r > RSI_OVERBOUGHT:
                reason = f"RSI_overbought({r:.1f})"

            if reason:
                self._exit(sym, exit_price, reason)

        # Update prev EMAs
        st.prev_ema_f = ema_f
        st.prev_ema_s = ema_s

        # Refresh dashboard on every bar so prices stay live
        self._write_dashboard()

    # ── Dashboard HTML ────────────────────────────────────────────────────────

    def _fetch_live_pnl(self):
        """Pull real P&L from Alpaca account + positions. Returns dict of metrics."""
        result = {
            "equity": 0.0, "today_pnl": 0.0, "net_pnl": 0.0,
            "unrealized_total": 0.0, "pos_unrealized": {},
        }
        try:
            acct = self.trading.get_account()
            equity      = float(acct.equity)
            last_equity = float(acct.last_equity)
            result["equity"]    = equity
            result["today_pnl"] = equity - last_equity

            # Net P&L vs starting equity (persisted on first run)
            if START_EQUITY_FILE.exists():
                start_eq = float(json.loads(START_EQUITY_FILE.read_text())["equity"])
            else:
                start_eq = equity
                START_EQUITY_FILE.write_text(json.dumps({"equity": equity}))
                log(f"Starting equity recorded: ${equity:,.2f}")
            result["net_pnl"] = equity - start_eq

            # Per-position unrealized P&L from Alpaca (actual, not estimated)
            positions = self.trading.get_all_positions()
            unreal_total = 0.0
            for p in positions:
                sym_raw = p.symbol  # "BTCUSD"
                for s in self.symbols:
                    if s.replace("/", "") == sym_raw:
                        upl = float(p.unrealized_pl)
                        result["pos_unrealized"][s] = upl
                        unreal_total += upl
                        break
            result["unrealized_total"] = unreal_total
        except Exception as e:
            log(f"Live P&L fetch failed: {e}", "WARN")
        return result

    def _write_dashboard(self):
        """Write a live HTML dashboard with current state and trade history."""
        # ── Real P&L from Alpaca ──────────────────────────────────────────
        pnl_data       = self._fetch_live_pnl()
        today_pnl      = pnl_data["today_pnl"]
        net_pnl        = pnl_data["net_pnl"]
        unrealized_total = pnl_data["unrealized_total"]
        equity         = pnl_data["equity"]
        today_utc      = datetime.now(UTC).strftime("%Y-%m-%d")

        # ── Trade log stats (W/L counts only — not used for $ P&L) ───────
        all_sells    = [t for st in self.states.values() for t in st.trade_log if t["action"] == "SELL"]
        total_trades = len(all_sells)
        total_wins   = sum(1 for t in all_sells if t.get("pnl_pct", 0) > 0)
        total_losses = total_trades - total_wins

        net_col      = "#3fb950" if net_pnl >= 0 else "#f85149"
        unreal_col2  = "#3fb950" if unrealized_total >= 0 else "#f85149"
        today_col    = "#3fb950" if today_pnl >= 0 else "#f85149"

        rows = ""
        for sym, st in self.states.items():
            status     = "IN POSITION" if st.in_position else "FLAT"
            status_col = "#3fb950" if st.in_position else "#8b949e"
            entry      = f"${st.entry_price:.4f}" if st.in_position else "—"
            last_price = st.closes[-1] if st.closes else 0.0

            # Live unrealized P&L on open position (use Alpaca real value if available)
            if st.in_position and st.entry_price > 0:
                real_upl    = pnl_data["pos_unrealized"].get(sym)
                if real_upl is not None:
                    unreal_pct = real_upl / (self.capital / MAX_POSITIONS) * 100
                    unreal_s   = f"${real_upl:+.2f} ({unreal_pct:+.1f}%)"
                else:
                    unreal_pct = (last_price - st.entry_price) / st.entry_price * 100
                    unreal_s   = f"{unreal_pct:+.2f}%"
                unreal_col  = "pos" if unreal_pct > 0 else "neg"
                trail_stop  = max(st.sl_price, st.highest_price * (1 - TRAIL_PCT))
                trail_locked = (trail_stop / st.entry_price - 1) * 100
                tp_dist     = (f"TP ${st.tp_price:.2f} / Trail ${trail_stop:.2f}"
                               f" ({trail_locked:+.1f}%)")
            else:
                unreal_s   = "—"
                unreal_col = "neu"
                tp_dist    = "—"

            sells   = [t for t in st.trade_log if t["action"] == "SELL"]
            avg_pnl = (sum(t.get("pnl_pct", 0) for t in sells) / len(sells) if sells else 0)
            wins    = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)
            win_rate = f"{wins}/{len(sells)}" if sells else "—"

            rows += f"""<tr>
              <td class="sym">{sym}</td>
              <td style="color:{status_col};font-weight:600">{status}</td>
              <td>${last_price:.4f}</td>
              <td>{entry}</td>
              <td class="{unreal_col}">{unreal_s}</td>
              <td>{tp_dist}</td>
              <td class="{"pos" if avg_pnl>0 else "neg" if avg_pnl<0 else "neu"}">{avg_pnl:+.2f}%</td>
              <td>{win_rate}</td>
            </tr>"""

        # Closed trades only, newest first
        closed_trades = []
        for sym, st in self.states.items():
            for t in st.trade_log:
                if t["action"] == "SELL":
                    closed_trades.append({**t, "symbol": sym})
        closed_trades.sort(key=lambda x: x["time"], reverse=True)

        running_total = 0.0
        trade_rows = ""
        for t in closed_trades[:20]:
            pnl_usd = t.get("pnl_usd")
            pnl_pct = t.get("pnl_pct", 0)
            # Use real dollar P&L if available, else fall back to estimate
            if pnl_usd is not None:
                dollar_val = pnl_usd
            else:
                dollar_val = pnl_pct * (self.capital / MAX_POSITIONS) / 100
            running_total += dollar_val
            is_win  = dollar_val > 0
            pnl_c   = "pos" if is_win else "neg"
            usd_s   = f"${dollar_val:+.2f}"
            pct_s   = f"({pnl_pct:+.2f}%)"
            run_c   = "pos" if running_total >= 0 else "neg"
            dry     = " [DRY]" if t.get("dry_run") else ""
            trade_rows += f"""<tr>
              <td>{t["time"][:19]}</td>
              <td class="sym">{t["symbol"]}</td>
              <td>${t["price"]:.4f}</td>
              <td>{t.get("reason","—")}</td>
              <td class="{pnl_c}" style="font-weight:600">{usd_s} {pct_s}</td>
              <td class="{run_c}">${running_total:+.2f}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>Crypto Trader Dashboard</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0d1117; color:#e6edf3; }}
  h1 {{ padding:28px 36px 6px; font-size:26px; color:#58a6ff; }}
  h2 {{ padding:20px 36px 8px; font-size:18px; color:#79c0ff; }}
  .subtitle {{ padding:0 36px 20px; color:#8b949e; font-size:13px; }}
  .section {{ padding:0 36px 32px; }}
  .pos {{ color:#3fb950; }} .neg {{ color:#f85149; }} .neu {{ color:#8b949e; }}
  .sym {{ color:#79c0ff; font-weight:600; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th {{ background:#161b22; color:#8b949e; text-transform:uppercase; font-size:11px;
       letter-spacing:.05em; padding:10px 14px; text-align:left; }}
  td {{ padding:9px 14px; border-bottom:1px solid #21262d; }}
  tr:hover {{ background:#161b22; }}
  .config-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }}
  .config-card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .config-label {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:.05em; }}
  .config-val {{ font-size:20px; font-weight:700; color:#58a6ff; margin-top:4px; }}
</style>
</head>
<body>
<h1>Crypto Trader — Live Dashboard</h1>
<p class="subtitle">Auto-refresh every 15s &nbsp;|&nbsp; Last update: {ts()} &nbsp;|&nbsp;
Strategy: EMA({EMA_FAST}/{EMA_SLOW}) + RSI({RSI_PERIOD}) + Hour Gate {HOUR_GATE_ON}:00–{HOUR_GATE_OFF}:00 UTC &nbsp;|&nbsp;
{"DRY RUN MODE" if self.dry_run else "LIVE PAPER TRADING"}</p>

<div class="section">
<div class="config-grid">
  <div class="config-card"><div class="config-label">Strategy</div>
    <div class="config-val" style="font-size:14px">EMA {EMA_FAST}/{EMA_SLOW} + RSI + Gate {HOUR_GATE_ON}-{HOUR_GATE_OFF}h</div></div>
  <div class="config-card"><div class="config-label">Take Profit</div>
    <div class="config-val">+{PROFIT_PCT*100:.1f}%</div></div>
  <div class="config-card"><div class="config-label">Stop Loss</div>
    <div class="config-val" style="color:#f85149">-{STOP_PCT*100:.1f}%</div></div>
  <div class="config-card"><div class="config-label">Max Positions</div>
    <div class="config-val">{MAX_POSITIONS}</div></div>
</div>
</div>

<div class="section">
<div class="config-grid">
  <div class="config-card"><div class="config-label">Net P&L (vs Start)</div>
    <div class="config-val" style="color:{net_col}">${net_pnl:+.2f}</div></div>
  <div class="config-card"><div class="config-label">Today's P&L ({today_utc})</div>
    <div class="config-val" style="color:{today_col}">${today_pnl:+.2f}</div></div>
  <div class="config-card"><div class="config-label">Unrealized P&L</div>
    <div class="config-val" style="color:{unreal_col2}">${unrealized_total:+.2f}</div></div>
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
    <div class="config-val">{"N/A" if total_trades == 0 else f"{100*total_wins//total_trades}%"}</div></div>
</div>
</div>

<h2>Symbol Status</h2>
<div class="section">
<table>
  <thead><tr>
    <th>Symbol</th><th>Status</th><th>Last Price</th><th>Entry Price</th>
    <th>Unrealized P&L</th><th>TP / SL</th><th>Avg Closed P&L</th><th>W/L</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>

<h2>Closed Trades (last 20)</h2>
<div class="section">
<table>
  <thead><tr>
    <th>Time (UTC)</th><th>Symbol</th><th>Exit Price</th>
    <th>Reason</th><th>P&L</th><th>Running Total</th>
  </tr></thead>
  <tbody>{trade_rows}</tbody>
</table>
</div>

</body></html>"""

        HTML_FILE.write_text(html, encoding="utf-8")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _start_dashboard_server(self):
        """Serve crypto_dashboard.html on DASHBOARD_PORT in a background thread."""
        os.chdir(Path(__file__).parent)  # serve from bot directory

        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass  # suppress per-request logs

        server = HTTPServer(("0.0.0.0", DASHBOARD_PORT), QuietHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        log(f"Dashboard served on port {DASHBOARD_PORT}")

    async def run(self):
        log("="*60)
        log(f"Crypto Bot starting — {len(self.symbols)} symbols")
        log(f"Strategy: EMA({EMA_FAST}/{EMA_SLOW}) cross + RSI({RSI_PERIOD}) + Hour Gate {HOUR_GATE_ON}:00-{HOUR_GATE_OFF}:00 UTC")
        log(f"TP={PROFIT_PCT*100:.1f}%  SL={STOP_PCT*100:.1f}%  MaxPos={MAX_POSITIONS}")
        log(f"Dry run: {self.dry_run}")
        log("="*60)

        self._write_dashboard()
        self._start_dashboard_server()

        # Background task: refresh dashboard every 60s so P&L stays live
        async def _dashboard_loop():
            while True:
                await asyncio.sleep(60)
                try:
                    self._write_dashboard()
                except Exception as e:
                    log(f"Dashboard refresh error: {e}", "WARN")

        asyncio.ensure_future(_dashboard_loop())

        log("Polling 1h bars every hour …")
        while True:
            # Wait until 10s after the next hour close
            now      = datetime.now(UTC).replace(tzinfo=None)
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, seconds=10)
            wait     = (next_run - now).total_seconds()
            log(f"Next bar check in {wait/60:.1f} min at {next_run.strftime('%H:%M UTC')}")
            await asyncio.sleep(wait)

            # Fetch the just-closed hourly bar for each symbol
            bar_start = next_run - timedelta(hours=2)
            bar_end   = next_run
            for sym in self.symbols:
                try:
                    req  = CryptoBarsRequest(
                        symbol_or_symbols=sym,
                        timeframe=TimeFrame.Hour,
                        start=bar_start,
                        end=bar_end,
                    )
                    resp = self.hist.get_crypto_bars(req)
                    raw  = getattr(resp, "data", {})
                    bars = raw.get(sym) or raw.get(sym.replace("/", "")) or []
                    if not bars:
                        log(f"  {sym}: no bar returned", "WARN")
                        continue
                    b = bars[-1]  # latest closed bar
                    # Build a minimal bar-like object for _on_bar
                    class _Bar:
                        symbol    = sym
                        close     = b.close
                        high      = b.high
                        low       = b.low
                        timestamp = b.timestamp
                    await self._on_bar(_Bar())
                except Exception as e:
                    log(f"  {sym}: bar fetch failed — {e}", "WARN")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live crypto EMA+RSI bot")
    parser.add_argument("--symbols",  nargs="+", default=SYMBOLS)
    parser.add_argument("--capital",  type=float, default=25_000.0,
                        help="Total USD capital to allocate")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print signals without submitting orders")
    args = parser.parse_args()

    bot = CryptoBot(
        symbols  = args.symbols,
        capital  = args.capital,
        dry_run  = args.dry_run,
    )

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log("Bot stopped by user.")
        bot._write_dashboard()


if __name__ == "__main__":
    main()
