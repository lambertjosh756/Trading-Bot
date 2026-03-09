"""
backtest.py -- 2-year historical backtest of the 5-min ORB strategy.

Pipeline:
  1. Download & cache daily + 1-min bars locally (data/ folder)
  2. Each trading day: build watchlist from prev-day volume
  3. Simulate ORB strategy bar-by-bar on 1-min data
  4. Track every trade with full bracket logic
  5. Write trade_log CSV + daily_summary CSV + print final report

Run:
    python backtest.py
    python backtest.py --start 2023-01-01 --end 2024-12-31
    python backtest.py --start 2023-01-01 --end 2024-12-31 --equity 200000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Alpaca imports ────────────────────────────────────────────────────────────
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# ─────────────────────────────────────────────────────────────────────────────
# SPY benchmark fetch (for alpha reporting)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_spy_return(start: date, end: date) -> Optional[float]:
    """
    Return SPY buy-and-hold return% over [start, end].
    Uses yfinance (already installed). Returns None silently on any failure.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        # Add a small buffer so the first/last trading days are captured
        hist = ticker.history(
            start=(start - timedelta(days=5)).isoformat(),
            end=(end   + timedelta(days=5)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if hist.empty or len(hist) < 2:
            return None
        # Closest available close on or after start
        hist.index = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)
        hist_start = hist[hist.index >= start_ts]
        hist_end   = hist[hist.index <= end_ts]
        if hist_start.empty or hist_end.empty:
            return None
        spy_start = float(hist_start.iloc[0]["Close"])
        spy_end   = float(hist_end.iloc[-1]["Close"])
        if spy_start == 0:
            return None
        return (spy_end - spy_start) / spy_start * 100
    except Exception:
        return None

# ── Strategy constants (must match live bot) ─────────────────────────────────
ET               = pytz.timezone("America/New_York")
MAX_POSITIONS    = 15
VOL_MULTIPLIER   = 1.5
RSI_LOW          = 40.0
RSI_HIGH         = 60.0
PROFIT_PCT       = 0.012     # +1.2%
STOP_PCT         = 0.003     # -0.3%
RSI_PERIOD       = 14
VOL_AVG_PERIOD   = 20
TOP_N            = 15
STARTING_EQUITY  = 200_000.0  # buying power

# Universe of stocks to screen
UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD",
    "NFLX", "CRM",  "PYPL", "INTC", "QCOM", "MU",   "PLTR",
    "SPY",  "QQQ",  "IWM",
    "BAC",  "JPM",  "GS",   "MS",
    "XOM",  "CVX",
    "PFE",  "JNJ",  "MRNA",
    "SOFI", "COIN", "MSTR", "HOOD",
    "F",    "GM",   "NIO",  "RIVN",
    "UBER", "SNAP", "PINS",
    "SQ",   "SHOP",
    "CRWD", "PANW", "OKTA",
    "DIS",  "BA",
]

DATA_DIR     = Path("data")
MINUTE_DIR   = DATA_DIR / "minute"
DAILY_DIR    = DATA_DIR / "daily"
RESULTS_DIR  = Path("results")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas    = np.diff(closes[-(period + 1):])
    gains     = np.where(deltas > 0, deltas, 0.0)
    losses    = np.where(deltas < 0, -deltas, 0.0)
    avg_gain  = gains.mean()
    avg_loss  = losses.mean()
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def trading_days_in_range(start: date, end: date) -> List[date]:
    """Return list of Mon–Fri dates (approximate trading days)."""
    days = []
    cur  = start
    while cur <= end:
        if cur.weekday() < 5:   # Mon=0 ... Fri=4
            days.append(cur)
        cur += timedelta(days=1)
    return days


def bars_response_to_df(bars_resp, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Convert StockBarsResponse to {symbol: DataFrame}."""
    result = {}
    data = getattr(bars_resp, "data", {}) or {}
    if not data:
        try:
            data = dict(bars_resp)
        except Exception:
            pass
    for sym in symbols:
        bars = data.get(sym, [])
        if not bars:
            continue
        rows = [{
            "timestamp": b.timestamp,
            "open":   b.open,
            "high":   b.high,
            "low":    b.low,
            "close":  b.close,
            "volume": b.volume,
        } for b in bars]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        result[sym] = df
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data manager -- fetch + cache
# ─────────────────────────────────────────────────────────────────────────────

class DataManager:

    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        MINUTE_DIR.mkdir(parents=True, exist_ok=True)
        DAILY_DIR.mkdir(parents=True, exist_ok=True)

    # ── Daily bars ────────────────────────────────────────────────────────────

    def get_daily_bars(self, symbols: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
        """
        Return daily OHLCV for symbols over the date range.
        Cached in data/daily/<sym>.parquet.
        """
        result = {}
        to_fetch = []

        for sym in symbols:
            cache = DAILY_DIR / f"{sym}.parquet"
            if cache.exists():
                df = pd.read_parquet(cache)
                # Check if our range is covered
                df.index = pd.to_datetime(df.index, utc=True)
                need_start = pd.Timestamp(start, tz="UTC")
                need_end   = pd.Timestamp(end,   tz="UTC")
                if df.index.min() <= need_start and df.index.max() >= need_end - timedelta(days=5):
                    result[sym] = df
                    continue
            to_fetch.append(sym)

        if to_fetch:
            print(f"  Downloading daily bars for {len(to_fetch)} symbols...")
            # Batch in groups of 30 to avoid URL length limits
            for i in range(0, len(to_fetch), 30):
                batch = to_fetch[i:i+30]
                try:
                    req  = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=str(start - timedelta(days=5)),
                        end=str(end + timedelta(days=1)),
                        feed="iex",
                    )
                    resp = self.client.get_stock_bars(req)
                    batch_df = bars_response_to_df(resp, batch)
                    for sym, df in batch_df.items():
                        cache = DAILY_DIR / f"{sym}.parquet"
                        df.to_parquet(cache)
                        result[sym] = df
                except Exception as exc:
                    print(f"  [WARN] Daily bars fetch failed for batch: {exc}")
                time.sleep(0.3)

        return result

    # ── Minute bars ───────────────────────────────────────────────────────────

    def get_minute_bars(self, symbols: List[str], day: date) -> Dict[str, pd.DataFrame]:
        """
        Return 1-min bars for symbols on a single trading day (9:30–13:30 ET).
        Cached in data/minute/YYYY-MM-DD.parquet (all symbols together).
        """
        cache = MINUTE_DIR / f"{day.isoformat()}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index, utc=True)
            # Return per-symbol dict
            result = {}
            for sym in symbols:
                if sym in df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else []:
                    result[sym] = df[sym]
                elif "symbol" in df.columns:
                    sym_df = df[df["symbol"] == sym].drop(columns=["symbol"])
                    if not sym_df.empty:
                        result[sym] = sym_df
            if result:
                return result

        # Fetch from API
        # Window: 9:25 ET (a bit early for pre-ORB context) to 14:00 ET
        day_et_start = ET.localize(datetime(day.year, day.month, day.day, 9, 25, 0))
        day_et_end   = ET.localize(datetime(day.year, day.month, day.day, 14, 0, 0))

        result = {}
        all_rows = []

        for i in range(0, len(symbols), 20):
            batch = symbols[i:i+20]
            try:
                req  = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Minute,
                    start=day_et_start.isoformat(),
                    end=day_et_end.isoformat(),
                    feed="iex",
                )
                resp = self.client.get_stock_bars(req)
                batch_df = bars_response_to_df(resp, batch)
                for sym, df in batch_df.items():
                    df["symbol"] = sym
                    all_rows.append(df)
                    result[sym] = df.drop(columns=["symbol"])
            except Exception as exc:
                print(f"  [WARN] Minute bars {day} batch failed: {exc}")
            time.sleep(0.2)

        # Cache combined
        if all_rows:
            combined = pd.concat(all_rows)
            combined.to_parquet(cache)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist builder (historical -- uses prev-day volume as proxy)
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_watchlist(
    day: date,
    daily_bars: Dict[str, pd.DataFrame],
    min_avg_vol: int = 1_000_000,
    price_min: float = 5.0,
    price_max: float = 500.0,
) -> List[str]:
    """Return top TOP_N symbols by previous trading day's volume."""
    prev_vols = {}
    for sym, df in daily_bars.items():
        if df.empty:
            continue
        # Get bars strictly before `day`
        day_ts = pd.Timestamp(day, tz="UTC")
        hist   = df[df.index < day_ts]
        if hist.empty:
            continue
        last   = hist.iloc[-1]
        # Price filter
        if not (price_min <= last["close"] <= price_max):
            continue
        # Liquidity filter -- avg volume over last 20 days
        recent = hist.tail(20)
        avg_v  = recent["volume"].mean()
        if avg_v < min_avg_vol:
            continue
        prev_vols[sym] = int(last["volume"])

    ranked = sorted(prev_vols, key=lambda s: prev_vols[s], reverse=True)
    return ranked[:TOP_N]


# ─────────────────────────────────────────────────────────────────────────────
# Day simulator
# ─────────────────────────────────────────────────────────────────────────────

class DaySimulator:

    def __init__(self, day: date, minute_bars: Dict[str, pd.DataFrame],
                 watchlist: List[str], start_equity: float):
        self.day          = day
        self.minute_bars  = minute_bars
        self.watchlist    = watchlist
        self.equity       = start_equity  # at start of day
        self.trades: List[dict] = []

    def run(self) -> Tuple[float, List[dict]]:
        """Simulate the strategy for one day. Returns (end_equity, trades)."""
        active: Dict[str, dict] = {}   # symbol -> position info
        orb_high:  Dict[str, float] = {}
        orb_low:   Dict[str, float] = {}
        orb_done:  Dict[str, bool]  = {}
        orb_buf:   Dict[str, list]  = defaultdict(list)

        # Rolling history buffers keyed by symbol
        close_hist:  Dict[str, list] = defaultdict(list)
        vol_hist:    Dict[str, list] = defaultdict(list)

        # Build a unified timeline: all 1-min bar timestamps across watchlist symbols
        all_times = set()
        sym_dfs: Dict[str, pd.DataFrame] = {}
        for sym in self.watchlist:
            df = self.minute_bars.get(sym)
            if df is None or df.empty:
                continue
            # Convert to ET for filtering
            df_et = df.copy()
            df_et.index = df_et.index.tz_convert(ET)
            sym_dfs[sym] = df_et
            all_times.update(df_et.index.tolist())

        if not all_times:
            return self.equity, []

        timeline = sorted(all_times)
        day_open  = ET.localize(datetime(self.day.year, self.day.month, self.day.day, 9, 30))
        orb_end   = ET.localize(datetime(self.day.year, self.day.month, self.day.day, 9, 35))
        time_exit = ET.localize(datetime(self.day.year, self.day.month, self.day.day, 13, 30))

        for ts in timeline:
            if ts < day_open:
                continue
            if ts >= time_exit:
                break

            for sym, df_et in sym_dfs.items():
                if ts not in df_et.index:
                    continue

                bar   = df_et.loc[ts]
                close = float(bar["close"])
                vol   = float(bar["volume"])

                close_hist[sym].append(close)
                vol_hist[sym].append(vol)

                # ── ORB recording (9:30 ≤ ts < 9:35) ──────────────────────
                if day_open <= ts < orb_end:
                    orb_buf[sym].append(bar)
                    continue

                # ── Finalise ORB at first bar at or after 9:35 ─────────────
                if ts >= orb_end and not orb_done.get(sym, False):
                    bufs = orb_buf.get(sym, [])
                    if bufs:
                        highs = [float(b["high"]) for b in bufs]
                        lows  = [float(b["low"])  for b in bufs]
                        orb_high[sym] = max(highs)
                        orb_low[sym]  = min(lows)
                    orb_done[sym] = True

                # ── Manage open position: check target / stop ──────────────
                if sym in active:
                    pos      = active[sym]
                    hit_tgt  = close >= pos["target"]
                    hit_stop = close <= pos["stop"] or float(bar["low"]) <= pos["stop"]

                    if hit_tgt or hit_stop:
                        exit_price = pos["target"] if hit_tgt else pos["stop"]
                        reason     = "TARGET" if hit_tgt else "STOP"
                        pnl        = (exit_price - pos["entry"]) * pos["qty"]
                        pnl_pct    = (exit_price - pos["entry"]) / pos["entry"] * 100
                        self.equity += pnl
                        trade = {
                            "date":       self.day.isoformat(),
                            "symbol":     sym,
                            "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                            "exit_time":  ts.strftime("%H:%M:%S"),
                            "entry":      round(pos["entry"], 4),
                            "exit":       round(exit_price,   4),
                            "target":     round(pos["target"], 4),
                            "stop":       round(pos["stop"],   4),
                            "qty":        pos["qty"],
                            "pnl":        round(pnl,     2),
                            "pnl_pct":    round(pnl_pct, 4),
                            "reason":     reason,
                            "orb_high":   round(orb_high.get(sym, 0), 4),
                            "rsi_entry":  round(pos["rsi_entry"], 2),
                            "vol_mult":   round(pos["vol_mult"],  2),
                        }
                        self.trades.append(trade)
                        del active[sym]
                    continue   # don't enter again same day

                # ── Entry signal check ─────────────────────────────────────
                if not orb_done.get(sym, False):
                    continue
                if sym in active or len(active) >= MAX_POSITIONS:
                    continue

                orb_h = orb_high.get(sym)
                if orb_h is None or close <= orb_h:
                    continue

                # Volume filter
                vols = vol_hist[sym]
                if len(vols) < 5:
                    continue
                avg_v = np.mean(vols[-VOL_AVG_PERIOD:])
                if avg_v == 0 or vol < VOL_MULTIPLIER * avg_v:
                    continue
                vol_mult = vol / avg_v

                # RSI filter
                closes = np.array(close_hist[sym])
                rsi = calc_rsi(closes)
                if not (RSI_LOW <= rsi <= RSI_HIGH):
                    continue

                # Position sizing: equity / MAX_POSITIONS
                alloc  = self.equity / MAX_POSITIONS
                qty    = max(1, int(alloc / close))
                target = round(close * (1 + PROFIT_PCT), 4)
                stop   = round(close * (1 - STOP_PCT),   4)

                active[sym] = {
                    "entry":      close,
                    "target":     target,
                    "stop":       stop,
                    "qty":        qty,
                    "entry_time": ts,
                    "rsi_entry":  rsi,
                    "vol_mult":   vol_mult,
                }

        # ── 13:30 time exit for any remaining positions ────────────────────
        for sym, pos in active.items():
            df_et = sym_dfs.get(sym)
            # Find last bar before 13:30
            exit_price = pos["entry"]
            if df_et is not None and not df_et.empty:
                before_exit = df_et[df_et.index < time_exit]
                if not before_exit.empty:
                    exit_price = float(before_exit.iloc[-1]["close"])

            pnl     = (exit_price - pos["entry"]) * pos["qty"]
            pnl_pct = (exit_price - pos["entry"]) / pos["entry"] * 100
            self.equity += pnl
            self.trades.append({
                "date":       self.day.isoformat(),
                "symbol":     sym,
                "entry_time": pos["entry_time"].strftime("%H:%M:%S"),
                "exit_time":  "13:30:00",
                "entry":      round(pos["entry"], 4),
                "exit":       round(exit_price,   4),
                "target":     round(pos["target"], 4),
                "stop":       round(pos["stop"],   4),
                "qty":        pos["qty"],
                "pnl":        round(pnl,     2),
                "pnl_pct":    round(pnl_pct, 4),
                "reason":     "TIME EXIT",
                "orb_high":   round(orb_high.get(sym, 0), 4),
                "rsi_entry":  round(pos["rsi_entry"], 2),
                "vol_mult":   round(pos["vol_mult"],  2),
            })

        return self.equity, self.trades


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ─────────────────────────────────────────────────────────────────────────────

class ORBBacktester:

    def __init__(self, api_key: str, api_secret: str,
                 start: date, end: date, starting_equity: float,
                 quiet: bool = False, json_out: str = "", label: str = "baseline"):
        self.data             = DataManager(api_key, api_secret)
        self.start            = start
        self.end              = end
        self.equity           = starting_equity
        self.starting_equity  = starting_equity  # preserved for report
        self.quiet            = quiet
        self.json_out         = json_out
        self.label            = label
        RESULTS_DIR.mkdir(exist_ok=True)

    def run(self) -> None:
        days = trading_days_in_range(self.start, self.end)
        print(f"\n{'='*60}")
        print(f"  ORB Backtest  |  {self.start} -> {self.end}")
        print(f"  Starting equity: ${self.equity:,.2f}")
        print(f"  Trading days   : {len(days)}")
        print(f"  Universe       : {len(UNIVERSE)} symbols")
        print(f"{'='*60}\n")

        # ── Step 1: Download all daily bars upfront ────────────────────────
        if not self.quiet:
            print("Step 1/3 -- Downloading daily bars (used for watchlist selection)...")
        daily_bars = self.data.get_daily_bars(
            UNIVERSE,
            self.start - timedelta(days=35),  # enough history for 20-day avg
            self.end,
        )
        if not self.quiet:
            print(f"  Got daily bars for {len(daily_bars)} symbols.\n")
            print("Step 2/3 -- Simulating trading days...")
        all_trades:   List[dict] = []
        daily_summary: List[dict] = []
        equity_curve: List[dict] = []

        for idx, day in enumerate(days):
            wl = build_historical_watchlist(day, daily_bars)
            if not wl:
                continue

            # Fetch 1-min bars for the day
            minute_bars = self.data.get_minute_bars(wl, day)
            if not minute_bars:
                continue

            day_start_equity = self.equity
            sim = DaySimulator(day, minute_bars, wl, self.equity)
            self.equity, trades = sim.run()

            all_trades.extend(trades)

            day_pnl    = self.equity - day_start_equity
            day_trades = len(trades)
            day_wins   = sum(1 for t in trades if t["pnl"] > 0)
            day_losses = sum(1 for t in trades if t["pnl"] <= 0)

            daily_summary.append({
                "date":          day.isoformat(),
                "watchlist_n":   len(wl),
                "trades":        day_trades,
                "wins":          day_wins,
                "losses":        day_losses,
                "day_pnl":       round(day_pnl, 2),
                "equity":        round(self.equity, 2),
            })
            equity_curve.append({"date": day.isoformat(), "equity": round(self.equity, 2)})

            progress = (idx + 1) / len(days) * 100
            if not self.quiet:
                print(f"  [{progress:5.1f}%] {day}  trades={day_trades:2d}  "
                      f"pnl=${day_pnl:+8.2f}  equity=${self.equity:,.2f}")

        # ── Step 3: Write results ──────────────────────────────────────────
        if not self.quiet:
            print("\nStep 3/3 -- Writing results...")
        self._write_results(all_trades, daily_summary, equity_curve)

    def _write_results(self, trades: List[dict], daily: List[dict], curve: List[dict]) -> None:
        date_tag = f"{self.start}_{self.end}"

        # Trade log CSV
        trades_path = RESULTS_DIR / f"trades_{date_tag}.csv"
        if trades:
            pd.DataFrame(trades).to_csv(trades_path, index=False)
            print(f"  Trade log    -> {trades_path}")

        # Daily summary CSV
        daily_path = RESULTS_DIR / f"daily_{date_tag}.csv"
        if daily:
            pd.DataFrame(daily).to_csv(daily_path, index=False)
            print(f"  Daily summary-> {daily_path}")

        # Equity curve CSV
        curve_path = RESULTS_DIR / f"equity_curve_{date_tag}.csv"
        if curve:
            pd.DataFrame(curve).to_csv(curve_path, index=False)
            print(f"  Equity curve -> {curve_path}")

        # ── Print final report ─────────────────────────────────────────────
        self._print_report(trades, daily, curve)

    def _print_report(self, trades: List[dict], daily: List[dict], curve: List[dict]) -> None:
        if not trades:
            print("\n  No trades generated in this period.")
            return

        df = pd.DataFrame(trades)

        total_trades  = len(df)
        wins          = df[df["pnl"] > 0]
        losses        = df[df["pnl"] <= 0]
        win_rate      = len(wins) / total_trades * 100
        net_pnl       = df["pnl"].sum()
        avg_win       = wins["pnl"].mean()     if not wins.empty   else 0
        avg_loss      = losses["pnl"].mean()   if not losses.empty else 0
        best_trade    = df.loc[df["pnl"].idxmax()]
        worst_trade   = df.loc[df["pnl"].idxmin()]
        avg_hold_bars = None  # minute bar count not tracked explicitly
        profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) \
                        if not losses.empty and losses["pnl"].sum() != 0 else float("inf")

        by_reason = df.groupby("reason")["pnl"].agg(["count", "sum", "mean"])

        # Drawdown from equity curve
        eq = pd.DataFrame(curve).set_index("date")["equity"]
        rolling_max = eq.cummax()
        drawdown    = (eq - rolling_max) / rolling_max * 100
        max_dd      = drawdown.min()

        # Sharpe (annualised, daily returns)
        if daily:
            daily_pnl = pd.DataFrame(daily).set_index("date")["day_pnl"]
            sharpe    = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) \
                        if daily_pnl.std() > 0 else 0.0
        else:
            sharpe = 0.0

        start_eq = self.starting_equity
        end_eq   = self.equity
        total_return_pct = (end_eq - start_eq) / start_eq * 100

        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS  |  {self.start} -> {self.end}")
        print(f"{'='*60}")
        print(f"  Starting equity  : ${start_eq:>12,.2f}")
        print(f"  Ending equity    : ${end_eq:>12,.2f}")
        print(f"  Total return     : {total_return_pct:>+10.2f}%")
        spy_ret = fetch_spy_return(self.start, self.end)
        if spy_ret is not None:
            alpha = total_return_pct - spy_ret
            print(f"  Alpha vs SPY B&H : {total_return_pct:>+10.2f}% (strategy) vs "
                  f"{spy_ret:>+.2f}% (SPY) = {alpha:>+.2f}% alpha")
        print(f"  Net P&L          : ${net_pnl:>+12,.2f}")
        print(f"  Max drawdown     : {max_dd:>+10.2f}%")
        print(f"  Sharpe ratio     : {sharpe:>10.2f}")
        print(f"  Profit factor    : {profit_factor:>10.2f}")
        print(f"{'─'*60}")
        print(f"  Total trades     : {total_trades:>10,}")
        print(f"  Wins             : {len(wins):>10,}  ({win_rate:.1f}%)")
        print(f"  Losses           : {len(losses):>10,}  ({100-win_rate:.1f}%)")
        print(f"  Avg win          : ${avg_win:>+11,.2f}")
        print(f"  Avg loss         : ${avg_loss:>+11,.2f}")
        print(f"  Best trade       : {best_trade['symbol']:6s}  ${best_trade['pnl']:>+9,.2f}  ({best_trade['date']})")
        print(f"  Worst trade      : {worst_trade['symbol']:6s}  ${worst_trade['pnl']:>+9,.2f}  ({worst_trade['date']})")
        print(f"{'─'*60}")
        print(f"  Exit breakdown:")
        for reason, row in by_reason.iterrows():
            print(f"    {reason:<12}  count={int(row['count']):4d}  "
                  f"total=${row['sum']:>+9,.2f}  avg=${row['mean']:>+7,.2f}")
        print(f"{'─'*60}")
        print(f"  Top 5 symbols by total P&L:")
        top_syms = df.groupby("symbol")["pnl"].sum().nlargest(5)
        for sym, pnl in top_syms.items():
            n = len(df[df["symbol"] == sym])
            print(f"    {sym:<6}  ${pnl:>+9,.2f}  ({n} trades)")
        print(f"{'─'*60}")
        print(f"  Bottom 5 symbols by total P&L:")
        bot_syms = df.groupby("symbol")["pnl"].sum().nsmallest(5)
        for sym, pnl in bot_syms.items():
            n = len(df[df["symbol"] == sym])
            print(f"    {sym:<6}  ${pnl:>+9,.2f}  ({n} trades)")
        print(f"{'='*60}\n")

        # Save summary text
        summary_path = RESULTS_DIR / f"summary_{self.start}_{self.end}.txt"
        with open(summary_path, "w") as f:
            f.write(f"BACKTEST RESULTS  {self.start} -> {self.end}\n")
            f.write(f"Starting equity : ${start_eq:,.2f}\n")
            f.write(f"Ending equity   : ${end_eq:,.2f}\n")
            f.write(f"Total return    : {total_return_pct:+.2f}%\n")
            f.write(f"Net P&L         : ${net_pnl:+,.2f}\n")
            f.write(f"Max drawdown    : {max_dd:+.2f}%\n")
            f.write(f"Sharpe ratio    : {sharpe:.2f}\n")
            f.write(f"Profit factor   : {profit_factor:.2f}\n")
            f.write(f"Total trades    : {total_trades}\n")
            f.write(f"Win rate        : {win_rate:.1f}%\n")
            f.write(f"Avg win         : ${avg_win:+,.2f}\n")
            f.write(f"Avg loss        : ${avg_loss:+,.2f}\n")
        print(f"  Summary text -> {summary_path}")

        # JSON output for programmatic consumption (sweep.py)
        if self.json_out:
            import json

            # Per-year breakdown
            df["_year"] = pd.to_datetime(df["date"]).dt.year
            year_by_year = {}
            eq_df = pd.DataFrame(curve).set_index("date")["equity"]
            for yr, grp in df.groupby("_year"):
                yr_pnl   = grp["pnl"].sum()
                yr_wins  = (grp["pnl"] > 0).sum()
                yr_wr    = yr_wins / len(grp) * 100 if len(grp) else 0
                # Sharpe from daily
                if daily:
                    dpnl = pd.DataFrame(daily)
                    dpnl["_yr"] = pd.to_datetime(dpnl["date"]).dt.year
                    dpnl_yr = dpnl[dpnl["_yr"] == yr]["day_pnl"]
                    yr_sh = (dpnl_yr.mean() / dpnl_yr.std() * np.sqrt(252)
                             if len(dpnl_yr) > 1 and dpnl_yr.std() > 0 else 0.0)
                else:
                    yr_sh = 0.0
                # Max DD from equity curve for that year
                yr_eq = eq_df[[d for d in eq_df.index if str(d).startswith(str(yr))]]
                if len(yr_eq) > 1:
                    rm = yr_eq.cummax()
                    yr_dd = ((yr_eq - rm) / rm * 100).min()
                else:
                    yr_dd = 0.0
                year_by_year[str(yr)] = {
                    "net_pnl":    round(float(yr_pnl), 2),
                    "win_rate":   round(float(yr_wr), 2),
                    "sharpe":     round(float(yr_sh), 4),
                    "max_dd_pct": round(float(yr_dd), 4),
                    "trades":     int(len(grp)),
                }

            payload = {
                "label":            self.label,
                "start":            str(self.start),
                "end":              str(self.end),
                "start_equity":     start_eq,
                "end_equity":       round(end_eq, 2),
                "total_return_pct": round(total_return_pct, 4),
                "net_pnl":          round(net_pnl, 2),
                "max_drawdown_pct": round(float(max_dd), 4),
                "sharpe":           round(float(sharpe), 4),
                "profit_factor":    round(float(profit_factor), 4),
                "total_trades":     total_trades,
                "wins":             len(wins),
                "losses":           len(losses),
                "win_rate_pct":     round(win_rate, 2),
                "avg_win":          round(float(avg_win), 2),
                "avg_loss":         round(float(avg_loss), 2),
                "stop_count":       int(by_reason.get("count", {}).get("STOP", 0)) if "STOP" in by_reason.index else 0,
                "target_count":     int(by_reason.get("count", {}).get("TARGET", 0)) if "TARGET" in by_reason.index else 0,
                "time_exit_count":  int(by_reason.get("count", {}).get("TIME EXIT", 0)) if "TIME EXIT" in by_reason.index else 0,
                "year_by_year":     year_by_year,
            }
            Path(self.json_out).write_text(json.dumps(payload, indent=2))
            if not self.quiet:
                print(f"  JSON output  -> {self.json_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Declare globals at the top of main() -- before any reference to them
    global PROFIT_PCT, STOP_PCT, RSI_LOW, RSI_HIGH, VOL_MULTIPLIER, UNIVERSE, MAX_POSITIONS

    parser = argparse.ArgumentParser(description="ORB Strategy Backtest")
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2024-12-31")
    parser.add_argument("--equity",    type=float, default=STARTING_EQUITY)
    # Strategy parameter overrides
    parser.add_argument("--target",    type=float, default=PROFIT_PCT,    help="Profit target pct e.g. 0.015")
    parser.add_argument("--stop",      type=float, default=STOP_PCT,      help="Stop loss pct e.g. 0.004")
    parser.add_argument("--rsi-low",   type=float, default=RSI_LOW,       help="RSI lower bound")
    parser.add_argument("--rsi-high",  type=float, default=RSI_HIGH,      help="RSI upper bound")
    parser.add_argument("--vol-mult",    type=float, default=VOL_MULTIPLIER, help="Volume multiplier threshold")
    parser.add_argument("--max-pos",     type=int,   default=MAX_POSITIONS,  help="Max concurrent positions")
    parser.add_argument("--blacklist",   type=str,   default="",             help="Comma-separated symbols to exclude")
    # Output control
    parser.add_argument("--json-out",  type=str,   default="",            help="Write summary JSON to this path")
    parser.add_argument("--label",     type=str,   default="baseline",    help="Config label for output")
    parser.add_argument("--quiet",     action="store_true",               help="Suppress per-day progress output")
    args = parser.parse_args()

    # Override module-level strategy constants so DaySimulator picks them up
    PROFIT_PCT     = args.target
    STOP_PCT       = args.stop
    RSI_LOW        = args.rsi_low
    RSI_HIGH       = args.rsi_high
    VOL_MULTIPLIER = args.vol_mult
    MAX_POSITIONS  = args.max_pos
    if args.blacklist:
        bl = {s.strip().upper() for s in args.blacklist.split(",")}
        UNIVERSE = [s for s in UNIVERSE if s not in bl]

    load_dotenv(".env")
    api_key    = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not api_secret:
        print("ERROR: API keys not found in .env -- run connectivity_test.py first.")
        sys.exit(1)

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    bt = ORBBacktester(
        api_key=api_key,
        api_secret=api_secret,
        start=start,
        end=end,
        starting_equity=args.equity,
        quiet=args.quiet,
        json_out=args.json_out,
        label=args.label,
    )
    bt.run()


if __name__ == "__main__":
    main()
