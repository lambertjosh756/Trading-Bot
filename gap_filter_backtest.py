"""
gap_filter_backtest.py -- Backtest 4 gap-filter configs on the full 5-year dataset.

Configs:
  1. Baseline   -- no gap filter
  2. Gap > 2%   -- skip if |gap| > 0.02
  3. Gap > 3%   -- skip if |gap| > 0.03
  4. Gap > 1.5% -- skip if |gap| > 0.015

Locked config:
  ORB 5-min | target 2.5% | stop 0.3% | vol 1.5x | RSI 40-60
  max 8 positions | $100k equity | blacklist SNAP,RIVN,HOOD,UBER
  Jan 2021 – Mar 2026

Run:
    python gap_filter_backtest.py
"""

from __future__ import annotations

import json
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

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── Locked strategy constants ─────────────────────────────────────────────────
ET             = pytz.timezone("America/New_York")
PROFIT_PCT     = 0.025    # 2.5% target
STOP_PCT       = 0.003    # 0.3% stop
VOL_MULTIPLIER = 1.5
RSI_LOW        = 40.0
RSI_HIGH       = 60.0
RSI_PERIOD     = 14
VOL_AVG_PERIOD = 20
MAX_POSITIONS  = 8
STARTING_EQUITY = 100_000.0
TOP_N          = 15

BACKTEST_START = date(2021, 1, 1)
BACKTEST_END   = date(2026, 3, 7)   # through Mar 2026

BLACKLIST: set = {"SNAP", "RIVN", "HOOD", "UBER"}

UNIVERSE: List[str] = [
    s for s in [
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
    ] if s not in BLACKLIST
]

DATA_DIR   = Path("data")
MINUTE_DIR = DATA_DIR / "minute"
DAILY_DIR  = DATA_DIR / "daily"
RESULTS_DIR = Path("results")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes[-(period + 1):])
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def trading_days_in_range(start: date, end: date) -> List[date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def bars_response_to_df(bars_resp, symbols: List[str]) -> Dict[str, pd.DataFrame]:
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
        rows = [{"timestamp": b.timestamp, "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume}
                for b in bars]
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        result[sym] = df
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data manager (reuses existing cache)
# ─────────────────────────────────────────────────────────────────────────────

class DataManager:
    def __init__(self, api_key: str, api_secret: str):
        self.client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        MINUTE_DIR.mkdir(parents=True, exist_ok=True)
        DAILY_DIR.mkdir(parents=True, exist_ok=True)

    def get_daily_bars(self, symbols: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
        result, to_fetch = {}, []
        for sym in symbols:
            cache = DAILY_DIR / f"{sym}.parquet"
            if cache.exists():
                df = pd.read_parquet(cache)
                df.index = pd.to_datetime(df.index, utc=True)
                need_start = pd.Timestamp(start, tz="UTC")
                need_end   = pd.Timestamp(end,   tz="UTC")
                if df.index.min() <= need_start and df.index.max() >= need_end - timedelta(days=5):
                    result[sym] = df
                    continue
            to_fetch.append(sym)

        if to_fetch:
            print(f"  Fetching daily bars for {len(to_fetch)} symbols...")
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
                        (DAILY_DIR / f"{sym}.parquet").write_bytes(
                            df.to_parquet())
                        result[sym] = df
                except Exception as exc:
                    print(f"  [WARN] Daily bars fetch error: {exc}")
                time.sleep(0.3)
        return result

    def get_minute_bars(self, symbols: List[str], day: date) -> Dict[str, pd.DataFrame]:
        cache = MINUTE_DIR / f"{day.isoformat()}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index, utc=True)
            result = {}
            for sym in symbols:
                if "symbol" in df.columns:
                    sym_df = df[df["symbol"] == sym].drop(columns=["symbol"])
                    if not sym_df.empty:
                        result[sym] = sym_df
            if result:
                return result

        day_et_start = ET.localize(datetime(day.year, day.month, day.day, 9, 25))
        day_et_end   = ET.localize(datetime(day.year, day.month, day.day, 14, 0))
        result, all_rows = {}, []

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
                print(f"  [WARN] Minute bars {day}: {exc}")
            time.sleep(0.2)

        if all_rows:
            pd.concat(all_rows).to_parquet(cache)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist builder
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_watchlist(day: date, daily_bars: Dict[str, pd.DataFrame]) -> List[str]:
    prev_vols = {}
    for sym, df in daily_bars.items():
        if df.empty:
            continue
        day_ts = pd.Timestamp(day, tz="UTC")
        hist   = df[df.index < day_ts]
        if hist.empty:
            continue
        last = hist.iloc[-1]
        if not (5.0 <= last["close"] <= 500.0):
            continue
        avg_v = hist.tail(20)["volume"].mean()
        if avg_v < 1_000_000:
            continue
        prev_vols[sym] = int(last["volume"])
    ranked = sorted(prev_vols, key=lambda s: prev_vols[s], reverse=True)
    return ranked[:TOP_N]


# ─────────────────────────────────────────────────────────────────────────────
# Gap calculator helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_gaps(
    day: date,
    watchlist: List[str],
    daily_bars: Dict[str, pd.DataFrame],
    minute_bars: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    Return {sym: gap_pct} for each symbol in watchlist.
    gap_pct = (today_open - yesterday_close) / yesterday_close
    Uses first 1-min bar open as today's open.
    Returns absolute value of gap.
    """
    gaps = {}
    day_ts = pd.Timestamp(day, tz="UTC")
    day_open_et = ET.localize(datetime(day.year, day.month, day.day, 9, 30))

    for sym in watchlist:
        # Previous close from daily bars
        df_daily = daily_bars.get(sym)
        if df_daily is None or df_daily.empty:
            continue
        hist = df_daily[df_daily.index < day_ts]
        if hist.empty:
            continue
        prev_close = float(hist.iloc[-1]["close"])
        if prev_close == 0:
            continue

        # Today's open from first minute bar
        df_min = minute_bars.get(sym)
        if df_min is None or df_min.empty:
            continue
        df_et = df_min.copy()
        df_et.index = df_et.index.tz_convert(ET)
        open_bars = df_et[df_et.index >= day_open_et]
        if open_bars.empty:
            continue
        today_open = float(open_bars.iloc[0]["open"])

        gap_pct = abs((today_open - prev_close) / prev_close)
        gaps[sym] = gap_pct

    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Day simulator (gap-filter aware)
# ─────────────────────────────────────────────────────────────────────────────

class DaySimulator:

    def __init__(self, day: date, minute_bars: Dict[str, pd.DataFrame],
                 watchlist: List[str], start_equity: float,
                 gap_threshold: Optional[float] = None,
                 gaps: Optional[Dict[str, float]] = None):
        self.day           = day
        self.minute_bars   = minute_bars
        self.watchlist     = watchlist
        self.equity        = start_equity
        self.gap_threshold = gap_threshold  # None = no filter
        self.gaps          = gaps or {}
        self.trades: List[dict] = []
        self.skipped_by_gap: List[str] = []  # symbols skipped this day

    def run(self) -> Tuple[float, List[dict], List[str]]:
        """Returns (end_equity, trades, skipped_symbols)."""

        # Apply gap filter: remove symbols exceeding threshold
        if self.gap_threshold is not None:
            filtered_wl = []
            for sym in self.watchlist:
                gap = self.gaps.get(sym)
                if gap is not None and gap > self.gap_threshold:
                    self.skipped_by_gap.append(sym)
                else:
                    filtered_wl.append(sym)
        else:
            filtered_wl = list(self.watchlist)

        active:   Dict[str, dict] = {}
        orb_high: Dict[str, float] = {}
        orb_low:  Dict[str, float] = {}
        orb_done: Dict[str, bool]  = {}
        orb_buf:  Dict[str, list]  = defaultdict(list)

        close_hist: Dict[str, list] = defaultdict(list)
        vol_hist:   Dict[str, list] = defaultdict(list)

        all_times = set()
        sym_dfs: Dict[str, pd.DataFrame] = {}
        for sym in filtered_wl:
            df = self.minute_bars.get(sym)
            if df is None or df.empty:
                continue
            df_et = df.copy()
            df_et.index = df_et.index.tz_convert(ET)
            sym_dfs[sym] = df_et
            all_times.update(df_et.index.tolist())

        if not all_times:
            return self.equity, [], self.skipped_by_gap

        timeline  = sorted(all_times)
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

                if day_open <= ts < orb_end:
                    orb_buf[sym].append(bar)
                    continue

                if ts >= orb_end and not orb_done.get(sym, False):
                    bufs = orb_buf.get(sym, [])
                    if bufs:
                        orb_high[sym] = max(float(b["high"]) for b in bufs)
                        orb_low[sym]  = min(float(b["low"])  for b in bufs)
                    orb_done[sym] = True

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
                        self.trades.append({
                            "date":       self.day.isoformat(),
                            "symbol":     sym,
                            "entry_time": pos["entry_time"].strftime("%H:%M"),
                            "exit_time":  ts.strftime("%H:%M"),
                            "entry":      round(pos["entry"],  4),
                            "exit":       round(exit_price,    4),
                            "qty":        pos["qty"],
                            "pnl":        round(pnl,     2),
                            "pnl_pct":    round(pnl_pct, 4),
                            "reason":     reason,
                            "gap_pct":    round(self.gaps.get(sym, 0) * 100, 3),
                        })
                        del active[sym]
                    continue

                if not orb_done.get(sym, False):
                    continue
                if sym in active or len(active) >= MAX_POSITIONS:
                    continue

                orb_h = orb_high.get(sym)
                if orb_h is None or close <= orb_h:
                    continue

                vols = vol_hist[sym]
                if len(vols) < 5:
                    continue
                avg_v = np.mean(vols[-VOL_AVG_PERIOD:])
                if avg_v == 0 or vol < VOL_MULTIPLIER * avg_v:
                    continue

                closes = np.array(close_hist[sym])
                rsi = calc_rsi(closes)
                if not (RSI_LOW <= rsi <= RSI_HIGH):
                    continue

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
                }

        # Time exit
        for sym, pos in active.items():
            df_et = sym_dfs.get(sym)
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
                "entry_time": pos["entry_time"].strftime("%H:%M"),
                "exit_time":  "13:30",
                "entry":      round(pos["entry"],  4),
                "exit":       round(exit_price,    4),
                "qty":        pos["qty"],
                "pnl":        round(pnl,     2),
                "pnl_pct":    round(pnl_pct, 4),
                "reason":     "TIME EXIT",
                "gap_pct":    round(self.gaps.get(sym, 0) * 100, 3),
            })

        return self.equity, self.trades, self.skipped_by_gap


# ─────────────────────────────────────────────────────────────────────────────
# Run one config
# ─────────────────────────────────────────────────────────────────────────────

def run_config(
    label: str,
    gap_threshold: Optional[float],
    data: DataManager,
    daily_bars: Dict[str, pd.DataFrame],
    days: List[date],
) -> dict:
    """Run a single config, return summary dict."""
    print(f"\n{'─'*60}")
    print(f"  Running: {label}")
    if gap_threshold is not None:
        print(f"  Gap filter: skip if |gap| > {gap_threshold*100:.1f}%")
    else:
        print(f"  Gap filter: NONE (baseline)")
    print(f"{'─'*60}")

    equity        = STARTING_EQUITY
    all_trades:   List[dict] = []
    daily_pnls:   List[float] = []
    equity_curve: List[float] = []
    total_skipped = 0
    day_count = 0

    for day in days:
        wl = build_historical_watchlist(day, daily_bars)
        if not wl:
            continue

        minute_bars = data.get_minute_bars(wl, day)
        if not minute_bars:
            continue

        # Compute gaps for all watchlist symbols
        gaps = compute_gaps(day, wl, daily_bars, minute_bars)

        day_start_eq = equity
        sim = DaySimulator(
            day=day,
            minute_bars=minute_bars,
            watchlist=wl,
            start_equity=equity,
            gap_threshold=gap_threshold,
            gaps=gaps,
        )
        equity, trades, skipped = sim.run()

        all_trades.extend(trades)
        day_pnl = equity - day_start_eq
        daily_pnls.append(day_pnl)
        equity_curve.append(equity)
        total_skipped += len(skipped)
        day_count += 1

        if day_count % 50 == 0:
            pct = day_count / len(days) * 100
            print(f"    [{pct:5.1f}%] {day}  eq=${equity:,.0f}  "
                  f"trades_so_far={len(all_trades)}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    if not all_trades:
        return {"label": label, "trades": 0, "win_pct": 0,
                "net_pnl": 0, "sharpe": 0, "max_dd_pct": 0,
                "skipped_total": total_skipped,
                "skipped_per_year": 0, "years": 5,
                "year_by_year": {}}

    df = pd.DataFrame(all_trades)
    total_trades  = len(df)
    wins          = df[df["pnl"] > 0]
    win_pct       = len(wins) / total_trades * 100
    net_pnl       = df["pnl"].sum()

    # Sharpe (annualised from daily P&L)
    dpnl = pd.Series(daily_pnls)
    sharpe = (dpnl.mean() / dpnl.std() * np.sqrt(252)
              if dpnl.std() > 0 else 0.0)

    # Max drawdown from equity curve
    eq_s  = pd.Series(equity_curve)
    rm    = eq_s.cummax()
    max_dd = float(((eq_s - rm) / rm * 100).min())

    # Year-by-year for beats-baseline reporting
    df["year"] = pd.to_datetime(df["date"]).dt.year
    year_by_year = {}
    for yr, grp in df.groupby("year"):
        yr_pnl = float(grp["pnl"].sum())
        yr_wr  = (grp["pnl"] > 0).mean() * 100
        year_by_year[int(yr)] = {
            "net_pnl": round(yr_pnl, 2),
            "win_pct": round(yr_wr, 1),
        }

    n_years = (BACKTEST_END - BACKTEST_START).days / 365.25
    skipped_per_year = round(total_skipped / n_years, 1)

    return {
        "label":            label,
        "trades":           total_trades,
        "win_pct":          round(win_pct, 1),
        "net_pnl":          round(net_pnl, 2),
        "sharpe":           round(float(sharpe), 4),
        "max_dd_pct":       round(max_dd, 2),
        "skipped_total":    total_skipped,
        "skipped_per_year": skipped_per_year,
        "year_by_year":     year_by_year,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(".env")
    api_key    = os.getenv("ALPACA_API_KEY", "").strip()
    api_secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not api_secret:
        print("ERROR: API keys not found in .env")
        sys.exit(1)

    data = DataManager(api_key, api_secret)
    days = trading_days_in_range(BACKTEST_START, BACKTEST_END)

    print(f"\n{'='*60}")
    print(f"  GAP FILTER BACKTEST  |  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Starting equity : ${STARTING_EQUITY:,.0f}")
    print(f"  Universe        : {len(UNIVERSE)} symbols (blacklist applied)")
    print(f"  Trading days    : {len(days)}")
    print(f"  Configs         : 4 (baseline + 3 gap thresholds)")
    print(f"{'='*60}")

    # Pre-download all daily bars (cached for subsequent config runs)
    print("\nDownloading / loading daily bars...")
    daily_bars = data.get_daily_bars(
        UNIVERSE,
        BACKTEST_START - timedelta(days=35),
        BACKTEST_END,
    )
    print(f"  Daily bars loaded for {len(daily_bars)} symbols.\n")

    configs = [
        ("Baseline (no filter)", None),
        ("Gap > 2%",             0.02),
        ("Gap > 3%",             0.03),
        ("Gap > 1.5%",           0.015),
    ]

    results = []
    for label, threshold in configs:
        res = run_config(label, threshold, data, daily_bars, days)
        results.append(res)
        print(f"  Done: {label}  |  trades={res['trades']}  "
              f"pnl=${res['net_pnl']:+,.0f}  sharpe={res['sharpe']:.3f}  "
              f"maxDD={res['max_dd_pct']:+.2f}%")

    # ── Save raw results JSON ─────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "gap_filter_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n  Raw results → {out_json}")

    # ─────────────────────────────────────────────────────────────────────────
    # Print comparison table
    # ─────────────────────────────────────────────────────────────────────────
    baseline = results[0]

    header = (
        f"\n{'='*80}\n"
        f"  GAP FILTER COMPARISON  |  $100k starting equity  |  Jan 2021 – Mar 2026\n"
        f"{'='*80}\n"
        f"  {'Config':<20}  {'Trades':>6}  {'Win%':>5}  {'Net P&L':>10}  "
        f"{'Sharpe':>7}  {'MaxDD%':>7}  {'Skipped/yr':>10}\n"
        f"  {'─'*20}  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*10}"
    )
    print(header)

    for r in results:
        marker = ""
        if r["label"] != baseline["label"]:
            if r["sharpe"] > baseline["sharpe"] and r["net_pnl"] > baseline["net_pnl"]:
                marker = " ✓ BEATS BASELINE"
            elif r["sharpe"] > baseline["sharpe"]:
                marker = " ~ higher Sharpe"
            elif r["net_pnl"] > baseline["net_pnl"]:
                marker = " ~ higher P&L"
            else:
                marker = " ✗"
        print(f"  {r['label']:<20}  {r['trades']:>6,}  {r['win_pct']:>5.1f}  "
              f"${r['net_pnl']:>+9,.0f}  {r['sharpe']:>7.3f}  "
              f"{r['max_dd_pct']:>+7.2f}  {r['skipped_per_year']:>10.1f}{marker}")

    print(f"  {'─'*75}")

    # ── Year-by-year for any config that beats baseline Sharpe ───────────────
    beats_baseline = [r for r in results[1:] if r["sharpe"] > baseline["sharpe"]]

    if beats_baseline:
        print(f"\n{'─'*80}")
        print(f"  YEAR-BY-YEAR for configs that beat baseline Sharpe ({baseline['sharpe']:.3f})")
        print(f"{'─'*80}")
        for r in beats_baseline:
            print(f"\n  Config: {r['label']}")
            print(f"  {'Year':>6}  {'Net P&L':>10}  {'Win%':>5}")
            print(f"  {'─'*6}  {'─'*10}  {'─'*5}")
            for yr in sorted(r["year_by_year"]):
                ydata = r["year_by_year"][yr]
                # baseline same year
                b_yr  = baseline["year_by_year"].get(yr, {})
                diff  = ydata["net_pnl"] - b_yr.get("net_pnl", 0)
                flag  = f"  Δ${diff:+,.0f} vs baseline"
                print(f"  {yr:>6}  ${ydata['net_pnl']:>+9,.0f}  {ydata['win_pct']:>5.1f}{flag}")
    else:
        print("\n  No gap-filter config beats baseline Sharpe.")

    # ── Direct answers ────────────────────────────────────────────────────────
    best = max(results[1:], key=lambda r: (r["sharpe"], r["net_pnl"]))
    best_vs_base_sharpe = best["sharpe"] - baseline["sharpe"]
    best_vs_base_pnl    = best["net_pnl"] - baseline["net_pnl"]

    print(f"\n{'='*80}")
    print(f"  DIRECT ANSWERS")
    print(f"{'='*80}")

    q1 = ("YES" if best["sharpe"] > baseline["sharpe"] and best["net_pnl"] > baseline["net_pnl"]
          else "NO" if best["sharpe"] <= baseline["sharpe"]
          else "MIXED")
    print(f"\n  1. Does gap filter improve Sharpe AND net return?")
    print(f"     → {q1}")
    print(f"       Best config '{best['label']}': "
          f"Sharpe {best['sharpe']:+.3f} vs baseline {baseline['sharpe']:.3f} "
          f"(Δ{best_vs_base_sharpe:+.3f})")
    print(f"       Net P&L ${best['net_pnl']:+,.0f} vs baseline ${baseline['net_pnl']:+,.0f} "
          f"(Δ${best_vs_base_pnl:+,.0f})")

    print(f"\n  2. Best threshold (if any)?")
    if best_vs_base_sharpe > 0.01:
        print(f"     → {best['label']} (Sharpe +{best_vs_base_sharpe:.3f} vs baseline)")
    else:
        print(f"     → No threshold shows meaningful Sharpe improvement (best Δ={best_vs_base_sharpe:+.3f})")

    skipped_yr = best["skipped_per_year"]
    print(f"\n  3. Trades removed per year on average (best config)?")
    print(f"     → ~{skipped_yr:.0f} watchlist slots skipped/yr by '{best['label']}'")
    print(f"       (Note: not all skipped slots would have generated trades)")

    worthit = best_vs_base_sharpe > 0.02 and best_vs_base_pnl > 0
    print(f"\n  4. Worth adding?")
    if worthit:
        print(f"     → YES — meaningful Sharpe gain with positive P&L delta.")
        print(f"       Recommend adding '{best['label']}' to production.")
    else:
        print(f"     → NO — gap filter does not produce a meaningful improvement.")
        print(f"       Baseline without filter remains the better config.")
        print(f"       (Same conclusion as regime filter / ATR filter investigations.)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
