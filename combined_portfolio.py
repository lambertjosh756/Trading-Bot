"""
combined_portfolio.py

Backtest ORB + Overnight Momentum combined portfolio.
Jan 2021 – Mar 2026 | $200,000 split 50/50

ORB: config 17 (target=2.5%, stop=0.3%, max_pos=8)
Overnight: buy top-N momentum stocks at close, exit next open.
5 configs varying momentum parameters.

Notes:
  - Overnight entry approximated as day's close price
  - Overnight exit = next trading day's open
  - Config E intraday exit = next day's close (daily bar proxy for 13:30)
  - Earnings avoidance not implemented (no free calendar)
  - Weekend/holiday gaps handled automatically via trading day calendar
"""

from __future__ import annotations

import os, sys, json, warnings, subprocess
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(".env")

# ── Alpaca ────────────────────────────────────────────────────────────────────
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY    = os.getenv("ALPACA_API_KEY", "").strip()
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "").strip()

# ── Constants ─────────────────────────────────────────────────────────────────
START            = "2021-01-01"
END              = "2026-03-07"
ORB_CAPITAL      = 100_000.0
OVERNIGHT_CAPITAL= 100_000.0
TOTAL_CAPITAL    = 200_000.0
DATA_DIR         = Path("data/daily")
RESULTS_DIR      = Path("results")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# S&P 500 proxy universe (~75 large-caps across all sectors)
OVERNIGHT_UNIVERSE = [
    # Tech
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD",
    "INTC","QCOM","MU","CRM","ORCL","ADBE","CSCO","TXN",
    "AMAT","LRCX","NOW","PANW","CRWD","FTNT",
    # Healthcare
    "JNJ","PFE","MRK","ABBV","AMGN","GILD","REGN","VRTX","ISRG","UNH",
    # Financials
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","SPGI",
    # Energy
    "XOM","CVX","COP","OXY",
    # Consumer Discretionary
    "HD","LOW","TGT","COST","MCD","SBUX","NKE","GM","F","LULU",
    # Consumer Staples
    "WMT","PG","KO","PEP",
    # Industrials
    "BA","LMT","GE","HON","CAT","DE","UNP",
    # Materials / Comms / Other
    "FCX","NEM","LIN","NFLX","DIS","CMCSA","T","VZ",
    # Broad ETFs
    "SPY","QQQ","IWM",
]

MIN_PRICE   = 10.0
MIN_AVG_VOL = 2_000_000.0

# ── RSI helper ────────────────────────────────────────────────────────────────
def calc_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    delta  = closes.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1/period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

# ── Sharpe / MaxDD helpers ────────────────────────────────────────────────────
def sharpe(daily_pnl: pd.Series) -> float:
    s = daily_pnl.std()
    return float(daily_pnl.mean() / s * np.sqrt(252)) if s > 0 else 0.0

def max_drawdown_pct(equity: pd.Series) -> float:
    rm = equity.cummax()
    return float(((equity - rm) / rm * 100).min())

def profit_factor(daily_pnl: pd.Series) -> float:
    gains  = daily_pnl[daily_pnl > 0].sum()
    losses = abs(daily_pnl[daily_pnl < 0].sum())
    return float(gains / losses) if losses > 0 else float("inf")

# ── Fetch / cache daily OHLCV ─────────────────────────────────────────────────
def load_daily_bars(symbol: str, start: str, end: str,
                    client: StockHistoricalDataClient) -> Optional[pd.DataFrame]:
    cache = DATA_DIR / f"{symbol}.parquet"
    needed_start = pd.Timestamp(start) - pd.Timedelta(days=40)

    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if not df.empty and df.index.min() <= needed_start and df.index.max() >= pd.Timestamp(end):
            return df

    try:
        req  = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=(pd.Timestamp(start) - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
            end=end,
            feed="iex",
        )
        resp = client.get_stock_bars(req)
        raw  = (getattr(resp, "data", {}) or {}).get(symbol, [])
        if not raw:
            return None
        df = pd.DataFrame([{
            "open":   b.open,
            "high":   b.high,
            "low":    b.low,
            "close":  b.close,
            "volume": b.volume,
        } for b in raw], index=pd.to_datetime([b.timestamp.date() for b in raw]))
        df.index.name = "date"
        df.to_parquet(cache)
        return df
    except Exception as e:
        print(f"  WARNING: {symbol} fetch failed: {e}")
        return None

# ── Build price matrices ───────────────────────────────────────────────────────
def build_price_matrices(universe: List[str], start: str, end: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Returns (closes, opens, volumes) DataFrames: index=date, columns=symbols."""
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    closes  = {}
    opens   = {}
    volumes = {}

    print(f"  Loading daily bars for {len(universe)} symbols...")
    for i, sym in enumerate(universe):
        df = load_daily_bars(sym, start, end, client)
        if df is not None and len(df) > 20:
            closes[sym]  = df["close"]
            opens[sym]   = df["open"]
            volumes[sym] = df["volume"]
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(universe)} done...")

    closes_df  = pd.DataFrame(closes).sort_index()
    opens_df   = pd.DataFrame(opens).sort_index()
    volumes_df = pd.DataFrame(volumes).sort_index()

    # Restrict to backtest window (keep extra history for momentum calc)
    backtest_start = pd.Timestamp(start) - pd.Timedelta(days=35)
    closes_df  = closes_df[closes_df.index >= backtest_start]
    opens_df   = opens_df[opens_df.index  >= backtest_start]
    volumes_df = volumes_df[volumes_df.index >= backtest_start]

    print(f"  Price matrices built: {closes_df.shape[0]} days x {closes_df.shape[1]} symbols")
    return closes_df, opens_df, volumes_df

# ── Overnight momentum simulator ──────────────────────────────────────────────
def simulate_overnight(
    closes:   pd.DataFrame,
    opens:    pd.DataFrame,
    volumes:  pd.DataFrame,
    capital:  float,
    top_n:    int,
    mom_window: int,
    rsi_filter: bool,
    intraday_exit: bool,   # Config E: use next close instead of next open
    label:    str,
) -> pd.Series:
    """
    Returns daily_pnl Series indexed by trade EXIT date (next day).
    """
    # Pre-compute RSI on close for all symbols (14-day)
    rsi_df = pd.DataFrame({sym: calc_rsi_series(closes[sym]) for sym in closes.columns})

    trading_dates = closes.index[closes.index >= pd.Timestamp(START)]
    trading_dates = trading_dates[trading_dates <= pd.Timestamp(END)]

    daily_pnl: Dict[pd.Timestamp, float] = {}

    for i, today in enumerate(trading_dates):
        # Need tomorrow's open/close for exit
        future = closes.index[closes.index > today]
        if len(future) == 0:
            continue
        tomorrow = future[0]

        # Must have enough history for momentum window
        past = closes.index[closes.index < today]
        if len(past) < mom_window:
            continue
        past_day = past[-mom_window]

        # Compute momentum for each symbol
        scores = {}
        for sym in closes.columns:
            try:
                close_t  = closes.loc[today, sym]
                close_t0 = closes.loc[past_day, sym]
                if pd.isna(close_t) or pd.isna(close_t0) or close_t0 == 0:
                    continue
                if close_t < MIN_PRICE:
                    continue
                # Volume filter: 20-day avg volume > 2M
                vol_slice = volumes[sym][volumes.index <= today].tail(20)
                if vol_slice.mean() < MIN_AVG_VOL:
                    continue
                # RSI filter (Config D only)
                if rsi_filter:
                    rsi_val = rsi_df.loc[today, sym] if today in rsi_df.index else 50
                    if rsi_val <= 50:
                        continue
                mom = (close_t - close_t0) / close_t0
                scores[sym] = mom
            except Exception:
                continue

        if not scores:
            continue

        # Select top N by momentum
        top_syms = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not top_syms:
            continue

        alloc = capital / top_n
        day_total = 0.0
        for sym in top_syms:
            try:
                entry = closes.loc[today, sym]
                if intraday_exit:
                    # Config E: buy at next open, sell at next close (≈13:30 proxy)
                    entry_p = opens.loc[tomorrow, sym]
                    exit_p  = closes.loc[tomorrow, sym]
                else:
                    # Standard overnight: buy at today close, sell at tomorrow open
                    entry_p = closes.loc[today, sym]
                    exit_p  = opens.loc[tomorrow, sym]
                if pd.isna(entry_p) or pd.isna(exit_p) or entry_p == 0:
                    continue
                qty    = alloc / entry_p
                pnl    = (exit_p - entry_p) * qty
                day_total += pnl
            except Exception:
                continue

        if tomorrow not in daily_pnl:
            daily_pnl[tomorrow] = 0.0
        daily_pnl[tomorrow] += day_total

    series = pd.Series(daily_pnl).sort_index()
    series.index = pd.to_datetime(series.index)
    return series

# ── Run ORB config 17 via subprocess ─────────────────────────────────────────
def run_orb_config17() -> pd.DataFrame:
    """Run ORB backtest with config 17 params, return daily P&L DataFrame."""
    print("\nRunning ORB config 17 (target=2.5%, stop=0.3%, max_pos=8)...")
    json_out = str(RESULTS_DIR / "orb_config17.json")
    cmd = [
        sys.executable, "backtest.py",
        "--start",     START,
        "--end",       END,
        "--equity",    str(ORB_CAPITAL),
        "--target",    "0.025",
        "--stop",      "0.003",
        "--max-pos",   "8",
        "--blacklist", "SNAP,RIVN,HOOD,UBER",
        "--json-out",  json_out,
        "--label",     "orb_config17",
        "--quiet",
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  ORB subprocess FAILED:\n{result.stderr[-500:]}")
        return pd.DataFrame()

    # Read daily CSV (written by backtest.py to results/daily_START_END.csv)
    daily_path = RESULTS_DIR / f"daily_{START}_{END}.csv"
    if not daily_path.exists():
        print(f"  ERROR: {daily_path} not found after ORB run")
        return pd.DataFrame()

    df = pd.read_csv(daily_path, parse_dates=["date"])
    df = df.set_index("date")[["day_pnl", "equity"]].copy()
    print(f"  ORB done: {len(df)} trading days, net P&L=${df['day_pnl'].sum():+,.0f}")
    return df

# ── Stats for one strategy ────────────────────────────────────────────────────
def strategy_stats(daily_pnl: pd.Series, start_cap: float, label: str) -> dict:
    if daily_pnl.empty:
        return {}
    equity    = start_cap + daily_pnl.cumsum()
    net_pnl   = daily_pnl.sum()
    trading_years = 5 + 67/252
    avg_annual = net_pnl / trading_years
    return {
        "label":       label,
        "net_pnl":     net_pnl,
        "avg_annual":  avg_annual,
        "total_ret_pct": net_pnl / start_cap * 100,
        "sharpe":      sharpe(daily_pnl),
        "max_dd":      max_drawdown_pct(equity),
        "pf":          profit_factor(daily_pnl),
        "final_eq":    equity.iloc[-1] if len(equity) else start_cap,
        "n_days":      len(daily_pnl),
    }

# ── Combine two strategies ────────────────────────────────────────────────────
def combine(orb_pnl: pd.Series, overnight_pnl: pd.Series) -> pd.Series:
    all_dates = orb_pnl.index.union(overnight_pnl.index)
    combined  = (orb_pnl.reindex(all_dates, fill_value=0)
                 + overnight_pnl.reindex(all_dates, fill_value=0))
    return combined.sort_index()

def correlation(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return float("nan")
    return float(a.reindex(common).corr(b.reindex(common)))

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

print("="*70)
print("COMBINED PORTFOLIO BACKTEST")
print("ORB + Overnight Momentum | Jan 2021 – Mar 2026 | $200k")
print("="*70)

# ── Step 1: Run ORB ───────────────────────────────────────────────────────────
orb_df    = run_orb_config17()
orb_daily = orb_df["day_pnl"] if not orb_df.empty else pd.Series(dtype=float)
orb_daily.index = pd.to_datetime(orb_daily.index)

# ── Step 2: Load overnight price data ────────────────────────────────────────
print("\nLoading overnight universe data...")
closes, opens, volumes = build_price_matrices(OVERNIGHT_UNIVERSE, START, END)

# ── Step 3: Define 5 configs ──────────────────────────────────────────────────
overnight_configs = [
    dict(label="A", desc="Top 10, 20-day mom",         top_n=10, mom_window=20, rsi_filter=False, intraday_exit=False),
    dict(label="B", desc="Top 5, 20-day mom",           top_n=5,  mom_window=20, rsi_filter=False, intraday_exit=False),
    dict(label="C", desc="Top 10, 5-day mom",            top_n=10, mom_window=5,  rsi_filter=False, intraday_exit=False),
    dict(label="D", desc="Top 10, 20-day mom + RSI>50", top_n=10, mom_window=20, rsi_filter=True,  intraday_exit=False),
    dict(label="E", desc="Top 10, 20-day, intraday",    top_n=10, mom_window=20, rsi_filter=False, intraday_exit=True),
]

# ── Step 4: Simulate all 5 overnight configs ──────────────────────────────────
print("\nSimulating overnight configs...")
overnight_results = {}
for cfg in overnight_configs:
    print(f"  Config {cfg['label']}: {cfg['desc']}...")
    pnl = simulate_overnight(
        closes=closes,
        opens=opens,
        volumes=volumes,
        capital=OVERNIGHT_CAPITAL,
        top_n=cfg["top_n"],
        mom_window=cfg["mom_window"],
        rsi_filter=cfg["rsi_filter"],
        intraday_exit=cfg["intraday_exit"],
        label=cfg["label"],
    )
    overnight_results[cfg["label"]] = pnl
    print(f"    -> {len(pnl)} days, net P&L=${pnl.sum():+,.0f}")

# ── Step 5: Combine and report ────────────────────────────────────────────────
print("\n" + "="*90)
print("RESULTS")
print("="*90)

# Per-strategy stats
orb_stats   = strategy_stats(orb_daily, ORB_CAPITAL, "ORB config17")

TRADING_YEARS = 5 + 67/252

print("\n--- ORB Standalone (config 17: 2.5% target, 0.3% stop, 8 positions) ---")
print(f"  Net P&L: ${orb_stats['net_pnl']:+,.0f}  |  "
      f"Avg annual: ${orb_stats['avg_annual']:+,.0f}/yr  |  "
      f"Sharpe: {orb_stats['sharpe']:.3f}  |  Max DD: {orb_stats['max_dd']:.2f}%")

all_combined = {}

print("\n--- Per-Config Summary ---")
print(f"\n{'Cfg':<4} {'Overnight desc':<32} {'ON P&L':>9} {'ON Annual':>11} {'ON Sharpe':>10} {'ON DD%':>8} "
      f"{'Corr':>6} {'Comb P&L':>10} {'Comb Annual':>12} {'Comb Sharpe':>12} {'Comb DD%':>9} "
      f"{'Hit 30k?':>9} {'DD<15%?':>8}")
print("-"*155)

results_list = []
for cfg in overnight_configs:
    lbl   = cfg["label"]
    on_pnl = overnight_results[lbl]
    on_st  = strategy_stats(on_pnl, OVERNIGHT_CAPITAL, f"ON-{lbl}")
    comb   = combine(orb_daily, on_pnl)
    comb_st= strategy_stats(comb, TOTAL_CAPITAL, f"Combined-{lbl}")
    corr   = correlation(orb_daily, on_pnl)

    meets_return = comb_st["avg_annual"] >= 30_000
    meets_dd     = abs(comb_st["max_dd"]) <= 15.0

    all_combined[lbl] = {
        "cfg":       cfg,
        "on_pnl":    on_pnl,
        "on_stats":  on_st,
        "comb_pnl":  comb,
        "comb_stats":comb_st,
        "corr":      corr,
        "meets_return": meets_return,
        "meets_dd":  meets_dd,
    }
    results_list.append(all_combined[lbl])

    hit_str = "YES***" if meets_return else "NO"
    dd_str  = "YES" if meets_dd else "NO"
    print(
        f"{lbl:<4} {cfg['desc']:<32} "
        f"${on_st['net_pnl']:>+8,.0f} "
        f"${on_st['avg_annual']:>+10,.0f} "
        f"{on_st['sharpe']:>10.3f} "
        f"{on_st['max_dd']:>7.2f}% "
        f"{corr:>6.3f} "
        f"${comb_st['net_pnl']:>+9,.0f} "
        f"${comb_st['avg_annual']:>+11,.0f} "
        f"{comb_st['sharpe']:>12.3f} "
        f"{comb_st['max_dd']:>8.2f}% "
        f"{hit_str:>9} "
        f"{dd_str:>8}"
    )

# ── Year-by-year for best 3 combined configs ──────────────────────────────────
results_list.sort(key=lambda r: r["comb_stats"]["avg_annual"], reverse=True)
top3 = results_list[:3]

print("\n" + "="*90)
print("YEAR-BY-YEAR — TOP 3 COMBINED CONFIGS")
print("="*90)

for rank, r in enumerate(top3, 1):
    lbl  = r["cfg"]["label"]
    desc = r["cfg"]["desc"]
    cs   = r["comb_stats"]
    print(f"\n  #{rank} Config {lbl} ({desc}): "
          f"5yr combined P&L=${cs['net_pnl']:+,.0f} | "
          f"Sharpe={cs['sharpe']:.3f} | Max DD={cs['max_dd']:.2f}%")

    orb_pnl  = orb_daily
    on_pnl   = r["on_pnl"]
    comb_pnl = r["comb_pnl"]

    print(f"  {'Year':<5} {'ORB P&L':>10} {'ON P&L':>10} {'Comb P&L':>11} {'Comb Ret%':>11} {'Sharpe':>8}")
    print(f"  {'-'*58}")
    for yr in [2021, 2022, 2023, 2024, 2025]:
        orb_yr  = float(orb_pnl[orb_pnl.index.year == yr].sum())
        on_yr   = float(on_pnl[on_pnl.index.year == yr].sum())  if not on_pnl.empty else 0.0
        comb_yr = float(comb_pnl[comb_pnl.index.year == yr].sum()) if not comb_pnl.empty else 0.0
        ret_yr  = comb_yr / TOTAL_CAPITAL * 100
        # Year Sharpe
        comb_d  = comb_pnl[comb_pnl.index.year == yr]
        yr_sh   = sharpe(comb_d) if len(comb_d) > 5 else 0.0
        flag    = " <-- target" if comb_yr >= 30_000 else ""
        print(f"  {yr:<5} ${orb_yr:>+9,.0f} ${on_yr:>+9,.0f} ${comb_yr:>+10,.0f} {ret_yr:>+9.1f}%  {yr_sh:>7.3f}{flag}")

    # 2026 partial
    orb_26  = float(orb_pnl[orb_pnl.index.year == 2026].sum())
    on_26   = float(on_pnl[on_pnl.index.year == 2026].sum())  if not on_pnl.empty else 0.0
    comb_26 = float(comb_pnl[comb_pnl.index.year == 2026].sum()) if not comb_pnl.empty else 0.0
    ret_26  = comb_26 / TOTAL_CAPITAL * 100
    comb_26d = comb_pnl[comb_pnl.index.year == 2026]
    yr_sh26  = sharpe(comb_26d) if len(comb_26d) > 5 else 0.0
    print(f"  {'2026Q1':<5} ${orb_26:>+9,.0f} ${on_26:>+9,.0f} ${comb_26:>+10,.0f} {ret_26:>+9.1f}%  {yr_sh26:>7.3f}")

# ── Q1 analysis ───────────────────────────────────────────────────────────────
best = results_list[0]
best_lbl = best["cfg"]["label"]
best_comb = best["comb_pnl"]
best_on   = best["on_pnl"]

print("\n" + "="*90)
print("Q1 ANALYSIS — Does overnight complement ORB's worst quarters?")
print("="*90)
q1_years = [2021, 2022, 2023, 2024, 2025, 2026]
print(f"\n  {'Quarter':<10} {'ORB Q1':>10} {'Overnight Q1':>14} {'Combined Q1':>13} {'Better?':>8}")
print(f"  {'-'*58}")
for yr in q1_years:
    q1_mask = lambda s: s[(s.index.year == yr) & (s.index.month.isin([1,2,3]))]
    orb_q1  = float(q1_mask(orb_daily).sum())
    on_q1   = float(q1_mask(best_on).sum())
    comb_q1 = float(q1_mask(best_comb).sum())
    better  = "YES" if comb_q1 > orb_q1 else "no"
    yr_lbl  = f"{yr} Q1" + (" (partial)" if yr == 2026 else "")
    print(f"  {yr_lbl:<10} ${orb_q1:>+9,.0f}  ${on_q1:>+12,.0f}  ${comb_q1:>+11,.0f}  {better:>8}")

# ── Key answers ───────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("KEY QUESTIONS ANSWERED")
print("="*90)

passing = [r for r in results_list if r["meets_return"] and r["meets_dd"]]
best_return = results_list[0]
best_sharpe = max(results_list, key=lambda r: r["comb_stats"]["sharpe"])
best_dd_safe = max([r for r in results_list if r["meets_dd"]],
                    key=lambda r: r["comb_stats"]["avg_annual"],
                    default=results_list[0])

print(f"\n  1. Hit $30k/yr on $200k?  {'YES — ' + str(len(passing)) + ' config(s) pass' if passing else 'NO'}")
print(f"     Best: Config {best_return['cfg']['label']} ({best_return['cfg']['desc']}) "
      f"-> ${best_return['comb_stats']['avg_annual']:+,.0f}/yr")

print(f"\n  2. Stay within 15% DD?    {'YES' if best_return['meets_dd'] else 'NO (best has ' + str(round(best_return['comb_stats']['max_dd'],1)) + '% DD)'}")
print(f"     Safest high-return: Config {best_dd_safe['cfg']['label']} "
      f"-> ${best_dd_safe['comb_stats']['avg_annual']:+,.0f}/yr | "
      f"DD={best_dd_safe['comb_stats']['max_dd']:.2f}%")

print(f"\n  3. ORB vs Overnight correlation (Config {best_lbl}):")
print(f"     Pearson r = {best['corr']:.3f}  "
      f"({'low — good diversification' if abs(best['corr']) < 0.2 else 'moderate' if abs(best['corr']) < 0.4 else 'high — limited diversification benefit'})")

print(f"\n  4. Best Sharpe:  Config {best_sharpe['cfg']['label']} ({best_sharpe['cfg']['desc']}) "
      f"-> Sharpe={best_sharpe['comb_stats']['sharpe']:.3f}")

print(f"\n  5. Q1 complementarity (Config {best_lbl}):")
for yr in [2025, 2026]:
    q1_mask = lambda s: s[(s.index.year == yr) & (s.index.month.isin([1,2,3]))]
    orb_q1  = float(q1_mask(orb_daily).sum())
    on_q1   = float(q1_mask(best_on).sum())
    net     = orb_q1 + on_q1
    print(f"     Q1 {yr}: ORB ${orb_q1:+,.0f}  +  Overnight ${on_q1:+,.0f}  =  Net ${net:+,.0f}")

# ── Final recommendation ──────────────────────────────────────────────────────
print("\n" + "="*90)
print("FINAL RECOMMENDATION")
print("="*90)

rec = best_dd_safe if not passing else sorted(passing, key=lambda r: r["comb_stats"]["sharpe"], reverse=True)[0]
rec_cs = rec["comb_stats"]
rec_on = rec["on_stats"]
gap    = 30_000 - rec_cs["avg_annual"]

print(f"\n  Recommended config: {rec['cfg']['label']} — {rec['cfg']['desc']}")
print(f"  ORB:        ${orb_stats['avg_annual']:+,.0f}/yr | Sharpe {orb_stats['sharpe']:.3f} | DD {orb_stats['max_dd']:.2f}%")
print(f"  Overnight:  ${rec_on['avg_annual']:+,.0f}/yr | Sharpe {rec_on['sharpe']:.3f} | DD {rec_on['max_dd']:.2f}%")
print(f"  Combined:   ${rec_cs['avg_annual']:+,.0f}/yr | Sharpe {rec_cs['sharpe']:.3f} | DD {rec_cs['max_dd']:.2f}%")
print(f"  Final equity: ${rec_cs['final_eq']:,.0f} (started $200,000)")

if gap > 0:
    print(f"\n  Gap to $30k/yr target: ${gap:,.0f}")
    print(f"  Honest assessment:")
    print(f"    The $30k/yr target requires consistent 15% annual on $200k.")
    print(f"    Combined best reaches ~${rec_cs['avg_annual']:,.0f}/yr ({rec_cs['total_ret_pct']/TRADING_YEARS:.1f}%/yr).")
    print(f"    To close the gap without major DD increase:")
    print(f"      - Scale capital: at this return rate, need ~${30000/max(rec_cs['avg_annual']/200000,0.01):,.0f} to generate $30k/yr")
    print(f"      - Or: add a 3rd non-correlated strategy (trend-following futures, options premium)")
else:
    print(f"\n  TARGET MET: ${rec_cs['avg_annual']:,.0f}/yr on $200k within {abs(rec_cs['max_dd']):.1f}% max DD.")

print("\nDone.")
