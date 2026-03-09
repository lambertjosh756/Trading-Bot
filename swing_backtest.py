"""
swing_backtest.py -- 9-combo swing trading backtest
  3 styles x 3 hold periods on S&P 500 universe
  $75,000 capital | Jan 2021 - Mar 2026 | yfinance daily data

Styles:
  Momentum      -- price above SMA50, top-20% 20d momentum, 1.5x volume
  Mean Reversion-- 3-7% pullback from 10d high, above SMA200, RSI 35-50, low vol
  Breakout      -- close above 52-week high, 2x volume, RSI 50-70

Hold periods: 3, 7, 14 calendar-trading-days (hard 5% stop on all)
Entry: next-day open after signal | Exit: open on exit day or stop

Usage:
  python swing_backtest.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Config --------------------------------------------------------------------

START_DATE      = "2021-01-01"
END_DATE        = "2026-03-07"
STARTING_EQUITY = 75_000.0
MAX_POSITIONS   = 15
POS_SIZE        = 5_000.0
STOP_PCT        = 0.05
TC              = 0.0005        # per side

STYLES          = ["momentum", "mean_reversion", "breakout"]
HOLD_DAYS_LIST  = [3, 7, 14]

MIN_PRICE       = 20.0
MIN_AVG_VOL     = 3_000_000

UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","NFLX","ORCL","IBM",
    "INTC","AMD","QCOM","MU","AVGO","TXN","AMAT","LRCX","KLAC","ADI",
    "JPM","BAC","WFC","GS","MS","C","BLK","AXP","USB","PNC","COF","SCHW",
    "JNJ","PFE","ABT","UNH","MRK","LLY","TMO","MDT","ABBV","AMGN","GILD",
    "HD","NKE","MCD","SBUX","TGT","LOW","COST","WMT","DIS","CMCSA",
    "XOM","CVX","CAT","HON","GE","MMM","BA","DE","RTX","LIN","UPS","FDX",
    "VZ","T","KO","PEP","PM","PG","MO","CL",
    "PYPL","CRM","UBER","SQ","SHOP","ZM","NOW","SNOW","DDOG","NET",
    "NIO","RIVN","F","GM",
]
# Deduplicate preserving order
_seen: set = set()
UNIVERSE = [s for s in UNIVERSE if s not in _seen and not _seen.add(s)]  # type: ignore


# -- Data download -------------------------------------------------------------

def download_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV for all symbols. Returns {sym: df} filtered by price/vol."""
    import yfinance as yf

    # Extend end by a few days to get full close data
    print(f"  Downloading {len(symbols)} symbols from yfinance...")
    raw = yf.download(
        symbols,
        start=START_DATE,
        end="2026-03-10",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        print("ERROR: yfinance returned empty DataFrame")
        sys.exit(1)

    # Flatten MultiIndex if multiple symbols
    if isinstance(raw.columns, pd.MultiIndex):
        closes  = raw["Close"]
        opens   = raw["Open"]
        highs   = raw["High"]
        lows    = raw["Low"]
        volumes = raw["Volume"]
    else:
        # Single symbol -- wrap in dict-like structure
        closes  = raw[["Close"]].rename(columns={"Close": symbols[0]})
        opens   = raw[["Open"]].rename(columns={"Open": symbols[0]})
        highs   = raw[["High"]].rename(columns={"High": symbols[0]})
        lows    = raw[["Low"]].rename(columns={"Low": symbols[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": symbols[0]})

    result: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = pd.DataFrame({
                "open":   opens[sym],
                "high":   highs[sym],
                "low":    lows[sym],
                "close":  closes[sym],
                "volume": volumes[sym],
            }).dropna()

            if len(df) < 260:   # need at least ~1 year
                continue

            # Apply price and volume filters based on first 20 days avg
            recent_price = df["close"].iloc[-20:].mean()
            recent_vol   = df["volume"].iloc[-20:].mean()
            if recent_price < MIN_PRICE or recent_vol < MIN_AVG_VOL:
                continue

            result[sym] = df
        except Exception:
            continue

    print(f"  {len(result)} symbols passed filters (price > ${MIN_PRICE}, avg vol > {MIN_AVG_VOL/1e6:.0f}M)")
    return result


# -- Indicators ----------------------------------------------------------------

def wilder_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma50"]   = df["close"].rolling(50).mean()
    df["sma200"]  = df["close"].rolling(200).mean()
    df["rsi14"]   = wilder_rsi(df["close"], 14)
    df["mom20"]   = df["close"].pct_change(20)        # 20-day momentum
    df["vol20"]   = df["volume"].rolling(20).mean()   # 20-day avg volume
    df["hi52w"]   = df["close"].rolling(252).max()    # 52-week high
    df["hi10d"]   = df["close"].rolling(10).max()     # 10-day high (for pullback)
    df["vol_mult"]= df["volume"] / df["vol20"]        # today's vol relative to avg
    return df


# -- Signal generation ---------------------------------------------------------

def generate_signals(data: Dict[str, pd.DataFrame], style: str) -> Dict[str, pd.Series]:
    """
    Returns {symbol: pd.Series[bool]} -- True on days with a valid entry signal.
    Signals are computed on the signal day; entry is next-day open.
    """
    signals: Dict[str, pd.Series] = {}

    # For momentum: need cross-sectional percentile rank per day
    if style == "momentum":
        # Build cross-sectional mom20 per date
        mom_panel = pd.DataFrame({s: d["mom20"] for s, d in data.items()})
        # Percentile rank (0-1) across universe per row
        mom_rank  = mom_panel.rank(axis=1, pct=True)

    for sym, df in data.items():
        df = df.dropna()
        if len(df) < 252:
            continue

        if style == "momentum":
            rank = mom_rank[sym].reindex(df.index)
            sig = (
                (df["close"] > df["sma50"]) &       # in uptrend
                (rank >= 0.80) &                     # top 20% momentum
                (df["vol_mult"] >= 1.5)              # high volume confirmation
            )

        elif style == "mean_reversion":
            pullback = (df["hi10d"] - df["close"]) / df["hi10d"]
            sig = (
                (pullback >= 0.03) & (pullback <= 0.07) &  # 3-7% pullback
                (df["close"] > df["sma200"]) &              # long-term uptrend
                (df["rsi14"] >= 35) & (df["rsi14"] <= 50) &# oversold but not broken
                (df["vol_mult"] < 1.0)                      # low volume pullback
            )

        elif style == "breakout":
            # Must exceed prior 52w high (not just equal)
            prior_hi52w = df["close"].rolling(252).max().shift(1)
            sig = (
                (df["close"] > prior_hi52w) &              # new 52w high breakout
                (df["vol_mult"] >= 2.0) &                  # heavy volume
                (df["rsi14"] >= 50) & (df["rsi14"] <= 70)  # not overbought
            )
        else:
            sig = pd.Series(False, index=df.index)

        # Only keep signals within backtest window
        sig = sig.fillna(False)
        sig = sig[(sig.index >= pd.Timestamp(START_DATE)) &
                  (sig.index <= pd.Timestamp(END_DATE))]
        if sig.any():
            signals[sym] = sig

    n = sum(s.sum() for s in signals.values())
    print(f"    {style:16s}: {n:,} raw signals across {len(signals)} symbols")
    return signals


# -- Simulation ----------------------------------------------------------------

def simulate(
    data: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.Series],
    hold_days: int,
) -> dict:
    """
    Simulate one (style, hold_days) combo.
    Returns dict with trades list, equity curve, and metrics.
    """
    # Build sorted master timeline from union of all dates
    all_dates = sorted(set(
        d for df in data.values() for d in df.index
        if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)
    ))

    # Pre-build signal lookup: {date: [(symbol, entry_open_next_day)]}
    # Signal on day D -- try to enter on day D+1's open
    # Build a date-to-index map per symbol
    date_idx: Dict[str, Dict] = {
        sym: {d: i for i, d in enumerate(df.index)}
        for sym, df in data.items()
    }

    # signal_queue[D+1] = list of symbols that signaled on D
    from collections import defaultdict
    signal_queue: Dict[pd.Timestamp, List[str]] = defaultdict(list)

    for sym, sig in signals.items():
        sig_dates = sig[sig].index
        df = data[sym]
        df_idx = date_idx[sym]
        for sd in sig_dates:
            if sd not in df_idx:
                continue
            next_i = df_idx[sd] + 1
            if next_i < len(df):
                next_date = df.index[next_i]
                signal_queue[next_date].append(sym)

    equity        = STARTING_EQUITY
    equity_curve  = []
    open_pos      = []   # list of position dicts
    trades        = []

    for today in all_dates:
        # -- Step 1: Check exits ----------------------------------------------
        still_open = []
        for pos in open_pos:
            sym   = pos["symbol"]
            df    = data[sym]
            df_ix = date_idx[sym]

            if today not in df_ix:
                still_open.append(pos)
                continue

            row          = df.iloc[df_ix[today]]
            stop_hit     = row["low"] <= pos["stop_price"]
            time_exit    = (today >= pos["exit_date"])

            if stop_hit or time_exit:
                if stop_hit:
                    exit_price  = max(pos["stop_price"], row["open"])
                    exit_reason = "stop"
                else:
                    exit_price  = row["open"]
                    exit_reason = "time"

                gross_pnl   = pos["shares"] * (exit_price - pos["entry_price"])
                tc_cost     = pos["shares"] * exit_price * TC
                net_pnl     = gross_pnl - tc_cost
                equity      += net_pnl
                hold_actual  = (today - pos["entry_date"]).days

                trades.append({
                    "symbol":      sym,
                    "entry_date":  pos["entry_date"],
                    "exit_date":   today,
                    "entry_price": pos["entry_price"],
                    "exit_price":  exit_price,
                    "exit_reason": exit_reason,
                    "shares":      pos["shares"],
                    "pnl":         net_pnl,
                    "pnl_pct":     (exit_price / pos["entry_price"] - 1) * 100,
                    "hold_days":   hold_actual,
                    "entry_idx":   pos["entry_df_idx"],   # for day-by-day analysis
                    "sym_for_lookup": sym,
                })
            else:
                still_open.append(pos)

        open_pos = still_open

        # -- Step 2: Open new positions ---------------------------------------
        candidates = signal_queue.get(today, [])
        # Shuffle to avoid systematic bias (alphabetical order)
        import random
        random.seed(int(today.timestamp()) % 2**32)
        random.shuffle(candidates)

        for sym in candidates:
            if len(open_pos) >= MAX_POSITIONS:
                break
            if any(p["symbol"] == sym for p in open_pos):
                continue  # already holding this symbol

            df    = data[sym]
            df_ix = date_idx[sym]
            if today not in df_ix:
                continue

            i   = df_ix[today]
            row = df.iloc[i]

            entry_price = row["open"]
            if entry_price <= 0:
                continue

            shares      = int(POS_SIZE / entry_price)
            if shares == 0:
                continue

            stop_price  = entry_price * (1 - STOP_PCT)
            tc_cost     = shares * entry_price * TC

            # Find exit date (hold_days trading sessions after entry)
            exit_i      = i + hold_days
            exit_date   = df.index[exit_i] if exit_i < len(df) else df.index[-1]

            equity     -= tc_cost  # entry transaction cost
            open_pos.append({
                "symbol":      sym,
                "entry_date":  today,
                "exit_date":   exit_date,
                "entry_price": entry_price,
                "stop_price":  stop_price,
                "shares":      shares,
                "entry_df_idx": df_ix[today],
            })

        equity_curve.append({"date": today, "equity": equity})

    # Force-close any remaining open positions at last price
    for pos in open_pos:
        sym = pos["symbol"]
        df  = data[sym]
        last_row    = df.iloc[-1]
        exit_price  = last_row["close"]
        gross_pnl   = pos["shares"] * (exit_price - pos["entry_price"])
        tc_cost     = pos["shares"] * exit_price * TC
        net_pnl     = gross_pnl - tc_cost
        equity      += net_pnl
        trades.append({
            "symbol":      sym,
            "entry_date":  pos["entry_date"],
            "exit_date":   df.index[-1],
            "entry_price": pos["entry_price"],
            "exit_price":  exit_price,
            "exit_reason": "forced",
            "shares":      pos["shares"],
            "pnl":         net_pnl,
            "pnl_pct":     (exit_price / pos["entry_price"] - 1) * 100,
            "hold_days":   (df.index[-1] - pos["entry_date"]).days,
            "entry_idx":   pos["entry_df_idx"],
            "sym_for_lookup": sym,
        })

    eq_df   = pd.DataFrame(equity_curve).set_index("date")["equity"]
    t_df    = pd.DataFrame(trades) if trades else pd.DataFrame()

    return {
        "equity":  eq_df,
        "trades":  t_df,
        "final_equity": equity,
    }


# -- Metrics -------------------------------------------------------------------

def compute_metrics(result: dict, hold_days: int) -> dict:
    eq     = result["equity"]
    t_df   = result["trades"]
    final  = result["final_equity"]

    total_ret  = (final - STARTING_EQUITY) / STARTING_EQUITY * 100
    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max * 100
    max_dd      = float(drawdown.min())

    # Sharpe (daily returns, annualized)
    daily_ret = eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    if t_df.empty:
        return {
            "total_ret": total_ret, "max_dd": max_dd, "sharpe": sharpe,
            "n_trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "stop_rate": 0,
        }

    wins   = t_df[t_df["pnl"] > 0]["pnl"]
    losses = t_df[t_df["pnl"] <= 0]["pnl"]
    stops  = t_df[t_df["exit_reason"] == "stop"]

    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    return {
        "total_ret":     total_ret,
        "max_dd":        max_dd,
        "sharpe":        sharpe,
        "n_trades":      len(t_df),
        "win_rate":      len(wins) / len(t_df) * 100,
        "avg_win":       float(wins.mean()) if len(wins) else 0,
        "avg_loss":      float(losses.mean()) if len(losses) else 0,
        "profit_factor": profit_factor,
        "stop_rate":     len(stops) / len(t_df) * 100,
    }


def yearly_return(result: dict, year: int) -> float:
    eq = result["equity"]
    yr = eq[eq.index.year == year]
    if len(yr) < 2:
        return 0.0
    return (float(yr.iloc[-1]) - float(yr.iloc[0])) / float(yr.iloc[0]) * 100


# -- Holding-day P&L analysis --------------------------------------------------

def holding_day_returns(result: dict, data: Dict[str, pd.DataFrame]) -> Dict[int, float]:
    """
    For each trade, compute avg % return at day 1..14 after entry.
    Uses the 14-day hold results (or all trades if mixed).
    """
    t_df = result["trades"]
    if t_df.empty:
        return {}

    day_returns: Dict[int, List[float]] = {d: [] for d in range(1, 15)}

    for _, row in t_df.iterrows():
        sym        = row.get("sym_for_lookup", row["symbol"])
        entry_idx  = row.get("entry_idx")
        entry_price= row["entry_price"]
        df         = data.get(sym)
        if df is None or entry_idx is None:
            continue

        for hold_d in range(1, 15):
            lookup_i = entry_idx + hold_d
            if lookup_i >= len(df):
                break
            day_close = df.iloc[lookup_i]["close"]
            ret = (day_close - entry_price) / entry_price * 100
            day_returns[hold_d].append(ret)

    return {d: float(np.mean(v)) if v else 0.0 for d, v in day_returns.items()}


# -- Output sections -----------------------------------------------------------

def fmt_pct(v: float) -> str:
    return f"{v:>+7.2f}%"

def fmt_dd(v: float) -> str:
    return f"{v:>7.2f}%"

def print_results_matrix(all_metrics: Dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print("SECTION 1: RESULTS MATRIX  (9 combos -- $75k starting equity)")
    print("=" * 90)
    header = f"{'Combo':<22} {'Return':>8} {'MaxDD':>8} {'Sharpe':>7} {'Trades':>7} {'WinRate':>8} {'PFactor':>8} {'StopRate':>9}"
    print(header)
    print("-" * 90)
    for key, m in sorted(all_metrics.items()):
        style, hold = key.rsplit("_", 1)
        label = f"{style[:10]}-{hold}d"
        print(
            f"{label:<22} "
            f"{fmt_pct(m['total_ret'])} "
            f"{fmt_dd(m['max_dd'])} "
            f"{m['sharpe']:>7.3f} "
            f"{m['n_trades']:>7} "
            f"{m['win_rate']:>7.1f}% "
            f"{m['profit_factor']:>8.2f} "
            f"{m['stop_rate']:>8.1f}%"
        )
    print("=" * 90)


def print_best_hold_per_style(all_metrics: Dict[str, dict]) -> None:
    print("\n" + "=" * 60)
    print("SECTION 2: BEST HOLD PERIOD PER STYLE (by Sharpe)")
    print("=" * 60)
    for style in STYLES:
        style_keys = {k: v for k, v in all_metrics.items() if k.startswith(style)}
        best_key   = max(style_keys, key=lambda k: style_keys[k]["sharpe"])
        best        = style_keys[best_key]
        hold        = best_key.rsplit("_", 1)[1]
        print(
            f"  {style:<16} best hold = {hold:>3}  |  "
            f"Return {fmt_pct(best['total_ret'])}  |  "
            f"Sharpe {best['sharpe']:.3f}  |  "
            f"MaxDD {fmt_dd(best['max_dd'])}"
        )
    print("=" * 60)


def print_year_by_year(all_metrics: Dict[str, dict], all_results: Dict[str, dict]) -> None:
    years = [2021, 2022, 2023, 2024, 2025]
    print("\n" + "=" * 90)
    print("SECTION 3: YEAR-BY-YEAR RETURNS (top 3 combos per year by return)")
    print("=" * 90)
    for yr in years:
        yr_rets = {
            k: yearly_return(all_results[k], yr)
            for k in all_results
        }
        top3 = sorted(yr_rets.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  {yr}:")
        for rank, (key, ret) in enumerate(top3, 1):
            style, hold = key.rsplit("_", 1)
            print(f"    #{rank}  {style[:10]}-{hold:>3}  {fmt_pct(ret)}")
    print("=" * 90)


def print_2022_bear(all_metrics: Dict[str, dict], all_results: Dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("SECTION 4: 2022 BEAR MARKET TEST (Jan-Dec 2022 return)")
    print("=" * 70)
    print(f"  SPY 2022 return: -18.1%  (benchmark)")
    print()
    yr_rets = {k: yearly_return(all_results[k], 2022) for k in all_results}
    for key, ret in sorted(yr_rets.items(), key=lambda x: x[1], reverse=True):
        style, hold = key.rsplit("_", 1)
        label = f"{style[:10]}-{hold}"
        verdict = "BEAT SPY" if ret > -18.1 else "      "
        print(f"  {label:<22} {fmt_pct(ret)}  {verdict}")
    print("=" * 70)


def print_holding_day_chart(
    all_day_returns: Dict[str, Dict[int, float]]
) -> None:
    print("\n" + "=" * 65)
    print("SECTION 5: HOLDING-DAY P&L ANALYSIS (avg return by day held)")
    print("=" * 65)
    print("  Day  |  " + "  ".join(f"{s[:4]:>6}" for s in STYLES))
    print("  -----|--" + "--".join(["------" for _ in STYLES]))
    for d in range(1, 15):
        vals = []
        for style in STYLES:
            # Use 14-day combo for most granular view
            key  = f"{style}_14"
            avgs = all_day_returns.get(key, {})
            vals.append(avgs.get(d, 0.0))
        bar_str = "  ".join(
            f"{v:>+6.2f}%" for v in vals
        )
        # ASCII spark for momentum (first style)
        spark_val = vals[0]
        spark = "^" if spark_val > 0 else "v"
        bar_width = max(0, min(20, int(abs(spark_val) * 3)))
        bar = ("#" * bar_width).ljust(20)
        print(f"  {d:>3}d |  {bar_str}   {spark}{bar}")
    print()
    print("  (All three styles shown; 14-day hold variant used for full curve)")
    print("=" * 65)


def print_recommendation(all_metrics: Dict[str, dict]) -> None:
    print("\n" + "=" * 65)
    print("SECTION 6: FINAL RECOMMENDATION")
    print("=" * 65)

    # Score: composite of Sharpe (weight 3) + return (weight 1) + MaxDD (negative weight 2)
    scores = {}
    rets   = [m["total_ret"] for m in all_metrics.values()]
    sharpes= [m["sharpe"]    for m in all_metrics.values()]
    dds    = [m["max_dd"]    for m in all_metrics.values()]

    r_min, r_max   = min(rets), max(rets)
    s_min, s_max   = min(sharpes), max(sharpes)
    d_min, d_max   = min(dds), max(dds)   # more negative = worse

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi != lo else 0.5

    for key, m in all_metrics.items():
        scores[key] = (
            3 * norm(m["sharpe"],    s_min, s_max) +
            1 * norm(m["total_ret"], r_min, r_max) +
            2 * norm(-m["max_dd"],   -d_max, -d_min)  # flip sign: less negative = better
        )

    best_key   = max(scores, key=lambda k: scores[k])
    best_style, best_hold = best_key.rsplit("_", 1)
    best_m     = all_metrics[best_key]

    print(f"\n  RECOMMENDED COMBO: {best_style.upper().replace('_',' ')} -- {best_hold}-day hold")
    print(f"\n  Rationale (composite score of Sharpe x3, Return x1, MaxDD x2):")
    print(f"    Return      : {fmt_pct(best_m['total_ret'])}")
    print(f"    Max DD      : {fmt_dd(best_m['max_dd'])}")
    print(f"    Sharpe      : {best_m['sharpe']:.3f}")
    print(f"    Win rate    : {best_m['win_rate']:.1f}%")
    print(f"    Profit factor: {best_m['profit_factor']:.2f}")
    print(f"    Stop rate   : {best_m['stop_rate']:.1f}%")
    print()

    print("  All combos ranked:")
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        style, hold = key.rsplit("_", 1)
        m = all_metrics[key]
        marker = " <-- RECOMMENDED" if key == best_key else ""
        print(f"    {style[:10]}-{hold:>3}  score={score:.3f}  "
              f"ret={fmt_pct(m['total_ret'])}  sharpe={m['sharpe']:.3f}  "
              f"dd={fmt_dd(m['max_dd'])}{marker}")

    print()
    print("  IMPORTANT CAVEATS:")
    print("   * Backtest uses next-day open entry (realistic) but assumes fills at open.")
    print("   * Universe fixed at ~80 S&P 500 stocks -- survivorship bias present.")
    print("   * 5% hard stop may not fill at exact price during gap-downs.")
    print("   * Commission model: 0.05%/side (SEC fee + spread approximation).")
    print("=" * 65)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("SWING TRADING BACKTEST  -- 9 combos (3 styles x 3 hold periods)")
    print(f"Capital: ${STARTING_EQUITY:,.0f}  |  {START_DATE} to {END_DATE}")
    print("=" * 70)

    # 1. Download data
    data = download_data(UNIVERSE)
    if not data:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    # 2. Add indicators
    print("  Computing indicators...")
    data = {sym: add_indicators(df) for sym, df in data.items()}

    # 3. Generate signals (once per style)
    print("  Generating signals...")
    all_signals: Dict[str, Dict[str, pd.Series]] = {}
    for style in STYLES:
        print(f"    {style}:")
        all_signals[style] = generate_signals(data, style)

    # 4. Run all 9 combos
    all_results: Dict[str, dict] = {}
    all_metrics: Dict[str, dict] = {}
    all_day_returns: Dict[str, Dict[int, float]] = {}

    for style in STYLES:
        for hold_days in HOLD_DAYS_LIST:
            key = f"{style}_{hold_days}"
            print(f"  Simulating {style}-{hold_days}d...", end=" ", flush=True)
            result            = simulate(data, all_signals[style], hold_days)
            metrics           = compute_metrics(result, hold_days)
            day_rets          = holding_day_returns(result, data)
            all_results[key]  = result
            all_metrics[key]  = metrics
            all_day_returns[key] = day_rets
            print(
                f"return={metrics['total_ret']:+.1f}%  "
                f"sharpe={metrics['sharpe']:.3f}  "
                f"trades={metrics['n_trades']}"
            )

    # 5. Print 6 output sections
    print_results_matrix(all_metrics)
    print_best_hold_per_style(all_metrics)
    print_year_by_year(all_metrics, all_results)
    print_2022_bear(all_metrics, all_results)
    print_holding_day_chart(all_day_returns)
    print_recommendation(all_metrics)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
