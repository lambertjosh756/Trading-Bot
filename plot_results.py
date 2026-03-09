"""
plot_results.py — 3-panel backtest chart for the ORB strategy.

Panels:
  1 (top)    — Equity curve: strategy vs SPY buy-and-hold (same starting equity)
  2 (middle) — Drawdown: rolling max drawdown % over time
  3 (bottom) — Daily P&L: green bars for positive days, red for negative

Usage:
  python plot_results.py
      Auto-detects most recent equity_curve + daily CSVs in results/

  python plot_results.py --equity results/equity_curve_2021-01-01_2026-03-07.csv
      Specific equity curve file (daily CSV auto-matched by date tag)

  python plot_results.py --equity <path> --daily <path>
      Fully explicit

Chart is saved to results/equity_chart_YYYY-MM-DD.png automatically.
Interactive window shown if the matplotlib backend supports it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_pair() -> tuple[Path, Path]:
    """
    Auto-detect the most recent equity_curve + daily CSV pair in results/.
    Pairs are matched by date tag (e.g. '2021-01-01_2026-03-07').
    Returns (equity_path, daily_path).
    """
    eq_files = sorted(RESULTS_DIR.glob("equity_curve_*.csv"), reverse=True)
    if not eq_files:
        print(f"ERROR: No equity_curve_*.csv found in {RESULTS_DIR}/")
        sys.exit(1)

    for eq_path in eq_files:
        # Extract date tag from filename
        tag = eq_path.stem.replace("equity_curve_", "")
        daily_path = RESULTS_DIR / f"daily_{tag}.csv"
        if daily_path.exists():
            return eq_path, daily_path

    # Fallback: just use equity curve without daily
    return eq_files[0], None


def match_daily_to_equity(equity_path: Path) -> Path | None:
    """Given an equity curve path, find the matching daily summary CSV."""
    tag = equity_path.stem.replace("equity_curve_", "")
    daily_path = RESULTS_DIR / f"daily_{tag}.csv"
    return daily_path if daily_path.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# SPY benchmark
# ─────────────────────────────────────────────────────────────────────────────

def fetch_spy_curve(start: date, end: date, starting_equity: float) -> pd.Series | None:
    """
    Fetch SPY daily closes from yfinance and scale to starting_equity.
    Returns a pd.Series indexed by date, or None on failure.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker("SPY").history(
            start=(start - timedelta(days=5)).isoformat(),
            end=(end   + timedelta(days=5)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        closes = hist["Close"]
        # Trim to backtest window
        closes = closes[(closes.index >= pd.Timestamp(start)) &
                        (closes.index <= pd.Timestamp(end))]
        if closes.empty:
            return None
        # Scale so first day = starting_equity
        spy_curve = starting_equity * (closes / closes.iloc[0])
        spy_curve.index = spy_curve.index.date  # match equity curve index type
        return spy_curve
    except Exception as exc:
        print(f"  [warn] SPY fetch failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest(equity_path: Path, daily_path: Path | None) -> Path:
    """
    Generate 3-panel chart. Returns path to the saved PNG.
    """
    # ── Load equity curve ─────────────────────────────────────────────────────
    eq_df = pd.read_csv(equity_path, parse_dates=["date"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date").sort_index()
    eq_series = eq_df["equity"]

    start_date   = eq_series.index[0].date()
    end_date     = eq_series.index[-1].date()
    start_equity = float(eq_series.iloc[0])
    end_equity   = float(eq_series.iloc[-1])
    total_ret    = (end_equity - start_equity) / start_equity * 100

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max = eq_series.cummax()
    drawdown    = (eq_series - rolling_max) / rolling_max * 100
    max_dd      = float(drawdown.min())

    # ── Daily P&L ─────────────────────────────────────────────────────────────
    if daily_path is not None:
        daily_df = pd.read_csv(daily_path, parse_dates=["date"])
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.set_index("date").sort_index()
        day_pnl  = daily_df["day_pnl"]
    else:
        # Derive from equity curve differences
        day_pnl = eq_series.diff().dropna()
        day_pnl.name = "day_pnl"

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    print(f"  Fetching SPY benchmark ({start_date} → {end_date})...")
    spy_curve = fetch_spy_curve(start_date, end_date, start_equity)
    spy_ret = None
    if spy_curve is not None:
        spy_end = float(spy_curve.iloc[-1])
        spy_ret = (spy_end - start_equity) / start_equity * 100
        alpha   = total_ret - spy_ret
        print(f"  SPY return : {spy_ret:+.2f}%  |  Strategy: {total_ret:+.2f}%  |  Alpha: {alpha:+.2f}%")
    else:
        print("  SPY benchmark unavailable — plotting strategy only.")

    # ── Metrics for titles ────────────────────────────────────────────────────
    tag = equity_path.stem.replace("equity_curve_", "")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1.2, 1.5]},
        sharex=False,
    )
    fig.suptitle(
        f"ORB Strategy Backtest  |  {start_date} → {end_date}  |  "
        f"$100k starting equity",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────
    ax1 = axes[0]
    strat_label = f"Strategy  {total_ret:+.1f}%"
    ax1.plot(eq_series.index, eq_series.values,
             color="#1976D2", linewidth=1.8, label=strat_label, zorder=3)
    ax1.fill_between(eq_series.index, start_equity, eq_series.values,
                     where=(eq_series.values >= start_equity),
                     alpha=0.10, color="#1976D2", interpolate=True)
    ax1.fill_between(eq_series.index, start_equity, eq_series.values,
                     where=(eq_series.values < start_equity),
                     alpha=0.10, color="#E53935", interpolate=True)
    ax1.axhline(start_equity, color="#888888", linewidth=0.7, linestyle=":", zorder=1)

    if spy_curve is not None:
        spy_idx = pd.to_datetime(spy_curve.index)
        ax1.plot(spy_idx, spy_curve.values,
                 color="#9E9E9E", linewidth=1.3, linestyle="--",
                 label=f"SPY B&H  {spy_ret:+.1f}%", zorder=2)
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    alpha_str = f"  |  Alpha {alpha:+.2f}%" if spy_ret is not None else ""
    ax1.set_title(
        f"Net P&L ${end_equity - start_equity:+,.0f}  |  "
        f"Return {total_ret:+.2f}%{alpha_str}  |  Max DD {max_dd:.2f}%",
        fontsize=10,
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#E53935", alpha=0.55, label="Drawdown")
    ax2.plot(drawdown.index, drawdown.values, color="#B71C1C", linewidth=0.8)
    ax2.axhline(0, color="#888888", linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title(f"Rolling Max Drawdown  |  Worst: {max_dd:.2f}%", fontsize=10)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # ── Panel 3: Daily P&L bars ───────────────────────────────────────────────
    ax3 = axes[2]
    colors = ["#43A047" if v >= 0 else "#E53935" for v in day_pnl.values]
    ax3.bar(day_pnl.index, day_pnl.values, color=colors,
            width=1.0, alpha=0.80, linewidth=0)
    ax3.axhline(0, color="#888888", linewidth=0.6)

    n_pos  = int((day_pnl > 0).sum())
    n_neg  = int((day_pnl <= 0).sum())
    n_days = len(day_pnl)
    pos_rate = n_pos / n_days * 100 if n_days else 0

    # Rolling 20-day avg P&L overlay
    rolling_avg = day_pnl.rolling(20, min_periods=5).mean()
    ax3.plot(rolling_avg.index, rolling_avg.values,
             color="#FF8F00", linewidth=1.4, label="20-day avg", zorder=3)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax3.set_ylabel("Daily P&L ($)")
    ax3.set_title(
        f"Daily P&L  |  Positive days: {n_pos}/{n_days} ({pos_rate:.0f}%)  |  "
        f"Avg: ${day_pnl.mean():+.0f}/day",
        fontsize=10,
    )
    ax3.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:+,.0f}")
    )
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"equity_chart_{date.today().isoformat()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {out_path}")

    # ── Show interactively if backend allows ──────────────────────────────────
    try:
        backend = matplotlib.get_backend()
        if backend.lower() not in ("agg", "cairo", "pdf", "ps", "svg", "template"):
            plt.show()
        else:
            print(f"  Backend '{backend}' is non-interactive — chart saved to file only.")
    except Exception:
        pass

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ORB backtest 3-panel chart")
    parser.add_argument("--equity", type=str, default="",
                        help="Path to equity_curve CSV (auto-detected if omitted)")
    parser.add_argument("--daily",  type=str, default="",
                        help="Path to daily summary CSV (auto-matched if omitted)")
    args = parser.parse_args()

    if args.equity:
        equity_path = Path(args.equity)
        if not equity_path.exists():
            print(f"ERROR: File not found: {equity_path}")
            sys.exit(1)
        daily_path = Path(args.daily) if args.daily else match_daily_to_equity(equity_path)
    else:
        equity_path, daily_path = find_latest_pair()

    print(f"\n  Equity curve : {equity_path}")
    print(f"  Daily summary: {daily_path or '(none — deriving from equity curve)'}")

    out = plot_backtest(equity_path, daily_path)
    print(f"\n  Done. Chart → {out}\n")


if __name__ == "__main__":
    main()
