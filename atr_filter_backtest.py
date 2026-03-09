"""
atr_filter_backtest.py

Backtest a 5-day SPY ATR% choppiness filter on the full 5-year dataset.
Pure analysis on existing trade CSV — no new simulations required.

For each trading day:
  ATR% = mean(true_range[-5:]) / SPY_close * 100
  If ATR% > threshold -> skip that day (remove its trades)

6 configurations tested:
  0. No filter (baseline)
  1. ATR% > 0.8
  2. ATR% > 1.0
  3. ATR% > 1.2
  4. ATR% > 1.5
  5. ATR% > 2.0
"""
import os, warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

TRADES_FILE   = "results/trades_2021-01-01_2026-03-07.csv"
START_EQUITY  = 200_000.0
THRESHOLDS    = [None, 0.8, 1.0, 1.2, 1.5, 2.0]

# ── 1. Load trades ────────────────────────────────────────────────────────────
df = pd.read_csv(TRADES_FILE, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"]).dt.date
all_trading_days = sorted(df["date"].unique())
print(f"Trades loaded: {len(df):,} across {len(all_trading_days)} trading days")

# ── 2. Fetch SPY daily OHLC ───────────────────────────────────────────────────
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

api_key    = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_SECRET_KEY")
client     = StockHistoricalDataClient(api_key, api_secret)

print("Fetching SPY daily OHLC 2020-12-01 -> 2026-03-10 ...")
req = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start="2020-12-01",
    end="2026-03-10",
    feed="iex",
)
resp = client.get_stock_bars(req)
spy_raw = (getattr(resp, "data", {}) or {}).get("SPY", [])
spy = pd.DataFrame([{
    "date":  b.timestamp.date(),
    "high":  b.high,
    "low":   b.low,
    "close": b.close,
} for b in spy_raw])
spy["date"] = pd.to_datetime(spy["date"]).dt.date
spy = spy.sort_values("date").reset_index(drop=True)
print(f"  SPY bars: {len(spy)} days loaded")

# ── 3. Compute 5-day ATR% for each day ───────────────────────────────────────
# True Range uses: max(H-L, |H-prev_C|, |L-prev_C|)
spy["prev_close"] = spy["close"].shift(1)
spy["tr"] = spy.apply(
    lambda r: max(
        r["high"] - r["low"],
        abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
        abs(r["low"]  - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
    ), axis=1
)
# 5-day ATR = rolling mean of last 5 TRs (use prior 5 days, not including today)
spy["atr5"] = spy["tr"].rolling(5).mean().shift(1)   # shift(1) = use only prior days
spy["atr_pct"] = spy["atr5"] / spy["close"] * 100

# Build lookup: trading_date -> atr_pct
spy_lookup = spy.set_index("date")[["atr_pct", "close"]].to_dict("index")

# For each trading day in the backtest, get ATR%
day_atr = {}
for d in all_trading_days:
    info = spy_lookup.get(d)
    day_atr[d] = info["atr_pct"] if info and pd.notna(info["atr_pct"]) else None

# ── 4. Stats helper ───────────────────────────────────────────────────────────

def calc_stats(trades_subset, all_days, skipped_days):
    n_all   = len(all_days)
    n_skip  = len(skipped_days)
    n_trade_days = n_all - n_skip

    if len(trades_subset) == 0:
        return {
            "n_trades": 0, "n_days_traded": n_trade_days,
            "n_days_skipped": n_skip, "win_pct": 0.0,
            "net_pnl": 0.0, "sharpe": 0.0, "max_dd": 0.0,
        }

    daily_pnl = trades_subset.groupby("date")["pnl"].sum().reindex(
        pd.Index([d for d in all_days if d not in skipped_days]),
        fill_value=0.0
    )

    # Sharpe (annualised daily)
    if daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown on equity curve
    equity = START_EQUITY + daily_pnl.cumsum()
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    return {
        "n_trades":       len(trades_subset),
        "n_days_traded":  n_trade_days,
        "n_days_skipped": n_skip,
        "win_pct":        (trades_subset["pnl"] > 0).mean() * 100,
        "net_pnl":        trades_subset["pnl"].sum(),
        "sharpe":         sharpe,
        "max_dd":         max_dd,
        "daily_pnl":      daily_pnl,
    }

# ── 5. Specific date ranges for spot-checks ──────────────────────────────────
import datetime

feb_blowup_days = {
    d for d in all_trading_days
    if datetime.date(2026, 2, 9) <= d <= datetime.date(2026, 2, 22)
}
oct2022_days = {d for d in all_trading_days if d.year == 2022 and d.month == 10}
mar2022_days = {d for d in all_trading_days if d.year == 2022 and d.month == 3}
jan2023_days = {d for d in all_trading_days if d.year == 2023 and d.month == 1}

# ── 6. Run all configurations ─────────────────────────────────────────────────
print("\n" + "="*80)
print("ATR CHOPPINESS FILTER — 6-CONFIG BACKTEST")
print("="*80)

results = []

for thr in THRESHOLDS:
    label = f"No filter (baseline)" if thr is None else f"ATR% > {thr}"

    if thr is None:
        skipped = set()
    else:
        skipped = {d for d, a in day_atr.items() if a is not None and a > thr}

    kept_trades = df[~df["date"].isin(skipped)]
    stats = calc_stats(kept_trades, all_trading_days, skipped)

    # Spot checks
    feb_skipped    = feb_blowup_days & skipped
    feb_total      = len(feb_blowup_days)
    feb_caught_pct = len(feb_skipped) / feb_total * 100 if feb_total else 0

    oct22_skipped  = len(oct2022_days & skipped)
    mar22_skipped  = len(mar2022_days & skipped)
    jan23_skipped  = len(jan2023_days & skipped)

    oct22_pnl = df[df["date"].isin(oct2022_days) & ~df["date"].isin(skipped)]["pnl"].sum()
    mar22_pnl = df[df["date"].isin(mar2022_days) & ~df["date"].isin(skipped)]["pnl"].sum()
    jan23_pnl = df[df["date"].isin(jan2023_days) & ~df["date"].isin(skipped)]["pnl"].sum()

    feb_pnl_saved = df[df["date"].isin(feb_blowup_days) & df["date"].isin(skipped)]["pnl"].sum()

    # ATR% on feb blowup days
    feb_atrs = [(d, day_atr.get(d)) for d in sorted(feb_blowup_days)]

    r = {
        "label":           label,
        "threshold":       thr,
        "n_days_skipped":  stats["n_days_skipped"],
        "n_days_traded":   stats["n_days_traded"],
        "n_trades":        stats["n_trades"],
        "win_pct":         stats["win_pct"],
        "net_pnl":         stats["net_pnl"],
        "sharpe":          stats["sharpe"],
        "max_dd":          stats["max_dd"],
        "feb_caught_pct":  feb_caught_pct,
        "feb_days_caught": len(feb_skipped),
        "feb_pnl_saved":   -feb_pnl_saved,   # positive = saved loss
        "oct22_skipped":   oct22_skipped,
        "mar22_skipped":   mar22_skipped,
        "jan23_skipped":   jan23_skipped,
        "oct22_pnl_kept":  oct22_pnl,
        "mar22_pnl_kept":  mar22_pnl,
        "jan23_pnl_kept":  jan23_pnl,
        "feb_atrs":        feb_atrs,
    }
    results.append(r)

# ── 7. Per-config detailed output ─────────────────────────────────────────────
baseline = results[0]

for r in results:
    thr = r["threshold"]
    pnl_delta   = r["net_pnl"]  - baseline["net_pnl"]
    sharpe_delta = r["sharpe"] - baseline["sharpe"]

    print(f"\n{'─'*70}")
    print(f"  CONFIG: {r['label']}")
    print(f"{'─'*70}")
    print(f"  Days traded:    {r['n_days_traded']} / {len(all_trading_days)}  "
          f"({r['n_days_skipped']} skipped, {r['n_days_skipped']/len(all_trading_days)*100:.1f}%)")
    print(f"  Trades:         {r['n_trades']:,}  |  Win%: {r['win_pct']:.1f}%")
    print(f"  Net P&L:        ${r['net_pnl']:+,.0f}  |  Sharpe: {r['sharpe']:.3f}  |  Max DD: {r['max_dd']:.2f}%")
    if thr is not None:
        print(f"  vs baseline:    P&L {pnl_delta:+,.0f}  |  Sharpe {sharpe_delta:+.3f}")

    # Feb blowup check
    print(f"\n  Feb 9-22 2026 blowup check:")
    print(f"    Trading days in window: {len(feb_blowup_days)}")
    if thr is None:
        print(f"    ATR% on those days:")
        for d, a in r["feb_atrs"]:
            astr = f"{a:.3f}%" if a is not None else "N/A"
            print(f"      {d}: ATR% = {astr}")
    else:
        print(f"    Days caught (skipped): {r['feb_days_caught']} / {len(feb_blowup_days)} "
              f"({r['feb_caught_pct']:.0f}%)")
        print(f"    P&L saved from skips:  ${r['feb_pnl_saved']:+,.0f}")

    # Big winning months check
    print(f"\n  Big winning months impact:")
    print(f"    Oct 2022: {r['oct22_skipped']} days skipped -> P&L kept: ${r['oct22_pnl_kept']:+,.0f}")
    print(f"    Mar 2022: {r['mar22_skipped']} days skipped -> P&L kept: ${r['mar22_pnl_kept']:+,.0f}")
    print(f"    Jan 2023: {r['jan23_skipped']} days skipped -> P&L kept: ${r['jan23_pnl_kept']:+,.0f}")

# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Threshold':<22} {'Days skip':>10} {'Net P&L':>10} {'Sharpe':>8} "
      f"{'vs base $':>11} {'vs base Sh':>11} {'Feb caught':>11}")
print("-"*85)

for r in results:
    thr = r["threshold"]
    pnl_delta    = r["net_pnl"]  - baseline["net_pnl"]
    sharpe_delta = r["sharpe"] - baseline["sharpe"]
    feb_str      = f"{r['feb_days_caught']}/{len(feb_blowup_days)} ({r['feb_caught_pct']:.0f}%)"

    label_str    = r["label"]
    skip_str     = f"{r['n_days_skipped']} ({r['n_days_skipped']/len(all_trading_days)*100:.1f}%)"
    pnl_str      = f"${r['net_pnl']:+,.0f}"
    base_pnl_str = f"{pnl_delta:+,.0f}" if thr is not None else "---"
    base_sh_str  = f"{sharpe_delta:+.3f}" if thr is not None else "---"

    print(f"{label_str:<22} {skip_str:>10} {pnl_str:>10} {r['sharpe']:>8.3f} "
          f"{base_pnl_str:>11} {base_sh_str:>11} {feb_str:>11}")

# ── 9. ATR% distribution ─────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SPY 5-DAY ATR% DISTRIBUTION ON BACKTEST TRADING DAYS")
print("="*80)
atrs = [a for d, a in day_atr.items() if a is not None and d in set(all_trading_days)]
atrs = sorted(atrs)
pcts = [10, 25, 50, 75, 90, 95, 99]
print(f"  Count: {len(atrs)} days with ATR data")
print(f"  Min: {min(atrs):.3f}%  |  Max: {max(atrs):.3f}%  |  Mean: {np.mean(atrs):.3f}%")
print(f"\n  Percentile distribution:")
for p in pcts:
    v = np.percentile(atrs, p)
    # How many days have ATR > v
    above = sum(1 for a in atrs if a > v)
    print(f"    p{p:2d}: {v:.3f}%  ->  {above} days above ({above/len(atrs)*100:.1f}%)")

# ── 10. Monthly ATR% for key periods ─────────────────────────────────────────
print("\n\n" + "="*80)
print("MONTHLY AVG ATR% — KEY PERIODS")
print("="*80)
key_months = [
    ("2022-03", mar2022_days, "Mar 2022 (big win +$3,008)"),
    ("2022-10", oct2022_days, "Oct 2022 (big win +$2,115)"),
    ("2023-01", jan2023_days, "Jan 2023 (big win +$1,794)"),
    ("2026-02", {d for d in all_trading_days if d.year == 2026 and d.month == 2}, "Feb 2026 (blowup -$1,781)"),
]
for label, days, desc in key_months:
    month_atrs = [day_atr[d] for d in days if day_atr.get(d) is not None]
    if month_atrs:
        print(f"  {desc}:")
        print(f"    ATR% range: {min(month_atrs):.3f}% - {max(month_atrs):.3f}%  |  mean: {np.mean(month_atrs):.3f}%")
        for thr in [0.8, 1.0, 1.2, 1.5]:
            blocked = sum(1 for a in month_atrs if a > thr)
            print(f"      > {thr}%: {blocked}/{len(month_atrs)} days blocked")

# ── 11. Final verdict ─────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("FINAL VERDICT")
print("="*80)
best = max(results[1:], key=lambda r: r["net_pnl"])
best_sharpe = max(results[1:], key=lambda r: r["sharpe"])
print(f"\n  Best P&L filter:    {best['label']} -> ${best['net_pnl']:+,.0f} "
      f"(vs baseline ${baseline['net_pnl']:+,.0f}, delta={best['net_pnl']-baseline['net_pnl']:+,.0f})")
print(f"  Best Sharpe filter: {best_sharpe['label']} -> {best_sharpe['sharpe']:.3f} "
      f"(vs baseline {baseline['sharpe']:.3f}, delta={best_sharpe['sharpe']-baseline['sharpe']:+.3f})")
print()
any_beat = [r for r in results[1:] if r["net_pnl"] > baseline["net_pnl"]]
any_beat_sh = [r for r in results[1:] if r["sharpe"] > baseline["sharpe"]]
if any_beat:
    print(f"  BEATS BASELINE P&L:    {[r['label'] for r in any_beat]}")
else:
    print(f"  NO threshold beats baseline P&L")
if any_beat_sh:
    print(f"  BEATS BASELINE SHARPE: {[r['label'] for r in any_beat_sh]}")
else:
    print(f"  NO threshold beats baseline Sharpe")

print("\nDone.")
