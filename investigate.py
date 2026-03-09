"""
Parallel investigations on 5-year backtest data.
Investigation 1: 2026 degradation analysis
Investigation 2: Regime filter effectiveness + threshold sensitivity
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

TRADES_FILE = "results/trades_2021-01-01_2026-03-07.csv"
DAILY_FILE  = "results/daily_2021-01-01_2026-03-07.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(TRADES_FILE, parse_dates=["date", "entry_time", "exit_time"])
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")
df["week"]  = df["date"].dt.to_period("W")
df["q"]     = df["date"].dt.to_period("Q")

# ── Fetch SPY daily bars via Alpaca ───────────────────────────────────────────
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

api_key    = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_SECRET_KEY")
data_client = StockHistoricalDataClient(api_key, api_secret)

print("Fetching SPY bars 2020-11-01 -> 2026-03-07 ...")
spy_req  = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start="2020-11-01",
    end="2026-03-07",
    feed="iex",
)
spy_resp = data_client.get_stock_bars(spy_req)
spy_raw  = (getattr(spy_resp, "data", {}) or {}).get("SPY", [])
spy_df   = pd.DataFrame([{
    "date":  b.timestamp.date(),
    "open":  b.open,
    "close": b.close,
} for b in spy_raw])
spy_df["date"] = pd.to_datetime(spy_df["date"])
spy_df = spy_df.sort_values("date").reset_index(drop=True)
print(f"  SPY bars loaded: {len(spy_df)}")

# ── Helper: SPY monthly returns ───────────────────────────────────────────────
def spy_monthly_returns(spy_df):
    """Return dict: period -> monthly return (prior-month close-to-close)."""
    spy_df = spy_df.copy()
    spy_df["month"] = spy_df["date"].dt.to_period("M")
    monthly = spy_df.groupby("month").agg(
        first_open=("open",  "first"),
        last_close=("close", "last"),
    )
    monthly["ret"] = (monthly["last_close"] - monthly["first_open"]) / monthly["first_open"]
    return monthly

spy_monthly = spy_monthly_returns(spy_df)

def get_prior_month_spy(period_m):
    """period_m is a pd.Period('M'). Return prior month SPY return or None."""
    prior = period_m - 1
    if prior in spy_monthly.index:
        return spy_monthly.loc[prior, "ret"]
    return None

# ═════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 1 — 2026 Degradation
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INVESTIGATION 1 — 2026 Degradation")
print("="*70)

df_2026 = df[df["year"] == 2026].copy()

# ── 1a. Week-by-week breakdown for 2026 ──────────────────────────────────────
print("\n--- 1a. 2026 Week-by-Week ---")
weeks_2026 = df_2026.groupby("week").apply(lambda g: pd.Series({
    "trades":    len(g),
    "win_pct":   (g["pnl"] > 0).mean() * 100,
    "net_pnl":   g["pnl"].sum(),
    "stop_pct":  (g["reason"] == "STOP").mean() * 100,
    "target_pct":(g["reason"] == "TARGET").mean() * 100,
    "time_pct":  (g["reason"] == "TIME EXIT").mean() * 100,
})).reset_index()

print(f"{'Week':<20} {'Trades':>6} {'Win%':>7} {'Net P&L':>10} {'Stop%':>8} {'Target%':>9} {'TimeExit%':>10}")
print("-"*72)
for _, r in weeks_2026.iterrows():
    print(f"{str(r['week']):<20} {r['trades']:>6.0f} {r['win_pct']:>6.1f}% {r['net_pnl']:>+10.0f} {r['stop_pct']:>7.1f}% {r['target_pct']:>8.1f}% {r['time_pct']:>9.1f}%")
print(f"\n2026 TOTAL: trades={len(df_2026)}, net={df_2026['pnl'].sum():+.0f}, win%={(df_2026['pnl']>0).mean()*100:.1f}%")

# ── 1b. Q1 comparison across all years ───────────────────────────────────────
print("\n--- 1b. Q1 Comparison Across Years ---")
q1_data = df[df["date"].dt.month.isin([1, 2, 3])].copy()
q1_data["year"] = q1_data["date"].dt.year

def sharpe_from_trades(grp):
    daily_pnl = grp.groupby("date")["pnl"].sum()
    if len(daily_pnl) < 5 or daily_pnl.std() == 0:
        return 0.0
    return daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)

q1_summary = q1_data.groupby("year").apply(lambda g: pd.Series({
    "trades":    len(g),
    "win_pct":   (g["pnl"] > 0).mean() * 100,
    "net_pnl":   g["pnl"].sum(),
    "stop_pct":  (g["reason"] == "STOP").mean() * 100,
    "sharpe":    sharpe_from_trades(g),
})).reset_index()

print(f"\n{'Year':<6} {'Sharpe':>8} {'Net P&L':>10} {'Win%':>7} {'Stop%':>8} {'Trades':>7}")
print("-"*50)
for _, r in q1_summary.iterrows():
    flag = " <-- WORST" if r["year"] == 2026 else ""
    print(f"{int(r['year']):<6} {r['sharpe']:>8.3f} {r['net_pnl']:>+10.0f} {r['win_pct']:>6.1f}% {r['stop_pct']:>7.1f}% {r['trades']:>7.0f}{flag}")

# ── 1c. SPY Jan-Mar 2026 monthly returns + regime filter simulation ──────────
print("\n--- 1c. SPY Jan-Mar 2026 + Regime Filter Impact ---")
months_2026 = ["2026-01", "2026-02", "2026-03"]

print(f"\n{'Month':<10} {'SPY prior mo%':>14} {'Filter?':>8} {'Actual P&L':>12} {'P&L if suppressed':>18}")
print("-"*66)
total_suppressed_loss = 0.0
for m_str in months_2026:
    m_period = pd.Period(m_str, freq="M")
    prior_ret = get_prior_month_spy(m_period)
    prior_pct = prior_ret * 100 if prior_ret is not None else None
    filter_active = (prior_ret is not None) and (prior_ret < -0.03)

    month_trades = df_2026[df_2026["month"] == m_period]
    actual_pnl = month_trades["pnl"].sum()

    if filter_active and actual_pnl < 0:
        avoided = -actual_pnl
        total_suppressed_loss += avoided
    elif filter_active and actual_pnl >= 0:
        avoided = -actual_pnl  # would have missed good months

    prior_str = f"{prior_pct:+.2f}%" if prior_pct is not None else "N/A"
    filter_str = "ACTIVE" if filter_active else "off"
    avoided_str = f"{-actual_pnl:+.0f}" if filter_active else "n/a"
    print(f"{m_str:<10} {prior_str:>14} {filter_str:>8} {actual_pnl:>+12.0f} {avoided_str:>18}")

# Show Jan/Feb/Mar 2026 SPY returns from data
print("\n  SPY actual monthly returns (from data):")
for m_str in ["2025-12", "2026-01", "2026-02", "2026-03"]:
    p = pd.Period(m_str, freq="M")
    if p in spy_monthly.index:
        r = spy_monthly.loc[p, "ret"]
        print(f"    {m_str}: {r*100:+.2f}%")

# ── 1d. Stop rate analysis 2026 vs full period ───────────────────────────────
print("\n--- 1d. Exit Rate Comparison ---")
def exit_stats(grp, label):
    n = len(grp)
    s = (grp["reason"] == "STOP").sum()
    t = (grp["reason"] == "TARGET").sum()
    te = (grp["reason"] == "TIME EXIT").sum()
    wins = (grp["pnl"] > 0).sum()
    print(f"  {label:<22}: n={n:>5} | STOP={s/n*100:.1f}% | TARGET={t/n*100:.1f}% | TIME={te/n*100:.1f}% | WIN={wins/n*100:.1f}% | Net={grp['pnl'].sum():>+9.0f}")

exit_stats(df,        "5-yr average")
for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
    exit_stats(df[df["year"] == yr], f"  {yr}")

# ═════════════════════════════════════════════════════════════════════════════
# INVESTIGATION 2 — Regime Filter Effectiveness (full 5 years)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("INVESTIGATION 2 — Regime Filter Effectiveness (Full 5 Years)")
print("="*70)

# Build month-by-month table
all_months = pd.period_range("2021-01", "2026-03", freq="M")

records = []
for m in all_months:
    prior_ret = get_prior_month_spy(m)
    month_trades = df[df["month"] == m]
    actual_pnl = month_trades["pnl"].sum() if len(month_trades) > 0 else 0.0
    n_trades = len(month_trades)
    records.append({
        "month":      str(m),
        "prior_spy":  prior_ret * 100 if prior_ret is not None else None,
        "filter_3":   (prior_ret is not None) and (prior_ret < -0.03),
        "actual_pnl": actual_pnl,
        "n_trades":   n_trades,
    })

month_df = pd.DataFrame(records)

# ── 2a. Full monthly table ────────────────────────────────────────────────────
print("\n--- 2a. Full Monthly Filter Table ---")
print(f"{'Month':<10} {'Prior SPY%':>11} {'Filter?':>8} {'Actual P&L':>12} {'Impact if filtered':>20}")
print("-"*65)
for _, r in month_df.iterrows():
    spy_str  = f"{r['prior_spy']:+.2f}%" if r["prior_spy"] is not None else "  N/A  "
    filt_str = "ACTIVE" if r["filter_3"] else "off"
    if r["filter_3"]:
        # Positive impact = saved a loss | Negative impact = missed a gain
        impact = -r["actual_pnl"]  # if we blocked this month, net impact on equity = -actual_pnl
        impact_str = f"{impact:>+.0f} ({'saved' if impact >= 0 else 'missed'})"
    else:
        impact_str = "n/a"
    print(f"{r['month']:<10} {spy_str:>11} {filt_str:>8} {r['actual_pnl']:>+12.0f} {impact_str:>20}")

# ── 2b. Aggregate stats ───────────────────────────────────────────────────────
print("\n--- 2b. Aggregate Stats (threshold = -3%) ---")
blocked  = month_df[month_df["filter_3"]]
unblocked = month_df[~month_df["filter_3"]]

blocked_pnl     = blocked["actual_pnl"].sum()
blocked_bad     = blocked[blocked["actual_pnl"] < 0]["actual_pnl"].sum()
blocked_good    = blocked[blocked["actual_pnl"] > 0]["actual_pnl"].sum()
total_months    = len(month_df)
blocked_months  = len(blocked)

print(f"  Total months evaluated:     {total_months}")
print(f"  Months filter would block:  {blocked_months} ({blocked_months/total_months*100:.1f}%)")
print(f"  P&L in blocked months:      {blocked_pnl:+.0f}")
print(f"    - Of which losses saved:  {-blocked_bad:+.0f}")
print(f"    - Of which gains missed:  {blocked_good:+.0f}")
print(f"  Net impact of -3% filter:   {-blocked_pnl:+.0f}")
print(f"  5-yr total P&L (actual):    {month_df['actual_pnl'].sum():+.0f}")
print(f"  5-yr P&L with filter:       {unblocked['actual_pnl'].sum():+.0f}")

# ── 2c. Sensitivity: vary threshold ──────────────────────────────────────────
print("\n--- 2c. Threshold Sensitivity ---")
thresholds = [-0.01, -0.02, -0.03, -0.04, -0.05]

print(f"\n{'Threshold':>10} {'Months blocked':>14} {'Losses saved':>14} {'Gains missed':>14} {'Net impact':>12} {'Equity w/filter':>16}")
print("-"*75)

total_actual_pnl = month_df["actual_pnl"].sum()
for t in thresholds:
    blocked_t  = month_df[month_df["prior_spy"].notna() & (month_df["prior_spy"]/100 < t)]
    kept_t     = month_df[~month_df.index.isin(blocked_t.index)]
    blocked_pnl_t  = blocked_t["actual_pnl"].sum()
    saved_losses   = -blocked_t[blocked_t["actual_pnl"] < 0]["actual_pnl"].sum()
    missed_gains   =  blocked_t[blocked_t["actual_pnl"] > 0]["actual_pnl"].sum()
    net_impact     = -blocked_pnl_t   # if we remove these months from our P&L
    equity_w_filt  = 200000 + kept_t["actual_pnl"].sum()
    print(f"{t*100:>9.0f}% {len(blocked_t):>14} {saved_losses:>+14.0f} {missed_gains:>+14.0f} {net_impact:>+12.0f} {equity_w_filt:>+16.0f}")

# Show months blocked by each threshold for -1% and -5% to illustrate over/under filtering
print("\n  Months blocked at -1% threshold:")
b1 = month_df[month_df["prior_spy"].notna() & (month_df["prior_spy"] < -1.0)]
for _, r in b1.iterrows():
    print(f"    {r['month']}: prior SPY {r['prior_spy']:+.2f}% -> actual P&L {r['actual_pnl']:+.0f}")

print("\n  Months blocked at -5% threshold:")
b5 = month_df[month_df["prior_spy"].notna() & (month_df["prior_spy"] < -5.0)]
for _, r in b5.iterrows():
    print(f"    {r['month']}: prior SPY {r['prior_spy']:+.2f}% -> actual P&L {r['actual_pnl']:+.0f}")

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL ANALYSIS")
print("="*70)

# Q1 average Sharpe
q1_sharpes = q1_summary["sharpe"].values
print(f"\nQ1 Sharpe across years: {[f'{x:.2f}' for x in q1_sharpes]}")
print(f"Q1 avg Sharpe (2021-2025): {q1_sharpes[:-1].mean():.3f}")
print(f"Q1 2026 Sharpe: {q1_sharpes[-1]:.3f}")

# Best threshold
best_net = None
best_t = None
for t in thresholds:
    blocked_t = month_df[month_df["prior_spy"].notna() & (month_df["prior_spy"]/100 < t)]
    net_impact = -blocked_t["actual_pnl"].sum()
    if best_net is None or net_impact > best_net:
        best_net = net_impact
        best_t = t

print(f"\nBest threshold by net impact: {best_t*100:.0f}% (net impact: {best_net:+.0f})")
print("\nDone.")
