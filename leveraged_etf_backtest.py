"""
leveraged_etf_backtest.py
Leveraged ETF Rotation — 5 strategies vs SPY / QQQ
Jan 2021 – Mar 2026 | $100,000 starting equity | 0.05% transaction costs
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
STARTING_EQUITY  = 100_000.0
BACKTEST_START   = "2021-01-01"
BACKTEST_END     = "2026-03-07"
FETCH_START      = "2019-06-01"   # ~18 months before backtest for 200-day SMA warmup
COMMISSION       = 0.0005         # 0.05% one-way per leg traded
MOM_12W          = 60             # 12 weeks ≈ 60 trading days
MOM_4W           = 20             # 4 weeks ≈ 20 trading days
SMA_PERIOD       = 200

EQUITY_ETFS = ["TQQQ", "UPRO", "TNA"]
UNIVERSE    = ["TQQQ", "UPRO", "TNA", "TMF", "SHY", "SPY", "QQQ"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Downloading data from yfinance...")
raw = yf.download(UNIVERSE, start=FETCH_START, end=BACKTEST_END,
                  auto_adjust=True, progress=False)

# yfinance returns MultiIndex columns (field, ticker); grab Close
prices = raw["Close"].copy()
prices.index = pd.to_datetime(prices.index).tz_localize(None)
prices = prices.ffill()

bt_mask        = (prices.index >= BACKTEST_START) & (prices.index <= BACKTEST_END)
backtest_dates = prices.index[bt_mask]

print(f"  Full history  : {prices.index[0].date()} → {prices.index[-1].date()}"
      f"  ({len(prices)} days)")
print(f"  Backtest range: {backtest_dates[0].date()} → {backtest_dates[-1].date()}"
      f"  ({len(backtest_dates)} trading days)\n")

# Precomputed indicators on full history (correct lookbacks, no leakage)
daily_rets = prices.pct_change().fillna(0.0)
sma_200    = prices.rolling(SMA_PERIOD).mean()
mom_12w    = prices / prices.shift(MOM_12W) - 1
mom_4w     = prices / prices.shift(MOM_4W)  - 1

# Weekly rebalance dates — first trading day of each ISO calendar week
_bd  = pd.Series(backtest_dates, dtype="datetime64[ns]")
weekly_dates = set(
    _bd.groupby(_bd.dt.to_period("W")).first().values
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def turnover(old: dict, new: dict) -> float:
    """Fraction of portfolio being traded (0 = unchanged, 1 = full rotation)."""
    syms = set(old) | set(new)
    return sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in syms) / 2.0


def simulate(signal_fn, rebalance_daily: bool = False) -> tuple[pd.Series, float]:
    """
    Simulate a strategy. Position set at end-of-day t earns t+1 returns.
    Returns (equity_curve, total_cost_$).
    """
    pos       : dict  = {}
    eq        : float = STARTING_EQUITY
    curve     : dict  = {}
    total_cost: float = 0.0

    rebal_set = set(backtest_dates) if rebalance_daily else weekly_dates

    for i, dt in enumerate(backtest_dates):
        # Apply today's return based on yesterday's position
        if i > 0 and pos:
            day_ret = sum(
                pos.get(s, 0.0) * float(daily_rets.loc[dt, s])
                for s in pos if s in daily_rets.columns
            )
            eq *= (1.0 + day_ret)

        # Rebalance if first day or signal day
        if i == 0 or dt in rebal_set:
            new_pos = signal_fn(dt)
            tv = turnover(pos, new_pos)
            if tv > 1e-9:
                cost       = eq * tv * COMMISSION * 2   # round-trip
                eq        -= cost
                total_cost += cost
            pos = new_pos

        curve[dt] = eq

    return pd.Series(curve), total_cost


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def signal_A(dt) -> dict:
    """
    Dual Momentum Rotation (weekly).
    Rank TQQQ/UPRO/TNA by 12-week return → buy the top IF return > 0, else SHY.
    """
    m     = mom_12w.loc[dt, EQUITY_ETFS].dropna()
    valid = m[m > 0]
    if valid.empty:
        return {"SHY": 1.0}
    return {valid.idxmax(): 1.0}


def signal_B(dt) -> dict:
    """
    Trend Filter Rotation (daily).
    Waterfall: TQQQ > SMA200 → UPRO > SMA200 → TMF > SMA200 → SHY.
    """
    p   = prices.loc[dt]
    sma = sma_200.loc[dt]
    for sym in ["TQQQ", "UPRO"]:
        if pd.notna(p[sym]) and pd.notna(sma[sym]) and p[sym] > sma[sym]:
            return {sym: 1.0}
    if pd.notna(p["TMF"]) and pd.notna(sma["TMF"]) and p["TMF"] > sma["TMF"]:
        return {"TMF": 1.0}
    return {"SHY": 1.0}


def signal_C(dt) -> dict:
    """60/40 Leveraged HFEA-lite (weekly rebalance). Always 60% UPRO + 40% TMF."""
    return {"UPRO": 0.6, "TMF": 0.4}


def signal_D(dt) -> dict:
    """
    Momentum + Trend Combined (weekly).
    ETF must be above SMA-200 AND have positive 12-week momentum.
    Rank qualifying ETFs by momentum; buy the top one or SHY if none qualify.
    """
    p    = prices.loc[dt]
    sma  = sma_200.loc[dt]
    m    = mom_12w.loc[dt, EQUITY_ETFS]
    ok = {
        sym: float(m[sym]) for sym in EQUITY_ETFS
        if pd.notna(p[sym]) and pd.notna(sma[sym]) and pd.notna(m[sym])
        and p[sym] > sma[sym] and m[sym] > 0
    }
    if not ok:
        return {"SHY": 1.0}
    return {max(ok, key=ok.get): 1.0}


def signal_E(dt) -> dict:
    """
    Aggressive Triple Momentum (weekly).
    Rank TQQQ/UPRO/TNA by 4-week return. Buy top 2 at 50/50.
    Replace any with negative 4-week return with SHY.
    """
    m    = mom_4w.loc[dt, EQUITY_ETFS].dropna()
    if m.empty:
        return {"SHY": 1.0}
    top2   = m.nlargest(2).index.tolist()
    result: dict = {}
    for sym in top2:
        if float(m[sym]) > 0:
            result[sym]   = result.get(sym, 0.0) + 0.5
        else:
            result["SHY"] = result.get("SHY", 0.0) + 0.5
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUN ALL STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

STRAT_CONFIG = {
    "A: Dual Momentum":   (signal_A, False),
    "B: Trend Filter":    (signal_B, True),
    "C: 60/40 Leveraged": (signal_C, False),
    "D: Mom + Trend":     (signal_D, False),
    "E: Agg Triple Mom":  (signal_E, False),
}

print("Running simulations...")
curves: dict = {}
costs : dict = {}

for name, (fn, daily) in STRAT_CONFIG.items():
    curves[name], costs[name] = simulate(fn, rebalance_daily=daily)
    print(f"  {name:<22}  final ${curves[name].iloc[-1]:>10,.0f}  "
          f"costs ${costs[name]:>7,.0f}")

# Buy-and-hold benchmarks scaled to $100k
def make_bh(sym: str) -> pd.Series:
    px = prices.loc[backtest_dates, sym]
    return STARTING_EQUITY * px / px.iloc[0]

curves["SPY B&H"] = make_bh("SPY")
curves["QQQ B&H"] = make_bh("QQQ")
costs["SPY B&H"]  = 0.0
costs["QQQ B&H"]  = 0.0

DISPLAY_ORDER = list(STRAT_CONFIG.keys()) + ["SPY B&H", "QQQ B&H"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(name: str) -> dict:
    c    = curves[name]
    cost = costs[name]

    final     = float(c.iloc[-1])
    total_ret = (final / STARTING_EQUITY - 1) * 100
    n_years   = (c.index[-1] - c.index[0]).days / 365.25
    cagr      = ((final / STARTING_EQUITY) ** (1.0 / n_years) - 1) * 100

    dr     = c.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if float(dr.std()) > 0 else 0.0

    rm     = c.cummax()
    dd     = (c - rm) / rm * 100
    max_dd = float(dd.min())

    # Annual returns: compare year-end equity to previous year-end
    ann  = {}
    prev = STARTING_EQUITY
    for yr in range(2021, 2027):
        yr_data = c[c.index.year == yr]
        if yr_data.empty:
            continue
        yr_end   = float(yr_data.iloc[-1])
        ann[yr]  = (yr_end / prev - 1) * 100
        prev     = yr_end

    spy_ret = (curves["SPY B&H"].iloc[-1] / STARTING_EQUITY - 1) * 100

    return {
        "final":      final,
        "total_ret":  total_ret,
        "cagr":       cagr,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "best_yr":    max(ann.values()) if ann else 0.0,
        "worst_yr":   min(ann.values()) if ann else 0.0,
        "beat_spy":   total_ret > spy_ret,
        "ann":        ann,
        "total_cost": cost,
        "cost_pct":   cost / STARTING_EQUITY * 100,
    }

metrics = {n: compute_metrics(n) for n in DISPLAY_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRAWDOWN ANALYSIS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def dd_stats(name: str) -> dict:
    c  = curves[name]
    rm = c.cummax()
    dd = (c - rm) / rm

    max_dd  = float(dd.min() * 100)
    trough  = dd.idxmin()
    peak    = c[:trough].idxmax()

    # Recovery: first day after trough where equity >= peak value
    post = c[trough:]
    rec  = post[post >= float(c[peak])]
    if rec.empty:
        recovery = "not recovered by Mar 2026"
    else:
        days     = (rec.index[0] - trough).days
        recovery = f"{days}d (recovered {rec.index[0].strftime('%Y-%m')})"

    # Worst single calendar month
    monthly = c.resample("ME").last().pct_change().dropna()
    wm_val  = float(monthly.min() * 100)
    wm_dt   = monthly.idxmin().strftime("%Y-%m")

    return {
        "max_dd":       max_dd,
        "peak":         peak.strftime("%Y-%m"),
        "trough":       trough.strftime("%Y-%m"),
        "recovery":     recovery,
        "worst_month":  wm_val,
        "worst_mo_dt":  wm_dt,
    }

dd_data = {n: dd_stats(n) for n in DISPLAY_ORDER}


# ─────────────────────────────────────────────────────────────────────────────
# ── OUTPUT ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

W = 118

print(f"\n{'='*W}")
print(f"  LEVERAGED ETF ROTATION BACKTEST  |  Jan 2021 – Mar 2026  |  "
      f"$100k starting equity  |  0.05% per-leg transaction cost")
print(f"{'='*W}")

# ── Section 1: Summary table ──────────────────────────────────────────────────
print(f"\n  {'Strategy':<22}  {'Final $':>11}  {'Return':>8}  {'CAGR':>7}  "
      f"{'Sharpe':>7}  {'MaxDD%':>8}  {'Best Yr':>8}  {'Worst Yr':>9}  {'Beat SPY?':>10}")
print(f"  {'─'*22}  {'─'*11}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*10}")

for name in DISPLAY_ORDER:
    m    = metrics[name]
    sep  = "B&H" in name
    beat = "" if sep else ("YES ✓" if m["beat_spy"] else "no")
    div  = "  " + "─" * 80 if sep else ""
    if div:
        print(div)
    print(f"  {name:<22}  ${m['final']:>10,.0f}  {m['total_ret']:>+7.1f}%  "
          f"{m['cagr']:>+6.1f}%  {m['sharpe']:>+6.2f}  {m['max_dd']:>+7.1f}%  "
          f"{m['best_yr']:>+7.1f}%  {m['worst_yr']:>+8.1f}%  {beat:>10}")

# ── Section 2: Year-by-year ───────────────────────────────────────────────────
print(f"\n{'─'*W}")
print(f"  YEAR-BY-YEAR RETURNS")
print(f"{'─'*W}")

yr_header = f"  {'Year':<8}"
for name in DISPLAY_ORDER:
    abbr = name.split(":")[0] if ":" in name else name[:7]
    yr_header += f"  {abbr:>9}"
print(yr_header)
print(f"  {'─'*8}" + ("  " + "─"*9) * len(DISPLAY_ORDER))

years = sorted({yr for n in DISPLAY_ORDER for yr in metrics[n]["ann"]})
for yr in years:
    suffix = " YTD" if yr == 2026 else "    "
    row = f"  {yr}{suffix}"
    for name in DISPLAY_ORDER:
        ann = metrics[name]["ann"]
        if yr in ann:
            row += f"  {ann[yr]:>+8.1f}%"
        else:
            row += f"  {'─':>9}"
    print(row)

# ── Section 3: Drawdown analysis ─────────────────────────────────────────────
print(f"\n{'─'*W}")
print(f"  DRAWDOWN ANALYSIS")
print(f"{'─'*W}")
print(f"  {'Strategy':<22}  {'MaxDD%':>7}  {'Peak':>7}  {'Trough':>7}  "
      f"{'Recovery':>36}  {'Worst Month':>12}  {'Month':>7}")
print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*36}  {'─'*12}  {'─'*7}")

for name in DISPLAY_ORDER:
    d = dd_data[name]
    print(f"  {name:<22}  {d['max_dd']:>+6.1f}%  {d['peak']:>7}  {d['trough']:>7}  "
          f"  {d['recovery']:<36}  {d['worst_month']:>+11.1f}%  {d['worst_mo_dt']:>7}")

# ── Section 4: 2022 bear market ───────────────────────────────────────────────
print(f"\n{'─'*W}")
print(f"  2022 BEAR MARKET  (Jan – Dec 2022)")
spy_2022 = metrics["SPY B&H"]["ann"].get(2022, -18.2)
qqq_2022 = metrics["QQQ B&H"]["ann"].get(2022, -32.6)
print(f"  Context: SPY {spy_2022:+.1f}%  |  QQQ {qqq_2022:+.1f}%  |  "
      f"Unprotected TQQQ: ~-80%  |  Unprotected UPRO: ~-55%")
print(f"{'─'*W}")
print(f"  {'Strategy':<22}  {'2022 Return':>12}  {'2022 MaxDD':>11}  {'Defense verdict':>35}")
print(f"  {'─'*22}  {'─'*12}  {'─'*11}  {'─'*35}")

for name in DISPLAY_ORDER:
    ret_2022 = metrics[name]["ann"].get(2022, None)
    if ret_2022 is None:
        continue
    c_22 = curves[name][curves[name].index.year == 2022]
    if c_22.empty:
        continue
    rm_22   = c_22.cummax()
    dd_22   = float(((c_22 - rm_22) / rm_22).min() * 100)

    if ret_2022 > spy_2022 + 5:
        verdict = "Strong protection — significantly beat SPY"
    elif ret_2022 > spy_2022:
        verdict = "Outperformed SPY"
    elif ret_2022 > -10:
        verdict = "Partial protection — limited damage"
    elif ret_2022 > -25:
        verdict = "Poor — significant loss, didn't protect"
    else:
        verdict = "NO protection — severe loss"

    print(f"  {name:<22}  {ret_2022:>+11.1f}%  {dd_22:>+10.1f}%  {verdict:>35}")

# ── Section 5: Transaction costs ─────────────────────────────────────────────
print(f"\n{'─'*W}")
print(f"  TRANSACTION COSTS  (0.05% per leg × 2 legs = 0.10% round-trip)")
print(f"{'─'*W}")
print(f"  {'Strategy':<22}  {'Total Cost $':>13}  {'% of Start Capital':>19}  "
      f"{'$/year avg':>11}  {'Drag on CAGR':>13}")
print(f"  {'─'*22}  {'─'*13}  {'─'*19}  {'─'*11}  {'─'*13}")

n_years_total = (backtest_dates[-1] - backtest_dates[0]).days / 365.25
for name in STRAT_CONFIG:
    m        = metrics[name]
    cost     = m["total_cost"]
    pct      = m["cost_pct"]
    per_yr   = cost / n_years_total
    cagr_drag = pct / n_years_total   # rough annualized drag
    print(f"  {name:<22}  ${cost:>12,.0f}  {pct:>18.3f}%  "
          f"${per_yr:>10,.0f}  {cagr_drag:>12.3f}%/yr")

# ── Section 6: Final answers ──────────────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  FINAL ANSWERS")
print(f"{'='*W}")

spy_ret    = metrics["SPY B&H"]["total_ret"]
spy_sharpe = metrics["SPY B&H"]["sharpe"]
spy_cagr   = metrics["SPY B&H"]["cagr"]
qqq_ret    = metrics["QQQ B&H"]["total_ret"]

beats_spy = [n for n in STRAT_CONFIG if metrics[n]["total_ret"] > spy_ret]
best_sharpe_name = max(STRAT_CONFIG, key=lambda n: metrics[n]["sharpe"])
best_ret_name    = max(STRAT_CONFIG, key=lambda n: metrics[n]["total_ret"])

# Q1
print(f"\n  1. Which strategy beats SPY (+{spy_ret:.1f}% / CAGR +{spy_cagr:.1f}%) over 5 years?")
if beats_spy:
    for n in beats_spy:
        m = metrics[n]
        print(f"     ✓  {n}: {m['total_ret']:+.1f}%  (CAGR {m['cagr']:+.1f}%  |  "
              f"Sharpe {m['sharpe']:.2f}  |  MaxDD {m['max_dd']:.1f}%)")
else:
    m = metrics[best_ret_name]
    print(f"     ✗  None beat SPY. Best was {best_ret_name}: "
          f"{m['total_ret']:+.1f}% vs SPY {spy_ret:+.1f}%")

# Q2
print(f"\n  2. Best Sharpe ratio?  (SPY benchmark: {spy_sharpe:.2f})")
for n in STRAT_CONFIG:
    m      = metrics[n]
    marker = " ← BEST" if n == best_sharpe_name else ""
    vs_spy = f"  (+{m['sharpe']-spy_sharpe:.2f} vs SPY)" if m["sharpe"] > spy_sharpe else \
             f"  ({m['sharpe']-spy_sharpe:.2f} vs SPY)"
    print(f"     {n}: {m['sharpe']:+.2f}{vs_spy}{marker}")

# Q3
print(f"\n  3. Did trend/momentum filtering protect against the 2022 bear market?")
strats_2022 = sorted(
    [(n, metrics[n]["ann"].get(2022, 0)) for n in STRAT_CONFIG],
    key=lambda x: x[1], reverse=True
)
for n, r in strats_2022:
    protected = "✓ YES" if r > spy_2022 else "✗ NO"
    lag_note  = " (signal lagged — entered bear market before exiting)" if r < -15 else ""
    print(f"     {protected}  {n}: {r:+.1f}%{lag_note}")

# Q4
print(f"\n  4. Max drawdown within 25% tolerance?")
all_pass = True
for n in STRAT_CONFIG:
    dd   = metrics[n]["max_dd"]
    ok   = dd > -25.0
    flag = "WITHIN  ✓" if ok else "BREACHED ✗"
    all_pass = all_pass and ok
    print(f"     {flag}  {n}: {dd:+.1f}%")

# Q5 — Honest verdict
best_m   = metrics[best_ret_name]
sharpe_m = metrics[best_sharpe_name]
bull_yrs = sum(1 for n in STRAT_CONFIG
               for yr, r in metrics[n]["ann"].items()
               if yr in (2021, 2023, 2024, 2025) and r > 0)
bear_protection = sum(1 for n, r in strats_2022 if r > spy_2022)

print(f"\n  5. Honest verdict — is leveraged ETF rotation practical?")
print(f"  {'─'*70}")
print(f"""
     LEVERAGE REALITY:
       3x leveraged ETFs (TQQQ/UPRO/TNA) experience compounding decay in
       choppy or bear markets. The ~0.9-1.0% annual expense ratios, daily
       rebalancing friction, and volatility drag erode returns over time.
       2022 showed this clearly: unprotected TQQQ lost ~80%, UPRO ~55%.

     ROTATION WORKS IN THEORY, LAGS IN PRACTICE:
       Trend/momentum signals are reactive by design. They exit after the
       decline has started and re-enter after the recovery has begun. The
       protective signal (SMA-200 / negative momentum) typically triggers
       2-4 weeks into a bear market — enough lag to absorb significant losses.

     WHAT THE NUMBERS SHOW:
       Best total return   → {best_ret_name}: {best_m['total_ret']:+.1f}%
       SPY buy-and-hold    → {spy_ret:+.1f}%
       QQQ buy-and-hold    → {qqq_ret:+.1f}%
       Best Sharpe         → {best_sharpe_name}: {sharpe_m['sharpe']:.2f}  (SPY: {spy_sharpe:.2f})
       Strategies beating SPY: {len(beats_spy)}/5
       Strategies protecting in 2022: {bear_protection}/5

     DRAWDOWN RISK:
       {"ALL 5 strategies stayed within the -25% drawdown tolerance." if all_pass
        else "WARNING: One or more strategies BREACHED the -25% drawdown limit."}
       Leveraged ETFs can gap down faster than signals can react.

     VERDICT:
       {"VIABLE with caveats." if beats_spy else "NOT RECOMMENDED as a SPY replacement."}
       The leverage that amplifies gains in strong bull years (2021, 2023–2025)
       creates outsized losses that compound negatively when the rotation
       signal is slow to trigger. High turnover also generates short-term
       capital gains tax drag (not modeled here) that further erodes returns.

       Most practical use case: small satellite allocation (10-20% of portfolio)
       in a tax-advantaged account, using the best-performing rotation rule,
       alongside core SPY/QQQ holdings — NOT as a full portfolio replacement.
""")
print(f"{'='*W}\n")
