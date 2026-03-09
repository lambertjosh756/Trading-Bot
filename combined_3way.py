"""
combined_3way.py

Three-strategy combined portfolio backtest.
ORB + Overnight Momentum (top 5, 20-day, $50k) + Iron Condors on SPX ($50k)
Jan 2021 - Mar 2026 | $200,000 total capital

Iron condor pricing: Black-Scholes with VIX as annualized IV proxy.
10% bid-ask haircut on opening credit.
Daily monitoring for stop (2x credit) and profit target (50% credit).
"""
from __future__ import annotations

import warnings, os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(".env")

# ── Capital split ─────────────────────────────────────────────────────────────
ORB_CAPITAL        = 100_000.0
OVERNIGHT_CAPITAL  =  50_000.0
SHORTVOL_CAPITAL   =  50_000.0
TOTAL_CAPITAL      = 200_000.0
START              = "2021-01-01"
END                = "2026-03-07"
TRADING_YEARS      = 5 + 67/252

DATA_DIR = Path("data/daily")

# ── Black-Scholes helpers ─────────────────────────────────────────────────────
RISK_FREE = 0.045   # ~avg fed funds 2021-2026

def bs_price(S, K, T, r, sigma, opt):
    if T <= 1e-6:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def strike_for_delta(S, T, r, sigma, target_delta, opt):
    """Find the strike producing target_delta for call (+) or put (-)."""
    if T <= 1e-6:
        return S
    if opt == "call":
        d1 = norm.ppf(target_delta)
    else:
        d1 = norm.ppf(1.0 + target_delta)  # target_delta is negative
    return S * np.exp((r + 0.5 * sigma**2) * T - d1 * sigma * np.sqrt(T))

def condor_value(S, K_sc, K_lc, K_sp, K_lp, T, r, sigma):
    """Current theoretical value of the short iron condor (seller receives this)."""
    if T <= 1e-6:
        cs = max(S - K_sc, 0.0) - max(S - K_lc, 0.0)
        ps = max(K_sp - S, 0.0) - max(K_lp - S, 0.0)
        return cs + ps   # intrinsic cost to seller (loss from sold condor)
    cs = bs_price(S, K_sc, T, r, sigma, "call") - bs_price(S, K_lc, T, r, sigma, "call")
    ps = bs_price(S, K_sp, T, r, sigma, "put")  - bs_price(S, K_lp, T, r, sigma, "put")
    return cs + ps

# ── Helpers ───────────────────────────────────────────────────────────────────
def sharpe(pnl: pd.Series) -> float:
    s = pnl.std()
    return float(pnl.mean() / s * np.sqrt(252)) if s > 0 else 0.0

def max_dd(equity: pd.Series) -> float:
    rm = equity.cummax()
    return float(((equity - rm) / rm * 100).min())

def pf(pnl: pd.Series) -> float:
    g = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return float(g / l) if l > 0 else float("inf")

def stats(pnl: pd.Series, cap: float, label: str) -> dict:
    if pnl.empty:
        return dict(label=label, net=0, annual=0, sharpe=0, dd=0, pf=0, final=cap)
    eq  = cap + pnl.cumsum()
    net = float(pnl.sum())
    return dict(
        label=label,
        net=net,
        annual=net / TRADING_YEARS,
        sharpe=sharpe(pnl),
        dd=max_dd(eq),
        pf=pf(pnl),
        final=float(eq.iloc[-1]),
        pnl_series=pnl,
    )

def corr(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.intersection(b.index)
    return float(a[idx].corr(b[idx])) if len(idx) > 20 else float("nan")

def combine(*series) -> pd.Series:
    idx = series[0].index
    for s in series[1:]:
        idx = idx.union(s.index)
    result = pd.Series(0.0, index=idx)
    for s in series:
        result = result.add(s.reindex(idx, fill_value=0), fill_value=0)
    return result.sort_index()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — ORB daily P&L (already run at $100k config 17)
# ═════════════════════════════════════════════════════════════════════════════
print("="*70)
print("THREE-STRATEGY COMBINED PORTFOLIO")
print("="*70)

orb_csv = Path("results/daily_2021-01-01_2026-03-07.csv")
if not orb_csv.exists():
    print("ERROR: ORB daily CSV not found. Run combined_portfolio.py first.")
    import sys; sys.exit(1)

orb_raw   = pd.read_csv(orb_csv, parse_dates=["date"]).set_index("date")
orb_pnl   = orb_raw["day_pnl"].copy()
orb_pnl.index = pd.to_datetime(orb_pnl.index)
print(f"[1/4] ORB loaded: {len(orb_pnl)} days | net=${orb_pnl.sum():+,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Overnight momentum at $50k (Config B: top 5, 20-day)
# ═════════════════════════════════════════════════════════════════════════════
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

OVERNIGHT_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AMD",
    "INTC","QCOM","MU","CRM","ORCL","ADBE","CSCO","TXN",
    "AMAT","LRCX","NOW","PANW","CRWD","FTNT",
    "JNJ","PFE","MRK","ABBV","AMGN","GILD","REGN","VRTX","ISRG","UNH",
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","SPGI",
    "XOM","CVX","COP","OXY",
    "HD","LOW","TGT","COST","MCD","SBUX","NKE","GM","F","LULU",
    "WMT","PG","KO","PEP",
    "BA","LMT","GE","HON","CAT","DE","UNP",
    "FCX","NEM","LIN","NFLX","DIS","CMCSA","T","VZ",
    "SPY","QQQ","IWM",
]

def load_parquet(sym: str) -> Optional[pd.DataFrame]:
    p = DATA_DIR / f"{sym}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def calc_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    ag = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50)

def simulate_overnight(capital, top_n, mom_window, rsi_filter=False):
    closes_d, opens_d, vols_d = {}, {}, {}
    for sym in OVERNIGHT_UNIVERSE:
        df = load_parquet(sym)
        if df is not None and len(df) > mom_window + 5:
            closes_d[sym] = df["close"]
            opens_d[sym]  = df["open"]
            vols_d[sym]   = df["volume"]

    closes  = pd.DataFrame(closes_d).sort_index()
    opens   = pd.DataFrame(opens_d).sort_index()
    volumes = pd.DataFrame(vols_d).sort_index()

    rsi_df = pd.DataFrame({s: calc_rsi_series(closes[s]) for s in closes.columns})

    trade_dates = closes.index[
        (closes.index >= pd.Timestamp(START)) & (closes.index <= pd.Timestamp(END))
    ]
    daily_pnl: Dict = {}
    alloc = capital / top_n

    for today in trade_dates:
        future = closes.index[closes.index > today]
        if not len(future): continue
        tomorrow = future[0]
        past = closes.index[closes.index < today]
        if len(past) < mom_window: continue
        past_day = past[-mom_window]

        scores = {}
        for sym in closes.columns:
            try:
                ct, c0 = closes.loc[today, sym], closes.loc[past_day, sym]
                if pd.isna(ct) or pd.isna(c0) or c0 == 0 or ct < 10: continue
                if volumes[sym][volumes.index <= today].tail(20).mean() < 2e6: continue
                if rsi_filter and rsi_df.loc[today, sym] <= 50: continue
                scores[sym] = (ct - c0) / c0
            except: continue

        if not scores: continue
        top = sorted(scores, key=scores.get, reverse=True)[:top_n]
        day_pnl = 0.0
        for sym in top:
            try:
                ep = closes.loc[today, sym]
                xp = opens.loc[tomorrow, sym]
                if pd.isna(ep) or pd.isna(xp) or ep == 0: continue
                day_pnl += (xp - ep) / ep * alloc
            except: continue
        if day_pnl != 0:
            daily_pnl[tomorrow] = daily_pnl.get(tomorrow, 0) + day_pnl

    s = pd.Series(daily_pnl).sort_index()
    s.index = pd.to_datetime(s.index)
    return s

print("[2/4] Running overnight momentum ($50k, top 5, 20-day)...")
on_pnl = simulate_overnight(OVERNIGHT_CAPITAL, top_n=5, mom_window=20)
print(f"      {len(on_pnl)} exit days | net=${on_pnl.sum():+,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Fetch SPX and VIX from yfinance
# ═════════════════════════════════════════════════════════════════════════════
print("[3/4] Fetching SPX and VIX from yfinance...")
_spx = yf.download("^GSPC", start="2020-11-01", end="2026-03-10", progress=False, auto_adjust=True)
_vix = yf.download("^VIX",  start="2020-11-01", end="2026-03-10", progress=False, auto_adjust=True)
_spy = yf.download("SPY",   start="2020-11-01", end="2026-03-10", progress=False, auto_adjust=True)

# Flatten multi-level columns if present
def clean_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

_spx = clean_yf(_spx); _vix = clean_yf(_vix); _spy = clean_yf(_spy)

spx_close = _spx["Close"].dropna()
vix_close = _vix["Close"].dropna()
spy_ret   = _spy["Close"].pct_change().dropna()

print(f"      SPX: {len(spx_close)} days | VIX: {len(vix_close)} days")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Iron Condor simulator
# ═════════════════════════════════════════════════════════════════════════════
SPREAD_WIDTH   = 25.0    # SPX points
ALLOC_PER_COND = 0.40    # 40% of short-vol capital per condor
CONTRACTS      = int(SHORTVOL_CAPITAL * ALLOC_PER_COND / (SPREAD_WIDTH * 100))  # 8
HAIRCUT        = 0.10
STOP_MULT      = 2.0     # close if condor costs 2x initial credit
PROFIT_TARGET  = 0.50    # close if condor worth 50% of initial credit

def simulate_iron_condors(
    short_delta: float,
    vix_lo: float,
    vix_hi: float,
    entry_weekday: int,  # 0=Mon, 2=Wed
    label: str,
) -> Tuple[pd.Series, dict]:
    """
    Simulate weekly iron condors on SPX.
    Returns daily P&L series and trade log summary.
    """
    long_delta = short_delta / 3.2   # rough: 5-delta for 16, ~3-delta for 10

    # Build trading day calendar from SPX data
    trading_days = sorted(spx_close.index[
        (spx_close.index >= pd.Timestamp(START)) &
        (spx_close.index <= pd.Timestamp(END))
    ])

    # Find entry days: first occurrence of entry_weekday in each calendar week
    entry_days = []
    seen_weeks = set()
    for d in trading_days:
        iso_week = (d.year, d.isocalendar()[1])
        if iso_week not in seen_weeks:
            # Find the entry_weekday day in this week or next available trading day
            week_start = d - timedelta(days=d.weekday())  # Monday of that calendar week
            target = week_start + timedelta(days=entry_weekday)
            # Find actual trading day on or after target within same week
            week_candidates = [t for t in trading_days
                               if week_start <= t <= week_start + timedelta(days=4)
                               and t >= target]
            if week_candidates:
                seen_weeks.add(iso_week)
                entry_days.append(week_candidates[0])

    daily_pnl: Dict[pd.Timestamp, float] = {}
    stats_log = dict(trades=0, expired=0, stopped=0, targeted=0,
                     total_credit=0.0, total_pnl=0.0, skipped_vix=0)

    for entry in entry_days:
        S = spx_close.get(entry)
        V = vix_close.get(entry)
        if S is None or V is None or pd.isna(S) or pd.isna(V):
            continue

        # VIX filters
        if V < vix_lo or V > vix_hi:
            stats_log["skipped_vix"] += 1
            continue

        # Find Friday expiry of same calendar week
        week_start = entry - timedelta(days=entry.weekday())
        friday = week_start + timedelta(days=4)
        # Find actual last trading day of the week
        expiry_candidates = [t for t in trading_days
                             if entry < t <= friday + timedelta(days=3)
                             and t.isocalendar()[1] == entry.isocalendar()[1]]
        if not expiry_candidates:
            continue
        expiry = expiry_candidates[-1]   # last trading day of the week

        T0 = max((expiry - entry).days / 365.0, 1/365)
        sigma = V / 100.0

        # Find strikes
        K_sc = strike_for_delta(S, T0, RISK_FREE, sigma,  short_delta, "call")
        K_lc = K_sc + SPREAD_WIDTH
        K_sp = strike_for_delta(S, T0, RISK_FREE, sigma, -short_delta, "put")
        K_lp = K_sp - SPREAD_WIDTH

        # Initial condor theoretical value (credit seller collects)
        cv0 = condor_value(S, K_sc, K_lc, K_sp, K_lp, T0, RISK_FREE, sigma)
        if cv0 <= 0:
            continue

        credit_per_contract = cv0 * (1 - HAIRCUT) * 100   # dollars per contract
        total_credit = credit_per_contract * CONTRACTS
        stats_log["total_credit"] += total_credit
        stats_log["trades"] += 1

        # Monitor daily until expiry
        exit_pnl = None
        exit_reason = "EXPIRY"

        # Days after entry up to and including expiry
        monitor_days = [t for t in trading_days if entry < t <= expiry]
        for d in monitor_days:
            Sd = spx_close.get(d)
            Vd = vix_close.get(d)
            if Sd is None or Vd is None or pd.isna(Sd) or pd.isna(Vd):
                continue

            T_rem = max((expiry - d).days / 365.0, 0.0)
            sigma_d = Vd / 100.0

            cv_now = condor_value(Sd, K_sc, K_lc, K_sp, K_lp, T_rem, RISK_FREE, sigma_d)
            cv_dollars = cv_now * 100 * CONTRACTS

            if d == expiry:
                # Expiry: use intrinsic (T=0)
                cv_expiry = condor_value(Sd, K_sc, K_lc, K_sp, K_lp, 0.0, RISK_FREE, 0.01)
                exit_pnl = total_credit - cv_expiry * 100 * CONTRACTS
                exit_reason = "EXPIRY"
                break

            # Stop loss: condor worth 2x initial credit
            if cv_dollars >= STOP_MULT * total_credit:
                exit_pnl = total_credit - cv_dollars
                exit_reason = "STOP"
                break

            # Profit target: condor worth <= 50% of initial credit
            if cv_dollars <= PROFIT_TARGET * total_credit:
                exit_pnl = total_credit - cv_dollars
                exit_reason = "TARGET"
                break

        if exit_pnl is None:
            # Shouldn't happen; mark as expired worthless
            exit_pnl = total_credit
            exit_reason = "EXPIRY"

        # Record on exit date
        exit_date = monitor_days[-1] if monitor_days else entry
        daily_pnl[exit_date] = daily_pnl.get(exit_date, 0.0) + exit_pnl
        stats_log["total_pnl"] += exit_pnl
        if exit_reason == "EXPIRY":   stats_log["expired"]  += 1
        elif exit_reason == "STOP":   stats_log["stopped"]  += 1
        elif exit_reason == "TARGET": stats_log["targeted"] += 1

    series = pd.Series(daily_pnl).sort_index()
    series.index = pd.to_datetime(series.index)
    return series, stats_log


# ── 4 iron condor configs ─────────────────────────────────────────────────────
ic_configs = [
    dict(label="A", desc="16-delta, VIX 12-999, Monday",
         short_delta=0.16, vix_lo=12, vix_hi=999, entry_weekday=0),
    dict(label="B", desc="10-delta, VIX 12-999, Monday",
         short_delta=0.10, vix_lo=12, vix_hi=999, entry_weekday=0),
    dict(label="C", desc="16-delta, VIX 12-25,  Monday",
         short_delta=0.16, vix_lo=12, vix_hi=25,  entry_weekday=0),
    dict(label="D", desc="16-delta, VIX 12-999, Wednesday",
         short_delta=0.16, vix_lo=12, vix_hi=999, entry_weekday=2),
]

print("[4/4] Simulating iron condors (4 configs)...")
ic_results = {}
for cfg in ic_configs:
    pnl, log = simulate_iron_condors(
        short_delta=cfg["short_delta"],
        vix_lo=cfg["vix_lo"],
        vix_hi=cfg["vix_hi"],
        entry_weekday=cfg["entry_weekday"],
        label=cfg["label"],
    )
    ic_results[cfg["label"]] = (pnl, log, cfg)
    wr = log["expired"] + log["targeted"]
    print(f"  Config {cfg['label']}: {log['trades']} trades | "
          f"expired={log['expired']} target={log['targeted']} stop={log['stopped']} "
          f"skipped={log['skipped_vix']} | net=${pnl.sum():+,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Combine and report
# ═════════════════════════════════════════════════════════════════════════════
orb_st  = stats(orb_pnl, ORB_CAPITAL, "ORB")
on_st   = stats(on_pnl,  OVERNIGHT_CAPITAL, "Overnight")

print("\n" + "="*90)
print("RESULTS — ALL 4 CONFIGS")
print("="*90)

# Header
HDR = (f"{'Cfg':<4} {'Short vol desc':<35} {'SV net':>8} {'SV ann':>9} "
       f"{'SV Sh':>7} {'SV DD%':>8} {'COMB net':>10} {'COMB ann':>10} "
       f"{'COMB Sh':>8} {'COMB DD%':>9} {'>=30k?':>7} {'DD<15?':>7}")
print(HDR)
print("-"*120)

all_rows = []
for cfg in ic_configs:
    lbl = cfg["label"]
    sv_pnl, log, _ = ic_results[lbl]
    sv_st   = stats(sv_pnl, SHORTVOL_CAPITAL, f"SV-{lbl}")
    comb    = combine(orb_pnl, on_pnl, sv_pnl)
    comb_st = stats(comb, TOTAL_CAPITAL, f"Combined-{lbl}")
    hit     = comb_st["annual"] >= 30_000
    safe    = abs(comb_st["dd"]) <= 15.0
    all_rows.append(dict(lbl=lbl, cfg=cfg, sv_st=sv_st, comb_st=comb_st,
                         comb_pnl=comb, sv_pnl=sv_pnl, hit=hit, safe=safe,
                         log=log))
    print(f"{lbl:<4} {cfg['desc']:<35} "
          f"${sv_st['net']:>+7,.0f} ${sv_st['annual']:>+8,.0f} "
          f"{sv_st['sharpe']:>7.3f} {sv_st['dd']:>7.2f}% "
          f"${comb_st['net']:>+9,.0f} ${comb_st['annual']:>+9,.0f} "
          f"{comb_st['sharpe']:>8.3f} {comb_st['dd']:>8.2f}% "
          f"{'YES***' if hit else 'NO':>7} {'YES' if safe else 'NO':>7}")

# ── Three-way split table ─────────────────────────────────────────────────────
best = sorted(all_rows, key=lambda r: r["comb_st"]["annual"], reverse=True)[0]
print(f"\n{'='*90}")
print(f"THREE-WAY SPLIT TABLE — Best Config: {best['lbl']} ({best['cfg']['desc']})")
print(f"{'='*90}")
print(f"{'Metric':<18} {'ORB ($100k)':>14} {'Overnight ($50k)':>18} "
      f"{'Short Vol ($50k)':>18} {'Combined ($200k)':>18}")
print(f"{'-'*72}")

sv_best = best["sv_st"]
cb_best = best["comb_st"]
rows_3way = [
    ("Avg annual",  f"${orb_st['annual']:>+,.0f}",   f"${on_st['annual']:>+,.0f}",
                    f"${sv_best['annual']:>+,.0f}",   f"${cb_best['annual']:>+,.0f}"),
    ("5yr Net P&L", f"${orb_st['net']:>+,.0f}",       f"${on_st['net']:>+,.0f}",
                    f"${sv_best['net']:>+,.0f}",       f"${cb_best['net']:>+,.0f}"),
    ("Sharpe",      f"{orb_st['sharpe']:.3f}",        f"{on_st['sharpe']:.3f}",
                    f"{sv_best['sharpe']:.3f}",        f"{cb_best['sharpe']:.3f}"),
    ("Max DD%",     f"{orb_st['dd']:.2f}%",           f"{on_st['dd']:.2f}%",
                    f"{sv_best['dd']:.2f}%",           f"{cb_best['dd']:.2f}%"),
    ("Profit factor",f"{orb_st['pf']:.2f}",           f"{on_st['pf']:.2f}",
                    f"{sv_best['pf']:.2f}",            f"{cb_best['pf']:.2f}"),
    ("Final equity",f"${orb_st['final']:,.0f}",       f"${on_st['final']:,.0f}",
                    f"${sv_best['final']:,.0f}",       f"${cb_best['final']:,.0f}"),
]
for m, o, n, v, c in rows_3way:
    print(f"{m:<18} {o:>14} {n:>18} {v:>18} {c:>18}")

# ── Correlation matrix ────────────────────────────────────────────────────────
sv_best_pnl = best["sv_pnl"]

# SPY daily P&L series
spy_daily = spy_ret.reindex(
    pd.to_datetime(spy_ret.index).tz_localize(None)
).dropna() * 200_000   # scale to $200k notional

print(f"\n{'='*90}")
print("CORRELATION MATRIX (daily P&L)")
print(f"{'='*90}")
r_on_orb = corr(orb_pnl, on_pnl)
r_sv_orb = corr(orb_pnl, sv_best_pnl)
r_sv_on  = corr(on_pnl,  sv_best_pnl)
comb_best_pnl = best["comb_pnl"]
spy_idx  = spy_daily.index[spy_daily.index >= pd.Timestamp(START)]
spy_sub  = spy_daily.reindex(spy_idx)
r_comb_spy = corr(comb_best_pnl, spy_sub)
print(f"  ORB vs Overnight    : r = {r_on_orb:+.3f}")
print(f"  ORB vs Short Vol    : r = {r_sv_orb:+.3f}")
print(f"  Overnight vs Short Vol: r = {r_sv_on:+.3f}")
print(f"  Combined vs SPY     : r = {r_comb_spy:+.3f}")

# ── Year-by-year for best config ──────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"YEAR-BY-YEAR — Config {best['lbl']} ({best['cfg']['desc']})")
print(f"{'='*90}")
print(f"{'Year':<7} {'ORB':>10} {'Overnight':>11} {'Short Vol':>11} "
      f"{'Combined':>11} {'Return%':>9} {'Sharpe':>8}")
print(f"{'-'*65}")
for yr in [2021, 2022, 2023, 2024, 2025]:
    ym = lambda s: s[s.index.year == yr] if not s.empty else pd.Series(dtype=float)
    o = float(ym(orb_pnl).sum())
    n = float(ym(on_pnl).sum())
    v = float(ym(sv_best_pnl).sum())
    c = o + n + v
    ret = c / TOTAL_CAPITAL * 100
    yr_sh = sharpe(ym(comb_best_pnl)) if len(ym(comb_best_pnl)) > 5 else 0.0
    print(f"{yr:<7} ${o:>+9,.0f} ${n:>+9,.0f}  ${v:>+9,.0f}  ${c:>+9,.0f}  "
          f"{ret:>+7.1f}%  {yr_sh:>7.3f}")
# 2026 partial
ym = lambda s: s[s.index.year == 2026] if not s.empty else pd.Series(dtype=float)
o, n, v = float(ym(orb_pnl).sum()), float(ym(on_pnl).sum()), float(ym(sv_best_pnl).sum())
c = o + n + v
ret = c / TOTAL_CAPITAL * 100
yr_sh = sharpe(ym(comb_best_pnl)) if len(ym(comb_best_pnl)) > 5 else 0.0
print(f"{'2026Q1':<7} ${o:>+9,.0f} ${n:>+9,.0f}  ${v:>+9,.0f}  ${c:>+9,.0f}  "
      f"{ret:>+7.1f}%  {yr_sh:>7.3f}")

# ── Short vol 2022 vs 2024 ────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("SHORT VOL PERFORMANCE: 2022 (high vol) vs 2024 (low vol) — all configs")
print(f"{'='*90}")
print(f"{'Config':<8} {'2022 P&L':>12} {'2022 ann%':>11} {'2024 P&L':>12} {'2024 ann%':>11} {'VIX filter impact'}")
print(f"{'-'*60}")
for r in all_rows:
    sv_p = r["sv_pnl"]
    p22 = float(sv_p[sv_p.index.year == 2022].sum())
    p24 = float(sv_p[sv_p.index.year == 2024].sum())
    ann22 = p22 / SHORTVOL_CAPITAL * 100
    ann24 = p24 / SHORTVOL_CAPITAL * 100
    skip_str = f"skipped {r['log']['skipped_vix']} entries"
    print(f"{r['lbl']:<8} ${p22:>+11,.0f} {ann22:>+10.1f}%  ${p24:>+11,.0f} {ann24:>+10.1f}%  {skip_str}")

# ── Key questions ─────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("KEY QUESTIONS")
print(f"{'='*90}")

passing = [r for r in all_rows if r["hit"] and r["safe"]]
best_safe = sorted([r for r in all_rows if r["safe"]], key=lambda r: r["comb_st"]["annual"], reverse=True)

print(f"\n  1. Any config hits $30k/yr on $200k?  {'YES — ' + ', '.join(r['lbl'] for r in passing) if passing else 'NO'}")
for r in all_rows:
    gap = 30_000 - r["comb_st"]["annual"]
    print(f"     Config {r['lbl']}: ${r['comb_st']['annual']:+,.0f}/yr  (gap: ${gap:+,.0f})")

print(f"\n  2. Does short vol add genuine alpha?")
orb_on_only = stats(combine(orb_pnl, on_pnl), TOTAL_CAPITAL - SHORTVOL_CAPITAL, "ORB+ON")
# Compare best combined vs ORB+overnight alone (same total capital $200k)
# Recompute with ORB+ON on $200k for fair comparison
orb_on_200k = combine(orb_pnl * 2.0 / 2.0, on_pnl * 1.0)  # same capital, just ORB+ON
orb_on_sh = sharpe(combine(orb_pnl, on_pnl))
best_comb_sh = best["comb_st"]["sharpe"]
print(f"     ORB+Overnight Sharpe (no short vol): {orb_on_sh:.3f}")
print(f"     Best 3-way Sharpe (with short vol):  {best_comb_sh:.3f}")
added_sh = best_comb_sh - orb_on_sh
print(f"     Sharpe delta: {added_sh:+.3f}  ({'adds alpha' if added_sh > 0.05 else 'minimal improvement' if added_sh > 0 else 'hurts'})")

print(f"\n  3. Short vol 2022 vs 2024 — see table above.")

print(f"\n  4. VIX filter (Config C: skip >25) vs baseline (Config A):")
c_a = next(r for r in all_rows if r["lbl"] == "A")
c_c = next(r for r in all_rows if r["lbl"] == "C")
print(f"     Config A (no upper limit): ${c_a['sv_st']['net']:+,.0f} net | "
      f"Sharpe {c_a['sv_st']['sharpe']:.3f}")
print(f"     Config C (skip VIX >25):  ${c_c['sv_st']['net']:+,.0f} net | "
      f"Sharpe {c_c['sv_st']['sharpe']:.3f}")
delta = c_c['sv_st']['net'] - c_a['sv_st']['net']
print(f"     Delta: {delta:+,.0f}  ({'filter helps' if delta > 0 else 'filter hurts — vol spikes are profitable'})")

print(f"\n  5. Combined max drawdown:")
for r in all_rows:
    flag = " -- SAFE" if abs(r["comb_st"]["dd"]) <= 15 else " -- EXCEEDS 15%"
    print(f"     Config {r['lbl']}: {r['comb_st']['dd']:.2f}%{flag}")

# ── Final recommendation ──────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("FINAL RECOMMENDATION")
print(f"{'='*90}")
rec = best_safe[0] if best_safe else all_rows[0]
rc  = rec["comb_st"]
rs  = rec["sv_st"]
gap = 30_000 - rc["annual"]

print(f"\n  Recommended config: {rec['lbl']} — {rec['cfg']['desc']}")
print(f"\n  Capital allocation:")
print(f"    ORB (config 17)    : $100,000 (50%) -> ${orb_st['annual']:+,.0f}/yr | Sharpe {orb_st['sharpe']:.3f}")
print(f"    Overnight (top 5)  :  $50,000 (25%) -> ${on_st['annual']:+,.0f}/yr | Sharpe {on_st['sharpe']:.3f}")
print(f"    Iron Condors (SPX) :  $50,000 (25%) -> ${rs['annual']:+,.0f}/yr | Sharpe {rs['sharpe']:.3f}")
print(f"\n  Combined:")
print(f"    Avg annual return  : ${rc['annual']:+,.0f}/yr ({rc['annual']/TOTAL_CAPITAL*100:.1f}%)")
print(f"    5yr total P&L      : ${rc['net']:+,.0f}")
print(f"    Sharpe             : {rc['sharpe']:.3f}")
print(f"    Max drawdown       : {rc['dd']:.2f}%")
print(f"    Final equity       : ${rc['final']:,.0f}")

print(f"\n  Gap to $30k/yr target: ${gap:+,.0f}")
if gap <= 0:
    print(f"  TARGET MET.")
else:
    needed = 30_000 / (rc["annual"] / TOTAL_CAPITAL) if rc["annual"] > 0 else float("inf")
    print(f"  Honest verdict:")
    print(f"    Three strategies combined produce ~${rc['annual']:,.0f}/yr ({rc['annual']/TOTAL_CAPITAL*100:.1f}%/yr).")
    print(f"    To generate $30k/yr at this return rate: need ~${needed:,.0f} capital.")
    print(f"    The 15% annual target is not achievable on $200k with these three strategies.")
    print(f"    What would get there:")
    print(f"      a) Scale capital to ~${needed:,.0f} maintaining same allocation ratios.")
    print(f"      b) Add leverage on the short vol leg (2x condor size increases SV returns ~2x).")
    print(f"      c) Add a 4th strategy: trend-following (managed futures / CTA style)")
    print(f"         which is non-correlated to all three and performs best in trending years.")

print("\nDone.")
