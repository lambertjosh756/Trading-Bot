"""
final_portfolio.py

Three-strategy combined portfolio — corrected short vol sizing.
ORB ($100k) + Overnight Momentum ($50k) + Iron Condors 2 contracts ($50k)
Jan 2021 – Mar 2026 | $200,000

Change from previous run: CONTRACTS = 2 (was 8)
Max loss per trade: 2 * 25 * 100 = $5,000 = 10% of short vol capital.
"""
from __future__ import annotations
import warnings, os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(".env")

# ── Constants ─────────────────────────────────────────────────────────────────
ORB_CAPITAL       = 100_000.0
OVERNIGHT_CAPITAL =  50_000.0
SHORTVOL_CAPITAL  =  50_000.0
TOTAL_CAPITAL     = 200_000.0
START, END        = "2021-01-01", "2026-03-07"
TRADING_YEARS     = 5 + 67/252
DATA_DIR          = Path("data/daily")
RISK_FREE         = 0.045
SPREAD_WIDTH      = 25.0
CONTRACTS         = 2          # KEY CHANGE: was 8, now 2
HAIRCUT           = 0.10
STOP_MULT         = 2.0
PROFIT_TARGET     = 0.50

# ── Black-Scholes ─────────────────────────────────────────────────────────────
def bs(S, K, T, r, sig, opt):
    if T <= 1e-6:
        return max(S-K,0) if opt=="call" else max(K-S,0)
    d1 = (np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    if opt=="call": return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def delta_strike(S, T, r, sig, tgt_delta, opt):
    if T <= 1e-6: return S
    d1 = norm.ppf(tgt_delta) if opt=="call" else norm.ppf(1+tgt_delta)
    return S * np.exp((r+.5*sig**2)*T - d1*sig*np.sqrt(T))

def condor_tv(S, Ksc, Klc, Ksp, Klp, T, r, sig):
    if T <= 1e-6:
        cs = max(S-Ksc,0)-max(S-Klc,0)
        ps = max(Ksp-S,0)-max(Klp-S,0)
        return cs+ps
    cs = bs(S,Ksc,T,r,sig,"call")-bs(S,Klc,T,r,sig,"call")
    ps = bs(S,Ksp,T,r,sig,"put") -bs(S,Klp,T,r,sig,"put")
    return cs+ps

# ── Portfolio stats helpers ───────────────────────────────────────────────────
def sharpe(pnl):
    s = pnl.std()
    return float(pnl.mean()/s*np.sqrt(252)) if s>0 else 0.0

def mdd(pnl, cap):
    eq = cap + pnl.cumsum()
    rm = eq.cummax()
    return float(((eq-rm)/rm*100).min())

def pf(pnl):
    g = pnl[pnl>0].sum(); l = abs(pnl[pnl<0].sum())
    return float(g/l) if l>0 else float("inf")

def sumstats(pnl, cap, lbl):
    if pnl.empty: return dict(lbl=lbl,net=0,ann=0,sh=0,dd=0,pf=0,final=cap)
    eq  = cap + pnl.cumsum()
    net = float(pnl.sum())
    return dict(lbl=lbl, net=net, ann=net/TRADING_YEARS,
                sh=sharpe(pnl), dd=mdd(pnl,cap),
                pf=pf(pnl), final=float(eq.iloc[-1]))

def combine(*series):
    idx = series[0].index
    for s in series[1:]: idx = idx.union(s.index)
    out = pd.Series(0.0, index=idx)
    for s in series: out = out.add(s.reindex(idx, fill_value=0), fill_value=0)
    return out.sort_index()

def corr(a, b):
    idx = a.index.intersection(b.index)
    return float(a[idx].corr(b[idx])) if len(idx)>20 else float("nan")

# ═════════════════════════════════════════════════════════════════════════════
# 1. ORB — load existing daily CSV (config 17, $100k)
# ═════════════════════════════════════════════════════════════════════════════
print("="*72)
print("FINAL PORTFOLIO BACKTEST — 2 CONTRACTS SHORT VOL")
print("="*72)

orb_pnl = pd.read_csv(
    "results/daily_2021-01-01_2026-03-07.csv",
    parse_dates=["date"]
).set_index("date")["day_pnl"]
orb_pnl.index = pd.to_datetime(orb_pnl.index)
print(f"[1/4] ORB:       {len(orb_pnl)} days  net=${orb_pnl.sum():+,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Overnight momentum — $50k, top 5, 20-day (re-simulate)
# ═════════════════════════════════════════════════════════════════════════════
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

def load_parquet(sym):
    p = DATA_DIR / f"{sym}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def rsi_series(closes, p=14):
    d = closes.diff()
    ag = d.clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1/p,adjust=False).mean()
    return (100-100/(1+ag/al.replace(0,np.nan))).fillna(50)

def run_overnight(capital, top_n=5, mom_window=20):
    closes_d, opens_d, vols_d = {}, {}, {}
    for sym in OVERNIGHT_UNIVERSE:
        df = load_parquet(sym)
        if df is not None and len(df) > mom_window+5:
            closes_d[sym]=df["close"]; opens_d[sym]=df["open"]; vols_d[sym]=df["volume"]
    closes  = pd.DataFrame(closes_d).sort_index()
    opens   = pd.DataFrame(opens_d).sort_index()
    volumes = pd.DataFrame(vols_d).sort_index()
    trade_dates = closes.index[(closes.index>=pd.Timestamp(START)) & (closes.index<=pd.Timestamp(END))]
    alloc = capital / top_n
    daily: Dict = {}
    for today in trade_dates:
        future = closes.index[closes.index > today]
        if not len(future): continue
        tmrw = future[0]
        past = closes.index[closes.index < today]
        if len(past) < mom_window: continue
        p0 = past[-mom_window]
        scores = {}
        for sym in closes.columns:
            try:
                ct,c0 = closes.loc[today,sym], closes.loc[p0,sym]
                if pd.isna(ct) or pd.isna(c0) or c0==0 or ct<10: continue
                if volumes[sym][volumes.index<=today].tail(20).mean() < 2e6: continue
                scores[sym]=(ct-c0)/c0
            except: continue
        if not scores: continue
        top = sorted(scores, key=scores.get, reverse=True)[:top_n]
        dp = 0.0
        for sym in top:
            try:
                ep=closes.loc[today,sym]; xp=opens.loc[tmrw,sym]
                if pd.isna(ep) or pd.isna(xp) or ep==0: continue
                dp += (xp-ep)/ep * alloc
            except: continue
        if dp: daily[tmrw] = daily.get(tmrw,0)+dp
    s = pd.Series(daily).sort_index()
    s.index = pd.to_datetime(s.index)
    return s

on_pnl = run_overnight(OVERNIGHT_CAPITAL)
print(f"[2/4] Overnight: {len(on_pnl)} days  net=${on_pnl.sum():+,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Market data — SPX, VIX, SPY
# ═════════════════════════════════════════════════════════════════════════════
def clean_yf(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

_spx = clean_yf(yf.download("^GSPC", start="2020-11-01", end="2026-03-10", progress=False, auto_adjust=True))
_vix = clean_yf(yf.download("^VIX",  start="2020-11-01", end="2026-03-10", progress=False, auto_adjust=True))
_spy = clean_yf(yf.download("SPY",   start="2020-11-01", end="2026-03-10", progress=False, auto_adjust=True))
spx_close = _spx["Close"].dropna()
vix_close = _vix["Close"].dropna()
spy_ret   = _spy["Close"].pct_change().dropna()
spy_ret.index = pd.to_datetime(spy_ret.index).tz_localize(None)
print(f"[3/4] Market:    SPX {len(spx_close)}d | VIX {len(vix_close)}d")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Iron condor simulator — 2 contracts
# ═════════════════════════════════════════════════════════════════════════════
trading_days = sorted(
    spx_close.index[(spx_close.index>=pd.Timestamp(START)) & (spx_close.index<=pd.Timestamp(END))]
)

def run_condors(short_delta, vix_lo, vix_hi, entry_weekday, label):
    # Build entry days: first occurrence of entry_weekday in each ISO week
    entry_days, seen = [], set()
    for d in trading_days:
        wk = (d.year, d.isocalendar()[1])
        if wk not in seen:
            week_mon = d - timedelta(days=d.weekday())
            target   = week_mon + timedelta(days=entry_weekday)
            cands    = [t for t in trading_days
                        if week_mon <= t <= week_mon+timedelta(days=4) and t >= target]
            if cands:
                seen.add(wk); entry_days.append(cands[0])

    daily_pnl: Dict = {}
    log = dict(trades=0, expired=0, targeted=0, stopped=0, skipped=0,
               total_credit=0.0)

    for entry in entry_days:
        S = spx_close.get(entry); V = vix_close.get(entry)
        if S is None or V is None or pd.isna(S) or pd.isna(V): continue
        if V < vix_lo or V > vix_hi:
            log["skipped"] += 1; continue

        # Expiry = last trading day of same ISO week
        wmon = entry - timedelta(days=entry.weekday())
        fri  = wmon + timedelta(days=4)
        expiry_cands = [t for t in trading_days
                        if entry < t <= fri+timedelta(days=3)
                        and t.isocalendar()[1] == entry.isocalendar()[1]]
        if not expiry_cands: continue
        expiry = expiry_cands[-1]

        T0    = max((expiry-entry).days/365.0, 1/365)
        sigma = V/100.0

        Ksc = delta_strike(S, T0, RISK_FREE, sigma,  short_delta, "call")
        Klc = Ksc + SPREAD_WIDTH
        Ksp = delta_strike(S, T0, RISK_FREE, sigma, -short_delta, "put")
        Klp = Ksp - SPREAD_WIDTH

        cv0 = condor_tv(S, Ksc, Klc, Ksp, Klp, T0, RISK_FREE, sigma)
        if cv0 <= 0: continue

        credit = cv0 * (1-HAIRCUT) * 100 * CONTRACTS   # total dollars
        log["total_credit"] += credit
        log["trades"] += 1

        monitor = [t for t in trading_days if entry < t <= expiry]
        exit_pnl    = None
        exit_reason = "EXPIRY"
        exit_date   = monitor[-1] if monitor else entry

        for d in monitor:
            Sd = spx_close.get(d); Vd = vix_close.get(d)
            if Sd is None or Vd is None or pd.isna(Sd) or pd.isna(Vd): continue
            T_rem  = max((expiry-d).days/365.0, 0.0)
            sig_d  = Vd/100.0
            cv_now = condor_tv(Sd, Ksc, Klc, Ksp, Klp, T_rem, RISK_FREE, sig_d)
            cv_usd = cv_now * 100 * CONTRACTS

            if d == expiry:
                cv_exp = condor_tv(Sd, Ksc, Klc, Ksp, Klp, 0.0, RISK_FREE, 0.01)
                exit_pnl = credit - cv_exp*100*CONTRACTS
                exit_reason = "EXPIRY"; exit_date = d; break

            if cv_usd >= STOP_MULT * credit:
                exit_pnl = credit - cv_usd
                exit_reason = "STOP"; exit_date = d; break

            if cv_usd <= PROFIT_TARGET * credit:
                exit_pnl = credit - cv_usd
                exit_reason = "TARGET"; exit_date = d; break

        if exit_pnl is None:
            exit_pnl = credit; exit_reason = "EXPIRY"

        daily_pnl[exit_date] = daily_pnl.get(exit_date, 0.0) + exit_pnl
        if exit_reason == "EXPIRY":   log["expired"]  += 1
        elif exit_reason == "STOP":   log["stopped"]  += 1
        elif exit_reason == "TARGET": log["targeted"] += 1

    s = pd.Series(daily_pnl).sort_index()
    s.index = pd.to_datetime(s.index)
    return s, log

configs = [
    dict(label="A", desc="2ct, 16-delta, Monday,    VIX>12",
         sd=0.16, vlo=12, vhi=999, wd=0),
    dict(label="B", desc="2ct, 16-delta, Wednesday, VIX>12",
         sd=0.16, vlo=12, vhi=999, wd=2),
    dict(label="C", desc="2ct, 16-delta, Monday,    VIX>20",
         sd=0.16, vlo=20, vhi=999, wd=0),
]

print("[4/4] Iron condors (2 contracts)...")
ic = {}
for cfg in configs:
    pnl, log = run_condors(cfg["sd"], cfg["vlo"], cfg["vhi"], cfg["wd"], cfg["label"])
    ic[cfg["label"]] = (pnl, log, cfg)
    print(f"  Config {cfg['label']}: {log['trades']} trades | "
          f"expired={log['expired']} target={log['targeted']} stop={log['stopped']} "
          f"skipped={log['skipped']} | net=${pnl.sum():+,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. Reporting
# ═════════════════════════════════════════════════════════════════════════════
orb_st = sumstats(orb_pnl, ORB_CAPITAL, "ORB")
on_st  = sumstats(on_pnl,  OVERNIGHT_CAPITAL, "Overnight")

# Two-strategy baseline (ORB + overnight only, $150k invested, $50k idle)
base2  = combine(orb_pnl, on_pnl)
base2_st = sumstats(base2, TOTAL_CAPITAL, "ORB+ON")

print("\n" + "="*72)
print("RESULTS")
print("="*72)

all_rows = []
for cfg in configs:
    lbl = cfg["label"]
    sv_pnl, log, _ = ic[lbl]
    sv_st   = sumstats(sv_pnl, SHORTVOL_CAPITAL, f"SV-{lbl}")
    comb    = combine(orb_pnl, on_pnl, sv_pnl)
    comb_st = sumstats(comb, TOTAL_CAPITAL, f"Comb-{lbl}")
    all_rows.append(dict(lbl=lbl, cfg=cfg, sv_st=sv_st,
                         comb_st=comb_st, sv_pnl=sv_pnl,
                         comb_pnl=comb, log=log))

# ── 1. Three-way split table ──────────────────────────────────────────────────
best = sorted(all_rows, key=lambda r: r["comb_st"]["sh"], reverse=True)[0]
bsv  = best["sv_st"]; bcs = best["comb_st"]
bsvp = best["sv_pnl"]; bcp = best["comb_pnl"]

print(f"\n--- Three-Way Split Table (Best Config: {best['lbl']}) ---")
W = 16
print(f"{'Metric':<18} {'ORB ($100k)':>{W}} {'Overnight ($50k)':>{W}} "
      f"{'Short Vol ($50k)':>{W}} {'Combined ($200k)':>{W}}")
print("-"*74)
for row in [
    ("Avg annual",    f"${orb_st['ann']:>+,.0f}",   f"${on_st['ann']:>+,.0f}",
                      f"${bsv['ann']:>+,.0f}",       f"${bcs['ann']:>+,.0f}"),
    ("5yr Net P&L",   f"${orb_st['net']:>+,.0f}",   f"${on_st['net']:>+,.0f}",
                      f"${bsv['net']:>+,.0f}",       f"${bcs['net']:>+,.0f}"),
    ("Sharpe",        f"{orb_st['sh']:.3f}",         f"{on_st['sh']:.3f}",
                      f"{bsv['sh']:.3f}",            f"{bcs['sh']:.3f}"),
    ("Max DD%",       f"{orb_st['dd']:.2f}%",        f"{on_st['dd']:.2f}%",
                      f"{bsv['dd']:.2f}%",           f"{bcs['dd']:.2f}%"),
    ("Profit factor", f"{orb_st['pf']:.2f}",         f"{on_st['pf']:.2f}",
                      f"{bsv['pf']:.2f}",            f"{bcs['pf']:.2f}"),
    ("Final equity",  f"${orb_st['final']:,.0f}",    f"${on_st['final']:,.0f}",
                      f"${bsv['final']:,.0f}",       f"${bcs['final']:,.0f}"),
]:
    print(f"{row[0]:<18} {row[1]:>{W}} {row[2]:>{W}} {row[3]:>{W}} {row[4]:>{W}}")

# ── 2. All configs summary ────────────────────────────────────────────────────
print(f"\n--- All Config Summary ---")
print(f"{'Cfg':<4} {'Description':<38} {'SV net':>9} {'SV ann':>9} {'SV Sh':>7} "
      f"{'SV DD':>8} {'Comb ann':>10} {'Comb Sh':>9} {'Comb DD':>9}")
print("-"*107)
for r in all_rows:
    sv = r["sv_st"]; co = r["comb_st"]
    print(f"{r['lbl']:<4} {r['cfg']['desc']:<38} "
          f"${sv['net']:>+8,.0f} ${sv['ann']:>+8,.0f} {sv['sh']:>7.3f} "
          f"{sv['dd']:>7.2f}% ${co['ann']:>+9,.0f} {co['sh']:>9.3f} {co['dd']:>8.2f}%")

# ── 3. Year-by-year for best config ──────────────────────────────────────────
print(f"\n--- Year-by-Year: Config {best['lbl']} ({best['cfg']['desc']}) ---")
print(f"{'Year':<7} {'ORB':>10} {'Overnight':>11} {'Short Vol':>11} "
      f"{'Combined':>11} {'Return%':>9} {'Sharpe':>8}")
print("-"*65)
for yr in [2021,2022,2023,2024,2025]:
    ym = lambda s: s[s.index.year==yr]
    o = float(ym(orb_pnl).sum()); n = float(ym(on_pnl).sum())
    v = float(ym(bsvp).sum());    c = o+n+v
    yr_sh = sharpe(ym(bcp)) if len(ym(bcp))>5 else 0.0
    ann_flag = " <-- near 15%" if c >= 25000 else ""
    print(f"{yr:<7} ${o:>+9,.0f} ${n:>+9,.0f}  ${v:>+9,.0f}  ${c:>+9,.0f}  "
          f"{c/TOTAL_CAPITAL*100:>+7.1f}%  {yr_sh:>7.3f}{ann_flag}")
ym26 = lambda s: s[s.index.year==2026]
o,n,v = float(ym26(orb_pnl).sum()),float(ym26(on_pnl).sum()),float(ym26(bsvp).sum())
c = o+n+v
yr_sh = sharpe(ym26(bcp)) if len(ym26(bcp))>5 else 0.0
print(f"{'2026Q1':<7} ${o:>+9,.0f} ${n:>+9,.0f}  ${v:>+9,.0f}  ${c:>+9,.0f}  "
      f"{c/TOTAL_CAPITAL*100:>+7.1f}%  {yr_sh:>7.3f}")

# ── 4. Correlation matrix ─────────────────────────────────────────────────────
spy_daily = spy_ret[spy_ret.index >= pd.Timestamp(START)] * TOTAL_CAPITAL
print(f"\n--- Correlation Matrix (Config {best['lbl']}) ---")
print(f"  ORB vs Overnight    : r = {corr(orb_pnl, on_pnl):+.3f}")
print(f"  ORB vs Short Vol    : r = {corr(orb_pnl, bsvp):+.3f}")
print(f"  Overnight vs SV     : r = {corr(on_pnl,  bsvp):+.3f}")
print(f"  Combined vs SPY     : r = {corr(bcp, spy_daily):+.3f}")

# ── 5. Comparison vs two-strategy baseline ────────────────────────────────────
print(f"\n--- Comparison vs Two-Strategy Baseline (ORB + Overnight only) ---")
print(f"{'Metric':<20} {'Two strategies':>16} {'Three strategies':>18} {'Delta':>10}")
print("-"*68)
for metric, b2, b3, fmt in [
    ("Avg annual P&L",  base2_st["ann"],  bcs["ann"],  "${:>+,.0f}"),
    ("5yr Net P&L",     base2_st["net"],  bcs["net"],  "${:>+,.0f}"),
    ("Sharpe",          base2_st["sh"],   bcs["sh"],   "{:>.3f}"),
    ("Max DD%",         base2_st["dd"],   bcs["dd"],   "{:>.2f}%"),
    ("Final equity",    base2_st["final"],bcs["final"],"${:>,.0f}"),
]:
    delta = b3 - b2
    if "%" in fmt:
        print(f"{metric:<20} {b2:>14.2f}%  {b3:>16.2f}%  {delta:>+8.2f}pp")
    elif "$" in fmt:
        print(f"{metric:<20} ${b2:>+14,.0f}  ${b3:>+15,.0f}  ${delta:>+8,.0f}")
    else:
        print(f"{metric:<20} {b2:>16.3f}  {b3:>18.3f}  {delta:>+10.3f}")

# ── 6. Key questions ──────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("KEY QUESTIONS")
print(f"{'='*72}")

best_cfg_a = next(r for r in all_rows if r["lbl"]=="A")
print(f"\n  1. Does re-sizing fix the drawdown problem?")
prev_sv_dd = -76.88   # from 8-contract run
print(f"     Previous (8 contracts): SV DD = {prev_sv_dd:.1f}%  |  Combined DD = -35.1%")
print(f"     This run (2 contracts): SV DD = {bsv['dd']:.2f}%  |  Combined DD = {bcs['dd']:.2f}%")
dd_ok = abs(bcs["dd"]) <= 15.0
print(f"     Within 15% DD constraint: {'YES' if dd_ok else 'NO — ' + str(round(bcs['dd'],1)) + '%'}")

print(f"\n  2. Does short vol still add meaningful alpha at 2 contracts?")
sv_annual = bsv["ann"]
add_vs_base = bcs["ann"] - base2_st["ann"]
print(f"     Short vol avg annual P&L: ${sv_annual:+,.0f}/yr")
print(f"     Added to combined vs ORB+overnight: ${add_vs_base:+,.0f}/yr")
print(f"     Sharpe impact: {base2_st['sh']:.3f} -> {bcs['sh']:.3f} ({bcs['sh']-base2_st['sh']:+.3f})")
print(f"     Alpha per dollar of SV capital: {sv_annual/SHORTVOL_CAPITAL*100:+.2f}%/yr on $50k")

print(f"\n  3. Realistic annual return and max DD (best config: {best['lbl']}):")
print(f"     Avg annual return : ${bcs['ann']:+,.0f}/yr  ({bcs['ann']/TOTAL_CAPITAL*100:.1f}%/yr on $200k)")
print(f"     Max drawdown      : {bcs['dd']:.2f}%")
print(f"     Sharpe            : {bcs['sh']:.3f}")
print(f"     Final equity      : ${bcs['final']:,.0f}")

print(f"\n  4. Is this the best achievable portfolio at $200k?")
best_ann    = bcs["ann"]
target_gap  = 30_000 - best_ann
print(f"     Best three-strategy config: ${best_ann:+,.0f}/yr ({best_ann/TOTAL_CAPITAL*100:.1f}%)")
print(f"     Gap to $30k/yr target:      ${target_gap:+,.0f}")
if target_gap > 0:
    scale_needed = 30_000 / (best_ann / TOTAL_CAPITAL) if best_ann > 0 else float("inf")
    print(f"     Capital needed at this rate: ~${scale_needed:,.0f}")

# ── 7. Final recommendation ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("FINAL RECOMMENDATION")
print(f"{'='*72}")
rec = sorted(all_rows, key=lambda r: (abs(r["comb_st"]["dd"])<=15)*10 + r["comb_st"]["sh"],
             reverse=True)[0]
rc  = rec["comb_st"]; rs = rec["sv_st"]
gap = 30_000 - rc["ann"]
dd_safe = abs(rc["dd"]) <= 15.0

print(f"\n  Best config: {rec['lbl']} — {rec['cfg']['desc']}")
print(f"\n  ┌─{'─'*18}─┬─{'─'*11}─┬─{'─'*13}─┬─{'─'*13}─┬─{'─'*14}─┐")
print(f"  │ {'Metric':<18} │ {'ORB ($100k)':>11} │ {'Overnight ($50k)':>13} │ "
      f"{'Short Vol ($50k)':>13} │ {'Combined ($200k)':>14} │")
print(f"  ├─{'─'*18}─┼─{'─'*11}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*14}─┤")
for m, o, n, v, c in [
    ("Avg annual",
     f"${orb_st['ann']:>+,.0f}", f"${on_st['ann']:>+,.0f}",
     f"${rs['ann']:>+,.0f}",     f"${rc['ann']:>+,.0f}"),
    ("Sharpe",
     f"{orb_st['sh']:.3f}", f"{on_st['sh']:.3f}",
     f"{rs['sh']:.3f}",     f"{rc['sh']:.3f}"),
    ("Max DD%",
     f"{orb_st['dd']:.2f}%", f"{on_st['dd']:.2f}%",
     f"{rs['dd']:.2f}%",    f"{rc['dd']:.2f}%"),
    ("Final equity",
     f"${orb_st['final']:,.0f}", f"${on_st['final']:,.0f}",
     f"${rs['final']:,.0f}",     f"${rc['final']:,.0f}"),
]:
    print(f"  │ {m:<18} │ {o:>11} │ {n:>13} │ {v:>13} │ {c:>14} │")
print(f"  └─{'─'*18}─┴─{'─'*11}─┴─{'─'*13}─┴─{'─'*13}─┴─{'─'*14}─┘")

print(f"\n  DD constraint (<= 15%): {'MET' if dd_safe else 'NOT MET (' + str(round(rc['dd'],1)) + '%)'}")
print(f"  Gap to $30k/yr:         ${gap:+,.0f}")
if gap > 0:
    scale = 30000/(rc['ann']/TOTAL_CAPITAL) if rc['ann']>0 else float("inf")
    print(f"\n  The three strategies at $200k produce ${rc['ann']:,.0f}/yr ({rc['ann']/TOTAL_CAPITAL*100:.1f}%).")
    print(f"  This is the best risk-adjusted combination available with these strategies.")
    print(f"  To reach $30k/yr at the same return rate: ~${scale:,.0f} total capital.")
    print(f"\n  Paths to close the gap on $200k:")
    print(f"    a) Accept lower target: $10–13k/yr at Sharpe ~1.1 is genuinely strong.")
    print(f"    b) Increase condor contracts gradually (3-4) as account grows.")
    print(f"    c) Add a 4th strategy (e.g., trend-following ETF rotation)")
    print(f"       to capture trending-year alpha ORB and short vol both miss.")
else:
    print(f"\n  TARGET MET: ${rc['ann']:,.0f}/yr on $200k.")

print("\nDone.")
