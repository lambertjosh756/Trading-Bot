#!/usr/bin/env python3
"""
orb_walkforward.py — Expanding Monthly Walk-Forward Backtest
Fixed config: 5-min ORB | 1.2% target | 0.3% stop | 1.5x vol | 13:30 exit | RSI 40-60
Date range : Jan 14 2025 → Mar 6 2026
"""
import sys, io, time, calendar
from datetime import date, timedelta
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Constants ────────────────────────────────────────────────────────────────
SEED      = 2025
N_STOCKS  = 15
INIT_EQ   = 100_000.0
POS_FRAC  = 1.0 / N_STOCKS
BAR_TIMES = np.array([9*60 + 30 + m for m in range(390)])
CFG = dict(target_pct=0.012, stop_pct=0.003, orb_mins=5,
           vol_mult=1.5, exit_hour=13, exit_min=30, use_rsi=True)

# ─── Date setup ───────────────────────────────────────────────────────────────
def bdays(s, e):
    out, d = [], s
    while d <= e:
        if d.weekday() < 5: out.append(d)
        d += timedelta(days=1)
    return out

START  = date(2025, 1, 14)
END    = date(2026, 3,  6)
DATES  = bdays(START, END)
N_DAYS = len(DATES)
MONTH_TAG = [(d.year, d.month) for d in DATES]      # (yr,mo) per day idx

# ─── Data generation ─────────────────────────────────────────────────────────
def generate_data():
    rng = np.random.default_rng(SEED)
    data = {}
    for s in range(N_STOCKS):
        avg_p  = float(rng.uniform(30, 200))
        pmvol  = float(rng.uniform(0.018, 0.040)) / 390**0.5
        avg_dv = int(rng.uniform(600_000, 4_500_000))
        data[s] = {}
        price = avg_p
        for d in range(N_DAYS):
            r = float(rng.random())
            if   r < 0.22: drift = float(rng.uniform(0.004, 0.018)) / 390
            elif r < 0.38: drift = float(rng.uniform(-0.018, -0.004)) / 390
            else:          drift = float(rng.normal(0, 0.001)) / 390
            open_p = max(price + float(rng.normal(0, price * 0.005)), price * 0.4)
            hi = np.empty(390); lo = np.empty(390)
            cl = np.empty(390); vl = np.empty(390, dtype=np.int64)
            p = open_p
            for m in range(390):
                vm = (1.9 - m*0.022) if m < 30 else ((1.4+(m-355)*0.015) if m > 355 else 0.82)
                c  = max(p * (1 + drift + float(rng.normal(0, pmvol*vm))), 0.01)
                sp = abs(float(rng.normal(0, pmvol*0.45)))
                hi[m] = max(p, c) * (1+sp); lo[m] = min(p, c) * (1-sp*0.65); cl[m] = c
                vf = (3.5-m*0.05) if m < 30 else ((1.5+(m-355)*0.12) if m > 355 else
                      0.62+abs(float(rng.normal(0, 0.28))))
                vl[m] = max(100, int(avg_dv/390 * max(0.05,vf) * float(rng.lognormal(0,0.48))))
                p = c
            data[s][d] = {'hi': hi, 'lo': lo, 'cl': cl, 'vl': vl, 'avg_dv': avg_dv}
            price = p
    return data

# ─── Indicators ───────────────────────────────────────────────────────────────
def rsi14(arr):
    n = len(arr); out = np.full(n, 50.0)
    if n < 15: return out
    d = np.diff(arr); g = np.maximum(d, 0.); l = np.maximum(-d, 0.)
    ag = float(np.mean(g[:14])); al = float(np.mean(l[:14]))
    for i in range(14, n):
        ag = (ag*13 + g[i-1]) / 14; al = (al*13 + l[i-1]) / 14
        out[i] = 100. if al == 0. else 100. - 100./(1. + ag/al)
    return out

# ─── Backtest on a list of day indices ───────────────────────────────────────
def run_days(data, idxs, eq_start):
    tgt = CFG['target_pct']; stp = CFG['stop_pct']; orb = CFG['orb_mins']
    vmult = CFG['vol_mult']; exit_t = CFG['exit_hour']*60 + CFG['exit_min']
    equity = eq_start; trades = []; daily_rets = []; eq_curve = [equity]

    for di in idxs:
        d0 = equity; dpnl = 0.
        for s in range(N_STOCKS):
            b = data[s][di]
            cl=b['cl']; hi=b['hi']; lo=b['lo']; vl=b['vl']
            avb = b['avg_dv'] / 390.
            orb_h = float(hi[:orb].max())
            rs = rsi14(cl)
            in_t = False; en = tp = sp = 0.
            for m in range(orb, 390):
                bt = BAR_TIMES[m]
                if in_t:
                    if hi[m] >= tp:
                        p = equity*POS_FRAC*tgt; dpnl+=p; equity+=p
                        trades.append(('T', p, tgt)); break
                    elif lo[m] <= sp:
                        p = -(equity*POS_FRAC*stp); dpnl+=p; equity+=p
                        trades.append(('S', p, -stp)); break
                    elif bt >= exit_t:
                        pp = (cl[m]-en)/en; p = equity*POS_FRAC*pp
                        dpnl+=p; equity+=p; trades.append(('X', p, pp)); break
                else:
                    if bt >= exit_t: break
                    if cl[m] <= orb_h or vl[m] < vmult*avb: continue
                    if not (40. <= rs[m] <= 60.): continue
                    in_t = True; en = cl[m]
                    tp = en*(1+tgt); sp = en*(1-stp)
        dr = dpnl/d0 if d0 > 0 else 0.
        daily_rets.append(dr); eq_curve.append(equity)

    return {'trades': trades, 'dr': np.array(daily_rets),
            'equity': equity, 'eqc': np.array(eq_curve)}

def metrics(res, eq0):
    t = res['trades']; n = len(t)
    if n == 0: return dict(n=0, wr=0., ret=0., sharpe=0., mdd=0.)
    wins = sum(1 for tp,p,pp in t if p > 0)
    dr = res['dr']; std = float(dr.std(ddof=1)) if len(dr) > 1 else 0.
    sharpe = float(dr.mean()/std*252**0.5) if std > 0 else 0.
    eq = res['eqc']; pk = np.maximum.accumulate(eq)
    mdd = float(((pk-eq)/pk).max())*100
    return dict(n=n, wr=wins/n*100, ret=(res['equity']-eq0)/eq0*100,
                sharpe=sharpe, mdd=mdd)

# ─── Build month → [day indices] map ─────────────────────────────────────────
month_days = {}
for idx, tag in enumerate(MONTH_TAG):
    month_days.setdefault(tag, []).append(idx)
all_months = sorted(month_days)

TRAIN_MONTHS = [(2025,1),(2025,2),(2025,3)]   # initial training window
OOS_MONTHS   = [m for m in all_months if m not in set(TRAIN_MONTHS)]

def month_name(ym): return f"{calendar.month_abbr[ym[1]]} {ym[0]}"

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"Period: {DATES[0]} → {DATES[-1]}  |  {N_DAYS} business days", flush=True)
    print(f"Stocks: {N_STOCKS}  |  OOS months: {len(OOS_MONTHS)}", flush=True)
    print("Generating data (seed 2025) ...", flush=True)
    data = generate_data()
    print(f"Data ready in {time.time()-t0:.1f}s\n", flush=True)

    # ── In-sample: initial training window (Jan14–Mar31 2025) ────────────────
    train_idxs = sorted(i for m in TRAIN_MONTHS for i in month_days[m])
    is_res = run_days(data, train_idxs, INIT_EQ)
    is_m   = metrics(is_res, INIT_EQ)
    print(f"IS (Jan14–Mar31 2025)  Trades={is_m['n']}  Win%={is_m['wr']:.1f}%  "
          f"Ret={is_m['ret']:+.2f}%  Sharpe={is_m['sharpe']:.3f}\n", flush=True)

    # ── OOS: month-by-month with equity continuity ────────────────────────────
    equity   = is_res['equity']
    oos_rows = []

    for ym in OOS_MONTHS:
        mo_idxs = month_days[ym]
        eq0     = equity
        mo_res  = run_days(data, mo_idxs, equity)
        m       = metrics(mo_res, eq0)
        equity  = mo_res['equity']
        oos_rows.append({'name': month_name(ym), 'eq_end': equity, **m})
        print(f"  OOS {month_name(ym):>8}: T={m['n']:>3}  W={m['wr']:>4.1f}%  "
              f"Ret={m['ret']:>+6.2f}%  Sh={m['sharpe']:>6.3f}  MDD={m['mdd']:>4.2f}%", flush=True)

    # ── Expanding IS sharpe for each OOS month ────────────────────────────────
    print("\nComputing expanding IS Sharpes ...", flush=True)
    expanding_is = []
    prior = list(TRAIN_MONTHS)
    for ym in OOS_MONTHS:
        idxs = sorted(i for m in prior for i in month_days[m])
        r = run_days(data, idxs, INIT_EQ)
        expanding_is.append(metrics(r, INIT_EQ)['sharpe'])
        prior.append(ym)

    # ── Aggregate OOS (all OOS months combined) ───────────────────────────────
    oos_all_idxs = sorted(i for ym in OOS_MONTHS for i in month_days[ym])
    oos_all_res  = run_days(data, oos_all_idxs, is_res['equity'])
    agg = metrics(oos_all_res, is_res['equity'])

    # ─── Print tables ─────────────────────────────────────────────────────────
    W = 102
    print(f"\n{'='*W}")
    print("WALK-FORWARD BACKTEST — MONTHLY OUT-OF-SAMPLE RESULTS")
    print(f"Strategy: 5-min ORB | 1.2% tgt | 0.3% stop | 1.5x vol | 13:30 exit | RSI 40-60")
    print(f"{'='*W}")
    print(f"{'Month':<12} {'Trades':>7} {'Win%':>7} {'NetRet%':>9} "
          f"{'Sharpe':>8} {'MaxDD%':>8} {'IS Sharpe':>11} {'IS/OOS':>8}")
    print("-"*W)

    for i, r in enumerate(oos_rows):
        is_sh = expanding_is[i]
        ratio = is_sh / r['sharpe'] if r['sharpe'] != 0 else float('nan')
        flag  = "  (!!)" if ratio > 2.0 and r['sharpe'] > 0 else ""
        ratio_s = f"{ratio:>7.2f}x" if ratio == ratio else "     N/A"
        print(f"{r['name']:<12} {r['n']:>7,} {r['wr']:>6.1f}% {r['ret']:>+8.2f}% "
              f"{r['sharpe']:>8.3f} {r['mdd']:>7.2f}% {is_sh:>11.3f} {ratio_s}{flag}")

    print("="*W)
    avg_is = float(np.mean(expanding_is))
    agg_ratio = avg_is / agg['sharpe'] if agg['sharpe'] != 0 else float('nan')
    print(f"{'AGGREGATE OOS':<12} {agg['n']:>7,} {agg['wr']:>6.1f}% {agg['ret']:>+8.2f}% "
          f"{agg['sharpe']:>8.3f} "
          f"{max(r['mdd'] for r in oos_rows):>7.2f}% "
          f"{avg_is:>11.3f} {agg_ratio:>7.2f}x")

    # ─── IS vs OOS comparison ─────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print(f"{'='*W}")
    print(f"  Initial IS Sharpe  (Jan14–Mar31 2025)          : {is_m['sharpe']:>8.3f}")
    print(f"  Average expanding IS Sharpe (all windows)       : {avg_is:>8.3f}")
    print(f"  Aggregate OOS Sharpe (Apr 2025–Mar 2026)        : {agg['sharpe']:>8.3f}")
    print(f"  IS/OOS ratio                                    : {agg_ratio:>7.2f}x")
    print(f"  IS Net Return                                   : {is_m['ret']:>+7.2f}%")
    print(f"  OOS Net Return                                  : {agg['ret']:>+7.2f}%")
    positive_oos = sum(1 for r in oos_rows if r['ret'] > 0)
    print(f"  Profitable OOS months                          : {positive_oos}/{len(oos_rows)}")

    # ─── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("VERDICT: DOES THE STRATEGY HOLD UP OOS?")
    print(f"{'='*W}")

    if agg_ratio < 1.4 and agg['sharpe'] > 1.0 and positive_oos >= len(oos_rows)*0.6:
        conclusion = "HOLDS UP WELL"
        detail = ("IS/OOS Sharpe gap is tight. Strategy shows consistent performance "
                  "across the walk-forward period with no evidence of curve-fitting. "
                  "The OOS Sharpe > 1.0 is meaningful in a real-trading context.")
    elif agg_ratio < 2.0 and agg['sharpe'] > 0 and positive_oos >= len(oos_rows)*0.5:
        conclusion = "MODERATE DEGRADATION — ACCEPTABLE"
        detail = ("IS/OOS gap is present but within expected range for any optimized strategy. "
                  "The majority of OOS months are profitable. Some degradation from "
                  "optimization-period to live period is normal; monitor ongoing performance.")
    elif agg['sharpe'] > 0:
        conclusion = "NOTICEABLE DEGRADATION — CAUTION WARRANTED"
        detail = ("Meaningful IS/OOS gap detected. Strategy is profitable OOS but performance "
                  "falls short of training. Consider: (1) the optimization may be slightly "
                  "overfit; (2) the date range is short (14 months); (3) run forward with "
                  "reduced position sizes until more OOS data accumulates.")
    else:
        conclusion = "OVERFITTING DETECTED — DO NOT TRADE LIVE"
        detail = ("OOS Sharpe is negative or near zero. The strategy's IS performance "
                  "does not translate OOS. The optimization was over-tuned to the "
                  "training environment.")

    print(f"\n  CONCLUSION  : {conclusion}")
    print(f"  IS/OOS ratio: {agg_ratio:.2f}x  |  "
          f"Profitable months: {positive_oos}/{len(oos_rows)}  |  "
          f"OOS Sharpe: {agg['sharpe']:.3f}")
    print(f"\n  {detail}")
    print(f"\n[DONE] ({time.time()-t0:.1f}s total)", flush=True)

if __name__ == "__main__":
    main()
