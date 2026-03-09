#!/usr/bin/env python3
"""
orb_kelly.py — Equal-Weight vs Half-Kelly Sizing Comparison
Full range: Jan 14 2025 → Mar 6 2026
Config: 5-min ORB | 1.2% target | 0.3% stop | 1.5x vol | 13:30 exit | RSI 40-60
"""
import sys, io, time, calendar
from datetime import date, timedelta
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Constants ────────────────────────────────────────────────────────────────
SEED       = 2025
N_STOCKS   = 15
INIT_EQ    = 100_000.0
EW_FRAC    = 1.0 / N_STOCKS   # Equal-weight: 6.67%
MAX_POS    = 0.20              # Kelly cap: 20%
SEED_KELLY = 0.25              # First month full-Kelly seed → half = 12.5%
BAR_TIMES  = np.array([9*60 + 30 + m for m in range(390)])
CFG = dict(target_pct=0.012, stop_pct=0.003, orb_mins=5,
           vol_mult=1.5, exit_hour=13, exit_min=30, use_rsi=True)

# ─── Date setup ───────────────────────────────────────────────────────────────
def bdays(s, e):
    out, d = [], s
    while d <= e:
        if d.weekday() < 5: out.append(d)
        d += timedelta(days=1)
    return out

START     = date(2025, 1, 14)
END       = date(2026, 3,  6)
DATES     = bdays(START, END)
N_DAYS    = len(DATES)
MONTH_TAG = [(d.year, d.month) for d in DATES]

# ─── Data generation (same seed → same data as walkforward script) ────────────
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
                vm = (1.9-m*0.022) if m < 30 else ((1.4+(m-355)*0.015) if m > 355 else 0.82)
                c  = max(p*(1 + drift + float(rng.normal(0, pmvol*vm))), 0.01)
                sp = abs(float(rng.normal(0, pmvol*0.45)))
                hi[m] = max(p,c)*(1+sp); lo[m] = min(p,c)*(1-sp*0.65); cl[m] = c
                vf = (3.5-m*0.05) if m < 30 else ((1.5+(m-355)*0.12) if m > 355 else
                      0.62+abs(float(rng.normal(0, 0.28))))
                vl[m] = max(100, int(avg_dv/390*max(0.05,vf)*float(rng.lognormal(0,0.48))))
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
        ag = (ag*13+g[i-1])/14; al = (al*13+l[i-1])/14
        out[i] = 100. if al == 0. else 100.-100./(1.+ag/al)
    return out

# ─── Kelly fraction from prior-month trade P&L percentages ───────────────────
def half_kelly(pct_list):
    """f = W - L/R  →  half_kelly = f/2, capped at MAX_POS."""
    if len(pct_list) < 5:
        return SEED_KELLY / 2   # Not enough data: use seed
    wins   = [p for p in pct_list if p > 0]
    losses = [abs(p) for p in pct_list if p < 0]
    if not wins or not losses:
        return SEED_KELLY / 2
    W = len(wins) / len(pct_list)
    R = float(np.mean(wins)) / float(np.mean(losses))
    f = W - (1.0 - W) / R
    return min(max(f / 2.0, 0.005), MAX_POS)   # floor at 0.5%, cap at 20%

# ─── Strategy runner (mode: 'ew' or 'hk') ────────────────────────────────────
def run_strategy(data, mode):
    """
    mode = 'ew'  → equal weight (EW_FRAC per trade, fixed)
    mode = 'hk'  → half Kelly, recalculated monthly from prior month's trades
    Returns: trades, daily_rets, equity_curve, month_equity, kelly_log
    """
    tgt = CFG['target_pct']; stp = CFG['stop_pct']; orb = CFG['orb_mins']
    vmult = CFG['vol_mult']; exit_t = CFG['exit_hour']*60 + CFG['exit_min']

    equity      = INIT_EQ
    trades      = []           # (type, pnl$, pnl%)
    daily_rets  = []
    eq_curve    = [equity]
    month_eq    = {}           # (yr,mo) -> equity at month-end
    kelly_log   = {}           # (yr,mo) -> {half_f, win_rate, rr, kelly_f}

    cur_month     = None
    cur_pcts      = []         # P&L% for trades in current month
    prev_pcts     = []         # P&L% from previous month → drives Kelly
    pos_frac      = SEED_KELLY / 2 if mode == 'hk' else EW_FRAC

    for di in range(N_DAYS):
        yr, mo = MONTH_TAG[di]
        mk = (yr, mo)

        # ── Month boundary: log previous month, recompute position size ──────
        if mk != cur_month:
            if cur_month is not None:
                month_eq[cur_month] = equity

            if mode == 'hk':
                # Compute Kelly from previous month's trade P&Ls
                wins_p  = [p for p in prev_pcts if p > 0]
                loss_p  = [abs(p) for p in prev_pcts if p < 0]
                if not prev_pcts:
                    pos_frac = SEED_KELLY / 2
                    kelly_log[mk] = {'half_f': pos_frac, 'wr': None,
                                     'rr': None, 'kelly_f': SEED_KELLY, 'seed': True}
                elif not wins_p or not loss_p:
                    pos_frac = SEED_KELLY / 2
                    wr_ = len(wins_p) / len(prev_pcts) if prev_pcts else None
                    kelly_log[mk] = {'half_f': pos_frac, 'wr': wr_,
                                     'rr': None, 'kelly_f': None, 'seed': True}
                else:
                    W_ = len(wins_p) / len(prev_pcts)
                    R_ = float(np.mean(wins_p)) / float(np.mean(loss_p))
                    f_ = W_ - (1-W_) / R_
                    pos_frac = min(max(f_/2, 0.005), MAX_POS)
                    kelly_log[mk] = {'half_f': pos_frac, 'wr': W_,
                                     'rr': R_, 'kelly_f': f_, 'seed': False}

            prev_pcts = cur_pcts
            cur_month = mk
            cur_pcts  = []

        # ── Trading day ───────────────────────────────────────────────────────
        d0 = equity; dpnl = 0.
        for s in range(N_STOCKS):
            b  = data[s][di]
            cl = b['cl']; hi = b['hi']; lo = b['lo']
            vl = b['vl']; avb = b['avg_dv'] / 390.
            orb_h = float(hi[:orb].max())
            rs    = rsi14(cl)
            in_t = False; en = tp = sp = 0.

            for m in range(orb, 390):
                bt = BAR_TIMES[m]
                if in_t:
                    if hi[m] >= tp:
                        p = equity*pos_frac*tgt; dpnl+=p; equity+=p
                        trades.append(('T', p, tgt))
                        cur_pcts.append(tgt); break
                    elif lo[m] <= sp:
                        p = -(equity*pos_frac*stp); dpnl+=p; equity+=p
                        trades.append(('S', p, -stp))
                        cur_pcts.append(-stp); break
                    elif bt >= exit_t:
                        pp = (cl[m]-en)/en; p = equity*pos_frac*pp
                        dpnl+=p; equity+=p; trades.append(('X', p, pp))
                        cur_pcts.append(pp); break
                else:
                    if bt >= exit_t: break
                    if cl[m] <= orb_h or vl[m] < vmult*avb: continue
                    if not (40. <= rs[m] <= 60.): continue
                    in_t = True; en = cl[m]
                    tp = en*(1+tgt); sp = en*(1-stp)

        daily_rets.append(dpnl/d0 if d0 > 0 else 0.)
        eq_curve.append(equity)

    if cur_month: month_eq[cur_month] = equity

    return dict(trades=trades, dr=np.array(daily_rets),
                equity=equity, eqc=np.array(eq_curve),
                month_eq=month_eq, kelly_log=kelly_log)

# ─── Aggregate metrics ────────────────────────────────────────────────────────
def agg(res):
    t = res['trades']; n = len(t)
    if n == 0: return {}
    wins = sum(1 for tp,p,pp in t if p > 0)
    dr = res['dr']; std = float(dr.std(ddof=1)) if len(dr)>1 else 0.
    sh = float(dr.mean()/std*252**0.5) if std > 0 else 0.
    eq = res['eqc']; pk = np.maximum.accumulate(eq)
    mdd = float(((pk-eq)/pk).max())*100
    net = (res['equity']-INIT_EQ)/INIT_EQ*100

    # Monthly returns for best/worst
    month_keys = sorted(res['month_eq'])
    prev = INIT_EQ; month_rets = []
    for mk in month_keys:
        eq_e = res['month_eq'][mk]
        month_rets.append((mk, (eq_e-prev)/prev*100))
        prev = eq_e
    best  = max(month_rets, key=lambda x: x[1]) if month_rets else (None, 0)
    worst = min(month_rets, key=lambda x: x[1]) if month_rets else (None, 0)

    def fmk(mk): return f"{calendar.month_abbr[mk[1]]} {mk[0]}" if mk else "N/A"

    return dict(n=n, wr=wins/n*100, net=net, eq=res['equity'],
                sh=sh, mdd=mdd, month_rets=month_rets,
                best=f"{best[1]:+.2f}% ({fmk(best[0])})",
                worst=f"{worst[1]:+.2f}% ({fmk(worst[0])})")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"Period: {DATES[0]} → {DATES[-1]}  |  {N_DAYS} business days", flush=True)
    print("Generating data (seed 2025) ...", flush=True)
    data = generate_data()
    print(f"Data ready in {time.time()-t0:.1f}s\n", flush=True)

    print("Running A: Equal-Weight ...", flush=True)
    res_A = run_strategy(data, 'ew')
    m_A   = agg(res_A)
    print(f"  EW done. Final equity: ${res_A['equity']:,.2f}", flush=True)

    print("Running B: Half Kelly ...", flush=True)
    res_B = run_strategy(data, 'hk')
    m_B   = agg(res_B)
    print(f"  HK done. Final equity: ${res_B['equity']:,.2f}\n", flush=True)

    elapsed = time.time() - t0

    # ─── Side-by-side comparison ──────────────────────────────────────────────
    W = 72
    print("=" * W)
    print("RUN A vs RUN B — FULL COMPARISON")
    print(f"Jan 14 2025 → Mar 6 2026  |  5-min ORB  |  $100,000 start")
    print("=" * W)
    rows = [
        ("Metric",        "Equal-Weight",              "Half Kelly"),
        ("─"*20,          "─"*22,                      "─"*22),
        ("Trades",        f"{m_A['n']:,}",              f"{m_B['n']:,}"),
        ("Win%",          f"{m_A['wr']:.1f}%",          f"{m_B['wr']:.1f}%"),
        ("Net Return%",   f"{m_A['net']:+.2f}%",        f"{m_B['net']:+.2f}%"),
        ("Final Equity",  f"${m_A['eq']:,.0f}",         f"${m_B['eq']:,.0f}"),
        ("Sharpe",        f"{m_A['sh']:.3f}",           f"{m_B['sh']:.3f}"),
        ("Max Drawdown",  f"{m_A['mdd']:.2f}%",         f"{m_B['mdd']:.2f}%"),
        ("Best Month",    m_A['best'],                   m_B['best']),
        ("Worst Month",   m_A['worst'],                  m_B['worst']),
    ]
    for label, va, vb in rows:
        print(f"  {label:<20} {va:<24} {vb}")

    # ─── Monthly equity curve ─────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("MONTHLY EQUITY CURVE COMPARISON")
    print("=" * W)
    unique_months = sorted(set(MONTH_TAG))
    print(f"  {'Month':<12}  {'Equal-Weight':>20}  {'Half Kelly':>20}  {'Diff':>10}")
    print(f"  {'-'*12}  {'-'*20}  {'-'*20}  {'-'*10}")
    prev_a = prev_b = INIT_EQ
    for mk in unique_months:
        eq_a = res_A['month_eq'].get(mk, prev_a)
        eq_b = res_B['month_eq'].get(mk, prev_b)
        ra = (eq_a-prev_a)/prev_a*100; rb = (eq_b-prev_b)/prev_b*100
        mn = f"{calendar.month_abbr[mk[1]]} {mk[0]}"
        print(f"  {mn:<12}  ${eq_a:>10,.0f} ({ra:>+5.2f}%)  "
              f"${eq_b:>10,.0f} ({rb:>+5.2f}%)  "
              f"{eq_b-eq_a:>+10,.0f}")
        prev_a = eq_a; prev_b = eq_b

    # ─── Kelly fractions by month ─────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("HALF KELLY FRACTIONS BY MONTH")
    print(f"  f = W - (1-W)/R   |   half_f = f/2   |   capped at {MAX_POS:.0%}")
    print("=" * W)
    print(f"  {'Month':<12}  {'Prev Win%':>10}  {'Prev R:R':>9}  "
          f"{'Kelly f':>9}  {'Half-f':>8}  {'Pos Size':>9}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}")

    for mk in unique_months:
        mn = f"{calendar.month_abbr[mk[1]]} {mk[0]}"
        kl = res_B['kelly_log'].get(mk)
        if kl is None:
            # Last month has no log entry (Kelly set at next month boundary)
            print(f"  {mn:<12}  (continued from previous month setting)")
            continue
        if kl.get('seed'):
            wr_s  = f"{'--':>10}"
            rr_s  = f"{'--':>9}"
            kf_s  = f"{'seed':>9}"
            hf_s  = f"{kl['half_f']:>7.1%}"
            ps_s  = f"{kl['half_f']:>8.1%}"
        else:
            wr_s  = f"{kl['wr']:>9.1%}" if kl['wr'] is not None else f"{'--':>9}"
            rr_s  = f"{kl['rr']:>9.2f}" if kl['rr'] is not None else f"{'--':>9}"
            kf_s  = f"{kl['kelly_f']:>8.1%}" if kl['kelly_f'] is not None else f"{'--':>8}"
            hf_s  = f"{kl['half_f']:>7.1%}"
            ps_s  = f"{kl['half_f']:>8.1%}"
        print(f"  {mn:<12}  {wr_s}  {rr_s}  {kf_s}  {hf_s}  {ps_s}")

    # ─── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("VERDICT: DOES HALF KELLY IMPROVE RISK-ADJUSTED RETURNS?")
    print("=" * W)
    sh_delta  = m_B['sh']  - m_A['sh']
    ret_delta = m_B['net'] - m_A['net']
    dd_delta  = m_B['mdd'] - m_A['mdd']

    if sh_delta > 0.3:
        verdict = ("HALF KELLY WINS DECISIVELY — meaningful Sharpe improvement. "
                   "Variable sizing successfully adapts to changing edge month-to-month, "
                   "compounding harder in strong periods and pulling back when edge shrinks.")
    elif sh_delta > 0.05:
        verdict = ("HALF KELLY MARGINAL WIN — small Sharpe gain. "
                   "The benefit is real but modest. Half Kelly is justified if you have "
                   "the discipline to follow the monthly recalibration religiously.")
    elif sh_delta > -0.1:
        verdict = ("TOSS-UP — similar risk-adjusted performance. "
                   "Equal-weight is simpler and nearly equivalent. "
                   "Half Kelly adds complexity without clear benefit at this win rate.")
    else:
        verdict = ("EQUAL-WEIGHT PREFERRED — Half Kelly underperforms. "
                   "With a ~24% win rate the Kelly fraction is small (~3-6%), "
                   "resulting in under-sizing vs the fixed 6.67% equal-weight slot. "
                   "Equal-weight is better here.")

    print(f"\n  Sharpe:      EW={m_A['sh']:.3f}  HK={m_B['sh']:.3f}  delta={sh_delta:>+.3f}")
    print(f"  Net Return:  EW={m_A['net']:+.2f}%  HK={m_B['net']:+.2f}%  delta={ret_delta:>+.2f}%")
    print(f"  Max DD:      EW={m_A['mdd']:.2f}%  HK={m_B['mdd']:.2f}%  delta={dd_delta:>+.2f}%")
    print(f"\n  VERDICT: {verdict}")
    print(f"\n[DONE] {elapsed:.1f}s total", flush=True)

if __name__ == "__main__":
    main()
