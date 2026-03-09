#!/usr/bin/env python3
"""
ORB Strategy — Phase 2 Optimization
12 parallel runs: Phase 1 (combined), Phase 2 (Stop×ORB grid), Phase 3 (combined variations)
Reuses same synthetic dataset as orb_backtest.py (SEED=42)
"""
import sys, time, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SEED       = 42
N_STOCKS   = 15
N_DAYS     = 252
INITIAL_EQ = 100_000.0
POS_FRAC   = 1.0 / N_STOCKS

# ─── Data Generation (identical to orb_backtest.py) ─────────────────────────

def generate_data():
    rng = np.random.default_rng(SEED)
    data = {}
    for s in range(N_STOCKS):
        avg_price     = float(rng.uniform(30, 200))
        per_min_vol   = float(rng.uniform(0.018, 0.040)) / (390 ** 0.5)
        avg_daily_vol = int(rng.uniform(600_000, 4_500_000))
        data[s] = {}
        price = avg_price
        for d in range(N_DAYS):
            roll = float(rng.random())
            if   roll < 0.22: daily_drift = float(rng.uniform(0.004, 0.018)) / 390
            elif roll < 0.38: daily_drift = float(rng.uniform(-0.018, -0.004)) / 390
            else:             daily_drift = float(rng.normal(0, 0.001)) / 390
            gap    = float(rng.normal(0, price * 0.005))
            open_p = max(price + gap, price * 0.4)
            highs  = np.empty(390); lows = np.empty(390)
            closes = np.empty(390); vols = np.empty(390, dtype=np.int64)
            p = open_p
            for m in range(390):
                if   m < 30:  vm = 1.9 - m * 0.022
                elif m > 355: vm = 1.4 + (m - 355) * 0.015
                else:         vm = 0.82
                ret   = daily_drift + float(rng.normal(0, per_min_vol * vm))
                c     = max(p * (1.0 + ret), 0.01)
                spike = abs(float(rng.normal(0, per_min_vol * 0.45)))
                h = max(p, c) * (1.0 + spike)
                l = min(p, c) * (1.0 - spike * 0.65)
                if   m < 30:  vf = 3.5 - m * 0.05
                elif m > 355: vf = 1.5 + (m - 355) * 0.12
                else:         vf = 0.62 + abs(float(rng.normal(0, 0.28)))
                vf = max(0.05, vf) * float(rng.lognormal(0, 0.48))
                v  = max(100, int(avg_daily_vol / 390 * vf))
                highs[m] = h; lows[m] = l; closes[m] = c; vols[m] = v
                p = c
            data[s][d] = {'high': highs, 'low': lows, 'close': closes,
                          'volume': vols, 'avg_daily_vol': avg_daily_vol}
            price = p
    return data

# ─── Indicators ──────────────────────────────────────────────────────────────

def ema(arr, period):
    k = 2.0 / (period + 1); out = np.empty(len(arr)); out[0] = arr[0]
    for i in range(1, len(arr)): out[i] = arr[i] * k + out[i-1] * (1.0 - k)
    return out

def rsi(arr, period=14):
    n = len(arr); out = np.full(n, 50.0)
    if n < period + 1: return out
    d = np.diff(arr); g = np.maximum(d, 0.0); l_ = np.maximum(-d, 0.0)
    ag = float(np.mean(g[:period])); al = float(np.mean(l_[:period]))
    for i in range(period, n):
        ag = (ag * (period-1) + g[i-1]) / period
        al = (al * (period-1) + l_[i-1]) / period
        out[i] = 100.0 if al == 0.0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out

# ─── Backtest Engine ─────────────────────────────────────────────────────────

BAR_TIMES = np.array([9*60 + 30 + m for m in range(390)])

def backtest(data, cfg):
    tgt     = cfg['target_pct']
    stp     = cfg['stop_pct']
    orb     = cfg['orb_mins']
    vmult   = cfg['vol_mult']
    exit_t  = cfg['exit_hour'] * 60 + cfg['exit_min']
    use_rsi = cfg.get('use_rsi', False)

    equity = INITIAL_EQ; trades = []; daily_rets = []

    for d in range(N_DAYS):
        day_eq_start = equity; day_pnl = 0.0
        for s in range(N_STOCKS):
            bars = data[s][d]
            cl = bars['close']; hi = bars['high']; lo = bars['low']
            vol = bars['volume']; avb = bars['avg_daily_vol'] / 390.0
            orb_high = float(hi[:orb].max())
            rsi14 = rsi(cl, 14) if use_rsi else None

            in_trade = False; entry_price = tgt_price = stp_price = 0.0
            for m in range(orb, 390):
                bt = BAR_TIMES[m]
                if in_trade:
                    if hi[m] >= tgt_price:
                        pnl = equity * POS_FRAC * tgt
                        day_pnl += pnl; equity += pnl; trades.append(('T', pnl)); break
                    elif lo[m] <= stp_price:
                        pnl = -(equity * POS_FRAC * stp)
                        day_pnl += pnl; equity += pnl; trades.append(('S', pnl)); break
                    elif bt >= exit_t:
                        pnl = equity * POS_FRAC * (cl[m] - entry_price) / entry_price
                        day_pnl += pnl; equity += pnl; trades.append(('X', pnl)); break
                else:
                    if bt >= exit_t: break
                    if cl[m] <= orb_high: continue
                    if vol[m] < vmult * avb: continue
                    if use_rsi and not (40.0 <= rsi14[m] <= 60.0): continue
                    in_trade = True
                    entry_price = cl[m]
                    tgt_price   = entry_price * (1.0 + tgt)
                    stp_price   = entry_price * (1.0 - stp)
        daily_rets.append(day_pnl / day_eq_start if day_eq_start > 0 else 0.0)

    n = len(trades)
    if n == 0:
        return dict(trades=0, win_rate=0.0, net_return=0.0, final_equity=equity,
                    sharpe=0.0, target_exits=0, time_exits=0, stop_exits=0)
    wins    = sum(1 for t, p in trades if p > 0)
    t_exits = sum(1 for t, _ in trades if t == 'T')
    x_exits = sum(1 for t, _ in trades if t == 'X')
    s_exits = sum(1 for t, _ in trades if t == 'S')
    dr  = np.array(daily_rets, dtype=float)
    std = float(dr.std(ddof=1))
    sharpe = float(dr.mean() / std * (252 ** 0.5)) if std > 0 else 0.0
    return dict(trades=n, win_rate=wins/n*100, net_return=(equity-INITIAL_EQ)/INITIAL_EQ*100,
                final_equity=equity, sharpe=sharpe,
                target_exits=t_exits, time_exits=x_exits, stop_exits=s_exits)

# ─── Config Definitions ───────────────────────────────────────────────────────

# Original baseline from Phase 1 study (for Phase 2 grid reference)
ORIG_BASE = dict(target_pct=0.012, stop_pct=0.005, orb_mins=15,
                 vol_mult=1.5, exit_hour=12, exit_min=30, use_rsi=False)

# Combined "winner" config (Phase 1 result)
COMBINED  = dict(target_pct=0.012, stop_pct=0.003, orb_mins=5,
                 vol_mult=2.0, exit_hour=13, exit_min=30, use_rsi=True)

CONFIGS = [
    # ── Phase 1 ──────────────────────────────────────────────────────────────
    ("P1_combined",         COMBINED),

    # ── Phase 2: Stop × ORB grid (no RSI, vol=1.5x, exit=12:30 from orig base)
    ("P2_orb5_stop0.3%",   {**ORIG_BASE, 'orb_mins': 5,  'stop_pct': 0.003}),
    ("P2_orb5_stop0.4%",   {**ORIG_BASE, 'orb_mins': 5,  'stop_pct': 0.004}),
    ("P2_orb5_stop0.5%",   {**ORIG_BASE, 'orb_mins': 5,  'stop_pct': 0.005}),
    ("P2_orb10_stop0.3%",  {**ORIG_BASE, 'orb_mins': 10, 'stop_pct': 0.003}),
    ("P2_orb10_stop0.4%",  {**ORIG_BASE, 'orb_mins': 10, 'stop_pct': 0.004}),
    ("P2_orb10_stop0.5%",  {**ORIG_BASE, 'orb_mins': 10, 'stop_pct': 0.005}),

    # ── Phase 3: Combined variations (one knob at a time) ────────────────────
    ("P3_no_rsi",           {**COMBINED, 'use_rsi': False}),
    ("P3_vol_2.5x",         {**COMBINED, 'vol_mult': 2.5}),
    ("P3_vol_1.5x",         {**COMBINED, 'vol_mult': 1.5}),
    ("P3_exit_12:30",       {**COMBINED, 'exit_hour': 12, 'exit_min': 30}),
    ("P3_stop_0.4%",        {**COMBINED, 'stop_pct': 0.004}),
]

# ─── Formatting ───────────────────────────────────────────────────────────────

W   = 110
HDR = (f"{'Config':<22} {'Trades':>7} {'Win%':>7} {'NetRet%':>9} "
       f"{'FinalEq':>12} {'Sharpe':>8} {'TARGET':>8} {'TIME_EXIT':>10} {'STOP':>7}")

def row(name, r, winner=False, label=None):
    tag = "  <<< BEST" if winner else ""
    nm  = label or name
    return (f"{nm:<22} {r['trades']:>7,} {r['win_rate']:>6.1f}% "
            f"{r['net_return']:>8.2f}% ${r['final_equity']:>10,.0f} "
            f"{r['sharpe']:>8.3f} {r['target_exits']:>8,} "
            f"{r['time_exits']:>10,} {r['stop_exits']:>7,}{tag}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Generating synthetic intraday data (same seed as Phase 1) ...", flush=True)
    data = generate_data()
    print(f"  {N_STOCKS} stocks x {N_DAYS} days x 390 bars ready  ({time.time()-t0:.1f}s)\n", flush=True)

    results = {}
    for name, cfg in CONFIGS:
        print(f"  Running {name} ...", flush=True)
        results[name] = backtest(data, cfg)

    elapsed = time.time() - t0
    print(f"\n[DONE] All 12 backtests in {elapsed:.1f}s\n", flush=True)

    combined_r = results["P1_combined"]

    # ── PHASE 1 ──────────────────────────────────────────────────────────────
    print("=" * W)
    print("PHASE 1 — Combined Winner Config")
    print("  (5-min ORB | 1.2% target | 0.3% stop | 2.0x vol | 13:30 exit | RSI 40-60)")
    print("=" * W)
    print(HDR)
    print("-" * W)
    print(row("P1_combined", combined_r, label="combined_all"))
    print()

    # ── PHASE 2: Stop x ORB grid ─────────────────────────────────────────────
    print("=" * W)
    print("PHASE 2 — Stop Loss x ORB Timeframe Grid")
    print("  (no RSI filter | vol=1.5x | exit=12:30 | target=1.2%)")
    print("=" * W)

    orbs  = [5, 10]
    stops = [0.3, 0.4, 0.5]
    grid  = {o: {s: None for s in stops} for o in orbs}
    for o in orbs:
        for s in stops:
            key = f"P2_orb{o}_stop{s}%"
            grid[o][s] = results[key]

    # Find best cell
    best_sharpe = -999; best_cell = (None, None)
    for o in orbs:
        for s in stops:
            sh = grid[o][s]['sharpe']
            if sh > best_sharpe:
                best_sharpe = sh; best_cell = (o, s)

    # Grid table — Sharpe
    print(f"\n  Sharpe Ratio Grid:")
    print(f"  {'':>12}  " + "  ".join(f"stop={s:.1f}%" .center(14) for s in stops))
    print(f"  {'-'*70}")
    for o in orbs:
        cells = []
        for s in stops:
            r_  = grid[o][s]
            mrk = " ***" if (o, s) == best_cell else "    "
            cells.append(f"{r_['sharpe']:>6.3f}{mrk}".center(14))
        print(f"  orb={o:>2}min  |  " + "  |  ".join(cells))

    # Grid table — Net Return
    print(f"\n  Net Return % Grid:")
    print(f"  {'':>12}  " + "  ".join(f"stop={s:.1f}%".center(14) for s in stops))
    print(f"  {'-'*70}")
    for o in orbs:
        cells = []
        for s in stops:
            r_  = grid[o][s]
            mrk = " ***" if (o, s) == best_cell else "    "
            cells.append(f"{r_['net_return']:>+6.2f}%{mrk}".center(14))
        print(f"  orb={o:>2}min  |  " + "  |  ".join(cells))

    # Full detail rows for each grid cell
    print(f"\n  Full detail (all grid cells):")
    print(HDR)
    print("-" * W)
    for o in orbs:
        for s in stops:
            key  = f"P2_orb{o}_stop{s}%"
            r_   = results[key]
            win  = (o, s) == best_cell
            lbl  = f"orb{o}m_stop{s:.1f}%"
            print(row(key, r_, winner=win, label=lbl))
    print()

    # ── PHASE 3: Combined variations ─────────────────────────────────────────
    print("=" * W)
    print("PHASE 3 — Combined Config Stress-Test (one knob at a time)")
    print("  Baseline = Phase 1 combined  (5-min ORB | 0.3% stop | 2.0x vol | 13:30 | RSI)")
    print("=" * W)
    print(HDR)
    print("-" * W)
    print(row("P1_combined", combined_r, label="[COMBINED BASE]"))

    p3_names = ["P3_no_rsi", "P3_vol_2.5x", "P3_vol_1.5x", "P3_exit_12:30", "P3_stop_0.4%"]
    p3_labels = {
        "P3_no_rsi":       "no_RSI_filter",
        "P3_vol_2.5x":     "vol=2.5x",
        "P3_vol_1.5x":     "vol=1.5x",
        "P3_exit_12:30":   "exit=12:30",
        "P3_stop_0.4%":    "stop=0.4%",
    }
    p3_best = max(p3_names, key=lambda n: results[n]['sharpe'])

    for n in p3_names:
        r_  = results[n]
        win = (n == p3_best) and (r_['sharpe'] > combined_r['sharpe'])
        print(row(n, r_, winner=win, label=p3_labels[n]))
    print()

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    print("=" * W)
    print("FINAL VERDICT")
    print("=" * W)

    # Collect all configs: P1 + P2 best + P3 best
    all_r = {name: results[name] for name, _ in CONFIGS}
    overall_best_name = max(all_r, key=lambda n: all_r[n]['sharpe'])
    overall_best_r    = all_r[overall_best_name]

    p3_best_r     = results[p3_best]
    p2_best_name  = f"P2_orb{best_cell[0]}_stop{best_cell[1]}%"
    p2_best_r     = results[p2_best_name]

    print(f"\n  Phase 1 combined   : Sharpe={combined_r['sharpe']:.3f}  Net={combined_r['net_return']:+.2f}%  Trades={combined_r['trades']}")
    print(f"  Phase 2 grid best  : {p2_best_name:<26} Sharpe={p2_best_r['sharpe']:.3f}  Net={p2_best_r['net_return']:+.2f}%  Trades={p2_best_r['trades']}")
    print(f"  Phase 3 best var.  : {p3_best:<26} Sharpe={p3_best_r['sharpe']:.3f}  Net={p3_best_r['net_return']:+.2f}%  Trades={p3_best_r['trades']}")

    print(f"\n  Overall best config: {overall_best_name}")
    print(f"    Sharpe      = {overall_best_r['sharpe']:.3f}")
    print(f"    Net Return  = {overall_best_r['net_return']:+.2f}%")
    print(f"    Final Eq    = ${overall_best_r['final_equity']:,.0f}")
    print(f"    Win Rate    = {overall_best_r['win_rate']:.1f}%")
    print(f"    Trades      = {overall_best_r['trades']:,}")
    print(f"    TARGET exits = {overall_best_r['target_exits']:,}")
    print(f"    TIME exits   = {overall_best_r['time_exits']:,}")
    print(f"    STOP exits   = {overall_best_r['stop_exits']:,}")

    if overall_best_name == "P1_combined":
        verdict = "YES — the combined config holds up. It is the best overall setting."
    elif overall_best_name.startswith("P3"):
        verdict = f"A VARIATION beats it: use {p3_labels.get(overall_best_name, overall_best_name)} instead."
    else:
        verdict = f"The Phase 2 grid cell ({p2_best_name}) outperforms the combined config — consider reverting the RSI/exit/vol changes."

    print(f"\n  VERDICT: {verdict}")

    # Print the single best config parameters
    cfg_map = dict(CONFIGS)
    best_cfg = cfg_map[overall_best_name]
    print(f"\n  Single best config to move forward with:")
    print(f"    ORB window   : {best_cfg['orb_mins']} min")
    print(f"    Target       : {best_cfg['target_pct']*100:.1f}%")
    print(f"    Stop         : {best_cfg['stop_pct']*100:.1f}%")
    print(f"    Volume filter: {best_cfg['vol_mult']:.1f}x")
    print(f"    Time exit    : {best_cfg['exit_hour']}:{best_cfg['exit_min']:02d} ET")
    print(f"    RSI filter   : {'YES (40-60)' if best_cfg.get('use_rsi') else 'NO'}")

if __name__ == "__main__":
    main()
