#!/usr/bin/env python3
"""
ORB Strategy Optimizer — 16 Parameter Variations
Baseline: 1.2% target | 15-min ORB | 1.5x vol | 0.5% stop | 12:30 exit | 15 stocks
"""
import sys, time, json, io
import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SEED          = 42
N_STOCKS      = 15
N_DAYS        = 252
INITIAL_EQ    = 100_000.0
POS_FRAC      = 1.0 / N_STOCKS   # equal-weight slot per stock

# ─── Data Generation ────────────────────────────────────────────────────────

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
            # --- Day character (drives strategy edge) ---
            roll = float(rng.random())
            if roll < 0.22:                        # trend-up day
                daily_drift = float(rng.uniform(0.004, 0.018)) / 390
            elif roll < 0.38:                      # trend-down day
                daily_drift = float(rng.uniform(-0.018, -0.004)) / 390
            else:                                  # choppy / range day
                daily_drift = float(rng.normal(0, 0.001)) / 390

            gap    = float(rng.normal(0, price * 0.005))
            open_p = max(price + gap, price * 0.4)

            highs  = np.empty(390)
            lows   = np.empty(390)
            closes = np.empty(390)
            vols   = np.empty(390, dtype=np.int64)

            p = open_p
            for m in range(390):
                # Volatility envelope: elevated open & close
                if   m < 30:  vm = 1.9 - m * 0.022
                elif m > 355: vm = 1.4 + (m - 355) * 0.015
                else:         vm = 0.82

                ret = daily_drift + float(rng.normal(0, per_min_vol * vm))
                c   = max(p * (1.0 + ret), 0.01)

                spike = abs(float(rng.normal(0, per_min_vol * 0.45)))
                h = max(p, c) * (1.0 + spike)
                l = min(p, c) * (1.0 - spike * 0.65)

                # U-shaped volume
                if   m < 30:  vf = 3.5 - m * 0.05
                elif m > 355: vf = 1.5 + (m - 355) * 0.12
                else:         vf = 0.62 + abs(float(rng.normal(0, 0.28)))
                vf = max(0.05, vf) * float(rng.lognormal(0, 0.48))
                v  = max(100, int(avg_daily_vol / 390 * vf))

                highs[m]  = h
                lows[m]   = l
                closes[m] = c
                vols[m]   = v
                p = c

            data[s][d] = {
                'high': highs, 'low': lows, 'close': closes,
                'volume': vols, 'avg_daily_vol': avg_daily_vol,
            }
            price = p

    return data

# ─── Indicators ─────────────────────────────────────────────────────────────

def ema(arr, period):
    k = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1.0 - k)
    return out

def rsi(arr, period=14):
    n   = len(arr)
    out = np.full(n, 50.0)
    if n < period + 1:
        return out
    d   = np.diff(arr)
    g   = np.maximum(d, 0.0)
    l_  = np.maximum(-d, 0.0)
    ag  = float(np.mean(g[:period]))
    al  = float(np.mean(l_[:period]))
    for i in range(period, n):
        ag = (ag * (period - 1) + g[i-1]) / period
        al = (al * (period - 1) + l_[i-1]) / period
        out[i] = 100.0 if al == 0.0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out

def macd(arr, fast=12, slow=26, sig=9):
    if len(arr) < slow:
        return np.zeros(len(arr)), np.zeros(len(arr))
    ef = ema(arr, fast)
    es = ema(arr, slow)
    m_ = ef - es
    return m_, ema(m_, sig)

def bollinger_lower(arr, period=20, n_std=2.0):
    n  = len(arr)
    lb = arr.copy()
    for i in range(period - 1, n):
        w = arr[i - period + 1 : i + 1]
        lb[i] = w.mean() - n_std * w.std(ddof=0)
    return lb

# ─── Backtest Engine ─────────────────────────────────────────────────────────

BAR_TIMES = np.array([9*60 + 30 + m for m in range(390)])   # minutes since midnight

def backtest(data, cfg):
    tgt      = cfg['target_pct']
    stp      = cfg['stop_pct']
    orb      = cfg['orb_mins']
    vmult    = cfg['vol_mult']
    exit_t   = cfg['exit_hour'] * 60 + cfg['exit_min']
    use_ema  = cfg.get('use_ema',  False)
    use_macd = cfg.get('use_macd', False)
    use_rsi  = cfg.get('use_rsi',  False)
    use_bb   = cfg.get('use_bb',   False)

    equity      = INITIAL_EQ
    trades      = []          # list of ('T'|'S'|'X', pnl_$)
    daily_rets  = []

    for d in range(N_DAYS):
        day_eq_start = equity
        day_pnl      = 0.0

        for s in range(N_STOCKS):
            bars  = data[s][d]
            cl    = bars['close']
            hi    = bars['high']
            lo    = bars['low']
            vol   = bars['volume']
            avb   = bars['avg_daily_vol'] / 390.0   # avg bar volume

            orb_high = float(hi[:orb].max())

            # Compute indicators once per (stock, day) — only if needed
            ema20     = ema(cl, 20)          if use_ema  else None
            rsi14     = rsi(cl, 14)          if use_rsi  else None
            ml, sl_   = macd(cl)             if use_macd else (None, None)
            bb_lo     = bollinger_lower(cl)  if use_bb   else None

            in_trade    = False
            entry_price = 0.0
            tgt_price   = 0.0
            stp_price   = 0.0

            for m in range(orb, 390):
                bt = BAR_TIMES[m]

                if in_trade:
                    if hi[m] >= tgt_price:
                        pnl = equity * POS_FRAC * tgt
                        day_pnl += pnl;  equity += pnl
                        trades.append(('T', pnl))
                        break
                    elif lo[m] <= stp_price:
                        pnl = -(equity * POS_FRAC * stp)
                        day_pnl += pnl;  equity += pnl
                        trades.append(('S', pnl))
                        break
                    elif bt >= exit_t:
                        pnl = equity * POS_FRAC * (cl[m] - entry_price) / entry_price
                        day_pnl += pnl;  equity += pnl
                        trades.append(('X', pnl))
                        break
                else:
                    if bt >= exit_t:
                        break
                    if cl[m] <= orb_high:
                        continue
                    if vol[m] < vmult * avb:
                        continue
                    if use_ema  and cl[m] <= ema20[m]:
                        continue
                    if use_macd and ml[m] <= sl_[m]:
                        continue
                    if use_rsi  and not (40.0 <= rsi14[m] <= 60.0):
                        continue
                    if use_bb   and cl[m] > bb_lo[m] * 1.04:
                        continue

                    in_trade    = True
                    entry_price = cl[m]
                    tgt_price   = entry_price * (1.0 + tgt)
                    stp_price   = entry_price * (1.0 - stp)

        daily_rets.append(day_pnl / day_eq_start if day_eq_start > 0 else 0.0)

    n = len(trades)
    if n == 0:
        return dict(trades=0, win_rate=0.0, net_return=0.0,
                    final_equity=equity, sharpe=0.0,
                    target_exits=0, time_exits=0, stop_exits=0)

    wins    = sum(1 for t, p in trades if p > 0)
    t_exits = sum(1 for t, _ in trades if t == 'T')
    x_exits = sum(1 for t, _ in trades if t == 'X')
    s_exits = sum(1 for t, _ in trades if t == 'S')

    dr  = np.array(daily_rets, dtype=float)
    std = float(dr.std(ddof=1))
    sharpe = float(dr.mean() / std * (252 ** 0.5)) if std > 0 else 0.0

    return dict(
        trades       = n,
        win_rate     = wins / n * 100,
        net_return   = (equity - INITIAL_EQ) / INITIAL_EQ * 100,
        final_equity = equity,
        sharpe       = sharpe,
        target_exits = t_exits,
        time_exits   = x_exits,
        stop_exits   = s_exits,
    )

# ─── Configurations ──────────────────────────────────────────────────────────

BASE = dict(target_pct=0.012, stop_pct=0.005, orb_mins=15,
            vol_mult=1.5, exit_hour=12, exit_min=30)

CONFIGS = [
    # Batch 1 — Stop Loss
    ("stop_0.3%",        {**BASE, 'stop_pct': 0.003}),
    ("stop_0.4%",        {**BASE, 'stop_pct': 0.004}),
    ("stop_0.7%",        {**BASE, 'stop_pct': 0.007}),
    # Batch 2 — Volume Filter
    ("vol_1.2x",         {**BASE, 'vol_mult': 1.2}),
    ("vol_2.0x",         {**BASE, 'vol_mult': 2.0}),
    ("vol_2.5x",         {**BASE, 'vol_mult': 2.5}),
    # Batch 3 — ORB Timeframe
    ("orb_5min",         {**BASE, 'orb_mins': 5}),
    ("orb_10min",        {**BASE, 'orb_mins': 10}),
    ("orb_30min",        {**BASE, 'orb_mins': 30}),
    # Batch 4 — Time Exit
    ("exit_11:00",       {**BASE, 'exit_hour': 11, 'exit_min':  0}),
    ("exit_12:00",       {**BASE, 'exit_hour': 12, 'exit_min':  0}),
    ("exit_13:30",       {**BASE, 'exit_hour': 13, 'exit_min': 30}),
    # Batch 5 — Indicator Filters
    ("ema20_filter",     {**BASE, 'use_ema':  True}),
    ("macd_filter",      {**BASE, 'use_macd': True}),
    ("rsi_40-60",        {**BASE, 'use_rsi':  True}),
    ("bb_reversal",      {**BASE, 'use_bb':   True}),
]

BATCHES = [
    ("BATCH 1 — Stop Loss",        ["stop_0.3%",    "stop_0.4%",   "stop_0.7%"]),
    ("BATCH 2 — Volume Filter",    ["vol_1.2x",     "vol_2.0x",    "vol_2.5x"]),
    ("BATCH 3 — ORB Timeframe",    ["orb_5min",     "orb_10min",   "orb_30min"]),
    ("BATCH 4 — Time Exit",        ["exit_11:00",   "exit_12:00",  "exit_13:30"]),
    ("BATCH 5 — Indicator Filters",["ema20_filter", "macd_filter", "rsi_40-60", "bb_reversal"]),
]

# ─── Formatting ──────────────────────────────────────────────────────────────

W  = 108
HDR = (f"{'Config':<20} {'Trades':>7} {'Win%':>7} {'NetRet%':>9} "
       f"{'FinalEq':>12} {'Sharpe':>8} {'TARGET':>8} {'TIME_EXIT':>10} {'STOP':>7}")

def row(name, r, winner=False, label_override=None):
    tag  = "  <<< WINNER" if winner else ""
    nm   = label_override or name
    return (f"{nm:<20} {r['trades']:>7,} {r['win_rate']:>6.1f}% "
            f"{r['net_return']:>8.2f}% ${r['final_equity']:>10,.0f} "
            f"{r['sharpe']:>8.3f} {r['target_exits']:>8,} "
            f"{r['time_exits']:>10,} {r['stop_exits']:>7,}{tag}")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Generating synthetic intraday data …", flush=True)
    data = generate_data()
    print(f"  {N_STOCKS} stocks × {N_DAYS} days × 390 bars/day ready  "
          f"({time.time()-t0:.1f}s)", flush=True)

    print("\nRunning BASELINE …", flush=True)
    baseline = backtest(data, BASE)

    results = {}
    for name, cfg in CONFIGS:
        print(f"  Running {name} …", flush=True)
        results[name] = backtest(data, cfg)

    elapsed = time.time() - t0
    print(f"\n[DONE] All 17 backtests done in {elapsed:.1f}s\n", flush=True)

    # ── Baseline summary ──────────────────────────────────────────────────
    b = baseline
    print("=" * W)
    print("BASELINE  (target=1.2% | stop=0.5% | orb=15min | vol=1.5x | exit=12:30)")
    print("=" * W)
    print(HDR)
    print("-" * W)
    print(row("BASELINE", b))
    print()

    # ── Per-batch tables ──────────────────────────────────────────────────
    batch_winners = {}

    for batch_name, names in BATCHES:
        print("=" * W)
        print(batch_name)
        print("=" * W)
        print(HDR)
        print("-" * W)
        print(row("[BASELINE]", b))

        winner = max(names, key=lambda n: results[n]['sharpe'])
        batch_winners[batch_name] = (winner, results[winner])

        for n in names:
            is_win = (n == winner)
            print(row(n, results[n], is_win))
        print()

    # ── Final recommendation ──────────────────────────────────────────────
    best_name, best_r = max(results.items(), key=lambda x: x[1]['sharpe'])

    print("=" * W)
    print("FINAL RECOMMENDATION")
    print("=" * W)
    print("\nBatch winners (by Sharpe):")
    for bn, (wn, wr) in batch_winners.items():
        print(f"  {bn}:  {wn:<20}  "
              f"Sharpe={wr['sharpe']:.3f}  Win%={wr['win_rate']:.1f}%  "
              f"Net={wr['net_return']:.2f}%  Trades={wr['trades']}")

    print(f"\n{'─'*W}")
    print(f"  Best overall config : {best_name}")
    print(f"  Trades={best_r['trades']}  |  Win%={best_r['win_rate']:.1f}%  |  "
          f"Net Return={best_r['net_return']:.2f}%  |  "
          f"Equity=${best_r['final_equity']:,.0f}  |  Sharpe={best_r['sharpe']:.3f}")
    print(f"\n  Baseline            : "
          f"Trades={b['trades']}  |  Win%={b['win_rate']:.1f}%  |  "
          f"Net Return={b['net_return']:.2f}%  |  "
          f"Equity=${b['final_equity']:,.0f}  |  Sharpe={b['sharpe']:.3f}")
    print(f"\n  Improvement vs baseline  → "
          f"Sharpe Δ={best_r['sharpe']-b['sharpe']:+.3f}  |  "
          f"Net Return Δ={best_r['net_return']-b['net_return']:+.2f}%")

    # ── Recommended combined config ───────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  RECOMMENDED COMBINED CONFIG (pick best from each batch, if compatible):")
    for bn, (wn, wr) in batch_winners.items():
        print(f"    {bn.split('—')[1].strip():<22}  →  {wn}")
    print()

if __name__ == "__main__":
    main()
