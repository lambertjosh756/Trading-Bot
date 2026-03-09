"""
sweep_15pct.py

Run 20 backtest configs in parallel targeting 15% annual return
($15,000/yr on $100k) with max 10% drawdown constraint.

All data is already cached in data/minute/ — no API downloads needed
beyond what's already on disk.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BLACKLIST   = "SNAP,RIVN,HOOD,UBER"
START       = "2021-01-01"
END         = "2026-03-07"
EQUITY      = 100_000.0
RESULTS_DIR = Path("results/sweep_15pct")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 20 configs ────────────────────────────────────────────────────────────────
CONFIGS = [
    # BATCH A — profit target sweep (stop=0.3%)
    dict(id=1,  batch="A", target=0.015, stop=0.003, max_pos=15),
    dict(id=2,  batch="A", target=0.020, stop=0.003, max_pos=15),
    dict(id=3,  batch="A", target=0.025, stop=0.003, max_pos=15),
    dict(id=4,  batch="A", target=0.030, stop=0.003, max_pos=15),
    # BATCH B — stop + target grid
    dict(id=5,  batch="B", target=0.015, stop=0.004, max_pos=15),
    dict(id=6,  batch="B", target=0.020, stop=0.004, max_pos=15),
    dict(id=7,  batch="B", target=0.025, stop=0.004, max_pos=15),
    dict(id=8,  batch="B", target=0.020, stop=0.005, max_pos=15),
    dict(id=9,  batch="B", target=0.025, stop=0.005, max_pos=15),
    dict(id=10, batch="B", target=0.030, stop=0.005, max_pos=15),
    # BATCH C — concentrated sizing (max_pos=5)
    dict(id=11, batch="C", target=0.012, stop=0.003, max_pos=5),
    dict(id=12, batch="C", target=0.015, stop=0.003, max_pos=5),
    dict(id=13, batch="C", target=0.020, stop=0.003, max_pos=5),
    dict(id=14, batch="C", target=0.025, stop=0.003, max_pos=5),
    # BATCH D — combined larger target + concentration
    dict(id=15, batch="D", target=0.015, stop=0.003, max_pos=8),
    dict(id=16, batch="D", target=0.020, stop=0.003, max_pos=8),
    dict(id=17, batch="D", target=0.025, stop=0.003, max_pos=8),
    dict(id=18, batch="D", target=0.020, stop=0.004, max_pos=8),
    dict(id=19, batch="D", target=0.025, stop=0.004, max_pos=8),
    dict(id=20, batch="D", target=0.020, stop=0.003, max_pos=10),
]


def label(c):
    return (f"B{c['batch']}-{c['id']:02d}_"
            f"T{c['target']*100:.1f}_S{c['stop']*100:.1f}_P{c['max_pos']}")


def run_config(c):
    lbl      = label(c)
    json_out = str(RESULTS_DIR / f"{lbl}.json")
    cmd = [
        sys.executable, "backtest.py",
        "--start",     START,
        "--end",       END,
        "--equity",    str(EQUITY),
        "--target",    str(c["target"]),
        "--stop",      str(c["stop"]),
        "--max-pos",   str(c["max_pos"]),
        "--blacklist", BLACKLIST,
        "--json-out",  json_out,
        "--label",     lbl,
        "--quiet",
    ]
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return c, None, result.stderr[-300:]
    try:
        data = json.loads(Path(json_out).read_text())
        return c, data, None
    except Exception as e:
        return c, None, str(e)


# ── Run in parallel ───────────────────────────────────────────────────────────
print(f"Launching {len(CONFIGS)} configs (4 workers, all data cached)...\n")
all_results = {}

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(run_config, c): c for c in CONFIGS}
    done = 0
    for fut in as_completed(futures):
        c, data, err = fut.result()
        done += 1
        lbl = label(c)
        if err:
            print(f"  [{done:2d}/20] FAILED  {lbl}: {err}")
        else:
            pnl   = data.get("net_pnl", 0)
            sh    = data.get("sharpe", 0)
            dd    = data.get("max_drawdown_pct", 0)
            print(f"  [{done:2d}/20] done    {lbl}  P&L={pnl:+,.0f}  Sh={sh:.3f}  DD={dd:.2f}%")
        all_results[c["id"]] = (c, data, err)

# ── Aggregate ─────────────────────────────────────────────────────────────────
TRADING_YEARS = (
    (2026 - 2021) + (67 / 252)   # Jan 2021 – Mar 7 2026 ≈ 5.27 years
)

rows = []
for cid in sorted(all_results):
    c, data, err = all_results[cid]
    if data is None:
        continue

    net_pnl    = data.get("net_pnl", 0)
    sharpe     = data.get("sharpe", 0)
    max_dd     = data.get("max_drawdown_pct", 0)   # negative value
    trades     = data.get("total_trades", 0)
    win_pct    = data.get("win_rate_pct", 0)
    avg_annual = net_pnl / TRADING_YEARS
    total_ret  = net_pnl / EQUITY * 100

    meets_return = avg_annual >= 15_000
    meets_dd     = abs(max_dd) <= 10.0
    meets_both   = meets_return and meets_dd

    rows.append({
        "id":          cid,
        "batch":       c["batch"],
        "target_pct":  c["target"] * 100,
        "stop_pct":    c["stop"]   * 100,
        "max_pos":     c["max_pos"],
        "trades":      trades,
        "win_pct":     win_pct,
        "avg_annual":  avg_annual,
        "total_ret":   total_ret,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "meets_return": meets_return,
        "meets_dd":    meets_dd,
        "meets_both":  meets_both,
        "data":        data,
    })

rows.sort(key=lambda r: r["avg_annual"], reverse=True)

# ── Print full results table ──────────────────────────────────────────────────
print("\n" + "="*115)
print("FULL RESULTS — sorted by avg annual return descending")
print("="*115)
print(f"{'#':>2} {'Bt':>2} {'Tgt%':>5} {'Stp%':>5} {'Pos':>4} "
      f"{'Trades':>7} {'Win%':>6} {'Avg Annual':>12} {'5yr Total%':>11} "
      f"{'Sharpe':>7} {'Max DD%':>8} {'>=15k?':>7} {'<=10%DD?':>9} {'PASS':>5}")
print("-"*115)

for r in rows:
    flag = "***" if r["meets_both"] else ("~15k" if r["meets_return"] else "")
    dd_flag = "YES" if r["meets_dd"] else "NO"
    ret_flag = "YES" if r["meets_return"] else "NO"
    print(
        f"{r['id']:>2} {r['batch']:>2}  "
        f"{r['target_pct']:>4.1f}%  {r['stop_pct']:>4.1f}%  {r['max_pos']:>3}  "
        f"{r['trades']:>7,}  {r['win_pct']:>5.1f}%  "
        f"${r['avg_annual']:>+10,.0f}  {r['total_ret']:>+9.1f}%  "
        f"{r['sharpe']:>7.3f}  {r['max_dd']:>7.2f}%  "
        f"{ret_flag:>7}  {dd_flag:>9}  {flag:>5}"
    )

# ── Highlight passing configs ─────────────────────────────────────────────────
passing = [r for r in rows if r["meets_both"]]
near_miss = [r for r in rows if r["meets_dd"] and not r["meets_return"]]
near_miss.sort(key=lambda r: r["avg_annual"], reverse=True)

print("\n" + "="*115)
if passing:
    print(f"CONFIGS MEETING BOTH CRITERIA (>=$15k/yr AND <=10% DD): {len(passing)}")
    print("Ranked by Sharpe:")
    for r in sorted(passing, key=lambda r: r["sharpe"], reverse=True):
        print(f"  Config {r['id']:2d} (Batch {r['batch']}): "
              f"target={r['target_pct']:.1f}% stop={r['stop_pct']:.1f}% pos={r['max_pos']} "
              f"-> ${r['avg_annual']:+,.0f}/yr | Sharpe={r['sharpe']:.3f} | DD={r['max_dd']:.2f}%")
else:
    print("NO CONFIG MEETS BOTH CRITERIA.")
    print(f"\nClosest within <=10% DD (best annual return while staying safe):")
    for r in near_miss[:5]:
        gap = 15_000 - r["avg_annual"]
        print(f"  Config {r['id']:2d}: target={r['target_pct']:.1f}% stop={r['stop_pct']:.1f}% "
              f"pos={r['max_pos']} -> ${r['avg_annual']:+,.0f}/yr  "
              f"(gap to 15k: ${gap:,.0f}) | Sharpe={r['sharpe']:.3f} | DD={r['max_dd']:.2f}%")

# ── Year-by-year for top 3 ────────────────────────────────────────────────────
if passing:
    top3 = sorted(passing, key=lambda r: r["sharpe"], reverse=True)[:3]
else:
    top3 = [r for r in rows if r["meets_dd"]][:3]

print("\n" + "="*115)
print("YEAR-BY-YEAR BREAKDOWN — TOP 3 CONFIGS")
print("="*115)

for rank, r in enumerate(top3, 1):
    d = r["data"]
    ybr = d.get("year_by_year", {})
    print(f"\n  #{rank} Config {r['id']} (Batch {r['batch']}): "
          f"target={r['target_pct']:.1f}% | stop={r['stop_pct']:.1f}% | "
          f"max_pos={r['max_pos']} | 5yr P&L=${d.get('net_pnl',0):+,.0f} | "
          f"Sharpe={r['sharpe']:.3f} | Max DD={r['max_dd']:.2f}%")
    print(f"  {'Year':<6} {'Net Return%':>12} {'Net P&L':>10} {'Sharpe':>8} {'Max DD%':>9}")
    print(f"  {'-'*48}")
    for yr in ["2021", "2022", "2023", "2024", "2025", "2026"]:
        if yr in ybr:
            y = ybr[yr]
            pnl  = y.get("net_pnl", 0)
            sh   = y.get("sharpe", 0)
            dd   = y.get("max_dd_pct", 0)
            ret  = pnl / EQUITY * 100
            flag = " <-- target met" if pnl >= 15_000 else ""
            print(f"  {yr:<6} {ret:>+11.1f}%  ${pnl:>+9,.0f}  {sh:>8.3f}  {dd:>8.2f}%{flag}")

# ── Final recommendation ──────────────────────────────────────────────────────
print("\n" + "="*115)
print("FINAL RECOMMENDATION")
print("="*115)

best = passing[0] if passing else None
if not best:
    best = near_miss[0] if near_miss else rows[0]

best_d = best["data"]
print(f"\n  Best config: #{best['id']} (Batch {best['batch']})")
print(f"  target={best['target_pct']:.1f}% | stop={best['stop_pct']:.1f}% | max_pos={best['max_pos']}")
print(f"  Avg annual return : ${best['avg_annual']:+,.0f}/yr")
print(f"  5-yr total P&L    : ${best_d.get('net_pnl',0):+,.0f} ({best['total_ret']:+.1f}%)")
print(f"  Sharpe            : {best['sharpe']:.3f}")
print(f"  Max drawdown      : {best['max_dd']:.2f}%")
print(f"  Meets >=15k/yr    : {'YES' if best['meets_return'] else 'NO'}")
print(f"  Meets <=10% DD    : {'YES' if best['meets_dd'] else 'NO'}")

if passing:
    print(f"\n  15% annual IS achievable within 10% DD constraint.")
else:
    best_annual = rows[0]["avg_annual"]
    gap = 15_000 - best_annual
    print(f"\n  15% annual is NOT achievable within 10% DD on this strategy.")
    print(f"  Best achievable within 10% DD: ${near_miss[0]['avg_annual']:+,.0f}/yr "
          f"(gap: ${gap:,.0f}) — config #{near_miss[0]['id']}")
    print(f"  To hit $15k/yr: either accept higher DD or use a different strategy.")

print("\nDone.")
