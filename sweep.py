"""
sweep.py -- Analysis 1: Parameter sweep across 9 configurations.

Runs baseline + 9 variant backtests on 2023-2024 data in parallel
(up to 4 workers). Each writes a JSON summary. Results printed as a
comparison table flagging configs that beat the current Sharpe of 0.71.

Usage:
    python sweep.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

BASELINE_SHARPE = 0.71   # from 2023-2024 run with current config
BASELINE_LABEL  = "baseline (current)"

CONFIGS: List[Dict[str, Any]] = [
    # label,                    extra CLI args
    {"label": BASELINE_LABEL,   "args": []},
    {"label": "target=1.5%",    "args": ["--target", "0.015"]},
    {"label": "target=2.0%",    "args": ["--target", "0.020"]},
    {"label": "stop=0.4%",      "args": ["--stop",   "0.004"]},
    {"label": "stop=0.5%",      "args": ["--stop",   "0.005"]},
    {"label": "rsi=45-55",      "args": ["--rsi-low", "45", "--rsi-high", "55"]},
    {"label": "rsi=35-65",      "args": ["--rsi-low", "35", "--rsi-high", "65"]},
    {"label": "vol=2.0x",       "args": ["--vol-mult", "2.0"]},
    {"label": "blacklist",      "args": ["--blacklist", "SNAP,RIVN,HOOD,UBER"]},
    {"label": "tgt1.5+blist",   "args": ["--target", "0.015",
                                          "--blacklist", "SNAP,RIVN,HOOD,UBER"]},
]

MAX_WORKERS = 4


def run_config(cfg: Dict[str, Any], tmp_dir: str) -> Dict[str, Any]:
    """Run one backtest config as a subprocess; return parsed JSON results."""
    label     = cfg["label"]
    json_path = os.path.join(tmp_dir, f"{label.replace(' ', '_').replace('=','_').replace('+','_')}.json")

    cmd = [
        sys.executable, "backtest.py",
        "--start", "2023-01-01",
        "--end",   "2024-12-31",
        "--equity", "200000",
        "--quiet",
        "--label", label,
        "--json-out", json_path,
    ] + cfg["args"]

    print(f"  [START] {label}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )

    if proc.returncode != 0:
        print(f"  [FAIL]  {label}: {proc.stderr[-300:] if proc.stderr else 'no output'}")
        return {"label": label, "error": True}

    if not Path(json_path).exists():
        print(f"  [FAIL]  {label}: no JSON written")
        return {"label": label, "error": True}

    data = json.loads(Path(json_path).read_text())
    print(f"  [DONE]  {label}  sharpe={data.get('sharpe','?'):.3f}  "
          f"return={data.get('total_return_pct','?'):+.2f}%")
    return data


def print_table(results: List[Dict[str, Any]]) -> None:
    # Sort: baseline first, then by sharpe descending
    baseline = next((r for r in results if r.get("label") == BASELINE_LABEL), None)
    others   = [r for r in results if r.get("label") != BASELINE_LABEL and not r.get("error")]
    others.sort(key=lambda r: r.get("sharpe", -999), reverse=True)
    rows = ([baseline] if baseline else []) + others

    COL = {
        "label":            22,
        "return":            9,
        "net_pnl":          11,
        "sharpe":            8,
        "win_rate":          8,
        "max_dd":            9,
        "profit_f":          9,
        "trades":            8,
        "flag":             20,
    }

    header = (
        f"{'Config':<{COL['label']}} "
        f"{'Return':>{COL['return']}} "
        f"{'Net P&L':>{COL['net_pnl']}} "
        f"{'Sharpe':>{COL['sharpe']}} "
        f"{'Win%':>{COL['win_rate']}} "
        f"{'MaxDD':>{COL['max_dd']}} "
        f"{'PFactor':>{COL['profit_f']}} "
        f"{'Trades':>{COL['trades']}} "
        f"{'Notes':<{COL['flag']}}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  ANALYSIS 1 -- PARAMETER SWEEP  (2023-01-01 to 2024-12-31, $200k equity)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for r in rows:
        if r.get("error"):
            print(f"  {r['label']:<{COL['label']}}  ERROR")
            continue

        sharpe    = r.get("sharpe", 0)
        ret_pct   = r.get("total_return_pct", 0)
        net_pnl   = r.get("net_pnl", 0)
        win_rate  = r.get("win_rate", 0)
        max_dd    = r.get("max_drawdown", 0)
        pf        = r.get("profit_factor", 0)
        trades    = r.get("total_trades", 0)
        label     = r.get("label", "")

        # Flags
        notes = []
        is_baseline = label == BASELINE_LABEL
        if not is_baseline:
            if sharpe > BASELINE_SHARPE:
                improvement = sharpe - BASELINE_SHARPE
                notes.append(f"BEATS baseline +{improvement:.3f}")
                if improvement > 0.30:
                    notes.append("[OVERFIT RISK]")
            elif sharpe >= BASELINE_SHARPE - 0.05:
                notes.append("~same")
            else:
                notes.append(f"worse by {BASELINE_SHARPE - sharpe:.3f}")
        else:
            notes.append("<-- current config")

        flag_str = "  ".join(notes)

        print(
            f"{'*' if (sharpe > BASELINE_SHARPE and not is_baseline) else ' '}"
            f"{label:<{COL['label']}} "
            f"{ret_pct:>+{COL['return']}.2f}% "
            f"${net_pnl:>{COL['net_pnl']-1},.0f} "
            f"{sharpe:>{COL['sharpe']}.3f} "
            f"{win_rate:>{COL['win_rate']}.1f}% "
            f"{max_dd:>{COL['max_dd']}.2f}% "
            f"{pf:>{COL['profit_f']}.3f} "
            f"{trades:>{COL['trades']},} "
            f"{flag_str}"
        )

    print(sep)
    print("  * = beats current config Sharpe of 0.71")
    print()

    # Recommendation
    winners = [r for r in others if r.get("sharpe", 0) > BASELINE_SHARPE]
    if winners:
        best = max(winners, key=lambda r: r["sharpe"])
        margin = best["sharpe"] - BASELINE_SHARPE
        print(f"  Best variant : {best['label']}  (Sharpe {best['sharpe']:.3f}, +{margin:.3f} vs baseline)")
        if margin > 0.30:
            print("  WARNING: Large improvement -- high overfitting risk on in-sample data.")
            print("           Validate on out-of-sample (2025+) before adopting.")
        else:
            print("  Moderate improvement -- plausible on out-of-sample, but verify.")
    else:
        print("  No variant beats the current config -- current parameters appear robust.")
    print()


def main():
    print("\n" + "="*60)
    print("  PARAMETER SWEEP -- starting 10 parallel backtest runs")
    print(f"  Workers: {MAX_WORKERS}  |  Configs: {len(CONFIGS)}")
    print("="*60 + "\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(run_config, cfg, tmp_dir): cfg for cfg in CONFIGS}
            for fut in as_completed(futures):
                results.append(fut.result())

        print_table(results)


if __name__ == "__main__":
    main()
