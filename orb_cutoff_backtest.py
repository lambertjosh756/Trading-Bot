"""
orb_cutoff_backtest.py — Find optimal entry time cutoff for ORB strategy.

Single-pass simulation: records entry_hour for every trade, then tests all
cutoffs by post-hoc filtering. Very fast — only reads data once.

Live bot parameters: 5-min ORB, 2.5% target, 0.3% stop, 1.5x vol, RSI 40-60.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

ET       = pytz.timezone("America/New_York")
DATA_DIR = Path("data/minute")
START    = date(2021, 1, 1)
END      = date(2026, 3, 7)

PROFIT_PCT = 0.025
STOP_PCT   = 0.003
VOL_MULT   = 1.5
VOL_PERIOD = 20
RSI_LOW    = 40
RSI_HIGH   = 60
CAPITAL    = 75_000
MAX_POS    = 8

BLACKLIST = {"SNAP", "RIVN", "HOOD", "UBER"}
UNIVERSE  = {
    "AAPL","MSFT","NVDA","AMZN","TSLA","META","GOOGL","AMD",
    "NFLX","BABA","CRM","PYPL","INTC","QCOM","MU",
    "SPY","QQQ","IWM","ARKK",
    "BAC","JPM","GS","MS",
    "XOM","CVX","SLB","EOG",
    "PFE","MRNA","JNJ",
    "PLTR","SOFI","COIN","MSTR",
    "F","GM","NIO","LCID",
    "AMC","GME","LYFT","PINS",
    "SQ","SHOP","SE","MELI",
    "ZM","DOCU","CRWD","PANW","OKTA",
    "DIS","CMCSA","NWSA",
    "BA","LMT","RTX",
    "GLD","SLV","USO",
} - BLACKLIST


def wilder_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    arr    = np.array(closes[-(period + 1):], dtype=float)
    deltas = np.diff(arr)
    avg_g  = np.where(deltas > 0, deltas, 0.0).mean()
    avg_l  = np.where(deltas < 0, -deltas, 0.0).mean()
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def simulate_all_trades() -> list[dict]:
    """
    Simulate with NO entry cutoff (13:30). Record entry_minute (minutes since
    midnight ET) on each trade so we can filter by cutoff post-hoc.
    """
    all_trades = []
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    total = len([f for f in parquet_files if START <= date.fromisoformat(f.stem) <= END])
    done  = 0

    for pf in parquet_files:
        day = date.fromisoformat(pf.stem)
        if not (START <= day <= END):
            continue

        df = pd.read_parquet(pf)
        df.index = df.index.tz_convert(ET)
        df = df[df["symbol"].isin(UNIVERSE)]
        if df.empty:
            done += 1
            continue

        day_start = df.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
        orb_end   = day_start.replace(hour=9,  minute=35)
        exit_time = day_start.replace(hour=13, minute=30)

        df = df[(df.index >= day_start) & (df.index <= exit_time)]
        if df.empty:
            done += 1
            continue

        orb_high:   dict[str, float]       = {}
        bar_closes: dict[str, list[float]] = defaultdict(list)
        bar_vols:   dict[str, list[float]] = defaultdict(list)
        positions:  dict[str, dict]        = {}
        placed:     set[str]               = set()

        sym_groups = {sym: grp.sort_index() for sym, grp in df.groupby("symbol")}
        all_times  = sorted(df.index.unique())

        for ts in all_times:
            ts_min = ts.hour * 60 + ts.minute  # minutes since midnight ET

            for sym, sym_df in sym_groups.items():
                if ts not in sym_df.index:
                    continue
                bar = sym_df.loc[ts]
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[0]

                c = float(bar["close"])
                v = float(bar["volume"])
                bar_closes[sym].append(c)
                bar_vols[sym].append(v)

                # Record ORB
                if sym not in orb_high and ts >= orb_end:
                    orb_bars = sym_df[(sym_df.index >= day_start) & (sym_df.index < orb_end)]
                    if not orb_bars.empty:
                        orb_high[sym] = float(orb_bars["high"].max())

                # Exits
                if sym in positions:
                    pos = positions[sym]
                    if c >= pos["target"]:
                        all_trades.append({**pos, "exit": pos["target"],
                                           "pnl": (pos["target"] - pos["entry"]) * pos["qty"],
                                           "exit_type": "target", "trade_date": day})
                        del positions[sym]
                        continue
                    elif c <= pos["stop"]:
                        all_trades.append({**pos, "exit": pos["stop"],
                                           "pnl": (pos["stop"] - pos["entry"]) * pos["qty"],
                                           "exit_type": "stop", "trade_date": day})
                        del positions[sym]
                        continue

                # Entry
                if sym in positions or sym in placed:
                    continue
                if len(positions) >= MAX_POS:
                    continue
                if sym not in orb_high or c <= orb_high[sym]:
                    continue

                vols  = bar_vols[sym]
                if len(vols) < 5:
                    continue
                avg_v = sum(vols[-VOL_PERIOD:]) / min(len(vols), VOL_PERIOD)
                if avg_v == 0 or v < VOL_MULT * avg_v:
                    continue

                rsi = wilder_rsi(bar_closes[sym])
                if rsi is None or not (RSI_LOW <= rsi <= RSI_HIGH):
                    continue

                qty    = max(1, int((CAPITAL / MAX_POS) / c))
                target = round(c * (1 + PROFIT_PCT), 2)
                stop   = min(round(c * (1 - STOP_PCT), 2), round(c - 0.01, 2))
                positions[sym] = {"entry": c, "target": target, "stop": stop,
                                  "qty": qty, "entry_min": ts_min, "trade_date": day}
                placed.add(sym)

        # Time exits
        for sym, pos in positions.items():
            sym_df  = sym_groups[sym]
            last_c  = float(sym_df.iloc[-1]["close"])
            all_trades.append({**pos, "exit": last_c,
                                "pnl": (last_c - pos["entry"]) * pos["qty"],
                                "exit_type": "time", "trade_date": day})

        done += 1
        if done % 100 == 0:
            print(f"  ...{done}/{total} days processed, {len(all_trades)} trades so far", flush=True)

    return all_trades


def analyse_cutoffs(trades: list[dict]) -> None:
    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)

    # Cutoffs to test: 9:45 to 13:00 in 15-min steps
    cutoffs = []
    h, m = 9, 45
    while (h, m) <= (13, 0):
        cutoffs.append(h * 60 + m)
        m += 15
        if m >= 60:
            m -= 60
            h += 1

    # All trading days in simulation
    all_days = df["trade_date"].unique()

    print(f"\n{'Cutoff':>8}  {'Net P&L':>10}  {'Trades':>7}  {'Win%':>6}  "
          f"{'Avg Win':>8}  {'Avg Loss':>9}  {'Sharpe':>7}  {'T/day':>6}")
    print("-" * 84)

    results = []
    for cut in cutoffs:
        sub   = df[df["entry_min"] < cut]
        if sub.empty:
            continue
        pnls  = sub["pnl"].values
        wins  = pnls[pnls > 0]
        losses= pnls[pnls <= 0]

        # Daily P&L (include zero-trade days for Sharpe denominator)
        daily = sub.groupby("trade_date")["pnl"].sum().reindex(all_days, fill_value=0)
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

        label = f"{cut//60:02d}:{cut%60:02d}"
        r = {
            "cutoff":   label,
            "net_pnl":  pnls.sum(),
            "trades":   len(pnls),
            "win_rate": len(wins) / len(pnls) * 100,
            "avg_win":  wins.mean()   if len(wins)   else 0,
            "avg_loss": losses.mean() if len(losses) else 0,
            "sharpe":   sharpe,
            "tpd":      len(pnls) / len(all_days),
        }
        results.append(r)
        print(f"  {r['cutoff']:>6}   ${r['net_pnl']:>9,.0f}  {r['trades']:>7,}  "
              f"{r['win_rate']:>5.1f}%  ${r['avg_win']:>7,.0f}  ${r['avg_loss']:>8,.0f}  "
              f"{r['sharpe']:>7.2f}  {r['tpd']:>5.1f}")

    if results:
        best_pnl    = max(results, key=lambda x: x["net_pnl"])
        best_sharpe = max(results, key=lambda x: x["sharpe"])
        print(f"\nBest by net P&L : {best_pnl['cutoff']}  → ${best_pnl['net_pnl']:,.0f}  Sharpe {best_pnl['sharpe']:.2f}")
        print(f"Best by Sharpe  : {best_sharpe['cutoff']}  → ${best_sharpe['net_pnl']:,.0f}  Sharpe {best_sharpe['sharpe']:.2f}")


if __name__ == "__main__":
    print("Simulating all trades (no cutoff)…")
    trades = simulate_all_trades()
    print(f"Total trades collected: {len(trades):,}")
    analyse_cutoffs(trades)
