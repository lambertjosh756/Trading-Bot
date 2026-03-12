"""
combined_4strategy.py
Four-strategy portfolio backtest — current live config
  ORB            $33,333   (33.3% — scaled from $100k CSV × 0.3333)
  Swing Mom-14   $33,333   (33.3%)
  Overnight Mom  $22,222   (22.2%)
  ETF Rotation E $11,111   (11.1%)
  ─────────────────────────
  TOTAL          $100,000
Proportions match live allocation (75/75/50/25 ratio) scaled to $100k.
Jan 2021 – Mar 2026 | vs QQQ buy & hold
"""
from __future__ import annotations
import os, warnings, random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(".env")

START      = "2021-01-01"
END        = "2026-03-07"
ORB_CAP    =  33_333.0
SWING_CAP  =  33_333.0
ON_CAP     =  22_222.0
ETF_CAP    =  11_111.0
TOTAL_CAP  = 100_000.0
RESULTS    = Path("results")
DATA_DIR   = Path("data/daily")

TRADING_YEARS = 5 + 67/252

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sharpe(pnl: pd.Series) -> float:
    s = pnl.std()
    return float(pnl.mean() / s * np.sqrt(252)) if s > 0 else 0.0

def max_dd(equity: pd.Series) -> float:
    rm = equity.cummax()
    return float(((equity - rm) / rm * 100).min())

def pf(pnl: pd.Series) -> float:
    g = pnl[pnl > 0].sum(); l = abs(pnl[pnl < 0].sum())
    return float(g / l) if l > 0 else float("inf")

def cagr(final: float, start: float, years: float) -> float:
    return ((final / start) ** (1 / years) - 1) * 100 if years > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. ORB — scale existing daily CSV from $100k → $75k
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("COMBINED 4-STRATEGY PORTFOLIO BACKTEST")
print("ORB $33k | Swing Mom-14 $33k | Overnight $22k | ETF Rotation $11k")
print("Jan 2021 – Mar 2026  |  $100k total")
print("="*70)

print("\n[1/4] Loading ORB daily P&L (scaled 0.75x)...")
orb_daily_path = RESULTS / "daily_2021-01-01_2026-03-07.csv"
orb_raw = pd.read_csv(orb_daily_path, parse_dates=["date"]).set_index("date")
orb_pnl = (orb_raw["day_pnl"] * (ORB_CAP / 100_000.0)).copy()
orb_pnl.index = pd.to_datetime(orb_pnl.index)
orb_pnl = orb_pnl[(orb_pnl.index >= START) & (orb_pnl.index <= END)]
orb_equity = ORB_CAP + orb_pnl.cumsum()
print(f"  ORB: {len(orb_pnl)} days | net=${orb_pnl.sum():+,.0f} | "
      f"final=${orb_equity.iloc[-1]:,.0f} | return={orb_pnl.sum()/ORB_CAP*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ETF ROTATION — Strategy E, $25k
#    Aggressive Triple Momentum: top-2 of TQQQ/UPRO/TNA by 4-week return
#    Weekly rebalance, replace negatives with SHY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] ETF Rotation Strategy E (TQQQ/UPRO/TNA, 4-week mom)...")
UNIVERSE_ETF = ["TQQQ", "UPRO", "TNA", "SHY", "QQQ"]
FETCH_START  = "2019-06-01"
MOM_4W       = 20
COMMISSION   = 0.0005

raw_etf = yf.download(UNIVERSE_ETF, start=FETCH_START, end="2026-03-10",
                      auto_adjust=True, progress=False)
prices_etf = raw_etf["Close"].copy()
prices_etf.index = pd.to_datetime(prices_etf.index).tz_localize(None)
prices_etf = prices_etf.ffill()

bt_mask_etf = (prices_etf.index >= START) & (prices_etf.index <= END)
etf_dates   = prices_etf.index[bt_mask_etf]

daily_rets_etf = prices_etf.pct_change().fillna(0.0)
mom_4w_etf     = prices_etf / prices_etf.shift(MOM_4W) - 1

EQUITY_ETFS = ["TQQQ", "UPRO", "TNA"]
_bd = pd.Series(etf_dates, dtype="datetime64[ns]")
weekly_dates_etf = set(_bd.groupby(_bd.dt.to_period("W")).first().values)

def signal_E(dt):
    m = mom_4w_etf.loc[dt, EQUITY_ETFS].dropna()
    if m.empty:
        return {"SHY": 1.0}
    top2 = m.nlargest(2).index.tolist()
    result: dict = {}
    for sym in top2:
        if float(m[sym]) > 0:
            result[sym] = result.get(sym, 0.0) + 0.5
        else:
            result["SHY"] = result.get("SHY", 0.0) + 0.5
    return result

def turnover(old, new):
    syms = set(old) | set(new)
    return sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in syms) / 2.0

pos_etf = {}
eq_etf  = ETF_CAP
etf_curve: dict = {}
for i, dt in enumerate(etf_dates):
    if i > 0 and pos_etf:
        day_ret = sum(pos_etf.get(s, 0.0) * float(daily_rets_etf.loc[dt, s])
                      for s in pos_etf if s in daily_rets_etf.columns)
        eq_etf *= (1.0 + day_ret)
    if i == 0 or dt in weekly_dates_etf:
        new_pos = signal_E(dt)
        tv = turnover(pos_etf, new_pos)
        if tv > 1e-9:
            eq_etf -= eq_etf * tv * COMMISSION * 2
        pos_etf = new_pos
    etf_curve[dt] = eq_etf

etf_equity = pd.Series(etf_curve)
etf_equity.index = pd.to_datetime(etf_equity.index)
etf_pnl = etf_equity.diff().fillna(etf_equity.iloc[0] - ETF_CAP)
print(f"  ETF: {len(etf_equity)} days | net=${etf_pnl.sum():+,.0f} | "
      f"final=${etf_equity.iloc[-1]:,.0f} | return={etf_pnl.sum()/ETF_CAP*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SWING MOM-14 — momentum style, 14-day hold, $75k
#    price > SMA50, top-20% 20d momentum, 1.5x volume, 5% stop
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Swing Momentum-14 ($75k)...")

SWING_UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","NFLX","ORCL","IBM",
    "INTC","AMD","QCOM","MU","AVGO","TXN","AMAT","LRCX","KLAC","ADI",
    "JPM","BAC","WFC","GS","MS","C","BLK","AXP","USB","PNC","COF","SCHW",
    "JNJ","PFE","ABT","UNH","MRK","LLY","TMO","MDT","ABBV","AMGN","GILD",
    "HD","NKE","MCD","SBUX","TGT","LOW","COST","WMT","DIS","CMCSA",
    "XOM","CVX","CAT","HON","GE","MMM","BA","DE","RTX","LIN","UPS","FDX",
    "VZ","T","KO","PEP","PM","PG","MO","CL",
    "PYPL","CRM","UBER","SQ","SHOP","ZM","NOW","SNOW","DDOG","NET",
    "NIO","RIVN","F","GM",
]
_seen: set = set()
SWING_UNIVERSE = [s for s in SWING_UNIVERSE if s not in _seen and not _seen.add(s)]

HOLD_DAYS  = 14
POS_SIZE   = 5_000.0
MAX_POS    = 15
STOP_PCT   = 0.05
TC         = 0.0005
MIN_PRICE  = 20.0
MIN_VOL    = 3_000_000

print(f"  Downloading {len(SWING_UNIVERSE)} swing universe symbols...")
raw_sw = yf.download(SWING_UNIVERSE, start=START, end="2026-03-10",
                     auto_adjust=True, progress=False, threads=True)

if isinstance(raw_sw.columns, pd.MultiIndex):
    sw_closes  = raw_sw["Close"]
    sw_opens   = raw_sw["Open"]
    sw_highs   = raw_sw["High"]
    sw_lows    = raw_sw["Low"]
    sw_volumes = raw_sw["Volume"]
else:
    s = SWING_UNIVERSE[0]
    sw_closes  = raw_sw[["Close"]].rename(columns={"Close": s})
    sw_opens   = raw_sw[["Open"]].rename(columns={"Open": s})
    sw_highs   = raw_sw[["High"]].rename(columns={"High": s})
    sw_lows    = raw_sw[["Low"]].rename(columns={"Low": s})
    sw_volumes = raw_sw[["Volume"]].rename(columns={"Volume": s})

sw_data: Dict[str, pd.DataFrame] = {}
for sym in SWING_UNIVERSE:
    try:
        df = pd.DataFrame({
            "open": sw_opens[sym], "high": sw_highs[sym],
            "low": sw_lows[sym],   "close": sw_closes[sym],
            "volume": sw_volumes[sym],
        }).dropna()
        if len(df) < 260: continue
        if df["close"].iloc[-20:].mean() < MIN_PRICE: continue
        if df["volume"].iloc[-20:].mean() < MIN_VOL: continue
        # Add indicators
        df["sma50"]    = df["close"].rolling(50).mean()
        df["mom20"]    = df["close"].pct_change(20)
        df["vol20"]    = df["volume"].rolling(20).mean()
        df["vol_mult"] = df["volume"] / df["vol20"]
        sw_data[sym] = df
    except Exception:
        continue

print(f"  {len(sw_data)} symbols passed filters")

# Cross-sectional momentum rank
mom_panel  = pd.DataFrame({s: sw_data[s]["mom20"] for s in sw_data})
mom_rank   = mom_panel.rank(axis=1, pct=True)

all_sw_dates = sorted(set(
    d for df in sw_data.values() for d in df.index
    if pd.Timestamp(START) <= d <= pd.Timestamp(END)
))

date_idx_sw: Dict[str, Dict] = {
    sym: {d: i for i, d in enumerate(df.index)}
    for sym, df in sw_data.items()
}

signal_queue_sw: Dict[pd.Timestamp, List[str]] = defaultdict(list)
for sym, df in sw_data.items():
    df_ix = date_idx_sw[sym]
    for d in df.index:
        if d < pd.Timestamp(START) or d > pd.Timestamp(END):
            continue
        if d not in mom_rank.index or sym not in mom_rank.columns:
            continue
        rank_val = mom_rank.loc[d, sym]
        row = df.loc[d]
        if (row["close"] > row["sma50"] and
                rank_val >= 0.80 and
                row["vol_mult"] >= 1.5):
            if d in df_ix:
                next_i = df_ix[d] + 1
                if next_i < len(df):
                    signal_queue_sw[df.index[next_i]].append(sym)

sw_equity_val = SWING_CAP
sw_eq_curve: dict = {}
sw_open_pos: list = []

for today in all_sw_dates:
    # Exits
    still_open = []
    for pos in sw_open_pos:
        sym   = pos["symbol"]
        df    = sw_data[sym]
        df_ix = date_idx_sw[sym]
        if today not in df_ix:
            still_open.append(pos); continue
        row       = df.iloc[df_ix[today]]
        stop_hit  = row["low"] <= pos["stop_price"]
        time_exit = today >= pos["exit_date"]
        if stop_hit or time_exit:
            exit_price = max(pos["stop_price"], row["open"]) if stop_hit else row["open"]
            net_pnl    = pos["shares"] * (exit_price - pos["entry_price"]) - pos["shares"] * exit_price * TC
            sw_equity_val += net_pnl
        else:
            still_open.append(pos)
    sw_open_pos = still_open

    # Entries
    candidates = signal_queue_sw.get(today, [])
    random.seed(int(today.timestamp()) % 2**32)
    random.shuffle(candidates)
    for sym in candidates:
        if len(sw_open_pos) >= MAX_POS: break
        if any(p["symbol"] == sym for p in sw_open_pos): continue
        df    = sw_data[sym]
        df_ix = date_idx_sw[sym]
        if today not in df_ix: continue
        i   = df_ix[today]
        row = df.iloc[i]
        entry_price = row["open"]
        if entry_price <= 0: continue
        shares = int(POS_SIZE / entry_price)
        if shares == 0: continue
        stop_price  = entry_price * (1 - STOP_PCT)
        exit_i      = i + HOLD_DAYS
        exit_date   = df.index[exit_i] if exit_i < len(df) else df.index[-1]
        sw_equity_val -= shares * entry_price * TC
        sw_open_pos.append({
            "symbol": sym, "entry_date": today, "exit_date": exit_date,
            "entry_price": entry_price, "stop_price": stop_price, "shares": shares,
        })
    sw_eq_curve[today] = sw_equity_val

# Force-close remaining
for pos in sw_open_pos:
    sym = pos["symbol"]
    df  = sw_data[sym]
    last_row = df.iloc[-1]
    net_pnl  = pos["shares"] * (last_row["close"] - pos["entry_price"]) - pos["shares"] * last_row["close"] * TC
    sw_equity_val += net_pnl

swing_equity = pd.Series(sw_eq_curve)
swing_equity.index = pd.to_datetime(swing_equity.index)
swing_pnl = swing_equity.diff().fillna(swing_equity.iloc[0] - SWING_CAP)
print(f"  Swing: {len(swing_equity)} days | net=${swing_pnl.sum():+,.0f} | "
      f"final=${swing_equity.iloc[-1]:,.0f} | return={swing_pnl.sum()/SWING_CAP*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. OVERNIGHT MOMENTUM — top-5, 20-day, $50k
#    buy at close, sell at next open
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Overnight Momentum (top-5, 20-day, $50k)...")

ON_UNIVERSE = [
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

on_closes_d, on_opens_d, on_vols_d = {}, {}, {}
for sym in ON_UNIVERSE:
    df = load_parquet(sym)
    if df is not None and len(df) > 25:
        on_closes_d[sym] = df["close"]
        on_opens_d[sym]  = df["open"]
        on_vols_d[sym]   = df["volume"]

on_closes  = pd.DataFrame(on_closes_d).sort_index()
on_opens   = pd.DataFrame(on_opens_d).sort_index()
on_volumes = pd.DataFrame(on_vols_d).sort_index()

trade_dates_on = on_closes.index[
    (on_closes.index >= pd.Timestamp(START)) &
    (on_closes.index <= pd.Timestamp(END))
]

ON_TOP_N   = 5
ON_MOM_WIN = 20
on_alloc   = ON_CAP / ON_TOP_N
on_daily: Dict = {}

for today in trade_dates_on:
    future = on_closes.index[on_closes.index > today]
    if not len(future): continue
    tmrw = future[0]
    past = on_closes.index[on_closes.index < today]
    if len(past) < ON_MOM_WIN: continue
    p0 = past[-ON_MOM_WIN]
    scores = {}
    for sym in on_closes.columns:
        try:
            ct = on_closes.loc[today, sym]; c0 = on_closes.loc[p0, sym]
            if pd.isna(ct) or pd.isna(c0) or c0 == 0 or ct < 10: continue
            if on_volumes[sym][on_volumes.index <= today].tail(20).mean() < 2e6: continue
            scores[sym] = (ct - c0) / c0
        except: continue
    if not scores: continue
    top = sorted(scores, key=scores.get, reverse=True)[:ON_TOP_N]
    dp = 0.0
    for sym in top:
        try:
            ep = on_closes.loc[today, sym]; xp = on_opens.loc[tmrw, sym]
            if pd.isna(ep) or pd.isna(xp) or ep == 0: continue
            dp += (xp - ep) / ep * on_alloc
        except: continue
    if dp: on_daily[tmrw] = on_daily.get(tmrw, 0.0) + dp

on_pnl = pd.Series(on_daily).sort_index()
on_pnl.index = pd.to_datetime(on_pnl.index)
on_equity = ON_CAP + on_pnl.cumsum()
print(f"  Overnight: {len(on_pnl)} days | net=${on_pnl.sum():+,.0f} | "
      f"final=${on_equity.iloc[-1]:,.0f} | return={on_pnl.sum()/ON_CAP*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINE ALL FOUR STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Combining strategies...")

all_idx = (orb_pnl.index.union(etf_pnl.index)
                          .union(swing_pnl.index)
                          .union(on_pnl.index))
all_idx = all_idx.sort_values()

combined_pnl = (
    orb_pnl.reindex(all_idx, fill_value=0) +
    etf_pnl.reindex(all_idx, fill_value=0) +
    swing_pnl.reindex(all_idx, fill_value=0) +
    on_pnl.reindex(all_idx, fill_value=0)
)
combined_equity = TOTAL_CAP + combined_pnl.cumsum()

# ─────────────────────────────────────────────────────────────────────────────
# 6. QQQ BENCHMARK (scale to $225k)
# ─────────────────────────────────────────────────────────────────────────────
qqq_px = prices_etf.loc[etf_dates, "QQQ"].dropna()
qqq_norm = qqq_px / qqq_px.iloc[0] * TOTAL_CAP
# Align to combined index
combined_df = pd.DataFrame({
    "portfolio": combined_equity,
    "QQQ": qqq_norm,
}).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 7. STATS
# ─────────────────────────────────────────────────────────────────────────────
port_ret  = (combined_df["portfolio"].iloc[-1] / TOTAL_CAP - 1) * 100
qqq_ret   = (combined_df["QQQ"].iloc[-1] / TOTAL_CAP - 1) * 100
port_dd   = max_dd(combined_df["portfolio"])
qqq_dd    = max_dd(combined_df["QQQ"])
port_net  = combined_df["portfolio"].iloc[-1] - TOTAL_CAP
port_ann  = port_net / TRADING_YEARS
port_cagr = cagr(combined_df["portfolio"].iloc[-1], TOTAL_CAP, TRADING_YEARS)
qqq_cagr  = cagr(combined_df["QQQ"].iloc[-1], TOTAL_CAP, TRADING_YEARS)
port_sh   = sharpe(combined_pnl.reindex(combined_df.index, fill_value=0))
qqq_daily = combined_df["QQQ"].pct_change().dropna()
qqq_sh    = float(qqq_daily.mean() / qqq_daily.std() * np.sqrt(252))

print(f"\n{'='*65}")
print(f"COMBINED PORTFOLIO RESULTS  (Jan 2021 – Mar 2026)")
print(f"{'='*65}")
strats = [
    ("ORB $33k",       orb_pnl.sum(),    ORB_CAP),
    ("Swing $33k",    swing_pnl.sum(),  SWING_CAP),
    ("Overnight $22k", on_pnl.sum(),    ON_CAP),
    ("ETF Rot $11k",  etf_pnl.sum(),    ETF_CAP),
    ("COMBINED $225k",port_net,         TOTAL_CAP),
]
print(f"\n{'Strategy':<20} {'Net P&L':>12} {'Return':>9} {'CAGR':>8}")
print(f"{'─'*20} {'─'*12} {'─'*9} {'─'*8}")
for lbl, net, cap in strats:
    r = net/cap*100
    c_val = cagr(cap+net, cap, TRADING_YEARS)
    sep = "\n" if lbl.startswith("COMBINED") else ""
    print(f"{sep}{lbl:<20} ${net:>+11,.0f} {r:>+8.1f}% {c_val:>+7.1f}%")

print(f"\n{'':20} {'Portfolio':>12} {'QQQ B&H':>10}")
print(f"{'─'*44}")
print(f"{'Total return':<20} {port_ret:>+11.1f}% {qqq_ret:>+9.1f}%")
print(f"{'CAGR':<20} {port_cagr:>+11.1f}% {qqq_cagr:>+9.1f}%")
print(f"{'Avg annual P&L':<20} ${port_ann:>+10,.0f}")
print(f"{'Max drawdown':<20} {port_dd:>+11.1f}% {qqq_dd:>+9.1f}%")
print(f"{'Sharpe (ann.)':<20} {port_sh:>12.2f} {qqq_sh:>10.2f}")
print(f"{'Final equity':<20} ${combined_df['portfolio'].iloc[-1]:>+10,.0f}")

# Year-by-year
print(f"\n{'Year':<7} {'ORB':>9} {'Swing':>9} {'Overnight':>10} {'ETF':>8} {'Total':>9} {'Ret%':>7}")
print(f"{'─'*60}")
for yr in [2021,2022,2023,2024,2025]:
    def yr_sum(s, idx=None):
        if idx is not None:
            s = s.reindex(idx, fill_value=0)
        return float(s[s.index.year == yr].sum())
    o = yr_sum(orb_pnl); sw = yr_sum(swing_pnl); n = yr_sum(on_pnl); e = yr_sum(etf_pnl)
    tot = o + sw + n + e
    print(f"{yr:<7} ${o:>+8,.0f} ${sw:>+8,.0f} ${n:>+9,.0f} ${e:>+7,.0f} ${tot:>+8,.0f} {tot/TOTAL_CAP*100:>+6.1f}%")
yr26 = lambda s: float(s[s.index.year == 2026].sum())
o,sw,n,e = yr26(orb_pnl),yr26(swing_pnl),yr26(on_pnl),yr26(etf_pnl)
tot=o+sw+n+e
print(f"{'2026Q1':<7} ${o:>+8,.0f} ${sw:>+8,.0f} ${n:>+9,.0f} ${e:>+7,.0f} ${tot:>+8,.0f} {tot/TOTAL_CAP*100:>+6.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SWING ALLOCATION SWEEP (vary swing from 30% to 75%, hold others proportional)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("SWING ALLOCATION SWEEP  (others rescaled proportionally)")
print(f"{'='*75}")
print(f"  DD limit: <= 10%  |  $100k total capital")
print(f"\n  {'Swing%':>7} {'ORB%':>6} {'ON%':>6} {'ETF%':>6} {'Return':>8} {'CAGR':>7} "
      f"{'MaxDD':>8} {'Sharpe':>8} {'Annl P&L':>10} {'Status':>8}")
print(f"  {'─'*80}")

# Per-dollar daily returns for each strategy
orb_ret   = orb_pnl   / ORB_CAP
swing_ret = swing_pnl / SWING_CAP
on_ret    = on_pnl    / ON_CAP
etf_ret   = etf_pnl   / ETF_CAP

# Baseline proportions for ORB, ON, ETF (when swing is reduced from 100%)
# Original ratio without swing: ORB 33.3 / ON 22.2 / ETF 11.1 → 3:2:1 of remaining
BASE_RATIO = np.array([33.333, 22.222, 11.111])  # ORB, ON, ETF
BASE_SUM   = BASE_RATIO.sum()   # 66.666

sweep_results = []
for sw_pct in range(30, 76, 5):  # 30% to 75% in 5% steps
    remaining   = 100.0 - sw_pct
    scaled      = BASE_RATIO * (remaining / BASE_SUM)
    a_orb, a_on, a_etf = scaled * 1000, scaled * 1000, scaled * 1000  # → dollars on $100k
    a_orb_d  = scaled[0] * 1000
    a_on_d   = scaled[1] * 1000
    a_etf_d  = scaled[2] * 1000
    a_sw_d   = sw_pct * 1000  # dollars

    # Combined daily P&L
    idx = orb_ret.index.union(swing_ret.index).union(on_ret.index).union(etf_ret.index)
    cpnl = (
        orb_ret.reindex(idx, fill_value=0)   * a_orb_d +
        swing_ret.reindex(idx, fill_value=0) * a_sw_d +
        on_ret.reindex(idx, fill_value=0)    * a_on_d +
        etf_ret.reindex(idx, fill_value=0)   * a_etf_d
    )
    ceq  = TOTAL_CAP + cpnl.cumsum()
    cret = (ceq.iloc[-1] / TOTAL_CAP - 1) * 100
    cdd  = max_dd(ceq)
    csh  = sharpe(cpnl)
    cnet = cpnl.sum()
    cc   = cagr(ceq.iloc[-1], TOTAL_CAP, TRADING_YEARS)
    status = "OK" if abs(cdd) <= 10.0 else "DD>10%"
    star   = " <-- BEST" if abs(cdd) <= 10.0 and cret > qqq_ret else ""

    sweep_results.append({
        "sw_pct": sw_pct, "a_orb": a_orb_d, "a_on": a_on_d, "a_etf": a_etf_d,
        "a_sw": a_sw_d, "ret": cret, "cagr_v": cc, "dd": cdd, "sharpe_v": csh,
        "net": cnet, "equity": ceq, "pnl": cpnl, "status": status,
    })
    orb_pct_show = a_orb_d / 1000
    on_pct_show  = a_on_d  / 1000
    etf_pct_show = a_etf_d / 1000
    print(f"  {sw_pct:>6}%  {orb_pct_show:>5.1f}% {on_pct_show:>5.1f}% {etf_pct_show:>5.1f}% "
          f"{cret:>+7.1f}% {cc:>+6.1f}% {cdd:>+7.1f}% {csh:>8.2f} ${cnet/TRADING_YEARS:>+9,.0f}"
          f"  {status}{star}")

# Find best: max return among DD <= 10%
passing = [r for r in sweep_results if abs(r["dd"]) <= 10.0]
best_sweep = max(passing, key=lambda r: r["ret"]) if passing else max(sweep_results, key=lambda r: r["ret"])

print(f"\n  RECOMMENDED: Swing = {best_sweep['sw_pct']}%  "
      f"(ORB {best_sweep['a_orb']/1000:.1f}% / ON {best_sweep['a_on']/1000:.1f}% / ETF {best_sweep['a_etf']/1000:.1f}%)")
print(f"  Return: {best_sweep['ret']:+.1f}%  |  CAGR: {best_sweep['cagr_v']:+.1f}%  |  "
      f"MaxDD: {best_sweep['dd']:+.1f}%  |  Sharpe: {best_sweep['sharpe_v']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CHART
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5),
                                gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#0d1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# Per-strategy equity (aligned to combined index)
orb_eq_full   = (ORB_CAP  + orb_pnl.reindex(combined_df.index, fill_value=0).cumsum())
swing_eq_full = (SWING_CAP + swing_pnl.reindex(combined_df.index, fill_value=0).cumsum())
on_eq_full    = (ON_CAP   + on_pnl.reindex(combined_df.index, fill_value=0).cumsum())
etf_eq_full   = (ETF_CAP  + etf_pnl.reindex(combined_df.index, fill_value=0).cumsum())

# Align best sweep equity to combined_df index
best_eq_aligned = best_sweep["equity"].reindex(combined_df.index, method="ffill")

# Three lines: baseline, best sweep, QQQ
ax1.plot(combined_df.index, combined_df["portfolio"], color="#58a6ff", lw=1.6,
         ls="--", alpha=0.7, label=f"Current allocation  ({port_ret:+.1f}%)", zorder=4)
ax1.plot(combined_df.index, best_eq_aligned, color="#3fb950", lw=2.2,
         label=f"Swing {best_sweep['sw_pct']}% allocation  ({best_sweep['ret']:+.1f}%)", zorder=5)
ax1.plot(combined_df.index, combined_df["QQQ"], color="#f78166", lw=1.8,
         label=f"QQQ Buy & Hold  ({qqq_ret:+.1f}%)", alpha=0.80, zorder=3)

# Component fills
ax1.fill_between(combined_df.index, orb_eq_full.reindex(combined_df.index),
                  ORB_CAP, alpha=0.06, color="#3fb950", label="ORB $75k")
ax1.fill_between(combined_df.index, swing_eq_full.reindex(combined_df.index),
                  SWING_CAP, alpha=0.06, color="#d2a8ff", label="Swing $75k")
ax1.fill_between(combined_df.index, etf_eq_full.reindex(combined_df.index),
                  ETF_CAP, alpha=0.08, color="#ffa657", label="ETF $25k")
ax1.axhline(TOTAL_CAP, color="#555", lw=0.8, ls="--")

ax1.set_title(
    f"4-Strategy Portfolio vs QQQ  |  Jan 2021–Mar 2026  |  $100k  "
    f"|  Best: Swing {best_sweep['sw_pct']}%",
    color="white", fontsize=13, pad=10, fontweight="bold"
)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
           fontsize=9.5, loc="upper left")
ax1.grid(color="#21262d", linewidth=0.6)

# Drawdown
port_dd_s = (combined_df["portfolio"] / combined_df["portfolio"].cummax()) - 1
best_dd_s = (best_eq_aligned / best_eq_aligned.cummax()) - 1
qqq_dd_s  = (combined_df["QQQ"] / combined_df["QQQ"].cummax()) - 1
ax2.fill_between(combined_df.index, qqq_dd_s * 100, 0,
                 color="#f78166", alpha=0.20, label="QQQ DD")
ax2.fill_between(combined_df.index, port_dd_s * 100, 0,
                 color="#58a6ff", alpha=0.35, label="Current DD")
ax2.fill_between(combined_df.index, best_dd_s * 100, 0,
                 color="#3fb950", alpha=0.35, label=f"Swing {best_sweep['sw_pct']}% DD")
ax2.axhline(0, color="#555", lw=0.8)
ax2.set_ylabel("Drawdown %", color="white")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
ax2.grid(color="#21262d", linewidth=0.6)

# Stats box
best_ann = best_sweep["net"] / TRADING_YEARS
stats_txt = (
    f"{'':20s}{'Current':>11s}{'Swing '+str(best_sweep['sw_pct'])+'%':>12s}{'QQQ':>9s}\n"
    f"{'Total return':20s}{port_ret:>+10.1f}%{best_sweep['ret']:>+11.1f}%{qqq_ret:>+8.1f}%\n"
    f"{'CAGR':20s}{port_cagr:>+10.1f}%{best_sweep['cagr_v']:>+11.1f}%{qqq_cagr:>+8.1f}%\n"
    f"{'Max drawdown':20s}{port_dd:>+10.1f}%{best_sweep['dd']:>+11.1f}%{qqq_dd:>+8.1f}%\n"
    f"{'Sharpe (ann.)':20s}{port_sh:>11.2f}{best_sweep['sharpe_v']:>12.2f}{qqq_sh:>9.2f}\n"
    f"{'Avg annual P&L':20s}${port_ann:>+9,.0f}  ${best_ann:>+9,.0f}"
)
ax1.text(0.01, 0.03, stats_txt, transform=ax1.transAxes,
         fontsize=9, color="white", verticalalignment="bottom",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                   edgecolor="#30363d", alpha=0.92),
         fontfamily="monospace")

plt.tight_layout(h_pad=0.5)
out = "results/portfolio_4strat_optimized_2026-03-10.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nChart saved → {out}")
plt.show()
