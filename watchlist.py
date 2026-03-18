"""
watchlist.py — Pre-market screener: top 15 stocks by pre-market volume.

Runs at 9:15 ET. Filters:
  - Price between $5 and $500
  - Average daily volume >= 1 000 000
Returns a list of up to 15 ticker symbols.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import List

import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

from logger import get_logger

ET = pytz.timezone("America/New_York")
logger = get_logger()

# Broad universe to screen — extend as desired.
# Using a practical set of high-volume, liquid US equities.
UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD",
    "NFLX", "BABA", "CRM", "PYPL", "INTC", "QCOM", "MU",
    "SPY",  "QQQ",  "IWM",  "ARKK",
    "BAC",  "JPM",  "GS",   "MS",
    "XOM",  "CVX",  "SLB",  "EOG",
    "PFE",  "MRNA", "JNJ",
    "PLTR", "SOFI", "HOOD", "COIN", "MSTR",
    "F",    "GM",   "NIO",  "RIVN", "LCID",
    "AMC",  "GME",
    "UBER", "LYFT", "SNAP", "PINS", "TWTR",
    "SQ",   "SHOP", "SE",   "MELI",
    "ZM",   "DOCU", "CRWD", "PANW", "OKTA",
    "DIS",  "CMCSA","NWSA",
    "BA",   "LMT",  "RTX",
    "GLD",  "SLV",  "USO",
]

MIN_PRICE = 5.0
MAX_PRICE = 500.0
MIN_AVG_DAILY_VOL = 1_000_000
TOP_N = 15

# Tickers permanently excluded from the watchlist regardless of pre-market volume.
# Based on backtest analysis: these symbols show structurally poor ORB behaviour
# (excessive whipsaws, low float, or strategy-incompatible volatility profiles).
BLACKLIST: List[str] = ["SNAP", "RIVN", "HOOD", "UBER"]


def build_watchlist(api_key: str, api_secret: str) -> List[str]:
    """
    Fetch pre-market volume for UNIVERSE stocks, filter by price and liquidity,
    and return top TOP_N symbols by today's pre-market volume.
    """
    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    today      = date.today()
    today_str  = today.isoformat()
    # Use 20 trading days back for average volume calc
    start_date = (today - timedelta(days=30)).isoformat()

    # ── 1. Fetch latest bar to get current price ─────────────────────────────
    logger.info(f"WATCHLIST: Fetching latest prices for {len(UNIVERSE)} symbols…")
    try:
        latest_req = StockLatestBarRequest(symbol_or_symbols=UNIVERSE)
        latest_bars = client.get_stock_latest_bar(latest_req)
    except Exception as exc:
        logger.error(f"WATCHLIST: Failed to fetch latest bars — {exc}")
        return []

    # Filter by price
    price_filtered = [
        sym for sym in UNIVERSE
        if sym in latest_bars and MIN_PRICE <= latest_bars[sym].close <= MAX_PRICE
    ]
    logger.info(f"WATCHLIST: {len(price_filtered)} symbols pass price filter (${MIN_PRICE}–${MAX_PRICE})")

    if not price_filtered:
        return []

    # ── 2. Fetch daily bars for avg volume calculation ───────────────────────
    try:
        bars_req = StockBarsRequest(
            symbol_or_symbols=price_filtered,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=today_str,
            feed="iex",
        )
        daily_bars = client.get_stock_bars(bars_req)
    except Exception as exc:
        logger.error(f"WATCHLIST: Failed to fetch daily bars — {exc}")
        return []

    avg_vol: dict[str, float] = {}
    # StockBarsResponse stores data in .data dict: {symbol: [Bar, ...]}
    bars_data: dict = getattr(daily_bars, "data", {}) or {}
    if not bars_data and hasattr(daily_bars, "__iter__"):
        # Some SDK versions return a plain dict directly
        try:
            bars_data = dict(daily_bars)
        except Exception:
            bars_data = {}
    for sym, sym_bars in bars_data.items():
        try:
            volumes = [b.volume for b in sym_bars]
            if volumes:
                avg_vol[sym] = sum(volumes) / len(volumes)
        except Exception:
            pass

    liq_filtered = [
        sym for sym in price_filtered
        if avg_vol.get(sym, 0) >= MIN_AVG_DAILY_VOL
    ]
    logger.info(f"WATCHLIST: {len(liq_filtered)} symbols pass liquidity filter (avg vol >= {MIN_AVG_DAILY_VOL:,})")

    # ── 3. Fetch today's pre-market 1-min bars and sum volume ────────────────
    if not liq_filtered:
        logger.warning("WATCHLIST: No symbols passed liquidity filter — returning top price-filtered symbols by latest volume.")
        # Fallback: use symbols that had any avg vol data, or just price_filtered[:TOP_N]
        fallback = sorted(price_filtered, key=lambda s: avg_vol.get(s, 0), reverse=True)[:TOP_N]
        _log_watchlist(fallback)
        return fallback

    now_et = datetime.now(ET)
    premarket_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
    premarket_end   = now_et.replace(hour=9, minute=15, second=0, microsecond=0)

    try:
        pm_req = StockBarsRequest(
            symbol_or_symbols=liq_filtered,
            timeframe=TimeFrame.Minute,
            start=premarket_start.isoformat(),
            end=premarket_end.isoformat(),
            feed="iex",
        )
        pm_bars = client.get_stock_bars(pm_req)
    except Exception as exc:
        logger.warning(f"WATCHLIST: Could not fetch pre-market bars — {exc}. Falling back to avg volume ranking.")
        # Fallback: rank by avg volume
        ranked = sorted(liq_filtered, key=lambda s: avg_vol.get(s, 0), reverse=True)
        watchlist = _apply_blacklist(ranked, TOP_N)
        _log_watchlist(watchlist)
        return watchlist

    pm_vol: dict[str, int] = {}
    for sym in liq_filtered:
        try:
            sym_bars = pm_bars[sym]
            pm_vol[sym] = sum(b.volume for b in sym_bars)
        except Exception:
            pm_vol[sym] = 0

    # Sort by pre-market volume descending (full ranked list kept for replacement pool)
    ranked = sorted(liq_filtered, key=lambda s: pm_vol.get(s, 0), reverse=True)
    watchlist = _apply_blacklist(ranked, TOP_N, pm_vol)
    _log_watchlist(watchlist, pm_vol)
    return watchlist


def _apply_blacklist(ranked: List[str], n: int, pm_vol: dict | None = None) -> List[str]:
    """
    Return the top-n symbols from `ranked` excluding BLACKLIST entries.
    Logs each exclusion and its replacement at INFO level.
    """
    blacklist_set  = set(BLACKLIST)
    original_top_n = ranked[:n]

    # Symbols that would have made the list but are blacklisted
    excluded = [s for s in original_top_n if s in blacklist_set]

    # Final list: top-n non-blacklisted symbols from the full ranked pool
    final = [s for s in ranked if s not in blacklist_set][:n]

    # Extra symbols that entered the list to replace excluded ones
    original_set = set(original_top_n)
    replacements = [s for s in final if s not in original_set]

    for i, sym in enumerate(excluded):
        replacement = replacements[i] if i < len(replacements) else None
        if replacement:
            vol_str = f" | vol: {pm_vol[replacement]:,}" if pm_vol and replacement in pm_vol else ""
            logger.info(f"BLACKLIST: {sym} excluded -- replaced with {replacement}{vol_str}")
        else:
            logger.info(f"BLACKLIST: {sym} excluded (no replacement available, list shortened)")

    return final


def _log_watchlist(symbols: List[str], pm_vol: dict | None = None) -> None:
    sym_str = ", ".join(symbols)
    logger.info(f"WATCHLIST: {sym_str} ({len(symbols)} stocks)")
    if pm_vol:
        for sym in symbols:
            vol = pm_vol.get(sym, 0)
            logger.info(f"  {sym:6s} pre-market vol: {vol:>10,}")
