"""
momentum_universe.py — Universe management for overnight momentum strategy.

Provides:
  - SP500_TICKERS: curated list of high-volume S&P 500 + liquid names
  - get_filtered_universe(): price > $10, avg 20-day vol > 2M
  - rank_by_momentum(): 20-day return, descending
  - has_earnings_soon(): True if earnings within N calendar days
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

ET = pytz.timezone("America/New_York")

# ── Universe ─────────────────────────────────────────────────────────────────
# Curated list of S&P 500 components + high-liquidity names.
# Filter (price > $10, avg vol > 2M) will trim to tradeable subset each day.
# Easy to update: add/remove tickers from this list.
SP500_TICKERS: List[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    # Semiconductors
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "ADI", "TXN", "AVGO", "MRVL",
    # Software / Cloud
    "ORCL", "CRM", "NOW", "SNOW", "PLTR", "ANET", "PANW", "CRWD", "FTNT",
    "ZS", "DDOG", "NET", "OKTA", "MDB", "TTD", "CFLT",
    # Internet / Consumer tech
    "NFLX", "SPOT", "ABNB", "DASH", "LYFT", "UBER",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP",
    "V", "MA", "PYPL", "SQ", "COIN", "SOFI",
    # Healthcare / Biotech
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "MRNA", "GILD",
    "REGN", "BIIB", "AMGN", "BMY", "CVS", "CI", "HUM", "ISRG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "DVN", "MPC", "VLO", "PSX",
    # Consumer staples / discretionary
    "WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "SBUX",
    "CMG", "DPZ", "YUM", "DKNG", "DIS", "CMCSA",
    # Industrial / Aerospace
    "BA", "LMT", "RTX", "GE", "CAT", "DE", "HON", "MMM", "FDX", "UPS",
    # Airlines
    "DAL", "UAL", "AAL", "LUV",
    # Auto / EV
    "F", "GM", "RIVN", "NIO", "LCID",
    # Communication
    "SNAP", "PINS", "TWTR",
    # Crypto adjacent
    "MSTR", "COIN",
    # High-vol ETFs (useful for momentum signal)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "ARKK", "GLD", "USO",
    "SOXL", "TQQQ",
    # Meme / retail
    "AMC", "GME",
]

# Deduplicate while preserving order
_seen: set = set()
SP500_TICKERS = [t for t in SP500_TICKERS if t not in _seen and not _seen.add(t)]

MIN_PRICE    = 10.0
# IEX feed captures ~2-3% of total market volume.
# 200_000 IEX shares ≈ 6-10M real shares — equivalent to the 2M live filter.
# If using SIP/full data feed, raise this to 2_000_000.
MIN_AVG_VOL  = 200_000
MOMENTUM_DAYS = 20


# ── Universe fetch & filter ───────────────────────────────────────────────────

def get_filtered_universe(
    api_key: str,
    api_secret: str,
    logger=None,
) -> Tuple[List[str], Dict[str, list]]:
    """
    Fetch 21 daily bars for the full universe, apply price + volume filters.

    Returns:
        (filtered_symbols, bars_data)
        bars_data maps symbol -> list of Bar objects (most recent last)
    """
    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    today      = date.today()
    end_str    = today.isoformat()
    start_str  = (today - timedelta(days=35)).isoformat()   # ~25 trading days

    if logger:
        logger.info(f"UNIVERSE: Fetching {MOMENTUM_DAYS}-day bars for {len(SP500_TICKERS)} symbols…")

    try:
        req = StockBarsRequest(
            symbol_or_symbols=SP500_TICKERS,
            timeframe=TimeFrame.Day,
            start=start_str,
            end=end_str,
            feed="iex",
        )
        resp = client.get_stock_bars(req)
    except Exception as exc:
        if logger:
            logger.error(f"UNIVERSE: Failed to fetch bars — {exc}")
        return [], {}

    # Normalise response to dict
    bars_data: Dict[str, list] = {}
    raw = getattr(resp, "data", None) or {}
    if not raw and hasattr(resp, "__iter__"):
        try:
            raw = dict(resp)
        except Exception:
            raw = {}
    for sym, sym_bars in raw.items():
        bars_data[sym] = list(sym_bars)

    # Filter: need enough bars + price > $10 + avg vol > 2M
    filtered: List[str] = []
    for sym in SP500_TICKERS:
        sym_bars = bars_data.get(sym, [])
        if len(sym_bars) < MOMENTUM_DAYS + 1:
            continue
        latest_close = sym_bars[-1].close
        if latest_close < MIN_PRICE:
            continue
        recent_vols = [b.volume for b in sym_bars[-MOMENTUM_DAYS:]]
        avg_vol = sum(recent_vols) / len(recent_vols)
        if avg_vol < MIN_AVG_VOL:
            continue
        filtered.append(sym)

    if logger:
        logger.info(
            f"UNIVERSE: {len(SP500_TICKERS)} loaded, "
            f"{len(filtered)} pass filters (price>${MIN_PRICE:.0f}, "
            f"avgvol>{MIN_AVG_VOL/1e6:.0f}M)"
        )

    return filtered, bars_data


# ── Momentum ranking ──────────────────────────────────────────────────────────

def rank_by_momentum(
    bars_data: Dict[str, list],
    universe: List[str],
) -> List[Tuple[str, float]]:
    """
    Return list of (symbol, 20d_return_pct) sorted best-to-worst.
    20d return = (latest_close / close_20_days_ago - 1) * 100
    """
    ranked: List[Tuple[str, float]] = []
    for sym in universe:
        sym_bars = bars_data.get(sym, [])
        if len(sym_bars) < MOMENTUM_DAYS + 1:
            continue
        price_now  = sym_bars[-1].close
        price_then = sym_bars[-(MOMENTUM_DAYS + 1)].close
        if price_then <= 0:
            continue
        ret_pct = (price_now / price_then - 1) * 100
        ranked.append((sym, ret_pct))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ── Earnings check ───────────────────────────────────────────────────────────

def has_earnings_soon(symbol: str, days: int = 2) -> bool:
    """
    Return True if `symbol` has an earnings announcement within `days`
    calendar days (before or after today).

    Uses yfinance. If yfinance is not installed or the call fails, returns
    False (conservative: do not block entry on unknown data).
    """
    try:
        import yfinance as yf
    except ImportError:
        return False   # yfinance not installed — skip filter

    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar

        if cal is None:
            return False

        earnings_date = None

        # yfinance >= 0.2.x returns a dict
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")

        # Older yfinance returns a DataFrame
        elif hasattr(cal, "loc"):
            try:
                earnings_date = cal.loc["Earnings Date"].iloc[0]
            except Exception:
                return False

        if earnings_date is None:
            return False

        # Normalise to date
        if hasattr(earnings_date, "date"):
            earnings_date = earnings_date.date()
        elif isinstance(earnings_date, str):
            try:
                earnings_date = date.fromisoformat(earnings_date[:10])
            except Exception:
                return False

        delta = abs((earnings_date - date.today()).days)
        return delta <= days

    except Exception:
        return False
