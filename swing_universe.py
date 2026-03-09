"""
swing_universe.py -- Universe management for the swing momentum strategy.

Extends momentum_universe.py with tighter filters:
  - Price > $20     (vs $10 overnight)
  - Avg volume > 3M (vs 2M overnight, IEX-adjusted to 300k)
  - SMA50 filter    (price must be above 50-day SMA at entry)
  - Volume today >= 1.5x 20-day average
  - Earnings filter: skip if within 5 calendar days (vs 2 overnight)
  - Momentum: top-20% of universe by 20-day return

Provides:
  get_swing_universe()  -- fetch bars, apply filters, rank momentum
  is_above_sma50()      -- quick check for an individual symbol
  has_earnings_soon()   -- re-exported with 5-day default
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = pytz.timezone("America/New_York")

# -- Universe ------------------------------------------------------------------
# Same curated S&P 500 base list as momentum_universe.py.
# Tighter price/volume filters are applied dynamically each scan.

SP500_TICKERS: List[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    # Semiconductors
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "ADI", "TXN", "AVGO", "MRVL",
    # Software / Cloud
    "ORCL", "CRM", "NOW", "SNOW", "PLTR", "ANET", "PANW", "CRWD", "FTNT",
    "ZS", "DDOG", "NET", "OKTA", "MDB", "TTD",
    # Internet / Consumer tech
    "NFLX", "SPOT", "ABNB", "DASH", "LYFT", "UBER",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP",
    "V", "MA", "PYPL", "COIN", "SOFI",
    # Healthcare / Biotech
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "MRNA", "GILD",
    "REGN", "BIIB", "AMGN", "BMY", "CVS", "CI", "HUM", "ISRG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "DVN", "MPC", "VLO", "PSX",
    # Consumer staples / discretionary
    "WMT", "TGT", "COST", "HD", "LOW", "NKE", "MCD", "SBUX",
    "CMG", "DPZ", "YUM", "DIS", "CMCSA",
    # Industrial / Aerospace
    "BA", "LMT", "RTX", "GE", "CAT", "DE", "HON", "MMM", "FDX", "UPS",
    # Airlines
    "DAL", "UAL", "AAL", "LUV",
    # Auto / EV
    "F", "GM", "RIVN", "NIO",
    # Communication
    "SNAP", "PINS",
]

# Deduplicate preserving order
_seen: set = set()
SP500_TICKERS = [t for t in SP500_TICKERS if t not in _seen and not _seen.add(t)]  # type: ignore

# -- Filter constants -----------------------------------------------------------
MIN_PRICE      = 20.0
# IEX captures ~2-3% of real volume. 300k IEX ~ 3M real shares.
MIN_AVG_VOL    = 300_000
VOL_MULT_MIN   = 1.5     # today's volume must be >= 1.5x 20d avg
MOMENTUM_DAYS  = 20
SMA_PERIOD     = 50
BARS_NEEDED    = SMA_PERIOD + MOMENTUM_DAYS + 5   # ~75 trading days


# -- Data structure ------------------------------------------------------------

class SwingCandidate(NamedTuple):
    symbol:       str
    momentum_pct: float   # 20-day return %
    price:        float   # latest close
    sma50:        float   # 50-day SMA
    vol_mult:     float   # today vol / 20d avg vol
    avg_vol:      float   # 20d avg IEX volume


# -- Universe fetch -------------------------------------------------------------

def get_swing_universe(
    api_key:    str,
    api_secret: str,
    logger=None,
) -> Tuple[List[SwingCandidate], Dict[str, list]]:
    """
    Fetch ~75 daily bars for the full universe. Apply filters and rank by
    20-day momentum. Return top-20% (by momentum) that also pass all
    other conditions (price, volume, SMA50, vol-today).

    Returns:
        (candidates, bars_data)
        candidates : SwingCandidate list, ranked by momentum desc, top-20% only
        bars_data  : {symbol: [Bar, ...]} for all symbols (use for stop checks etc.)
    """
    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    today     = date.today()
    end_str   = today.isoformat()
    start_str = (today - timedelta(days=BARS_NEEDED + 30)).isoformat()  # buffer for weekends

    if logger:
        logger.info(
            f"UNIVERSE: Fetching {BARS_NEEDED}-day bars for "
            f"{len(SP500_TICKERS)} symbols..."
        )

    try:
        req  = StockBarsRequest(
            symbol_or_symbols=SP500_TICKERS,
            timeframe=TimeFrame.Day,
            start=start_str,
            end=end_str,
            feed="iex",
        )
        resp = client.get_stock_bars(req)
    except Exception as exc:
        if logger:
            logger.error(f"UNIVERSE: Failed to fetch bars -- {exc}")
        return [], {}

    # Normalise response
    bars_data: Dict[str, list] = {}
    raw = getattr(resp, "data", None) or {}
    if not raw and hasattr(resp, "__iter__"):
        try:
            raw = dict(resp)
        except Exception:
            raw = {}
    for sym, sym_bars in raw.items():
        bars_data[sym] = list(sym_bars)

    # -- Step 1: Basic price + volume + bar-count filter -----------------------
    pool: List[SwingCandidate] = []

    for sym in SP500_TICKERS:
        sym_bars = bars_data.get(sym, [])
        if len(sym_bars) < SMA_PERIOD + 1:
            continue

        latest_close = sym_bars[-1].close
        if latest_close < MIN_PRICE:
            continue

        recent_vols = [b.volume for b in sym_bars[-MOMENTUM_DAYS:]]
        avg_vol     = sum(recent_vols) / len(recent_vols)
        if avg_vol < MIN_AVG_VOL:
            continue

        # SMA50
        sma50_closes = [b.close for b in sym_bars[-SMA_PERIOD:]]
        sma50        = sum(sma50_closes) / SMA_PERIOD
        if latest_close <= sma50:
            continue  # price below SMA50 -- not in uptrend

        # 20d momentum
        price_then   = sym_bars[-(MOMENTUM_DAYS + 1)].close
        if price_then <= 0:
            continue
        momentum_pct = (latest_close / price_then - 1) * 100

        # Today's volume multiplier (last bar = today's session)
        today_vol = sym_bars[-1].volume
        vol_mult  = today_vol / avg_vol if avg_vol > 0 else 0.0

        pool.append(SwingCandidate(
            symbol=sym,
            momentum_pct=momentum_pct,
            price=latest_close,
            sma50=sma50,
            vol_mult=vol_mult,
            avg_vol=avg_vol,
        ))

    # -- Step 2: Top-20% momentum threshold -----------------------------------
    if not pool:
        if logger:
            logger.warning("UNIVERSE: No symbols passed base filters.")
        return [], bars_data

    pool.sort(key=lambda c: c.momentum_pct, reverse=True)
    cutoff_idx = max(1, int(len(pool) * 0.20))  # top 20%
    threshold  = pool[cutoff_idx - 1].momentum_pct

    # -- Step 3: Apply volume-today filter on top candidates -------------------
    candidates: List[SwingCandidate] = []
    vol_skipped: List[str] = []

    for c in pool[:cutoff_idx]:
        if c.vol_mult >= VOL_MULT_MIN:
            candidates.append(c)
        else:
            vol_skipped.append(c.symbol)

    if logger:
        logger.info(
            f"UNIVERSE: {len(SP500_TICKERS)} tickers | "
            f"{len(pool)} pass base filters | "
            f"{cutoff_idx} in top 20% momentum (>={threshold:+.1f}%) | "
            f"{len(candidates)} pass vol filter ({VOL_MULT_MIN}x today)"
        )
        if candidates:
            top3 = candidates[:3]
            top_str = ", ".join(f"{c.symbol} {c.momentum_pct:+.1f}%" for c in top3)
            logger.info(f"SCAN: {len(candidates)} stocks pass filters | Top momentum: {top_str}")
        if vol_skipped:
            logger.info(f"UNIVERSE: Vol filter excluded: {', '.join(vol_skipped)}")

    return candidates, bars_data


# -- Earnings filter (5-day default for swing) ---------------------------------

def has_earnings_soon(symbol: str, days: int = 5) -> bool:
    """
    Return True if `symbol` has earnings within `days` calendar days.
    Uses yfinance. Returns False on any error (do not block on unknown data).
    """
    try:
        import yfinance as yf
    except ImportError:
        return False

    try:
        ticker = yf.Ticker(symbol)
        cal    = ticker.calendar

        if cal is None:
            return False

        earnings_date = None

        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
        elif hasattr(cal, "loc"):
            try:
                earnings_date = cal.loc["Earnings Date"].iloc[0]
            except Exception:
                return False

        if earnings_date is None:
            return False

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


# -- Utility -------------------------------------------------------------------

def spy_return_since(start_date: str, start_price: float) -> Optional[float]:
    """
    Return SPY % return from start_date/start_price to today.
    Uses yfinance. Returns None on failure.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker("SPY").history(
            start=start_date,
            end=(date.today() + timedelta(days=2)).isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])
        return (current_price - start_price) / start_price * 100
    except Exception:
        return None
