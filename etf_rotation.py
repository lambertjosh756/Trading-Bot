"""
etf_rotation.py — Strategy E: Aggressive Triple Momentum ETF Rotation
                  Live paper trading bot using Alpaca.

Strategy:
  Capital   : $25,000 (tracked separately from ORB / Overnight)
  Universe  : TQQQ, UPRO, TNA  (3x leveraged equity ETFs)
  Safe haven: SHY               (short-term bond ETF)
  Rebalance : Every Monday at 09:35 ET
  Selection : Top 2 ETFs by 20-trading-day momentum, equal-weight ($12,500 each)
              ETFs with negative momentum are replaced by SHY

Run schedule (Windows Task Scheduler, or cron on WSL):
  Every weekday at 09:35 ET → python etf_rotation.py
  (Bot self-selects rebalance vs. status-only based on day-of-week)

Usage:
  python etf_rotation.py            # auto mode (rebalance on Monday, status otherwise)
  python etf_rotation.py --status   # force status display only
  python etf_rotation.py --rebalance # force rebalance now (testing)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from logger import get_logger

# ── Constants ─────────────────────────────────────────────────────────────────
ET              = pytz.timezone("America/New_York")
ENV_FILE        = ".env"
STATE_FILE      = ".etf_rotation_state.json"

CAPITAL         = 25_000.0      # total ETF rotation allocation
PER_POSITION    = 12_500.0      # per holding ($25k / 2)
TOP_N           = 2             # number of ETF positions
MOMENTUM_DAYS   = 20            # lookback for momentum ranking

UNIVERSE        = ["TQQQ", "UPRO", "TNA"]   # ranked universe
SAFE_HAVEN      = "SHY"                      # replace negative-momentum ETFs

logger = get_logger("etf_rotation", log_prefix="etf_rotation")


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "last_rebalance":   None,
            "holdings":         [],
            "entries":          {},
            "start_equity":     CAPITAL,
            "week_start_value": CAPITAL,
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"STATE: Could not load — {exc}")
        return {
            "last_rebalance":   None,
            "holdings":         [],
            "entries":          {},
            "start_equity":     CAPITAL,
            "week_start_value": CAPITAL,
        }


def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        logger.warning(f"STATE: Could not save — {exc}")


# ── Momentum calculation ───────────────────────────────────────────────────────

def compute_momentum(
    data_client: StockHistoricalDataClient,
    symbols: List[str],
    lookback: int = MOMENTUM_DAYS,
) -> Dict[str, float]:
    """
    Fetch last (lookback + 5) daily bars for each symbol via Alpaca IEX feed.
    Returns {symbol: momentum_pct} where momentum = (close_now - close_N_ago) / close_N_ago.
    """
    import pandas as pd

    end_dt   = datetime.now(ET)
    start_dt = end_dt - timedelta(days=lookback + 14)  # extra buffer for weekends/holidays

    try:
        req  = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            feed="iex",
        )
        resp = data_client.get_stock_bars(req)
        data = getattr(resp, "data", {})
    except Exception as exc:
        logger.error(f"MOMENTUM: Data fetch failed — {exc}")
        return {s: 0.0 for s in symbols}

    momentum: Dict[str, float] = {}
    for sym in symbols:
        bars = data.get(sym, [])
        if len(bars) < lookback:
            logger.warning(f"MOMENTUM: {sym} only has {len(bars)} bars (need {lookback}) — using 0%")
            momentum[sym] = 0.0
            continue
        # Use last `lookback` bars
        closes = [b.close for b in bars[-lookback:]]
        start_price = closes[0]
        end_price   = closes[-1]
        if start_price == 0:
            momentum[sym] = 0.0
        else:
            momentum[sym] = (end_price - start_price) / start_price * 100
        logger.info(
            f"MOMENTUM: {sym:6s} {start_price:>8.2f} → {end_price:>8.2f} | "
            f"{momentum[sym]:>+.2f}% ({lookback}d)"
        )
    return momentum


def get_target_allocation(momentum: Dict[str, float]) -> List[str]:
    """
    Rank UNIVERSE ETFs by momentum. Take top TOP_N with positive momentum.
    Replace any negative-momentum slots with SAFE_HAVEN.
    Returns list of length TOP_N (e.g. ['TQQQ', 'UPRO'] or ['TQQQ', 'SHY']).
    """
    ranked = sorted(UNIVERSE, key=lambda s: momentum.get(s, 0.0), reverse=True)
    target = []
    for sym in ranked[:TOP_N]:
        if momentum.get(sym, 0.0) > 0:
            target.append(sym)
        else:
            target.append(SAFE_HAVEN)
    # Pad to TOP_N with SHY if universe < TOP_N
    while len(target) < TOP_N:
        target.append(SAFE_HAVEN)
    return target


# ── Main bot ──────────────────────────────────────────────────────────────────

class ETFRotationBot:

    def __init__(self, api_key: str, api_secret: str):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )

    # ── Account ───────────────────────────────────────────────────────────────

    def validate_account(self) -> Tuple[float, float]:
        account      = self.trading_client.get_account()
        equity       = float(account.equity)
        buying_power = float(account.buying_power)
        logger.info(
            f"ACCOUNT: Paper confirmed | Equity: ${equity:,.2f} | "
            f"Buying power: ${buying_power:,.2f}"
        )
        return equity, buying_power

    def get_open_positions(self) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        try:
            positions = self.trading_client.get_all_positions()
            for p in positions:
                result[p.symbol] = {
                    "qty":         abs(float(p.qty)),
                    "entry_price": float(p.avg_entry_price),
                    "current":     float(p.current_price),
                    "pnl":         float(p.unrealized_pl),
                    "market_value": float(p.market_value),
                }
        except Exception as exc:
            logger.error(f"POSITIONS: Could not fetch — {exc}")
        return result

    # ── Orders ────────────────────────────────────────────────────────────────

    def close_position(self, symbol: str, pos: dict) -> float:
        """Close a position, return realized P&L estimate."""
        try:
            self.trading_client.close_position(symbol)
            pnl     = pos["pnl"]
            pnl_pct = (pos["current"] - pos["entry_price"]) / pos["entry_price"] * 100
            sign    = "+" if pnl >= 0 else ""
            logger.info(
                f"SELL: {symbol} @ ${pos['current']:.2f} | "
                f"Entry: ${pos['entry_price']:.2f} | "
                f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)"
            )
            return pnl
        except Exception as exc:
            logger.error(f"SELL: Failed to close {symbol} — {exc}")
            return 0.0

    def buy_notional(self, symbol: str, notional: float) -> Optional[dict]:
        """Buy `notional` dollars worth of symbol via market order."""
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(req)
            logger.info(
                f"BUY: {symbol} ${notional:,.2f} notional | "
                f"Order ID: {order.id}"
            )
            return {"symbol": symbol, "notional": notional, "order_id": str(order.id)}
        except Exception as exc:
            logger.error(f"BUY: Failed to buy {symbol} — {exc}")
            return None

    # ── Core logic ────────────────────────────────────────────────────────────

    def rebalance(self, state: dict, forced: bool = False) -> dict:
        """
        Full rebalance: compute momentum, determine target, close/open positions.
        Updates and returns state.
        """
        today_str = date.today().isoformat()
        logger.info("=" * 60)
        logger.info(f"REBALANCE: {today_str} {'(forced)' if forced else ''}")
        logger.info("=" * 60)

        # 1. Compute momentum
        logger.info(f"MOMENTUM: Computing {MOMENTUM_DAYS}-day momentum for {', '.join(UNIVERSE)}...")
        momentum = compute_momentum(self.data_client, UNIVERSE)

        # 2. Determine target allocation
        target = get_target_allocation(momentum)
        logger.info(f"TARGET: {' + '.join(target)} @ ${PER_POSITION:,.0f} each")

        # 3. Get current open positions (Alpaca source of truth)
        current_positions = self.get_open_positions()
        etf_positions = {
            k: v for k, v in current_positions.items()
            if k in UNIVERSE + [SAFE_HAVEN]
        }

        # 4. Close positions not in target
        total_realized_pnl = 0.0
        symbols_to_close = [s for s in etf_positions if s not in target]
        for sym in symbols_to_close:
            pnl = self.close_position(sym, etf_positions[sym])
            total_realized_pnl += pnl

        # 5. Open new positions (skip if already holding at roughly right size)
        new_entries: Dict[str, dict] = {}
        for sym in target:
            if sym in etf_positions:
                existing = etf_positions[sym]
                logger.info(
                    f"HOLD: {sym} already held @ ${existing['entry_price']:.2f} | "
                    f"Market value: ${existing['market_value']:,.2f}"
                )
                new_entries[sym] = state.get("entries", {}).get(sym, {
                    "notional": PER_POSITION,
                    "entry_date": today_str,
                })
            else:
                result = self.buy_notional(sym, PER_POSITION)
                if result:
                    new_entries[sym] = {
                        "notional":   PER_POSITION,
                        "entry_date": today_str,
                    }

        # 6. Capture week-start value (for EOW summary)
        week_start_value = sum(
            v["market_value"] for v in self.get_open_positions().values()
            if v is not None
        )
        if not week_start_value:
            week_start_value = CAPITAL  # fallback

        # 7. Update state
        state["last_rebalance"]   = today_str
        state["holdings"]         = list(new_entries.keys())
        state["entries"]          = new_entries
        state["week_start_value"] = week_start_value
        if state.get("start_equity") is None:
            state["start_equity"] = CAPITAL

        save_state(state)
        logger.info(f"REBALANCE: Complete. Holdings: {', '.join(state['holdings'])}")
        return state

    def show_status(self, state: dict) -> None:
        """Display current holdings, P&L, and next Monday rebalance date."""
        logger.info("=" * 60)
        logger.info("ETF ROTATION STATUS")
        logger.info("=" * 60)

        # Next rebalance
        today      = date.today()
        days_until = (7 - today.weekday()) % 7  # days until Monday (0=Mon)
        if days_until == 0:
            next_monday = today
            next_label  = "TODAY"
        else:
            next_monday = today + timedelta(days=days_until)
            next_label  = next_monday.isoformat()
        logger.info(f"SCHEDULE: Next rebalance → Monday {next_label}")

        last_rb = state.get("last_rebalance", "never")
        logger.info(f"SCHEDULE: Last rebalance → {last_rb}")

        # Current Alpaca positions for ETF universe
        holdings = state.get("holdings", [])
        if not holdings:
            logger.info("HOLDINGS: None (not yet initialized)")
            return

        current_pos = self.get_open_positions()
        entries     = state.get("entries", {})
        total_value = 0.0
        week_start  = state.get("week_start_value", CAPITAL)
        start_eq    = state.get("start_equity", CAPITAL)

        logger.info(f"HOLDINGS: {', '.join(holdings)}")
        logger.info("-" * 60)

        for sym in holdings:
            pos     = current_pos.get(sym)
            entry   = entries.get(sym, {})
            notional = entry.get("notional", PER_POSITION)
            entry_dt = entry.get("entry_date", "?")

            if pos:
                market_val = pos["market_value"]
                pnl        = pos["pnl"]
                pnl_pct    = (pos["current"] - pos["entry_price"]) / pos["entry_price"] * 100
                sign       = "+" if pnl >= 0 else ""
                total_value += market_val
                logger.info(
                    f"  {sym:6s} | Held since {entry_dt} | "
                    f"Current: ${pos['current']:.2f} | "
                    f"Value: ${market_val:,.2f} | "
                    f"P&L: {sign}${pnl:,.2f} ({sign}{pnl_pct:.1f}%)"
                )
            else:
                logger.warning(f"  {sym:6s} | In state but no Alpaca position found")

        logger.info("-" * 60)
        week_pnl     = total_value - week_start
        total_pnl    = total_value - start_eq
        total_ret    = total_pnl / start_eq * 100 if start_eq else 0
        week_sign    = "+" if week_pnl >= 0 else ""
        total_sign   = "+" if total_pnl >= 0 else ""
        logger.info(f"PORTFOLIO: Total value     ${total_value:>10,.2f}")
        logger.info(f"PORTFOLIO: Week P&L        {week_sign}${abs(week_pnl):>9,.2f}")
        logger.info(f"PORTFOLIO: Total return    {total_sign}${abs(total_pnl):>9,.2f} ({total_sign}{total_ret:.2f}%)")

    def eow_summary(self, state: dict) -> None:
        """Friday end-of-week summary vs SHY benchmark."""
        logger.info("=" * 60)
        logger.info("ETF ROTATION — END OF WEEK SUMMARY")
        logger.info("=" * 60)

        current_pos = self.get_open_positions()
        entries     = state.get("entries", {})
        week_start  = state.get("week_start_value", CAPITAL)
        start_eq    = state.get("start_equity",    CAPITAL)

        total_value = 0.0
        for sym in state.get("holdings", []):
            pos = current_pos.get(sym)
            if pos:
                total_value += pos["market_value"]
                entry   = entries.get(sym, {})
                pnl     = pos["pnl"]
                pnl_pct = (pos["current"] - pos["entry_price"]) / pos["entry_price"] * 100
                sign    = "+" if pnl >= 0 else ""
                logger.info(
                    f"  {sym:6s} | Entry: ${pos['entry_price']:.2f} | "
                    f"Current: ${pos['current']:.2f} | "
                    f"Value: ${pos['market_value']:,.2f} | "
                    f"P&L: {sign}${pnl:,.2f} ({sign}{pnl_pct:.1f}%)"
                )

        # SHY benchmark comparison
        shy_mom = compute_momentum(self.data_client, [SAFE_HAVEN], lookback=5)
        shy_5d  = shy_mom.get(SAFE_HAVEN, 0.0)

        week_pnl  = total_value - week_start
        total_pnl = total_value - start_eq
        total_ret = total_pnl / start_eq * 100 if start_eq else 0
        week_ret  = week_pnl / week_start * 100  if week_start else 0

        logger.info("-" * 60)
        logger.info(f"WEEK P&L  : {'+'if week_pnl>=0 else ''}${week_pnl:,.2f} ({week_ret:+.2f}%)")
        logger.info(f"SHY 5d ret: {shy_5d:+.2f}% (benchmark)")
        logger.info(f"TOTAL RET : {total_ret:+.2f}% vs start (${start_eq:,.0f} → ${total_value:,.2f})")
        logger.info("=" * 60)

        # Update week_start for next week
        state["week_start_value"] = total_value
        save_state(state)


# ── Capital exposure summary ───────────────────────────────────────────────────

def print_combined_exposure(equity: float) -> None:
    logger.info("=" * 60)
    logger.info("COMBINED CAPITAL EXPOSURE (all bots)")
    logger.info("-" * 60)
    logger.info(f"  ORB strategy      $75,000")
    logger.info(f"  Overnight mom.    $50,000")
    logger.info(f"  ETF Rotation      $25,000")
    logger.info(f"  ─────────────────────────")
    logger.info(f"  Total allocated  $150,000")
    logger.info(f"  Account equity   ${equity:>10,.2f}")
    logger.info(f"  Free buffer      ${equity - 150_000:>+10,.2f}")
    logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ETF Rotation Bot — Strategy E")
    parser.add_argument("--status",    action="store_true", help="Show status only, no rebalance")
    parser.add_argument("--rebalance", action="store_true", help="Force rebalance regardless of day")
    parser.add_argument("--eow",       action="store_true", help="Show end-of-week summary")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not api_secret:
        logger.error("INIT: ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        sys.exit(1)

    bot   = ETFRotationBot(api_key, api_secret)
    state = load_state()

    # Validate account + show combined exposure
    equity, buying_power = bot.validate_account()
    print_combined_exposure(equity)

    now_et   = datetime.now(ET)
    weekday  = now_et.weekday()   # 0=Monday, 4=Friday

    if args.eow:
        bot.eow_summary(state)
        return

    if args.status:
        bot.show_status(state)
        # Show momentum scores as well
        logger.info("MOMENTUM: Current scores:")
        momentum = compute_momentum(bot.data_client, UNIVERSE)
        target   = get_target_allocation(momentum)
        logger.info(f"MOMENTUM: Would hold → {' + '.join(target)}")
        return

    if args.rebalance or weekday == 0:
        # Monday or forced rebalance
        if not args.rebalance:
            logger.info(f"SCHEDULE: Today is Monday — proceeding with rebalance.")
        state = bot.rebalance(state, forced=args.rebalance)
        bot.show_status(state)
        if weekday == 4:  # won't hit on Monday, but guard for Friday forced runs
            bot.eow_summary(state)
    else:
        day_name = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][weekday]
        logger.info(f"SCHEDULE: Today is {day_name} — no rebalance.")
        bot.show_status(state)
        # Show current momentum for situational awareness
        logger.info("MOMENTUM: Current scores:")
        momentum = compute_momentum(bot.data_client, UNIVERSE)
        target   = get_target_allocation(momentum)
        logger.info(f"MOMENTUM: Would hold if rebalancing today → {' + '.join(target)}")

    # EOW summary on Fridays
    if weekday == 4 and not args.rebalance:
        bot.eow_summary(state)


if __name__ == "__main__":
    main()
