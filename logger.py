"""
logger.py — Colored terminal + rotating file logger for the ORB trading bot.
"""

import logging
import os
from datetime import datetime, timezone
import pytz

ET = pytz.timezone("America/New_York")

# ANSI color codes
RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"
WHITE   = "\033[37m"


class ETFormatter(logging.Formatter):
    """Format log records with ET timestamp and level tag."""

    LEVEL_COLORS = {
        "INFO":    GREEN,
        "WARNING": YELLOW,
        "ERROR":   RED,
        "DEBUG":   CYAN,
    }

    def __init__(self, colored: bool = False):
        super().__init__()
        self.colored = colored

    def format(self, record: logging.LogRecord) -> str:
        now_et = datetime.now(ET)
        time_str = now_et.strftime("%H:%M:%S ET")
        level = record.levelname
        msg = record.getMessage()

        if self.colored:
            color = self.LEVEL_COLORS.get(level, WHITE)
            level_tag = f"{color}[{level}]{RESET}"
            time_tag  = f"{CYAN}[{time_str}]{RESET}"
        else:
            level_tag = f"[{level}]"
            time_tag  = f"[{time_str}]"

        return f"{time_tag} {level_tag} {msg}"


def get_logger(name: str = "orb_trader", log_prefix: str = "trading") -> logging.Logger:
    """
    Return a logger that writes:
    - colored output to the terminal
    - plain text to logs/{log_prefix}_YYYY-MM-DD.log

    Parameters
    ----------
    name       : logger identity (unique per script)
    log_prefix : filename prefix, e.g. "trading" or "overnight"
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.DEBUG)

    # ── terminal handler (colored) ──────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ETFormatter(colored=True))

    # ── file handler (plain) ────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    date_str  = datetime.now(ET).strftime("%Y-%m-%d")
    log_file  = os.path.join("logs", f"{log_prefix}_{date_str}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ETFormatter(colored=False))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def log_trade(logger: logging.Logger, symbol: str, side: str, qty: int,
              entry: float, target: float, stop: float) -> None:
    logger.info(
        f"ORDER: {side.upper()} {qty} {symbol} @ ${entry:.2f} | "
        f"Target: ${target:.2f} | Stop: ${stop:.2f}"
    )


def log_exit(logger: logging.Logger, symbol: str, reason: str,
             exit_price: float, pnl: float, pnl_pct: float) -> None:
    sign = "+" if pnl >= 0 else ""
    logger.info(
        f"EXIT: {symbol} {reason} @ ${exit_price:.2f} | "
        f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)"
    )


def log_signal(logger: logging.Logger, symbol: str, price: float,
               orb_high: float, vol_mult: float, rsi: float) -> None:
    logger.info(
        f"SIGNAL: {symbol} breakout at ${price:.2f} | "
        f"ORB high: ${orb_high:.2f} | Vol: {vol_mult:.1f}x | RSI: {rsi:.1f}"
    )
