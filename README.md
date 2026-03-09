# ORB Paper Trading Bot

A live paper trading bot using the Alpaca API implementing a 5-minute Opening Range Breakout (ORB) strategy.

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Opening range | First 5-min candle after 9:30 ET |
| Entry trigger | Close > ORB high AND volume > 1.5x 20-period avg AND RSI(14) between 40–60 |
| Profit target | +1.2% above entry |
| Stop loss | -0.3% below entry |
| Time exit | 13:30 ET (force-close all positions) |
| Max positions | 15 (equal-weight sizing) |
| Watchlist | Dynamic — top 15 pre-market volume stocks, fetched at 9:15 ET |

## Requirements

- Python 3.11+
- Alpaca paper trading account ([sign up free](https://alpaca.markets))
- API key + secret from the Alpaca dashboard

## Installation

```bash
# 1. Clone / download the bot files
cd orb-trader

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

On first run the bot prompts you for your Alpaca API key and secret, then saves them to a local `.env` file so you never have to enter them again.

Alternatively, copy `.env.example` to `.env` and fill in the values manually:

```bash
cp .env.example .env
# edit .env with your keys
```

The bot **only** uses `paper=True` endpoints — no real money is ever at risk.

## Running the Bot

```bash
python orb_trader.py
```

### What happens at each time:

| Time (ET) | Action |
|-----------|--------|
| Startup | Load `.env`, validate API keys, confirm paper account equity |
| 9:15 ET | Fetch top 15 pre-market volume stocks |
| 9:30–9:35 ET | Record 5-minute opening range (high/low) for each symbol |
| 9:35 ET+ | Stream live 1-min bars; check breakout + volume + RSI on every bar |
| Signal detected | Submit bracket order (entry → take-profit + stop-loss) |
| 13:30 ET | Cancel all open orders, market-close all positions |
| End of day | Write daily summary to log |

## Dry-Run / Connectivity Test

Run the standalone connectivity check without starting the full bot:

```bash
python connectivity_test.py
```

## Log Files

Logs are written to `logs/trading_YYYY-MM-DD.log` and also printed to the terminal with color coding.

### Log format
```
[HH:MM:SS ET] [LEVEL] MESSAGE
```

### Examples
```
[09:15:00 ET] [INFO] WATCHLIST: NVDA, TSLA, AMD, META... (15 stocks)
[09:35:12 ET] [INFO] ORB: AAPL range recorded | High: $192.10 | Low: $191.50
[09:37:45 ET] [INFO] SIGNAL: AAPL breakout at $192.45 | ORB high: $192.10 | Vol: 2.3x | RSI: 52.1
[09:37:46 ET] [INFO] ORDER: BUY 52 AAPL @ $192.45 | Target: $194.76 | Stop: $191.87
[10:14:22 ET] [INFO] EXIT: AAPL TARGET hit @ $194.76 | P&L: +$120.64 (+1.2%)
[13:30:00 ET] [INFO] TIME EXIT: Closing all positions
```

## Daily Summary

Written to the log at end of day:

```
============================================================
DAILY SUMMARY — 2024-01-15
============================================================
  Watchlist count : 15
  Signals         : 4
  Trades taken    : 3
  Wins / Losses   : 2 / 1
  Win rate        : 66.7%
  Net P&L         : +$187.42
  Best trade      : NVDA +$210.00
  Worst trade     : AMD -$22.58
  Starting equity : $100,000.00
  Ending equity   : $100,187.42
============================================================
```

## File Structure

```
orb-trader/
├── orb_trader.py        # Main bot
├── logger.py            # Logging module
├── watchlist.py         # Pre-market screener
├── connectivity_test.py # Standalone API connectivity test
├── requirements.txt     # Python dependencies
├── .env.example         # Key template
├── .env                 # Your real keys (never commit this)
└── logs/
    └── trading_YYYY-MM-DD.log
```

## Blacklist

Certain tickers are permanently excluded from the dynamic watchlist regardless of their pre-market volume. The blacklist is defined in `watchlist.py`:

```python
BLACKLIST = ["SNAP", "RIVN", "HOOD", "UBER"]
```

**Why these symbols were added:** Backtest analysis over 2023-2024 showed these four stocks were the largest P&L detractors (-$5,100 combined), driven by excessive whipsaws, low float, and volatility profiles incompatible with the ORB breakout logic.

**What happens when a blacklisted stock would have made the watchlist:**
The next highest pre-market volume stock replaces it, and the exclusion is logged:
```
[09:15:32 ET] [INFO] BLACKLIST: SNAP excluded -- replaced with COIN | vol: 1,240,000
```

**To modify the blacklist**, edit the `BLACKLIST` list in `watchlist.py`. To remove all filtering, set `BLACKLIST = []`.

---

## Strategy 2: Swing Momentum-14

**File:** `swing_trader.py` | **Universe:** `swing_universe.py`
**Capital:** $75,000 | **Max positions:** 15 | **Position size:** $5,000 each

Enters at 15:45 ET daily. Shares the $75k ORB allocation but operates in a non-overlapping window (ORB exits by 13:30, Swing enters at 15:45).

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500, price > $20, avg vol > 3M daily |
| Entry signal | Top-20% 20d momentum + above SMA50 + 1.5x vol today |
| Earnings filter | Skip if earnings within 5 days |
| Hold period | 14 calendar days |
| Stop loss | -5% from entry price |
| Scan time | 15:45 ET daily (Mon-Fri) |

**5-year backtest (Jan 2021-Mar 2026, $75k):**
- Return: +75.7% | Sharpe: 2.137 | MaxDD: -6.7%
- 2022 bear: +3.2% (beat SPY -18.1%)
- Win rate: 47.8% | Profit factor: 1.90

```bash
python swing_trader.py           # run continuously (self-scheduled at 15:45 ET)
python swing_trader.py --scan    # dry-run: show today's candidates, no orders
python swing_trader.py --status  # show open positions and stats
```

---

## Strategy 3: Overnight Momentum

**File:** `overnight_trader.py`
**Capital:** $50,000 | **Max positions:** 5 | **Position size:** $10,000 each

Enters at 15:30 ET Mon-Thu, exits at 09:35 ET next morning.

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500, price > $10, avg vol > 2M |
| Entry signal | Top-5 by 20d momentum |
| Earnings filter | Skip if earnings within 2 days |
| No Friday entries | Avoids weekend holds |

```bash
python overnight_trader.py
```

---

## Strategy 4: ETF Rotation (Strategy E)

**File:** `etf_rotation.py`
**Capital:** $25,000 | **Positions:** 2 x $12,500

Weekly rebalance every Monday at 09:35 ET. Holds top-2 leveraged ETFs by 20-day momentum, replacing negative-momentum ETFs with SHY.

| Parameter | Value |
|-----------|-------|
| Universe | TQQQ, UPRO, TNA |
| Safe haven | SHY (replaces negative-momentum ETFs) |
| Lookback | 20 trading days |
| Rebalance | Monday 09:35 ET |

**5-year backtest:** +191.4% return, Sharpe 0.73, MaxDD -48.0%

```bash
python etf_rotation.py            # auto mode (rebalance Monday, status otherwise)
python etf_rotation.py --status   # show holdings and current momentum scores
python etf_rotation.py --rebalance # force rebalance now
```

---

## Capital Coordination

| Bot | Capital | Window | Overlap risk |
|-----|---------|--------|--------------|
| ORB | $75,000 | 09:15-13:30 ET | None (exits before swing enters) |
| Swing | $75,000 | 15:45 ET entry, 14d hold | Shares $75k with ORB (non-overlapping) |
| Overnight | $50,000 | 15:30-09:35 ET next day | Overnight + ETF = $75k on Mondays |
| ETF Rotation | $25,000 | Monday 09:35 ET | See above |

**Worst-case simultaneous:** Swing $75k + Overnight $50k + ETF $25k = **$150k** (within $200k buying power limit).

---

## Supporting Files

| File | Purpose |
|------|---------|
| `swing_universe.py` | Swing universe: filters, SMA50, momentum rank, earnings check |
| `momentum_universe.py` | Overnight universe |
| `watchlist.py` | ORB pre-market screener |
| `logger.py` | Shared colored ET-timestamped logger |
| `backtest.py` | ORB historical backtest engine |
| `swing_backtest.py` | 9-combo swing backtest (3 styles x 3 hold periods) |
| `leveraged_etf_backtest.py` | Leveraged ETF rotation backtest (5 strategies) |
| `plot_results.py` | 3-panel backtest chart |
| `connectivity_test.py` | Alpaca API dry-run test |

---

## Disclaimer

This bot is for educational and paper trading purposes only. Past performance does not guarantee future results. Never risk money you cannot afford to lose.
