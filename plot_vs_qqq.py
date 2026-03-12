import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timezone

API_KEY = "PKPOUHYPFZRVQEXYBPD2QJVCRT"
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

# Load equity curve
eq = pd.read_csv(r"results/equity_curve_2021-01-01_2026-03-07.csv", parse_dates=["date"])
eq = eq.set_index("date").sort_index()
start = eq.index[0]
end   = eq.index[-1]

# Fetch QQQ daily bars
client = StockHistoricalDataClient(API_KEY, API_SECRET)
req = StockBarsRequest(
    symbol_or_symbols=["QQQ"],
    timeframe=TimeFrame.Day,
    start=start.to_pydatetime().replace(tzinfo=timezone.utc),
    end=end.to_pydatetime().replace(tzinfo=timezone.utc),
    feed="iex",
)
resp = client.get_stock_bars(req)
raw = getattr(resp, "data", {}).get("QQQ", [])
qqq = pd.DataFrame([{"date": b.timestamp.date(), "close": b.close} for b in raw])
qqq["date"] = pd.to_datetime(qqq["date"])
qqq = qqq.set_index("date").sort_index()

# Normalise QQQ to $100k start
qqq_norm = qqq["close"] / qqq["close"].iloc[0] * 100_000

# Align on trading days
combined = pd.DataFrame({"ORB": eq["equity"], "QQQ": qqq_norm}).dropna()

# Stats
orb_ret  = (combined["ORB"].iloc[-1] / combined["ORB"].iloc[0] - 1) * 100
qqq_ret  = (combined["QQQ"].iloc[-1] / combined["QQQ"].iloc[0] - 1) * 100
orb_dd   = ((combined["ORB"] / combined["ORB"].cummax()) - 1).min() * 100
qqq_dd   = ((combined["QQQ"] / combined["QQQ"].cummax()) - 1).min() * 100

# Sharpe (annualised, daily returns)
orb_daily = combined["ORB"].pct_change().dropna()
qqq_daily = combined["QQQ"].pct_change().dropna()
orb_sharpe = orb_daily.mean() / orb_daily.std() * np.sqrt(252)
qqq_sharpe = qqq_daily.mean() / qqq_daily.std() * np.sqrt(252)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#0d1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# Equity curves
ax1.plot(combined.index, combined["ORB"], color="#58a6ff", lw=1.8, label="ORB Strategy")
ax1.plot(combined.index, combined["QQQ"], color="#f78166", lw=1.8, label="QQQ (buy & hold)", alpha=0.85)
ax1.fill_between(combined.index, combined["ORB"], 100_000,
                 where=combined["ORB"] >= 100_000, alpha=0.08, color="#58a6ff")
ax1.axhline(100_000, color="#555", lw=0.8, ls="--")

ax1.set_title("ORB Strategy vs QQQ  |  Jan 2021 – Mar 2026  ($100k start)",
              color="white", fontsize=13, pad=10, fontweight="bold")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=10)
ax1.grid(color="#21262d", linewidth=0.6)

# Drawdown
orb_dd_series = (combined["ORB"] / combined["ORB"].cummax()) - 1
qqq_dd_series = (combined["QQQ"] / combined["QQQ"].cummax()) - 1
ax2.fill_between(combined.index, orb_dd_series * 100, 0,
                 color="#58a6ff", alpha=0.35, label="ORB DD")
ax2.fill_between(combined.index, qqq_dd_series * 100, 0,
                 color="#f78166", alpha=0.25, label="QQQ DD")
ax2.axhline(0, color="#555", lw=0.8)
ax2.set_ylabel("Drawdown %", color="white")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
ax2.grid(color="#21262d", linewidth=0.6)

# Stats box
stats = (
    f"{'':20s}{'ORB':>10s}{'QQQ':>10s}\n"
    f"{'Total return':20s}{orb_ret:>9.1f}%{qqq_ret:>9.1f}%\n"
    f"{'Max drawdown':20s}{orb_dd:>9.1f}%{qqq_dd:>9.1f}%\n"
    f"{'Sharpe (ann.)':20s}{orb_sharpe:>10.2f}{qqq_sharpe:>10.2f}\n"
    f"{'Win rate':20s}{'27.8%':>10s}{'N/A':>10s}\n"
    f"{'Trades':20s}{'8,877':>10s}{'N/A':>10s}"
)
ax1.text(0.01, 0.03, stats, transform=ax1.transAxes,
         fontsize=9, color="white", verticalalignment="bottom",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                   edgecolor="#30363d", alpha=0.92),
         fontfamily="monospace")

plt.tight_layout(h_pad=0.5)
out = "results/orb_vs_qqq_2026-03-10.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
