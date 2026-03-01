# Geopolitical Shock — Sector ETF Backtest

Quantifies how **XLE, ITA, PPA, JETS, QQQ, and SPY** behave in the days and
months following major Middle East / oil-shock events, from 9/11 through the
current Iran crisis (2026).

---

## Events Covered

| Event | Start Date | Notes |
|---|---|---|
| 9/11 Attacks | 2001-09-11 | S&P −12% initial; VIX spiked; airlines decimated |
| Iraq War Start | 2003-03-20 | Energy +20% in weeks; broad rally followed |
| Lebanon War | 2006-07-12 | Oil above $78; ~6% market drawdown then recovery |
| Hamas / Gaza War | 2023-10-07 | Defense +5–8%; markets resilient |
| Iran–Israel Direct Exchange | 2024-04-13 | 300+ drones; 1-week equity dip recovered |
| Iran Tensions (Oct 2024) | 2024-10-01 | Oil +5%; defense +3–5% |
| Iran Crisis (2026) | 2026-01-01 | Current baseline |

---

## Metrics Computed

| Metric | Description |
|---|---|
| **Absolute return %** | Price change over T+1 / T+5 / T+20 / T+60 windows |
| **Alpha vs SPY** | Excess return relative to S&P 500 |
| **Sharpe ratio** | Annualised return / volatility (risk-free = 0) |
| **Max drawdown %** | Peak-to-trough within each window |
| **Win rate %** | % of events where return was positive |

---

## Output Files  (`results/` folder)

| File | Description |
|---|---|
| `report.html` | **Interactive dashboard** — all charts, heatmap, full data table (self-contained, no server needed) |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the backtest (downloads ~25 years of data on first run)
python geopolitical_etf_backtest.py
```

Results are printed to the console and an interactive HTML dashboard is saved to `results/report.html`. Open it in any browser — no server required.

---

## Expected Patterns (Historical Average)

| ETF | T+5 (1wk) | T+20 (1mo) | T+60 (3mo) | Win Rate (T+5) |
|---|---|---|---|---|
| XLE (Energy) | +3 – +8% | +5 – +15% | +8 – +20% | ~70% |
| ITA (Defense) | +2 – +6% | +3 – +10% | +5 – +12% | ~75% |
| JETS (Airlines) | −3 – −8% | −2 – −5% | varies | ~30% |
| QQQ (Tech) | −1 – −5% | flat – +5% | +10 – +20% | ~45% |
| SPY (Broad) | −1 – −3% | flat – +5% | +8 – +15% | ~55% |

*Ranges are illustrative; actual script output is data-driven.*

---

## Customising

- **Add/change events** — edit the `EVENTS` list in `geopolitical_etf_backtest.py`
- **Add/change tickers** — edit the `TICKERS` dict
- **Change windows** — edit the `WINDOWS` dict (values are trading days)
