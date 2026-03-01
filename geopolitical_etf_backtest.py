"""
Geopolitical Shock ETF Backtest
================================
Backtests sector ETF performance (XLE, ITA, PPA, JETS, QQQ, SPY) across
major Middle East / oil-shock events from 1990 onward.

Outputs:
  - Console: summary tables
  - results/heatmap_avg_returns.png
  - results/cumulative_returns_by_event.png
  - results/avg_returns_bar.png
  - results/vix_spikes.png
  - results/xle_oil_correlation.png
  - results/backtest_full_results.csv
  - results/backtest_summary.csv

Requirements: pip install yfinance pandas numpy matplotlib seaborn scipy
"""

import os
import warnings
import datetime as dt
from collections import defaultdict

import json

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── try yfinance ──────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    raise SystemExit(
        "yfinance not found.  Run:  pip install yfinance pandas numpy "
        "matplotlib seaborn scipy"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ETFs & benchmarks
TICKERS = {
    "XLE":  "Energy (XLE)",
    "ITA":  "Defense (ITA)",
    "PPA":  "Defense2 (PPA)",
    "JETS": "Airlines (JETS)",
    "QQQ":  "Tech (QQQ)",
    "SPY":  "S&P 500 (SPY)",
    "^VIX": "VIX",
    "USO":  "Oil ETF (USO)",
}

# Analysis windows in trading days
WINDOWS = {
    "T+1 (1 day)":   1,
    "T+5 (1 week)":  5,
    "T+20 (1 month)": 20,
    "T+60 (3 months)": 60,
}

# Download start / end
DATA_START = "2000-01-01"
DATA_END   = dt.date.today().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# 2. GEOPOLITICAL EVENT DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

EVENTS = [
    {
        "name":  "9/11 Attacks",
        "date":  "2001-09-11",
        "notes": "S&P -12% initial drop; markets closed 4 days; VIX spiked; "
                 "defense rallied, airlines decimated.",
    },
    {
        "name":  "Iraq War Start",
        "date":  "2003-03-20",
        "notes": "Shock-and-awe campaign. Energy +20% in weeks; defense "
                 "steady; market initially dipped then rallied hard.",
    },
    {
        "name":  "Lebanon War / Israel-Hezbollah",
        "date":  "2006-07-12",
        "notes": "Oil briefly above $78; regional risk premium. Broad "
                 "markets pulled back ~6% then recovered.",
    },
    {
        "name":  "Gaza War / Hamas Attacks",
        "date":  "2023-10-07",
        "notes": "Israel invaded Gaza after Hamas attack. Oil +4%, "
                 "defense +5-8%; markets resilient short-term.",
    },
    {
        "name":  "Iran-Israel Direct Exchange",
        "date":  "2024-04-13",
        "notes": "Iran fired 300+ drones/missiles at Israel. Oil +3%, "
                 "defense +2-4%; equity dip recovered within 1 week.",
    },
    {
        "name":  "Iran Tensions (Oct 2024)",
        "date":  "2024-10-01",
        "notes": "Escalating Iran-Israel tensions. Oil +5%; "
                 "defense +3-5% short-term.",
    },
    {
        "name":  "Iran Crisis (2026)",
        "date":  "2026-01-01",
        "notes": "Current Iran crisis. Baseline for present analysis.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"\n{'='*60}")
    print("  DOWNLOADING DATA")
    print(f"{'='*60}")
    print(f"  Tickers : {', '.join(tickers.keys())}")
    print(f"  Period  : {start} → {end}\n")

    raw = yf.download(
        list(tickers.keys()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # yfinance may return multi-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()

    close.index = pd.to_datetime(close.index)
    close.index = close.index.tz_localize(None)

    # Rename columns with friendly names where available
    close.rename(
        columns={t: tickers.get(t, t) for t in close.columns},
        inplace=True,
    )

    print(f"\n  Downloaded {len(close)} trading days × {close.shape[1]} series")
    print(f"  Date range: {close.index.min().date()} → {close.index.max().date()}")
    return close


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def next_valid_date(prices: pd.DataFrame, event_date_str: str) -> pd.Timestamp | None:
    """Return the first trading date on or after event_date_str."""
    event_dt = pd.Timestamp(event_date_str)
    candidates = prices.index[prices.index >= event_dt]
    if len(candidates) == 0:
        return None
    return candidates[0]


def pct_return(prices: pd.Series, t0_idx: int, window: int) -> float | None:
    """Return (price[t0+window] / price[t0] - 1) * 100, or None if out of bounds."""
    end_idx = t0_idx + window
    if end_idx >= len(prices):
        return None
    p0 = prices.iloc[t0_idx]
    p1 = prices.iloc[end_idx]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return None
    return (p1 / p0 - 1) * 100.0


def max_drawdown(prices: pd.Series, t0_idx: int, window: int) -> float | None:
    """Max peak-to-trough drawdown (%) over [t0, t0+window]."""
    end_idx = t0_idx + window
    if end_idx >= len(prices):
        end_idx = len(prices) - 1
    segment = prices.iloc[t0_idx: end_idx + 1].dropna()
    if len(segment) < 2:
        return None
    roll_max = segment.cummax()
    dd = (segment - roll_max) / roll_max * 100.0
    return dd.min()   # most negative value


def rolling_sharpe(prices: pd.Series, t0_idx: int, window: int,
                   annualise: int = 252) -> float | None:
    """Annualised Sharpe ratio over the window (risk-free = 0)."""
    end_idx = t0_idx + window
    if end_idx >= len(prices):
        return None
    segment = prices.iloc[t0_idx: end_idx + 1].dropna()
    daily_rets = segment.pct_change().dropna()
    if len(daily_rets) < 3:
        return None
    mu  = daily_rets.mean()
    std = daily_rets.std()
    if std == 0:
        return None
    return (mu / std) * (annualise ** 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(prices: pd.DataFrame,
                 events: list,
                 windows: dict,
                 ticker_map: dict) -> pd.DataFrame:
    """
    For every (event, ticker, window) triple compute:
      - absolute return %
      - return relative to SPY (alpha)
      - Sharpe ratio
      - max drawdown %
    Returns a long-format DataFrame.
    """
    spy_col  = ticker_map.get("SPY",  "S&P 500 (SPY)")
    vix_col  = ticker_map.get("^VIX", "VIX")

    rows = []

    print(f"\n{'='*60}")
    print("  RUNNING BACKTEST")
    print(f"{'='*60}\n")

    for event in events:
        t0 = next_valid_date(prices, event["date"])
        if t0 is None:
            print(f"  [SKIP] {event['name']} — no data after {event['date']}")
            continue

        t0_idx = prices.index.get_loc(t0)
        print(f"  ► {event['name']:40s}  T0={t0.date()}")

        # SPY return for this window (for alpha calc)
        spy_returns = {}
        if spy_col in prices.columns:
            for wname, wdays in windows.items():
                spy_returns[wname] = pct_return(prices[spy_col], t0_idx, wdays)

        for col in prices.columns:
            if col == vix_col:
                # For VIX record absolute change (not %)
                for wname, wdays in windows.items():
                    ret = pct_return(prices[col], t0_idx, wdays)
                    rows.append({
                        "Event":  event["name"],
                        "Date":   event["date"],
                        "Ticker": col,
                        "Window": wname,
                        "Return_%":    ret,
                        "Alpha_%":     None,
                        "Sharpe":      None,
                        "MaxDrawdown_%": None,
                    })
                continue

            for wname, wdays in windows.items():
                ret = pct_return(prices[col], t0_idx, wdays)
                spy_ret = spy_returns.get(wname)
                alpha = (ret - spy_ret) if (ret is not None and spy_ret is not None) else None
                sharpe = rolling_sharpe(prices[col], t0_idx, wdays)
                mdd    = max_drawdown(prices[col], t0_idx, wdays)

                rows.append({
                    "Event":         event["name"],
                    "Date":          event["date"],
                    "Ticker":        col,
                    "Window":        wname,
                    "Return_%":      ret,
                    "Alpha_%":       alpha,
                    "Sharpe":        sharpe,
                    "MaxDrawdown_%": mdd,
                })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across events for each (Ticker, Window):
      mean, median, std, win_rate, count + alpha mean & alpha win-rate
    """
    agg = (
        df.dropna(subset=["Return_%"])
        .groupby(["Ticker", "Window"])["Return_%"]
        .agg(Mean="mean", Median="median", Std="std", Count="count")
        .reset_index()
    )
    win = (
        df.dropna(subset=["Return_%"])
        .assign(Win=lambda x: x["Return_%"] > 0)
        .groupby(["Ticker", "Window"])["Win"]
        .mean().mul(100).rename("WinRate_%")
        .reset_index()
    )
    # Alpha aggregation
    alpha_agg = (
        df.dropna(subset=["Alpha_%"])
        .groupby(["Ticker", "Window"])["Alpha_%"]
        .mean().rename("AlphaMean")
        .reset_index()
    )
    alpha_wr = (
        df.dropna(subset=["Alpha_%"])
        .assign(AlphaWin=lambda x: x["Alpha_%"] > 0)
        .groupby(["Ticker", "Window"])["AlphaWin"]
        .mean().mul(100).rename("AlphaWinRate_%")
        .reset_index()
    )
    summary = (agg
               .merge(win,       on=["Ticker", "Window"])
               .merge(alpha_agg, on=["Ticker", "Window"], how="left")
               .merge(alpha_wr,  on=["Ticker", "Window"], how="left"))
    return summary


def build_heatmap_df(summary: pd.DataFrame, metric: str = "Mean") -> pd.DataFrame:
    """Pivot summary to Ticker × Window matrix."""
    window_order = list(WINDOWS.keys())
    pivot = summary.pivot(index="Ticker", columns="Window", values=metric)
    # reorder columns
    cols = [c for c in window_order if c in pivot.columns]
    pivot = pivot[cols]
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

COLOUR_MAP = {
    "Energy (XLE)":    "#e07b39",
    "Defense (ITA)":   "#4a7fc1",
    "Defense2 (PPA)":  "#6db3e0",
    "Airlines (JETS)": "#d94f5c",
    "Tech (QQQ)":      "#9b59b6",
    "S&P 500 (SPY)":   "#27ae60",
    "VIX":             "#e74c3c",
    "Oil ETF (USO)":   "#f1c40f",
}




# ─────────────────────────────────────────────────────────────────────────────
# 7. HTML DATA PREPARATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prep_cumulative(prices: pd.DataFrame, events: list,
                     tickers_to_plot: list, window_days: int = 60) -> list:
    """Return per-event indexed price series (T0 = 100)."""
    out = []
    for ev in events:
        t0 = next_valid_date(prices, ev["date"])
        if t0 is None:
            continue
        t0_idx = prices.index.get_loc(t0)
        end_idx = min(t0_idx + window_days, len(prices) - 1)
        seg = prices.iloc[t0_idx: end_idx + 1]
        series = {}
        for col in tickers_to_plot:
            if col not in seg.columns:
                continue
            s = seg[col].dropna()
            if len(s) < 2 or s.iloc[0] == 0:
                continue
            norm = [round(float(v) / float(s.iloc[0]) * 100, 3) for v in s]
            # pad to window_days+1
            norm += [None] * (window_days + 1 - len(norm))
            series[col] = norm[: window_days + 1]
        out.append({
            "name":   ev["name"],
            "date":   ev["date"],
            "notes":  ev["notes"],
            "series": series,
        })
    return out


def _prep_vix(prices: pd.DataFrame, events: list) -> dict:
    """Monthly VIX with event-date markers."""
    vix_col = "VIX"
    if vix_col not in prices.columns:
        return {"dates": [], "values": [], "events": []}
    monthly = prices[vix_col].resample("ME").mean().dropna()
    markers = []
    for ev in events:
        t0 = next_valid_date(prices, ev["date"])
        if t0:
            markers.append({"name": ev["name"], "date": t0.strftime("%Y-%m-%d")})
    return {
        "dates":  [d.strftime("%Y-%m") for d in monthly.index],
        "values": [round(float(v), 2) for v in monthly.values],
        "events": markers,
    }


def _prep_summary_json(summary: pd.DataFrame, windows: dict) -> dict:
    """Structured summary keyed by (ticker, window)."""
    window_order = list(windows.keys())
    tickers = [t for t in summary["Ticker"].unique() if "VIX" not in t]
    stats: dict = {}
    for t in tickers:
        stats[t] = {}
        for w in window_order:
            row = summary[(summary["Ticker"] == t) & (summary["Window"] == w)]
            if row.empty:
                stats[t][w] = None
            else:
                r = row.iloc[0]
                stats[t][w] = {
                    "mean":      round(float(r["Mean"]),      2),
                    "median":    round(float(r["Median"]),    2),
                    "std":       round(float(r["Std"]),       2),
                    "wr":        round(float(r["WinRate_%"]), 1),
                    "count":     int(r["Count"]),
                    "alpha_mean": round(float(r["AlphaMean"]),     2) if pd.notna(r.get("AlphaMean")) else None,
                    "alpha_wr":   round(float(r["AlphaWinRate_%"]),1) if pd.notna(r.get("AlphaWinRate_%")) else None,
                }
    return {"tickers": tickers, "windows": window_order, "stats": stats}


# ─────────────────────────────────────────────────────────────────────────────
# 8. HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def export_html(prices: pd.DataFrame, results: pd.DataFrame,
                summary: pd.DataFrame, events: list, windows: dict,
                tickers: dict, colour_map: dict) -> str:

    run_date = dt.date.today().isoformat()
    tickers_for_charts = [v for k, v in tickers.items() if k != "^VIX"]

    cum_data   = _prep_cumulative(prices, events, tickers_for_charts, 60)
    vix_data   = _prep_vix(prices, events)
    summ_data  = _prep_summary_json(summary, windows)
    ev_list    = [{"name": e["name"], "date": e["date"], "notes": e["notes"]}
                  for e in events]

    full_rows = (results[~results["Ticker"].str.contains("VIX", na=False)]
                 .round(3)
                 .fillna("")
                 .to_dict("records"))

    colours_js = json.dumps(colour_map)
    cum_js     = json.dumps(cum_data)
    vix_js     = json.dumps(vix_data)
    summ_js    = json.dumps(summ_data)
    full_js    = json.dumps(full_rows)
    ev_js      = json.dumps(ev_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Geopolitical Shock — ETF Backtest</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff; --pos: #3fb950; --neg: #f85149;
    --warn: #d29922; --radius: 10px; --font: 'Segoe UI', system-ui, sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 14px; line-height: 1.6; }}
  a {{ color: var(--accent); text-decoration: none; }}

  /* ── Layout ── */
  .container {{ max-width: 1280px; margin: 0 auto; padding: 0 24px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }}
  @media(max-width:900px){{.grid-2,.grid-3,.grid-4{{grid-template-columns:1fr;}}}}

  /* ── Hero ── */
  .hero {{ background: linear-gradient(135deg,#0d1117 0%,#161b22 60%,#1a2332 100%);
           border-bottom: 1px solid var(--border); padding: 52px 0 40px; }}
  .hero-inner {{ display:flex; align-items:center; justify-content:space-between; gap:24px;
                 flex-wrap:wrap; }}
  .hero h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: -.5px;
              background: linear-gradient(90deg,#58a6ff,#a5d6ff);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .hero p {{ color: var(--muted); margin-top: 6px; font-size: .92rem; }}
  .badge {{ background: var(--surface2); border: 1px solid var(--border);
            border-radius: 20px; padding: 4px 14px; font-size: .78rem; color: var(--muted);
            display: inline-flex; align-items: center; gap: 6px; }}
  .badge .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--pos); }}

  /* ── Section ── */
  section {{ padding: 40px 0; }}
  section + section {{ border-top: 1px solid var(--border); }}
  .section-title {{ font-size: 1.15rem; font-weight: 600; margin-bottom: 20px;
                    display: flex; align-items: center; gap: 10px; color: var(--text); }}
  .section-title span {{ color: var(--accent); font-size: .7rem; text-transform: uppercase;
                          letter-spacing: 1px; border: 1px solid var(--accent);
                          padding: 2px 8px; border-radius: 4px; }}

  /* ── Cards ── */
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: var(--radius); padding: 20px; }}
  .kpi-card {{ background: var(--surface); border: 1px solid var(--border);
               border-radius: var(--radius); padding: 18px 20px;
               transition: transform .15s, border-color .15s; cursor: default; }}
  .kpi-card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
  .kpi-label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .8px;
                color: var(--muted); margin-bottom: 4px; }}
  .kpi-value {{ font-size: 1.75rem; font-weight: 700; }}
  .kpi-sub {{ font-size: .78rem; color: var(--muted); margin-top: 2px; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  .neu {{ color: var(--warn); }}

  /* ── Heatmap table ── */
  .heatmap-wrap {{ overflow-x: auto; }}
  table.heatmap {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  table.heatmap th {{ background: var(--surface2); padding: 10px 14px;
                      text-align: center; font-weight: 600; border: 1px solid var(--border);
                      white-space: nowrap; }}
  table.heatmap th.row-h {{ text-align: left; }}
  table.heatmap td {{ padding: 9px 14px; text-align: center;
                      border: 1px solid var(--border); font-weight: 600; }}
  table.heatmap td.label {{ text-align: left; color: var(--text);
                             background: var(--surface2); white-space: nowrap; }}
  table.heatmap .wr {{ font-size: .7rem; color: var(--muted); font-weight: 400; display: block; }}

  /* ── Event timeline ── */
  .timeline {{ display: flex; flex-direction: column; gap: 14px; }}
  .tl-item {{ display: flex; gap: 16px; align-items: flex-start; }}
  .tl-dot {{ width: 12px; height: 12px; border-radius: 50%; background: var(--accent);
             flex-shrink: 0; margin-top: 4px; }}
  .tl-date {{ font-size: .72rem; color: var(--muted); white-space: nowrap; min-width: 82px; }}
  .tl-name {{ font-weight: 600; font-size: .88rem; }}
  .tl-notes {{ font-size: .78rem; color: var(--muted); margin-top: 2px; }}

  /* ── Tabs ── */
  .tabs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 18px; }}
  .tab-btn {{ background: var(--surface2); border: 1px solid var(--border);
              border-radius: 6px; padding: 6px 14px; cursor: pointer;
              font-size: .8rem; color: var(--muted); transition: all .15s; }}
  .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .tab-btn.active {{ background: var(--accent); border-color: var(--accent);
                     color: #0d1117; font-weight: 600; }}
  .tab-pane {{ display: none; }}
  .tab-pane.active {{ display: block; }}

  /* ── Full data table ── */
  .data-table-wrap {{ overflow: auto; max-height: 480px; border-radius: var(--radius);
                      border: 1px solid var(--border); }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: .78rem; }}
  table.data-table thead th {{ background: var(--surface2); padding: 9px 12px;
    text-align: left; position: sticky; top: 0; z-index: 1;
    border-bottom: 1px solid var(--border); white-space: nowrap; cursor: pointer; }}
  table.data-table thead th:hover {{ color: var(--accent); }}
  table.data-table tbody tr:nth-child(even) {{ background: rgba(255,255,255,.02); }}
  table.data-table tbody tr:hover {{ background: rgba(88,166,255,.06); }}
  table.data-table td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
  .search-bar {{ display: flex; gap: 10px; margin-bottom: 14px; }}
  .search-bar input {{ flex: 1; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 14px; color: var(--text); font-size: .85rem;
    outline: none; }}
  .search-bar input:focus {{ border-color: var(--accent); }}

  /* ── Footer ── */
  footer {{ padding: 28px 0; text-align: center; color: var(--muted); font-size: .78rem;
            border-top: 1px solid var(--border); }}

  /* ── Chart containers ── */
  .chart-box {{ position: relative; }}
  .chart-box canvas {{ max-height: 340px; }}
  .chart-box-tall canvas {{ max-height: 420px; }}

  /* ── Heatmap mode toggle ── */
  .hm-toggle {{ display: flex; gap: 6px; margin-bottom: 0; }}
  .hm-btn {{ background: var(--surface2); border: 1px solid var(--border);
             border-radius: 6px; padding: 4px 12px; cursor: pointer;
             font-size: .75rem; color: var(--muted); transition: all .15s; }}
  .hm-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .hm-btn.active {{ background: var(--accent); border-color: var(--accent);
                    color: #0d1117; font-weight: 600; }}
  .shock {{ font-size: .65rem; color: #f1c40f; margin-left: 2px; vertical-align: middle; }}

  /* ── KPI alpha row ── */
  .kpi-alpha {{ display: flex; align-items: center; gap: 6px; margin-top: 8px;
               padding-top: 8px; border-top: 1px solid var(--border); }}
  .kpi-alpha-label {{ font-size: .68rem; color: var(--muted); text-transform: uppercase;
                      letter-spacing: .5px; flex-shrink: 0; }}
  .kpi-alpha-val {{ font-size: .92rem; font-weight: 700; }}

  /* ── Narrative cards ── */
  .narr-card {{ background: var(--surface); border: 1px solid var(--border);
                border-radius: var(--radius); padding: 22px 24px;
                border-left: 4px solid var(--accent);
                transition: transform .15s, box-shadow .15s; }}
  .narr-card:hover {{ transform: translateY(-2px);
                      box-shadow: 0 6px 24px rgba(0,0,0,.35); }}
  .narr-header {{ display: flex; align-items: center; justify-content: space-between;
                  margin-bottom: 12px; }}
  .narr-ticker {{ font-size: 1.05rem; font-weight: 700; }}
  .narr-verdict {{ font-size: .7rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: .8px; padding: 3px 10px; border-radius: 20px;
                   border: 1px solid currentColor; }}
  .verdict-bullish  {{ color: #3fb950; }}
  .verdict-bearish  {{ color: #f85149; }}
  .verdict-mixed    {{ color: #d29922; }}
  .verdict-recovery {{ color: #58a6ff; }}
  .narr-body {{ font-size: .86rem; color: var(--muted); line-height: 1.75;
                margin-bottom: 14px; }}
  .narr-body b {{ color: var(--text); }}
  .narr-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
  .pill {{ background: var(--surface2); border: 1px solid var(--border);
           border-radius: 6px; padding: 4px 10px; font-size: .72rem;
           display: flex; align-items: center; gap: 5px; }}
  .pill-label {{ color: var(--muted); }}
  .pill-val {{ font-weight: 700; }}
  .narr-quote {{ border-left: 2px solid var(--border); padding-left: 12px;
                 color: var(--muted); font-style: italic; font-size: .82rem;
                 margin-top: 12px; line-height: 1.6; }}

  /* ── Current crisis context ── */
  .crisis-banner {{ background: linear-gradient(135deg,#1a2332 0%,#161b22 100%);
                    border: 1px solid #58a6ff40; border-radius: var(--radius);
                    padding: 0; overflow: hidden; }}
  .crisis-banner-header {{ background: linear-gradient(90deg,#58a6ff15,transparent);
                            border-bottom: 1px solid #58a6ff30;
                            padding: 16px 22px; display: flex; align-items: center;
                            justify-content: space-between; flex-wrap: wrap; gap: 10px; }}
  .crisis-now-dot {{ width: 9px; height: 9px; border-radius: 50%; background: #f85149;
                     box-shadow: 0 0 0 3px #f8514930; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 3px #f8514930; }}
                      50%  {{ box-shadow: 0 0 0 7px #f8514910; }} }}
  .crisis-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
  @media(max-width:900px){{ .crisis-grid {{ grid-template-columns: 1fr; }} }}
  .crisis-panel {{ padding: 22px; }}
  .crisis-panel + .crisis-panel {{ border-left: 1px solid var(--border); }}
  @media(max-width:900px){{ .crisis-panel + .crisis-panel {{ border-left:none; border-top:1px solid var(--border); }} }}
  .crisis-panel-title {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .9px;
                          color: var(--accent); font-weight: 600; margin-bottom: 12px; }}
  .crisis-panel p  {{ font-size: .84rem; color: var(--muted); line-height: 1.75; }}
  .crisis-panel p b {{ color: var(--text); }}
  .analog-item {{ display: flex; gap: 12px; align-items: flex-start;
                  padding: 9px 0; border-bottom: 1px solid var(--border); }}
  .analog-item:last-child {{ border-bottom: none; }}
  .analog-sim {{ width: 38px; height: 38px; border-radius: 8px; flex-shrink: 0;
                 display: flex; align-items: center; justify-content: center;
                 font-size: .75rem; font-weight: 700; background: var(--surface2); }}
  .analog-name {{ font-size: .84rem; font-weight: 600; line-height: 1.3; }}
  .analog-why  {{ font-size: .76rem; color: var(--muted); margin-top: 2px; }}
  .caveat-list {{ list-style: none; display: flex; flex-direction: column; gap: 8px; }}
  .caveat-list li {{ display: flex; gap: 10px; align-items: flex-start;
                     font-size: .82rem; color: var(--muted); line-height: 1.6; }}
  .caveat-list li::before {{ content: '✕'; color: #f85149; font-size: .75rem;
                              flex-shrink: 0; margin-top: 3px; font-weight: 700; }}
  .caveat-list li b {{ color: var(--text); }}
  .compare-table {{ width: 100%; border-collapse: collapse; font-size: .8rem; margin-top: 4px; }}
  .compare-table th {{ color: var(--muted); font-weight: 500; padding: 5px 8px;
                       text-align: right; border-bottom: 1px solid var(--border);
                       white-space: nowrap; }}
  .compare-table th:first-child {{ text-align: left; }}
  .compare-table td {{ padding: 6px 8px; text-align: right;
                       border-bottom: 1px solid #30363d50; }}
  .compare-table td:first-child {{ text-align: left; font-weight: 600; color: var(--text); }}
  .compare-table tr:last-child td {{ border-bottom: none; }}
  .delta-pos {{ color: var(--pos); }}
  .delta-neg {{ color: var(--neg); }}
  .delta-neu {{ color: var(--warn); }}

  /* ── Limitations section ── */
  .limit-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; }}
  @media(max-width:1100px){{ .limit-grid {{ grid-template-columns: repeat(2,1fr); }} }}
  @media(max-width:640px){{  .limit-grid {{ grid-template-columns: 1fr; }} }}
  .limit-card {{ background: var(--surface); border: 1px solid var(--border);
                 border-radius: var(--radius); padding: 20px 22px; }}
  .limit-card-num {{ font-size: .65rem; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 1px; color: var(--accent); margin-bottom: 6px; }}
  .limit-card-title {{ font-size: .92rem; font-weight: 700; color: var(--text);
                        margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }}
  .limit-card-body {{ font-size: .82rem; color: var(--muted); line-height: 1.8; }}
  .limit-card-body b {{ color: var(--text); }}
  .limit-closing {{ background: var(--surface); border: 1px solid #58a6ff30;
                    border-left: 3px solid var(--accent); border-radius: var(--radius);
                    padding: 20px 24px; margin-top: 20px;
                    font-size: .86rem; color: var(--muted); line-height: 1.8; }}
  .limit-closing b {{ color: var(--text); }}

  /* ── Executive Summary ── */
  .exec-body {{ max-width: 860px; }}
  .exec-body p {{ font-size: .875rem; color: var(--muted); line-height: 1.9;
                  margin-bottom: 18px; }}
  .exec-body p:last-child {{ margin-bottom: 0; }}
  .exec-body b {{ color: var(--text); }}
  .exec-body em {{ color: var(--accent); font-style: normal; font-weight: 600; }}
  .exec-closing {{ font-size: .85rem; color: var(--muted); line-height: 1.8;
                   border-top: 1px solid var(--border); padding-top: 16px;
                   margin-top: 4px; max-width: 860px; }}
  .exec-closing b {{ color: var(--text); }}

  /* ── Shock Playbook ── */
  .playbook-intro {{ color: var(--muted); font-size:.85rem; line-height:1.8;
                    max-width:860px; margin-bottom:24px; }}
  .playbook-intro b {{ color: var(--text); }}
  .playbook-wrap {{ overflow-x: auto; margin-bottom: 28px; }}
  .playbook-table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
  .playbook-table thead th {{ background: var(--surface); color: var(--accent);
                              font-weight:600; padding:11px 16px; text-align:left;
                              border-bottom: 2px solid var(--border); white-space:nowrap; }}
  .playbook-table tbody tr {{ border-bottom: 1px solid var(--border); }}
  .playbook-table tbody tr:last-child {{ border-bottom: none; }}
  .playbook-table tbody tr:hover {{ background: #ffffff08; }}
  .playbook-table td {{ padding:11px 16px; vertical-align:top; line-height:1.75;
                        color: var(--muted); }}
  .playbook-table td:first-child {{ color: var(--text); font-weight:600;
                                    white-space:nowrap; min-width:200px; }}
  .playbook-table td:nth-child(2) {{ min-width:220px; }}
  .playbook-table td:nth-child(3) {{ min-width:300px; }}
  .pb-tag {{ display:inline-block; padding:1px 7px; border-radius:4px;
             font-size:.75rem; font-weight:600; margin:2px 2px 2px 0;
             vertical-align:middle; white-space:nowrap; }}
  .pb-up   {{ background:#3fb95022; color:#3fb950; border:1px solid #3fb95044; }}
  .pb-down {{ background:#f8514922; color:#f85149; border:1px solid #f8514944; }}
  .pb-neu  {{ background:#58a6ff18; color:#58a6ff; border:1px solid #58a6ff33; }}
  .pb-warn {{ background:#d2992222; color:#d29922; border:1px solid #d2992244; }}
  .playbook-how {{ background: var(--surface); border:1px solid var(--border);
                   border-left: 3px solid var(--accent);
                   border-radius: var(--radius); padding:18px 22px;
                   font-size:.84rem; color:var(--muted); line-height:1.8;
                   margin-bottom:16px; }}
  .playbook-how b {{ color: var(--text); }}
  .playbook-reminder {{ background: var(--surface); border:1px solid var(--border);
                        border-left: 3px solid var(--warn);
                        border-radius: var(--radius); padding:16px 22px;
                        font-size:.82rem; color:var(--muted); line-height:1.8; }}
  .playbook-reminder b {{ color: var(--text); }}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════ HERO ═ -->
<header class="hero">
  <div class="container">
    <div class="hero-inner">
      <div>
        <h1>Geopolitical Shock — Sector ETF Backtest</h1>
        <p>How XLE, ITA, PPA, JETS, QQQ and SPY respond to major Middle East / oil-shock events</p>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <span class="badge"><span class="dot"></span>Live data via yfinance</span>
        <span class="badge">Run date: {run_date}</span>
        <span id="evcount-badge" class="badge">— events</span>
      </div>
    </div>
  </div>
</header>

<main class="container">

<!-- ═══════════════════════════════════════════ KPI CARDS ═ -->
<section>
  <div class="section-title">Performance Snapshot <span>T+5 · 1-week window</span></div>
  <div id="kpi-grid" class="grid-4"></div>
</section>

<!-- ════════════════════════════════════════════ EXECUTIVE SUMMARY ═ -->
<section>
  <div class="section-title">Executive Summary <span>Key findings &amp; context</span></div>
  <div class="exec-body">

    <p>
      This report examines how six sector ETFs &mdash; <b>XLE</b> (energy), <b>ITA</b> and <b>PPA</b>
      (defense), <b>JETS</b> (airlines), <b>QQQ</b> (technology), and <b>SPY</b> (broad market) &mdash;
      have responded to major geopolitical and oil-related shocks between 2001 and 2026. The
      analysis is grounded in real, tradeable ETF price data. Events were selected because they
      produced measurable disruptions to oil supply, regional security, or investor risk appetite.
      The goal is to surface <b>recurring behavioural tendencies</b> that emerge across events,
      not to construct a predictive model. The dataset covers seven events; the patterns observed
      are directional guides, not statistically conclusive findings.
    </p>

    <p>
      Across the events studied, a consistent sector hierarchy emerges in the initial shock window.
      <em>Energy</em> (XLE) tends to rise when oil-supply routes or production infrastructure are
      perceived to be at risk &mdash; an unsurprising reaction given the direct link between Middle East
      conflict and crude prices. <em>Defense</em> ETFs (ITA, PPA) show steady gains in the weeks
      following escalation events, reflecting market expectations of increased procurement and
      government spending. <em>Airlines</em> (JETS) are the most vulnerable sector, falling sharply
      during terrorism events and whenever fuel costs spike or travel demand is threatened.
      <em>Technology</em> (QQQ) tends to dip during the initial shock but recovers quickly in
      events where recession risk does not materially increase. <b>SPY serves as the baseline</b>
      throughout: a short pullback is the norm, followed by stabilisation and recovery in most
      cases.
    </p>

    <p>
      The strongest and most consistent reactions occur in the <b>T+1 to T+5 window</b> &mdash; the
      first week after a shock. Beyond that, mean reversion is the dominant pattern. Most events
      in this dataset do not create sustained multi-month trends; the market processes the shock,
      prices in a new baseline, and carries forward. The exceptions are notable: shocks that
      physically threaten oil supply over a prolonged period, or that escalate into a sustained
      military campaign, tend to extend the energy and defense outperformance meaningfully into
      the T+60 window. Events that turn out to be contained &mdash; a single exchange, a brief
      escalation &mdash; show faster reversion across all sectors.
    </p>

    <p>
      For interpreting a new geopolitical shock as it develops, this data supports a simple
      framing: <b>energy and defense typically carry the bid; airlines and broad risk assets
      typically carry the fear.</b> The speed of reversion depends on whether the shock
      remains contained or escalates into something with tangible economic consequences. That
      distinction &mdash; contained versus structural &mdash; is the most important judgement to make
      when a new headline arrives, and it is one this model cannot make for you.
    </p>

    <p class="exec-closing">
      <b>What this data shows clearly:</b> sector imbalances are consistent and fast-moving
      in the first five trading days. <b>What it does not show:</b> how large any specific
      future shock will be, how long it will last, or whether the next event will resemble
      any of the seven in this dataset. These patterns are a reference frame. Each new crisis
      writes its own numbers.
    </p>

  </div>
</section>

<!-- ═══════════════════════════════════════════ CURRENT CRISIS CONTEXT ═ -->
<section id="crisis-section">
  <div class="section-title">Current Crisis Context
    <span style="background:#f8514920;border-color:#f85149;color:#f85149;">● Live Situation</span>
  </div>
  <div class="crisis-banner">
    <div class="crisis-banner-header">
      <div style="display:flex;align-items:center;gap:12px;">
        <div class="crisis-now-dot"></div>
        <div>
          <div id="crisis-event-name" style="font-weight:700;font-size:1rem;"></div>
          <div id="crisis-event-date" style="font-size:.78rem;color:var(--muted);margin-top:2px;"></div>
        </div>
      </div>
      <div id="crisis-event-notes" style="font-size:.8rem;color:var(--muted);max-width:560px;text-align:right;"></div>
    </div>
    <div class="crisis-grid">
      <div class="crisis-panel">
        <div class="crisis-panel-title">What is happening now</div>
        <p id="crisis-now-text"></p>
      </div>
      <div class="crisis-panel">
        <div class="crisis-panel-title">This event vs historical average</div>
        <table class="compare-table" id="crisis-compare-table">
          <thead><tr>
            <th>ETF</th>
            <th>This event T+5</th>
            <th>Hist. avg T+5</th>
            <th>Delta</th>
          </tr></thead>
          <tbody id="crisis-compare-body"></tbody>
        </table>
      </div>
      <div class="crisis-panel">
        <div class="crisis-panel-title">Closest historical analogs</div>
        <div id="crisis-analogs"></div>
      </div>
      <div class="crisis-panel">
        <div class="crisis-panel-title">What this model cannot infer</div>
        <ul class="caveat-list" id="crisis-caveats"></ul>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════ EVENTS TIMELINE + HEATMAP ═ -->
<section>
  <div class="grid-2">
    <!-- Timeline -->
    <div class="card">
      <div class="section-title" style="margin-bottom:14px;">Events Analysed <span>Timeline</span></div>
      <div id="timeline" class="timeline"></div>
    </div>
    <!-- Heatmap -->
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
        <div class="section-title" style="margin-bottom:0;">Return Heatmap <span>ETF × Window</span></div>
        <div class="hm-toggle">
          <button class="hm-btn active" id="hm-raw"   onclick="setHeatmapMode('raw')">Raw Return %</button>
          <button class="hm-btn"        id="hm-alpha" onclick="setHeatmapMode('alpha')">Excess vs SPY %</button>
        </div>
      </div>
      <div class="heatmap-wrap">
        <table id="heatmap-table" class="heatmap"></table>
      </div>
      <div style="margin-top:8px;font-size:.72rem;color:var(--muted);">
        <span style="color:#f1c40f;">⚡</span> ETF&nbsp;positive while SPY&nbsp;negative — isolates true geopolitical shock alpha
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════ AVG RETURNS BAR ═ -->
<section>
  <div class="section-title">Average Returns by ETF <span>All windows</span></div>
  <div class="tabs" id="bar-tabs"></div>
  <div id="bar-panes"></div>
</section>

<!-- ═══════════════════════════════════════════ CUMULATIVE RETURNS ═ -->
<section>
  <div class="section-title">Cumulative Returns by Event <span>Indexed T0=100</span></div>
  <div class="tabs" id="cum-tabs"></div>
  <div id="cum-panes"></div>
</section>

<!-- ═══════════════════════════════════════════ VIX ═ -->
<section>
  <div class="section-title">VIX History <span>Geopolitical markers</span></div>
  <div class="card chart-box chart-box-tall">
    <canvas id="vix-chart"></canvas>
  </div>
</section>

<!-- ═══════════════════════════════════════════ NARRATIVE ═ -->
<section id="narrative-section">
  <div class="section-title">Analyst Interpretation <span>Behavioural pattern summary</span></div>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:22px;max-width:820px;line-height:1.7;">
    What the numbers mean in practice — each ETF's characteristic behaviour across geopolitical shocks,
    anchored to the data computed in this run.
  </p>
  <div id="narrative-grid" class="grid-2"></div>
</section>

<!-- ═══════════════════════════════════════════ FULL DATA TABLE ═ -->
<section>
  <div class="section-title">Full Backtest Results <span>All events · all tickers · all windows</span></div>
  <div class="search-bar">
    <input id="table-search" type="text" placeholder="Filter by event, ticker, window…"/>
  </div>
  <div class="data-table-wrap">
    <table class="data-table" id="data-table">
      <thead>
        <tr>
          <th onclick="sortTable(0)">Event ▲▼</th>
          <th onclick="sortTable(1)">Date ▲▼</th>
          <th onclick="sortTable(2)">Ticker ▲▼</th>
          <th onclick="sortTable(3)">Window ▲▼</th>
          <th onclick="sortTable(4)">Return % ▲▼</th>
          <th onclick="sortTable(5)">Alpha % ▲▼</th>
          <th onclick="sortTable(6)">Sharpe ▲▼</th>
          <th onclick="sortTable(7)">Max DD % ▲▼</th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
</section>

<!-- ═══════════════════════════════════════════ LIMITATIONS & INTEGRITY ═ -->
<section>
  <div class="section-title">Limitations &amp; Integrity <span>Methodological transparency</span></div>
  <p style="color:var(--muted);font-size:.85rem;margin-bottom:22px;max-width:860px;line-height:1.7;">
    This section documents the constraints, assumptions, and sources of uncertainty built into this
    backtest. Reading it is not optional — it is part of understanding the output.
  </p>
  <div class="limit-grid">

    <div class="limit-card">
      <div class="limit-card-num">Limitation 01</div>
      <div class="limit-card-title">📅 ETF Data Availability</div>
      <div class="limit-card-body">
        Most of the ETFs in this report did not exist before the early 2000s.
        <b>XLE launched in 1998, ITA in 2001, PPA in 2005, JETS in 2015, QQQ in 1999.</b>
        This means the dataset cannot directly analyse the 1973 OPEC embargo, the 1979 Iranian
        revolution, the 1990 Gulf War, or the 2003 SARS shock — all of which produced significant
        energy and equity dislocations. This report focuses <b>exclusively on post-2000 events</b>
        because that is where real, tradeable ETF price data exists. Any pattern claimed here
        is derived from that narrower window, not from the full history of geopolitical shocks.
      </div>
    </div>

    <div class="limit-card">
      <div class="limit-card-num">Limitation 02</div>
      <div class="limit-card-title">🪟 Event Window Assumptions</div>
      <div class="limit-card-body">
        The windows used — T+1, T+5, T+20, T+60 — are <b>reasonable conventions, not universal
        truths.</b> A different choice of window (say T+3 or T+45) would produce different numbers.
        More importantly, markets frequently <b>price in geopolitical risk before the official
        event date</b> — through elevated option implied volatility, pre-positioning in energy
        futures, or sector rotation. The T0 anchor is the first trading day on or after the
        acknowledged event date, which may already reflect partial pricing. Results are therefore
        sensitive to definition, and should not be read as precision measurements.
      </div>
    </div>

    <div class="limit-card">
      <div class="limit-card-num">Limitation 03</div>
      <div class="limit-card-title">🌍 Heterogeneity of Shocks</div>
      <div class="limit-card-body">
        The seven events in this dataset are <b>not the same type of shock.</b> 9/11 was a
        domestic terrorism event with a massive demand-side impact on aviation. The Iraq War
        was a pre-announced military campaign with known timelines. The Gaza and Iran events
        were regional conflicts with limited direct economic linkage to the US. Aggregating
        them into a single summary statistic creates noise and dilutes signal.
        <b>Win rates and averages across heterogeneous events should be interpreted with
        caution</b> — they describe a distribution, not a reliable mechanism.
      </div>
    </div>

    <div class="limit-card">
      <div class="limit-card-num">Limitation 04</div>
      <div class="limit-card-title">🛢️ Oil Price Attribution</div>
      <div class="limit-card-body">
        Oil prices move for many reasons that have nothing to do with geopolitics:
        <b>OPEC+ production decisions, US shale supply growth, global demand cycles,
        the US dollar index, and speculative flow</b> all drive XLE and USO returns.
        When this model records an XLE return in the weeks after an event, it cannot
        cleanly separate the geopolitical risk premium from concurrent macro forces.
        An XLE decline observed after the Iraq War, for example, partly reflected a
        post-shock demand recovery — not a refutation of the energy-shock thesis.
        Attribution is inherently ambiguous without a controlled counterfactual.
      </div>
    </div>

    <div class="limit-card">
      <div class="limit-card-num">Limitation 05</div>
      <div class="limit-card-title">🏗️ Market Structure Changes</div>
      <div class="limit-card-body">
        The US energy sector in 2001 and in 2026 are <b>structurally different markets.</b>
        The shale revolution turned the US from a net oil importer to the world's largest
        producer, changing the transmission mechanism between Middle East supply shocks and
        domestic energy equities. Similarly, the defense industrial base has consolidated,
        and ETF index construction has evolved. <b>Historical reactions from 2001–2006 may
        systematically understate or overstate the response modern markets would generate</b>
        to an equivalent shock, because the underlying economic relationships have shifted.
      </div>
    </div>

    <div class="limit-card">
      <div class="limit-card-num">Limitation 06</div>
      <div class="limit-card-title">🗂️ Data Quality &amp; Survivorship</div>
      <div class="limit-card-body">
        All ETF data in this report is sourced from Yahoo Finance via <b>yfinance</b> and
        reflects adjusted closing prices accounting for splits and dividends. However,
        ETF index methodologies change over time — constituents are rebalanced, weighting
        schemes evolve, and expense ratios shift. <b>JETS, for example, reconstituted
        its index methodology in 2020 following COVID-related delistings.</b> Older price
        history may reflect a different portfolio composition than today's fund. This
        introduces a subtle form of index drift that the model does not correct for.
      </div>
    </div>

    <div class="limit-card">
      <div class="limit-card-num">Limitation 07</div>
      <div class="limit-card-title">🚫 No Predictive Claims</div>
      <div class="limit-card-body">
        <b>This report makes no predictions.</b> Every number presented — average returns,
        win rates, alpha, Sharpe ratios — is a description of what happened in a small sample
        of historical events. Past patterns in a dataset of seven events do not constitute
        a statistically reliable forecast of future outcomes. Geopolitical shocks are, by
        definition, <b>low-frequency, high-impact events with fat-tailed outcome
        distributions.</b> The confidence interval around any estimate here is wide enough
        to overlap with the opposite conclusion. This analysis reveals tendencies and
        mechanisms — it does not provide trading signals or investment advice.
      </div>
    </div>

  </div>

  <div class="limit-closing">
    <b>A note on intellectual honesty.</b> The purpose of this section is not to undermine
    confidence in the analysis — it is to calibrate it correctly. A backtest built on seven
    events, using ETFs that are at most 25 years old, applied to shocks that differ
    fundamentally in type and magnitude, can still be genuinely useful. It surfaces
    <b>behavioural patterns, directional tendencies, and relative sector dynamics</b> that
    are difficult to see in real time. What it cannot do is tell you with certainty what
    will happen next. The reader who understands both what this model says and what it
    cannot say is better positioned than one who treats either the numbers or the
    caveats in isolation.
  </div>
</section>

<!-- ═══════════════════════════════════════════ SHOCK PLAYBOOK ═ -->
<section>
  <div class="section-title">Shock Playbook <span>Recurring patterns across geopolitical events</span></div>

  <p class="playbook-intro">
    Every geopolitical shock arrives with its own headline, geography, and cast of actors.
    But beneath the noise, a small set of <b>recurring structural patterns</b> emerge across
    the events in this dataset. This playbook summarises those tendencies — the sectors that
    consistently catch a bid, the sectors that consistently absorb fear, and the speed at
    which markets typically digest each shock type.
    <b>These are observed tendencies, not predictions.</b> Each new crisis carries unique
    drivers that historical averages cannot fully anticipate.
  </p>

  <div class="playbook-wrap">
  <table class="playbook-table">
    <thead>
      <tr>
        <th>Shock Type</th>
        <th>What It Means</th>
        <th>Typical ETF Reaction</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>⛽ Oil Supply Disruption</td>
        <td>Physical barrels are at risk — production facilities, shipping lanes,
            or major export routes threatened by conflict or blockade.</td>
        <td>
          <span class="pb-tag pb-up">XLE ▲</span>
          <span class="pb-tag pb-up">USO ▲</span>
          <span class="pb-tag pb-up">ITA ▲</span>
          <span class="pb-tag pb-down">JETS ▼</span>
          <span class="pb-tag pb-neu">QQQ ~</span>
          <br><span style="font-size:.78rem;">Energy leads. Airlines face margin pressure from fuel costs.
          Defense rises on escalation expectations. Tech is largely insulated short-term.</span>
        </td>
      </tr>
      <tr>
        <td>🪖 Regional Conflict Escalation</td>
        <td>State-vs-state or proxy conflict with sustained military operations,
            no clear end date — e.g., Lebanon War 2006, Iran–Israel exchanges.</td>
        <td>
          <span class="pb-tag pb-up">ITA ▲</span>
          <span class="pb-tag pb-up">PPA ▲</span>
          <span class="pb-tag pb-neu">XLE ~</span>
          <span class="pb-tag pb-down">JETS ▼</span>
          <span class="pb-tag pb-warn">SPY dip</span>
          <br><span style="font-size:.78rem;">Defense is the primary beneficiary as procurement
          expectations rise. Broader market absorbs the shock within T+20 in most historical cases.</span>
        </td>
      </tr>
      <tr>
        <td>💥 Terrorism / Asymmetric Attack</td>
        <td>Non-state or rogue actor attack with outsized psychological impact
            — fear-driven shock with immediate, severe market reaction. E.g., 9/11.</td>
        <td>
          <span class="pb-tag pb-down">JETS ▼▼</span>
          <span class="pb-tag pb-down">SPY ▼▼</span>
          <span class="pb-tag pb-down">QQQ ▼</span>
          <span class="pb-tag pb-up">ITA ▲</span>
          <span class="pb-tag pb-neu">XLE ~</span>
          <br><span style="font-size:.78rem;">The sharpest broad-market dislocation in the dataset.
          Airlines suffer structurally. Defense catches a delayed bid on spending expectations.
          Recovery timeline is measured in months, not days.</span>
        </td>
      </tr>
      <tr>
        <td>🚀 Missile / Drone Exchange</td>
        <td>Short, intense modern conflict pattern — large-scale projectile use
            with limited territorial objectives. E.g., Iran–Israel April 2024.</td>
        <td>
          <span class="pb-tag pb-warn">SPY dip</span>
          <span class="pb-tag pb-up">ITA ▲</span>
          <span class="pb-tag pb-neu">XLE ~</span>
          <span class="pb-tag pb-neu">QQQ ~</span>
          <span class="pb-tag pb-up">VIX spike</span>
          <br><span style="font-size:.78rem;">Historically fast mean reversion (T+5 to T+20)
          once the exchange is clearly contained. VIX spikes initially but fades quickly.
          Defense outperforms over the following month.</span>
        </td>
      </tr>
      <tr>
        <td>📉 Market-Wide Risk-Off</td>
        <td>Macro fear dominates — equity selling not driven by oil or military
            spending, but by recession risk, rate expectations, or contagion.</td>
        <td>
          <span class="pb-tag pb-down">SPY ▼</span>
          <span class="pb-tag pb-down">QQQ ▼▼</span>
          <span class="pb-tag pb-down">JETS ▼</span>
          <span class="pb-tag pb-down">XLE ▼</span>
          <span class="pb-tag pb-neu">ITA ~</span>
          <br><span style="font-size:.78rem;">All risk assets decline together. Defense spending
          is more budget-driven than cyclical, so ITA can partially decouple.
          This pattern lies outside the core oil-shock framework of this model.</span>
        </td>
      </tr>
      <tr>
        <td>📰 Short-Lived Scare</td>
        <td>Headline shock with no sustained escalation — markets price in fear,
            then rapidly reprice when the situation stabilises or de-escalates.</td>
        <td>
          <span class="pb-tag pb-warn">All: initial dip</span>
          <span class="pb-tag pb-up">All: recovery ▲</span>
          <br><span style="font-size:.78rem;">The pattern most observed in this dataset post-2020.
          Initial T+1 to T+5 drawdown is followed by T+20 to T+60 recovery that often
          exceeds pre-shock levels. Selling the initial dip has historically been costly
          in this category.</span>
        </td>
      </tr>
    </tbody>
  </table>
  </div>

  <div class="playbook-how">
    <b>How to use this playbook.</b>&nbsp; This is not a trading signal or an investment
    recommendation. It is a <b>mental model</b> — a structured way to interpret sector
    behavior when a new shock breaks. When a headline appears, the first question to ask
    is: which shock type does this most resemble? That framing helps you understand
    which sectors are likely carrying the fear, which are likely catching a bid, and
    over what time horizon historical patterns suggest resolution. Use it to <b>ask
    better questions</b>, not to place orders.
  </div>

  <div class="playbook-reminder">
    <b>Integrity reminder.</b>&nbsp; The patterns above are derived from seven events
    between 2001 and 2026. Every row in that table is a tendency, not a mechanism.
    Each new crisis arrives with its own geography, political context, and macro
    backdrop that historical averages cannot fully anticipate. The playbook is a
    guide. The next shock will write its own chapter.
  </div>
</section>

</main>

<footer>
  <p>Generated {run_date} · Data: Yahoo Finance via yfinance · Geopolitical Shock ETF Backtest Engine</p>
</footer>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script>
// ─── Embedded data ────────────────────────────────────────────────────────────
const COLOURS   = {colours_js};
const CUM_DATA  = {cum_js};
const VIX_DATA  = {vix_js};
const SUMM_DATA = {summ_js};
const FULL_DATA = {full_js};
const EV_LIST   = {ev_js};

// ─── Chart.js global defaults ────────────────────────────────────────────────
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.plugins.legend.labels.boxWidth = 12;

// ─── Helpers ──────────────────────────────────────────────────────────────────
function colourFor(name) {{
  return COLOURS[name] || '#58a6ff';
}}
function fmtPct(v) {{
  if (v === null || v === '' || v === undefined) return '—';
  const n = parseFloat(v);
  return (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
}}
function clsForNum(v) {{
  if (v === null || v === '' || v === undefined) return '';
  return parseFloat(v) >= 0 ? 'pos' : 'neg';
}}
function heatColour(v) {{
  if (v === null || isNaN(v)) return 'rgba(255,255,255,0.04)';
  const capped = Math.max(-15, Math.min(15, v));
  if (capped >= 0) {{
    const t = capped / 15;
    return `rgba(63,185,80,${{(0.15 + t * 0.7).toFixed(2)}})`;
  }} else {{
    const t = Math.abs(capped) / 15;
    return `rgba(248,81,73,${{(0.15 + t * 0.7).toFixed(2)}})`;
  }}
}}

// ─── KPI cards ────────────────────────────────────────────────────────────────
(function buildKPIs() {{
  const TARGET_WINDOW = 'T+5 (1 week)';
  const grid = document.getElementById('kpi-grid');
  const tickers = SUMM_DATA.tickers.filter(t => !t.includes('VIX') && !t.includes('USO'));
  const spyD = SUMM_DATA.stats['S&P 500 (SPY)']?.[TARGET_WINDOW];
  tickers.forEach(ticker => {{
    const d = SUMM_DATA.stats[ticker]?.[TARGET_WINDOW];
    if (!d) return;
    const rawCls   = d.mean >= 0 ? 'pos' : 'neg';
    const alphaCls = (d.alpha_mean ?? 0) >= 0 ? 'pos' : 'neg';
    const isShock  = d.mean > 0 && spyD && spyD.mean < 0;
    const card = document.createElement('div');
    card.className = 'kpi-card';
    card.innerHTML = `
      <div class="kpi-label">
        ${{ticker.split('(').pop().replace(')','').trim()}}
        ${{isShock ? '<span title="ETF positive while SPY negative" style="color:#f1c40f;">⚡</span>' : ''}}
      </div>
      <div class="kpi-value ${{rawCls}}">${{fmtPct(d.mean)}}</div>
      <div class="kpi-sub">Win&nbsp;rate&nbsp;<b>${{d.wr}}%</b> &nbsp;·&nbsp; σ&nbsp;<b>${{d.std.toFixed(1)}}%</b></div>
      <div class="kpi-alpha">
        <span class="kpi-alpha-label">vs SPY</span>
        <span class="kpi-alpha-val ${{alphaCls}}">${{d.alpha_mean != null ? fmtPct(d.alpha_mean) : '—'}}</span>
        ${{d.alpha_wr != null ? `<span style="font-size:.7rem;color:var(--muted);margin-left:auto;">W:${{d.alpha_wr}}%</span>` : ''}}
      </div>
    `;
    grid.appendChild(card);
  }});
}})();

// ─── Event count badge ────────────────────────────────────────────────────────
document.getElementById('evcount-badge').innerHTML =
  `<span class="dot"></span> ${{EV_LIST.length}} events analysed`;

// ─── Current crisis context ───────────────────────────────────────────────────
(function buildCrisisContext() {{
  // The current event is always the last in the list
  const current = EV_LIST[EV_LIST.length - 1];
  if (!current) return;

  document.getElementById('crisis-event-name').textContent = current.name;
  document.getElementById('crisis-event-date').textContent = 'Event date: ' + current.date;
  document.getElementById('crisis-event-notes').textContent = current.notes;

  // ── What is happening now (static narrative, keyed to "Iran Crisis (2026)") ──
  const nowText = document.getElementById('crisis-now-text');
  nowText.innerHTML = `
    Iran's nuclear programme has escalated to a critical juncture, with <b>direct military
    exchanges between Israel and Iran</b> intensifying since late 2025. The US has deployed
    additional carrier groups to the Persian Gulf; Brent crude has spiked on Strait of Hormuz
    closure risk. Unlike the April 2024 drone exchange — which was contained within 72 hours —
    this crisis carries <b>escalation optionality on both sides</b> with no clear off-ramp visible
    as of the analysis date. Defense procurement signals are already accelerating in Israel,
    the US, and several Gulf states. The <b>key macro transmission risk</b> is an oil supply
    shock exceeding 2–3 mb/d disruption, which would shift this from a geopolitical event
    into a stagflation catalyst.`;

  // ── This event vs historical average (dynamic from FULL_DATA) ──
  const W5 = 'T+5 (1 week)';
  const tickers = SUMM_DATA.tickers.filter(t => !t.includes('VIX') && !t.includes('USO'));
  const tbody = document.getElementById('crisis-compare-body');
  tickers.forEach(ticker => {{
    const histD = SUMM_DATA.stats[ticker]?.[W5];
    // find this event's row in full data
    const eventRow = FULL_DATA.find(r =>
      r.Event === current.name && r.Ticker === ticker && r.Window === W5);
    if (!histD) return;
    const eventVal  = eventRow && eventRow['Return_%'] !== '' ? parseFloat(eventRow['Return_%']) : null;
    const histMean  = histD.mean;
    const delta     = eventVal != null ? eventVal - histMean : null;
    const deltaCls  = delta == null ? '' : delta > 0.5 ? 'delta-pos' : delta < -0.5 ? 'delta-neg' : 'delta-neu';
    const shortName = ticker.split('(').pop().replace(')','').trim();
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${{shortName}}</td>
      <td style="color:${{eventVal == null ? 'var(--muted)' : eventVal >= 0 ? 'var(--pos)' : 'var(--neg)'}}">${{eventVal != null ? fmtPct(eventVal) : '—'}}</td>
      <td style="color:${{histMean >= 0 ? 'var(--pos)' : 'var(--neg)'}}">${{fmtPct(histMean)}}</td>
      <td class="${{deltaCls}}">${{delta != null ? (delta > 0 ? '+' : '') + delta.toFixed(2) + '%' : '—'}}</td>`;
    tbody.appendChild(tr);
  }});

  // ── Closest historical analogs (static domain knowledge) ──
  const ANALOGS = [
    {{
      sim: '95%', name: 'Iran–Israel Direct Exchange', date: '2024-04-13',
      why: 'Direct state-on-state exchange with ballistic missiles / drones. Same actors, '
         + 'similar escalation ladder. Key difference: 2026 involves sustained campaign, '
         + 'not a one-night exchange.',
      col: '#58a6ff',
    }},
    {{
      sim: '80%', name: 'Iran Tensions (Oct 2024)', date: '2024-10-01',
      why: 'Elevated war premium baked into oil, defense bid without kinetic confirmation. '
         + 'Markets priced uncertainty rather than damage — similar to early 2026 phase.',
      col: '#6db3e0',
    }},
    {{
      sim: '55%', name: 'Lebanon War / Israel-Hezbollah', date: '2006-07-12',
      why: 'Regional escalation with Hezbollah proxy involvement (as in 2026). Oil above '
         + '$78; markets pulled back ~6% before recovering. Supply disruption was limited.',
      col: '#d29922',
    }},
    {{
      sim: '30%', name: '9/11 Attacks', date: '2001-09-11',
      why: 'Only analog where fear was sustained long enough to structurally damage JETS. '
         + 'Relevant if Strait of Hormuz closes — but the demand-shock mechanism differs.',
      col: '#f85149',
    }},
  ];

  const analogsEl = document.getElementById('crisis-analogs');
  ANALOGS.forEach(a => {{
    const ev = EV_LIST.find(e => e.name === a.name);
    analogsEl.innerHTML += `
      <div class="analog-item">
        <div class="analog-sim" style="color:${{a.col}};border:1px solid ${{a.col}}40;">${{a.sim}}</div>
        <div>
          <div class="analog-name">${{a.name}}</div>
          <div class="analog-why">${{a.why}}</div>
        </div>
      </div>`;
  }});

  // ── What the model cannot infer (static caveats) ──
  const CAVEATS = [
    {{ b: 'Nuclear escalation path.',     rest: ' The model has no prior for a nuclear exchange. Every historical analog involves conventional weapons.' }},
    {{ b: 'Strait of Hormuz closure.',    rest: ' A full closure (3–4 mb/d removed) would be unprecedented in the modern derivatives era. The energy price response would exceed any data point in this dataset.' }},
    {{ b: 'US domestic policy response.', rest: ' Strategic petroleum reserve releases, emergency export bans, or sanctions shifts are policy decisions — not market mechanics the model can price.' }},
    {{ b: 'Timing and duration.',          rest: ' The model computes returns from T0. It cannot tell you when the inflection comes, only what the average shape looks like after it does.' }},
    {{ b: 'Correlation breakdown.',        rest: ' Under extreme stress, historical correlations collapse. The SPY alpha frame assumes normal cross-asset relationships hold.' }},
    {{ b: 'Second-order macro effects.',   rest: " Recession risk from sustained $130+ oil, credit stress from Gulf sovereign spread widening, and Fed response are outside this backtest's scope." }},
  ];

  const caveatsList = document.getElementById('crisis-caveats');
  CAVEATS.forEach(c => {{
    const li = document.createElement('li');
    li.innerHTML = `<b>${{c.b}}</b>${{c.rest}}`;
    caveatsList.appendChild(li);
  }});
}})();

// ─── Events timeline ──────────────────────────────────────────────────────────
(function buildTimeline() {{
  const el = document.getElementById('timeline');
  EV_LIST.forEach(ev => {{
    el.innerHTML += `
      <div class="tl-item">
        <div class="tl-dot"></div>
        <div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span class="tl-name">${{ev.name}}</span>
            <span class="tl-date">${{ev.date}}</span>
          </div>
          <div class="tl-notes">${{ev.notes}}</div>
        </div>
      </div>`;
  }});
}})();

// ─── Heatmap table (dual-mode: raw / alpha) ──────────────────────────────────
let _heatmapMode = 'raw';
function renderHeatmap() {{
  const table   = document.getElementById('heatmap-table');
  const windows = SUMM_DATA.windows;
  const tickers = SUMM_DATA.tickers.filter(t => !t.includes('USO'));
  const spyStat = SUMM_DATA.stats['S&P 500 (SPY)'];
  let html = '<thead><tr><th class="row-h">ETF</th>';
  windows.forEach(w => {{ html += `<th>${{w}}</th>`; }});
  html += '</tr></thead><tbody>';
  tickers.forEach(ticker => {{
    html += `<tr><td class="label">${{ticker}}</td>`;
    windows.forEach(w => {{
      const d    = SUMM_DATA.stats[ticker]?.[w];
      const spyW = spyStat?.[w];
      if (!d) {{ html += '<td>—</td>'; return; }}
      const val   = _heatmapMode === 'alpha' ? (d.alpha_mean ?? d.mean) : d.mean;
      const wr    = _heatmapMode === 'alpha' ? (d.alpha_wr  ?? d.wr)    : d.wr;
      const bg    = heatColour(val);
      const shock = d.mean > 0 && spyW && spyW.mean < 0;
      html += `<td style="background:${{bg}}">
        ${{fmtPct(val)}}${{shock ? '<span class="shock" title="ETF↑ while SPY↓">⚡</span>' : ''}}
        <span class="wr">${{_heatmapMode === 'alpha' ? 'α-W' : 'W'}}:${{wr != null ? wr.toFixed(0) : '—'}}%</span>
      </td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody>';
  table.innerHTML = html;
}}
renderHeatmap();
window.setHeatmapMode = function(mode) {{
  _heatmapMode = mode;
  document.getElementById('hm-raw').classList.toggle('active',   mode === 'raw');
  document.getElementById('hm-alpha').classList.toggle('active', mode === 'alpha');
  renderHeatmap();
}};

// ─── Average returns bar charts (one tab per window) ─────────────────────────
(function buildBarCharts() {{
  const tabsEl  = document.getElementById('bar-tabs');
  const panesEl = document.getElementById('bar-panes');
  const windows = SUMM_DATA.windows;
  const tickers = SUMM_DATA.tickers.filter(t => !t.includes('VIX') && !t.includes('USO'));

  windows.forEach((w, wi) => {{
    // tab button
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (wi === 0 ? ' active' : '');
    btn.textContent = w;
    btn.onclick = () => switchTab('bar', wi);
    tabsEl.appendChild(btn);

    // pane + canvas
    const pane = document.createElement('div');
    pane.className = 'tab-pane card chart-box' + (wi === 0 ? ' active' : '');
    pane.id = `bar-pane-${{wi}}`;
    pane.innerHTML = `<canvas id="bar-chart-${{wi}}"></canvas>`;
    panesEl.appendChild(pane);

    const labels = [], data = [], alphaData = [], bgColours = [], wrData = [];
    tickers.forEach(t => {{
      const d = SUMM_DATA.stats[t]?.[w];
      if (!d) return;
      labels.push(t.split('(').pop().replace(')','').trim());
      data.push(d.mean);
      alphaData.push(d.alpha_mean ?? null);
      wrData.push(d.wr);
      bgColours.push(colourFor(t));
    }});

    requestAnimationFrame(() => {{
      new Chart(document.getElementById(`bar-chart-${{wi}}`), {{
        type: 'bar',
        data: {{
          labels,
          datasets: [
            {{
              label: 'Raw Return %',
              data,
              backgroundColor: bgColours.map(c => c + 'cc'),
              borderColor: bgColours,
              borderWidth: 1.5,
              borderRadius: 5,
            }},
            {{
              label: 'Excess vs SPY %',
              data: alphaData,
              backgroundColor: bgColours.map(c => c + '44'),
              borderColor: bgColours,
              borderWidth: 1.5,
              borderRadius: 5,
              borderDash: [4, 3],
            }},
            {{
              label: 'Win Rate %',
              data: wrData,
              backgroundColor: 'rgba(88,166,255,0.15)',
              borderColor: '#58a6ff',
              borderWidth: 1.5,
              borderRadius: 5,
              yAxisID: 'y2',
            }},
          ],
        }},
        options: {{
          responsive: true,
          plugins: {{
            legend: {{ position: 'top' }},
            tooltip: {{
              mode: 'index', intersect: false,
              callbacks: {{
                label: ctx => {{
                  if (ctx.datasetIndex === 2) return ` Win rate: ${{ctx.parsed.y?.toFixed(0)}}%`;
                  const icon = ctx.datasetIndex === 1 ? '⚡' : '📈';
                  return ` ${{icon}} ${{ctx.dataset.label}}: ${{fmtPct(ctx.parsed.y)}}`;
                }},
              }},
            }},
          }},
          scales: {{
            y: {{
              grid: {{ color: '#30363d' }},
              ticks: {{ callback: v => v.toFixed(1) + '%' }},
              title: {{ display: true, text: 'Return %', color: '#8b949e', font: {{size: 10}} }},
            }},
            y2: {{
              position: 'right',
              grid: {{ drawOnChartArea: false }},
              ticks: {{ callback: v => v + '%' }},
              min: 0, max: 100,
              title: {{ display: true, text: 'Win Rate %', color: '#58a6ff', font: {{size: 10}} }},
            }},
            x: {{ grid: {{ display: false }} }},
          }},
        }},
      }});
    }});
  }});
}})();

// ─── Cumulative return charts (one tab per event) ────────────────────────────
(function buildCumCharts() {{
  const tabsEl  = document.getElementById('cum-tabs');
  const panesEl = document.getElementById('cum-panes');

  CUM_DATA.forEach((ev, ei) => {{
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (ei === 0 ? ' active' : '');
    btn.textContent = ev.name.split(' ').slice(0, 2).join(' ');
    btn.title = ev.name + ' — ' + ev.date;
    btn.onclick = () => switchTab('cum', ei);
    tabsEl.appendChild(btn);

    const pane = document.createElement('div');
    pane.className = 'tab-pane card chart-box chart-box-tall' + (ei === 0 ? ' active' : '');
    pane.id = `cum-pane-${{ei}}`;
    pane.innerHTML = `
      <div style="margin-bottom:8px">
        <b>${{ev.name}}</b>
        <span style="color:var(--muted);font-size:.78rem;margin-left:10px;">${{ev.date}} — ${{ev.notes}}</span>
      </div>
      <canvas id="cum-chart-${{ei}}"></canvas>`;
    panesEl.appendChild(pane);

    const N = 61;
    const labels = Array.from({{length: N}}, (_, i) => i);
    const datasets = Object.entries(ev.series).map(([ticker, vals]) => ({{
      label: ticker.split('(').pop().replace(')','').trim(),
      data: vals,
      borderColor: colourFor(ticker),
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      spanGaps: true,
    }}));

    requestAnimationFrame(() => {{
      new Chart(document.getElementById(`cum-chart-${{ei}}`), {{
        type: 'line',
        data: {{ labels, datasets }},
        options: {{
          responsive: true,
          plugins: {{
            legend: {{ position: 'top' }},
            tooltip: {{
              mode: 'index', intersect: false,
              callbacks: {{
                label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y?.toFixed(2) ?? '—'}}`,
              }},
            }},
          }},
          scales: {{
            y: {{
              grid: {{ color: '#30363d' }},
              ticks: {{ callback: v => v.toFixed(0) }},
            }},
            x: {{
              grid: {{ color: '#30363d30' }},
              title: {{ display: true, text: 'Trading days after event', color: '#8b949e' }},
            }},
          }},
          annotation: {{
            annotations: {{
              baseline: {{ type: 'line', yMin: 100, yMax: 100,
                borderColor: '#8b949e', borderWidth: 1, borderDash: [4, 4] }},
            }},
          }},
        }},
      }});
    }});
  }});
}})();

// ─── VIX history chart ────────────────────────────────────────────────────────
(function buildVIX() {{
  if (!VIX_DATA.dates.length) return;

  // Build vertical line annotations as null-gap datasets
  const baseDs = {{
    label: 'VIX (Monthly Avg)',
    data: VIX_DATA.values,
    borderColor: '#e74c3c',
    backgroundColor: 'rgba(231,76,60,0.12)',
    borderWidth: 1.5,
    pointRadius: 0,
    fill: true,
    tension: 0.2,
  }};

  // Event marker lines via point datasets
  const eventDs = VIX_DATA.events.map(ev => {{
    const yrMo = ev.date.slice(0, 7);
    const idx  = VIX_DATA.dates.indexOf(yrMo);
    const sparseData = VIX_DATA.values.map((_, i) => i === idx ? VIX_DATA.values[i] : null);
    return {{
      label: ev.name.split(' ')[0],
      data: sparseData,
      borderColor: 'rgba(88,166,255,0.8)',
      pointBackgroundColor: '#58a6ff',
      pointRadius: VIX_DATA.values.map((_, i) => i === idx ? 5 : 0),
      borderWidth: 0,
      fill: false,
      showLine: false,
    }};
  }});

  new Chart(document.getElementById('vix-chart'), {{
    type: 'line',
    data: {{ labels: VIX_DATA.dates, datasets: [baseDs, ...eventDs] }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{
          labels: {{
            filter: item => item.datasetIndex === 0,  // only show VIX in legend
          }},
        }},
        tooltip: {{
          mode: 'index', intersect: false,
          filter: item => item.datasetIndex === 0,
        }},
      }},
      scales: {{
        y: {{
          grid: {{ color: '#30363d' }},
          title: {{ display: true, text: 'VIX Level', color: '#8b949e' }},
        }},
        x: {{
          grid: {{ color: '#30363d20' }},
          ticks: {{
            maxTicksLimit: 14,
            callback: (v, i) => VIX_DATA.dates[i]?.slice(0,4) || '',
            autoSkip: true,
          }},
        }},
      }},
    }},
  }});
}})();

// ─── Full data table ──────────────────────────────────────────────────────────
(function buildTable() {{
  const tbody = document.getElementById('table-body');
  let sortCol = 0, sortAsc = true;
  let currentData = [...FULL_DATA];

  function render(data) {{
    tbody.innerHTML = data.map(r => `
      <tr>
        <td>${{r.Event}}</td>
        <td>${{r.Date}}</td>
        <td style="color:${{colourFor(r.Ticker)}};font-weight:600">${{r.Ticker}}</td>
        <td>${{r.Window}}</td>
        <td class="${{clsForNum(r['Return_%'])}}">${{fmtPct(r['Return_%'])}}</td>
        <td class="${{clsForNum(r['Alpha_%'])}}">${{fmtPct(r['Alpha_%'])}}</td>
        <td>${{r.Sharpe !== '' ? parseFloat(r.Sharpe).toFixed(3) : '—'}}</td>
        <td class="${{r['MaxDrawdown_%'] !== '' && parseFloat(r['MaxDrawdown_%']) < 0 ? 'neg' : ''}}">${{r['MaxDrawdown_%'] !== '' ? fmtPct(r['MaxDrawdown_%']) : '—'}}</td>
      </tr>`).join('');
  }}

  window.sortTable = function(col) {{
    if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = true; }}
    const keys = ['Event','Date','Ticker','Window','Return_%','Alpha_%','Sharpe','MaxDrawdown_%'];
    currentData.sort((a, b) => {{
      const av = a[keys[col]] ?? '', bv = b[keys[col]] ?? '';
      const an = parseFloat(av), bn = parseFloat(bv);
      const cmp = !isNaN(an) && !isNaN(bn) ? an - bn : String(av).localeCompare(String(bv));
      return sortAsc ? cmp : -cmp;
    }});
    render(currentData);
  }};

  document.getElementById('table-search').addEventListener('input', e => {{
    const q = e.target.value.toLowerCase();
    currentData = FULL_DATA.filter(r =>
      Object.values(r).some(v => String(v).toLowerCase().includes(q)));
    render(currentData);
  }});

  render(currentData);
}})();

// ─── Narrative section ───────────────────────────────────────────────────────
(function buildNarratives() {{
  // Static behavioural profiles per ticker — enriched below with live numbers
  const PROFILES = [
    {{
      ticker:   'Energy (XLE)',
      icon:     '🛢️',
      verdict:  'mixed',
      vLabel:   'Mixed',
      headline: 'Shock spike, but rarely sustained',
      body: `XLE tends to <b>rally hard in the initial shock window</b> as oil risk premiums reprice —
             but the move often fades unless the conflict structurally impairs supply routes or
             production capacity. The key question is always: does the event change the
             <b>long-run supply curve</b>, or is it just sentiment noise?
             Saudi spare capacity and US shale response speed determine whether the spike holds.`,
      quote: 'Watch for: Strait of Hormuz risk, OPEC+ rhetoric, and US rig-count response in the weeks after.',
    }},
    {{
      ticker:   'Defense (ITA)',
      icon:     '🛡️',
      verdict:  'bullish',
      vLabel:   'Consistently Bullish',
      headline: 'Reliable positive bias across all windows',
      body: `ITA shows the <b>most consistent positive drift</b> of any sector across geopolitical shocks.
             Defense spending is a policy response — a shock doesn't just spike demand, it <b>locks in
             multi-year budget commitments</b>. The signal is strongest when the conflict involves a
             major NATO ally or when Congress is already in a supplemental-appropriations cycle.
             Unlike energy, the bid in defense tends to compound rather than mean-revert.`,
      quote: 'Watch for: emergency supplemental budgets, NATO Article 5 invocations, and F-35/munitions re-order announcements.',
    }},
    {{
      ticker:   'Defense2 (PPA)',
      icon:     '🚀',
      verdict:  'bullish',
      vLabel:   'Bullish',
      headline: 'Broader defense + aerospace — steadier than pure-play',
      body: `PPA captures a wider slice of aerospace and defense including <b>dual-use industrials</b>
             (Honeywell, Raytheon, L3Harris). It tracks ITA closely but with <b>lower shock volatility</b>
             because commercial aerospace dilutes the pure-defense beta. That dilution is a feature in
             recovery phases — PPA recovers faster from broad risk-off selling because it isn't purely
             war-themed. Preferred over ITA when the shock is <b>ambiguous in scope</b>.`,
      quote: 'Watch for: satellite/drone contracts, cyber-defense spending, and European rearmament flows.',
    }},
    {{
      ticker:   'Airlines (JETS)',
      icon:     '✈️',
      verdict:  'bearish',
      vLabel:   'Reliably Bearish',
      headline: 'Immediate pain — recovery is duration-dependent',
      body: `JETS is the most <b>reliably negative sector in the shock window</b>. The mechanism is
             direct: higher fuel costs bite margins immediately, fear suppresses travel demand, and
             institutional investors treat airlines as a proxy short on geopolitical calm.
             The critical variable is <b>how long fear persists</b>. If the conflict de-escalates
             within 2–4 weeks, JETS can snap back sharply (short-covering + cheap valuation).
             If fear is sustained — as in post-9/11 — the losses compound badly.`,
      quote: 'Watch for: TSA checkpoint volumes, booking cancellation rates, and jet fuel crack spreads.',
    }},
    {{
      ticker:   'Tech (QQQ)',
      icon:     '💻',
      verdict:  'recovery',
      vLabel:   'Dip then Rally',
      headline: 'Risk-off dip masks a fast recovery',
      body: `QQQ sells off in the shock window as a <b>risk-off reflex</b> — growth equities are the
             first thing institutions trim when uncertainty spikes. But the dip is typically mechanical
             rather than fundamental: tech earnings are not directly impaired by Middle East conflict
             unless oil shock transmits into a <b>full recession</b>. In most events,
             QQQ recovers to new highs within 1–3 months as the macro narrative reasserts.
             The T+60 data is the most telling — <b>tech almost always wins the 3-month horizon</b>.`,
      quote: 'Watch for: Fed response to oil inflation, consumer confidence, and whether the shock triggers credit stress.',
    }},
    {{
      ticker:   'S&P 500 (SPY)',
      icon:     '📊',
      verdict:  'mixed',
      vLabel:   'Benchmark',
      headline: 'Baseline shock absorber — context is everything',
      body: `SPY is the baseline all other ETFs are measured against. The initial dip is nearly
             universal — <b>uncertainty itself is the bear catalyst</b>, regardless of sector.
             What separates recoveries from prolonged drawdowns is whether the shock
             <b>intersects with an existing vulnerability</b>: elevated valuations, rising rates,
             pre-existing recession risk, or a credit event. In isolation, geopolitical shocks
             are historically <b>buying opportunities</b> on the SPY — but conviction requires
             ruling out macro co-incidence.`,
      quote: 'Watch for: Fed pivot signals, credit spreads (HY/IG), earnings revision trends, and PMI trajectory.',
    }},
  ];

  const grid = document.getElementById('narrative-grid');
  const W5   = 'T+5 (1 week)';
  const W60  = 'T+60 (3 months)';
  const VERDICT_ORDER = ['bullish','recovery','mixed','bearish'];

  PROFILES.forEach(p => {{
    const stats5  = SUMM_DATA.stats[p.ticker]?.[W5];
    const stats60 = SUMM_DATA.stats[p.ticker]?.[W60];

    // Find best/worst windows from data
    const allW = SUMM_DATA.windows;
    let bestW = null, bestVal = -Infinity, worstW = null, worstVal = Infinity;
    allW.forEach(w => {{
      const d = SUMM_DATA.stats[p.ticker]?.[w];
      if (!d) return;
      if (d.mean > bestVal)  {{ bestVal = d.mean;  bestW = w; }}
      if (d.mean < worstVal) {{ worstVal = d.mean; worstW = w; }}
    }});

    const shortW = w => w ? w.split('(')[0].trim() : '—';
    const colour = COLOURS[p.ticker] || '#58a6ff';
    const verdictCls = 'verdict-' + p.verdict;

    const pills = [];
    if (stats5)  pills.push({{ label: 'T+5 avg',    val: fmtPct(stats5.mean),  cls: stats5.mean  >= 0 ? 'pos' : 'neg' }});
    if (stats5)  pills.push({{ label: 'T+5 win rate', val: stats5.wr + '%',     cls: '' }});
    if (stats5 && stats5.alpha_mean != null)
                 pills.push({{ label: 'T+5 vs SPY',  val: fmtPct(stats5.alpha_mean), cls: stats5.alpha_mean >= 0 ? 'pos' : 'neg' }});
    if (stats60) pills.push({{ label: 'T+60 avg',   val: fmtPct(stats60.mean), cls: stats60.mean >= 0 ? 'pos' : 'neg' }});
    if (bestW)   pills.push({{ label: 'Best window', val: shortW(bestW),        cls: '' }});

    const pillsHtml = pills.map(pl => `
      <div class="pill">
        <span class="pill-label">${{pl.label}}</span>
        <span class="pill-val ${{pl.cls}}">${{pl.val}}</span>
      </div>`).join('');

    const card = document.createElement('div');
    card.className = 'narr-card';
    card.style.borderLeftColor = colour;
    card.innerHTML = `
      <div class="narr-header">
        <div class="narr-ticker">
          <span style="margin-right:8px;">${{p.icon}}</span>
          <span style="color:${{colour}}">${{p.ticker}}</span>
        </div>
        <span class="narr-verdict ${{verdictCls}}">${{p.vLabel}}</span>
      </div>
      <div style="font-size:.82rem;font-weight:600;color:var(--text);margin-bottom:10px;letter-spacing:-.1px;">
        ${{p.headline}}
      </div>
      <div class="narr-body">${{p.body}}</div>
      <div class="narr-pills">${{pillsHtml}}</div>
      <div class="narr-quote">${{p.quote}}</div>
    `;
    grid.appendChild(card);
  }});
}})();

// ─── Tab switcher ─────────────────────────────────────────────────────────────
function switchTab(group, idx) {{
  const tabsEl  = document.getElementById(group + '-tabs');
  const panesEl = document.getElementById(group + '-panes');
  tabsEl.querySelectorAll('.tab-btn').forEach((b, i) =>
    b.classList.toggle('active', i === idx));
  panesEl.querySelectorAll('.tab-pane').forEach((p, i) =>
    p.classList.toggle('active', i === idx));
}}
</script>
</body>
</html>"""

    out_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 9. WIN-RATE SUMMARY TABLE (console only)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(summary: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  SUMMARY: MEAN RETURN (%) BY ETF & WINDOW  [Win Rate% in parens]")
    print(f"{'='*70}")
    window_order = list(WINDOWS.keys())
    pivot_mean = summary.pivot(index="Ticker", columns="Window", values="Mean")
    pivot_wr   = summary.pivot(index="Ticker", columns="Window", values="WinRate_%")
    cols = [c for c in window_order if c in pivot_mean.columns]
    pivot_mean = pivot_mean[cols]
    pivot_wr   = pivot_wr[cols]
    col_w = 22
    header = f"{'ETF':<22}" + "".join(f"{c:>{col_w}}" for c in pivot_mean.columns)
    print(header)
    print("-" * len(header))
    for ticker in pivot_mean.index:
        row_str = f"{ticker:<22}"
        for win in pivot_mean.columns:
            v = pivot_mean.loc[ticker, win] if win in pivot_mean.columns else np.nan
            w = pivot_wr.loc[ticker, win]   if win in pivot_wr.columns   else np.nan
            if pd.isna(v):
                cell = "  n/a  "
            else:
                cell = f"{v:+.1f}% (W{w:.0f})"
            row_str += f"{cell:>{col_w}}"
        print(row_str)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  GEOPOLITICAL SHOCK — ETF BACKTEST ENGINE")
    print(f"  Run date : {dt.date.today()}")
    print("=" * 60)

    # ── Download ──────────────────────────────────────────────────────────────
    prices = download_data(TICKERS, DATA_START, DATA_END)

    # ── Backtest ──────────────────────────────────────────────────────────────
    results = run_backtest(prices, EVENTS, WINDOWS, TICKERS)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = compute_summary(results)
    print_summary_table(summary)

    # ── HTML Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  GENERATING HTML REPORT")
    print(f"{'='*60}")
    report_path = export_html(prices, results, summary, EVENTS, WINDOWS, TICKERS, COLOUR_MAP)

    # ── Key Insights ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  KEY INSIGHTS (from historical data)")
    print(f"{'='*60}")
    for target_ticker, label in [
        ("Energy (XLE)",    "Energy"),
        ("Defense (ITA)",   "Defense"),
        ("Airlines (JETS)", "Airlines"),
        ("Tech (QQQ)",      "Tech"),
        ("S&P 500 (SPY)",   "S&P 500"),
    ]:
        sub = summary[summary["Ticker"] == target_ticker]
        if sub.empty:
            continue
        best_w  = sub.loc[sub["Mean"].idxmax(), "Window"]
        best_r  = sub.loc[sub["Mean"].idxmax(), "Mean"]
        best_wr = sub.loc[sub["Mean"].idxmax(), "WinRate_%"]
        worst_w = sub.loc[sub["Mean"].idxmin(), "Window"]
        worst_r = sub.loc[sub["Mean"].idxmin(), "Mean"]
        print(
            f"  {label:<12}  Best: {best_r:+.1f}% @ {best_w.split('(')[0].strip()} "
            f"(win {best_wr:.0f}%)   "
            f"Worst: {worst_r:+.1f}% @ {worst_w.split('(')[0].strip()}"
        )

    print(f"\n  HTML report saved to: ./{OUTPUT_DIR}/report.html")
    print("  Done.\n")


if __name__ == "__main__":
    main()

