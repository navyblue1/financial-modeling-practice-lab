#!/usr/bin/env python3
"""
Run the W-bottom demo strategy on Magnificent Seven stocks YTD.

The script downloads daily OHLCV data from Yahoo Finance's chart endpoint,
applies the same W-bottom detector used in w_bottom_demo.py, and exports
summary CSVs plus a results chart.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd

from w_bottom_demo import StrategyConfig, backtest, detect_w_bottoms


MAG7 = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def unix_midnight_utc(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())


def yahoo_chart_url(symbol: str, start_date: date, end_date_exclusive: date) -> str:
    params = {
        "period1": str(unix_midnight_utc(start_date)),
        "period2": str(unix_midnight_utc(end_date_exclusive)),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{urllib.parse.urlencode(params)}"


def fetch_yahoo_daily(symbol: str, start_date: date, end_date_exclusive: date) -> tuple[pd.DataFrame, dict[str, object]]:
    url = yahoo_chart_url(symbol, start_date, end_date_exclusive)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 WBottomAudit/1.0",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo Finance returned an error for {symbol}: {error}")

    result = chart.get("result") or []
    if not result:
        raise RuntimeError(f"No chart data returned for {symbol}")

    data = result[0]
    timestamps = data.get("timestamp") or []
    quote = (data.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (data.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    meta = data.get("meta", {})

    rows = []
    for i, ts in enumerate(timestamps):
        try:
            row = {
                "Date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d"),
                "Open": quote["open"][i],
                "High": quote["high"][i],
                "Low": quote["low"][i],
                "Close": quote["close"][i],
                "Adj Close": adjclose[i] if i < len(adjclose) else None,
                "Volume": quote["volume"][i],
            }
        except (KeyError, IndexError):
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No usable rows returned for {symbol}")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    df["Volume"] = df["Volume"].astype("int64")

    strategy_df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    return strategy_df, meta


def percent(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value * 100.0


def max_drawdown_pct(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "drawdown_pct" not in equity_df:
        return 0.0
    return float(equity_df["drawdown_pct"].min()) * 100.0


def run_audit(output_dir: Path, start_date: date, end_date_exclusive: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices_dir = output_dir / "mag7_ytd_prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    cfg = StrategyConfig()
    summary_rows: list[dict[str, object]] = []
    all_patterns: list[dict[str, object]] = []
    all_trades: list[dict[str, object]] = []

    for symbol, company in MAG7.items():
        df, meta = fetch_yahoo_daily(symbol, start_date, end_date_exclusive)
        df.to_csv(prices_dir / f"{symbol}_ytd_ohlcv.csv", index=False)

        patterns = detect_w_bottoms(df, symbol, cfg)
        trades_df, equity_df, _tickets_df = backtest(df, patterns, cfg)

        for pattern in patterns:
            all_patterns.append(asdict(pattern))

        if not trades_df.empty:
            trades_for_symbol = trades_df.copy()
            trades_for_symbol.insert(1, "company", company)
            all_trades.extend(trades_for_symbol.to_dict("records"))

        prev_close = float(meta.get("chartPreviousClose") or df["Close"].iloc[0])
        first_close = float(df["Close"].iloc[0])
        last_close = float(df["Close"].iloc[-1])
        ytd_return = last_close / prev_close - 1.0 if prev_close > 0 else float("nan")

        ending_equity = cfg.starting_cash if equity_df.empty else float(equity_df["equity"].iloc[-1])
        strategy_return = ending_equity / cfg.starting_cash - 1.0

        if trades_df.empty:
            closed_trades = trades_df
            wins = 0
            hit_rate = None
            total_pnl = 0.0
        else:
            closed_trades = trades_df[trades_df["exit_date"].astype(str) != ""]
            pnl_series = pd.to_numeric(closed_trades["pnl"], errors="coerce") if not closed_trades.empty else pd.Series(dtype=float)
            wins = int((pnl_series > 0).sum())
            hit_rate = wins / len(closed_trades) if len(closed_trades) else None
            total_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0

        summary_rows.append(
            {
                "symbol": symbol,
                "company": company,
                "first_data_date": df["Date"].iloc[0],
                "last_data_date": df["Date"].iloc[-1],
                "prior_close": round(prev_close, 4),
                "first_ytd_close": round(first_close, 4),
                "latest_close": round(last_close, 4),
                "ytd_price_return_pct": round(percent(ytd_return) or 0.0, 2),
                "patterns_detected": len(patterns),
                "trades_taken": len(trades_df),
                "closed_trades": len(closed_trades),
                "winning_trades": wins,
                "signal_hit_rate_pct": None if hit_rate is None else round(percent(hit_rate) or 0.0, 2),
                "strategy_return_pct": round(percent(strategy_return) or 0.0, 2),
                "strategy_pnl": round(total_pnl, 2),
                "max_drawdown_pct": round(max_drawdown_pct(equity_df), 2),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    patterns_df = pd.DataFrame(all_patterns)
    trades_df = pd.DataFrame(all_trades)

    summary_df.to_csv(output_dir / "mag7_ytd_w_bottom_summary.csv", index=False)
    patterns_df.to_csv(output_dir / "mag7_ytd_w_bottom_patterns.csv", index=False)
    trades_df.to_csv(output_dir / "mag7_ytd_w_bottom_trades.csv", index=False)
    save_results_chart(summary_df, start_date, end_date_exclusive, output_dir / "mag7_ytd_w_bottom_results.png")
    return summary_df, patterns_df, trades_df


def save_results_chart(summary_df: pd.DataFrame, start_date: date, end_date_exclusive: date, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        section_font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        section_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    title = "Mag 7 YTD W-bottom strategy audit"
    subtitle = f"Daily bars from {start_date.isoformat()} through latest completed bar before {end_date_exclusive.isoformat()}"
    draw.text((60, 28), title, fill="#111827", font=title_font)
    draw.text((60, 66), subtitle, fill="#4b5563", font=label_font)

    ordered = summary_df.copy().sort_values("ytd_price_return_pct", ascending=True).reset_index(drop=True)

    chart_left, chart_right = 210, 1420
    top_a, bottom_a = 130, 475
    top_b, bottom_b = 570, 815

    def draw_axis(top: int, bottom: int, min_value: float, max_value: float) -> None:
        zero_x = value_to_x(0.0, min_value, max_value)
        draw.line((chart_left, bottom, chart_right, bottom), fill="#9ca3af", width=1)
        draw.line((zero_x, top, zero_x, bottom), fill="#6b7280", width=2)
        for tick in nice_ticks(min_value, max_value):
            x = value_to_x(tick, min_value, max_value)
            draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
            draw.text((x - 18, bottom + 8), f"{tick:.0f}%", fill="#4b5563", font=small_font)

    def value_to_x(value: float, min_value: float, max_value: float) -> float:
        if max_value <= min_value:
            return (chart_left + chart_right) / 2
        return chart_left + (value - min_value) / (max_value - min_value) * (chart_right - chart_left)

    def nice_ticks(min_value: float, max_value: float) -> list[float]:
        span = max_value - min_value
        if span <= 0:
            return [0.0]
        raw_step = span / 5
        magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
        step = math.ceil(raw_step / magnitude) * magnitude
        start = math.floor(min_value / step) * step
        ticks = []
        value = start
        while value <= max_value + step * 0.5:
            ticks.append(value)
            value += step
        return ticks

    return_values = list(ordered["ytd_price_return_pct"]) + list(ordered["strategy_return_pct"]) + [0.0]
    min_return = min(return_values)
    max_return = max(return_values)
    pad_return = max((max_return - min_return) * 0.15, 2.0)
    min_return -= pad_return
    max_return += pad_return

    draw.text((60, 105), "YTD price return vs W-bottom strategy return", fill="#111827", font=section_font)
    draw_axis(top_a, bottom_a, min_return, max_return)

    row_gap = (bottom_a - top_a) / len(ordered)
    bar_h = min(18, row_gap * 0.28)
    for idx, row in ordered.iterrows():
        y_center = top_a + row_gap * (idx + 0.5)
        label = f"{row['symbol']}  {row['company']}"
        draw.text((60, y_center - 9), label, fill="#111827", font=label_font)

        for offset, field, color in [(-bar_h * 0.75, "ytd_price_return_pct", "#2563eb"), (bar_h * 0.75, "strategy_return_pct", "#16a34a")]:
            value = float(row[field])
            zero_x = value_to_x(0.0, min_return, max_return)
            end_x = value_to_x(value, min_return, max_return)
            x0, x1 = sorted([zero_x, end_x])
            draw.rounded_rectangle((x0, y_center + offset - bar_h / 2, x1, y_center + offset + bar_h / 2), radius=3, fill=color)
            text_x = end_x + (6 if value >= 0 else -48)
            draw.text((text_x, y_center + offset - 8), f"{value:.1f}%", fill="#111827", font=small_font)

    draw.rectangle((1060, 105, 1080, 119), fill="#2563eb")
    draw.text((1088, 101), "YTD stock", fill="#374151", font=small_font)
    draw.rectangle((1170, 105, 1190, 119), fill="#16a34a")
    draw.text((1198, 101), "Strategy", fill="#374151", font=small_font)

    draw.text((60, 535), "Signal accuracy: closed-trade hit rate", fill="#111827", font=section_font)
    hit_values = [float(value) for value in ordered["signal_hit_rate_pct"].dropna()] + [0.0, 100.0]
    min_hit, max_hit = 0.0, 100.0
    draw_axis(top_b, bottom_b, min_hit, max_hit)

    row_gap_b = (bottom_b - top_b) / len(ordered)
    for idx, row in ordered.iterrows():
        y_center = top_b + row_gap_b * (idx + 0.5)
        draw.text((60, y_center - 9), str(row["symbol"]), fill="#111827", font=label_font)
        trades = int(row["closed_trades"])
        hit = row["signal_hit_rate_pct"]
        if pd.isna(hit):
            draw.text((chart_left, y_center - 9), "No closed W-bottom trades", fill="#6b7280", font=small_font)
            continue
        value = float(hit)
        x0 = value_to_x(0.0, min_hit, max_hit)
        x1 = value_to_x(value, min_hit, max_hit)
        draw.rounded_rectangle((x0, y_center - 9, x1, y_center + 9), radius=3, fill="#f59e0b")
        hit_label = f"{value:.0f}% ({trades} trades)"
        label_width = draw.textbbox((0, 0), hit_label, font=small_font)[2]
        label_x = x1 + 8
        label_fill = "#111827"
        if label_x + label_width > width - 60:
            label_x = x1 - label_width - 8
            label_fill = "white"
        draw.text((label_x, y_center - 8), hit_label, fill=label_fill, font=small_font)

    total_trades = int(ordered["closed_trades"].sum())
    total_wins = int(ordered["winning_trades"].sum())
    aggregate_hit = None if total_trades == 0 else total_wins / total_trades * 100.0
    footer = (
        f"Aggregate closed trades: {total_trades} | Wins: {total_wins} | "
        f"Hit rate: {'N/A' if aggregate_hit is None else f'{aggregate_hit:.1f}%'} | "
        "Source: Yahoo Finance chart endpoint"
    )
    draw.text((60, 850), footer, fill="#374151", font=label_font)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit W-bottom strategy on Magnificent Seven YTD data.")
    parser.add_argument("--start-date", default="2026-01-01", help="Inclusive YTD start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2026-06-08", help="Exclusive end date, YYYY-MM-DD. Use today's date to avoid partial daily bars.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_and_testing",
        help="Output folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = parse_date(args.start_date)
    end_date_exclusive = parse_date(args.end_date)
    if end_date_exclusive <= start_date:
        raise ValueError("--end-date must be after --start-date")

    summary_df, patterns_df, trades_df = run_audit(output_dir, start_date, end_date_exclusive)
    total_closed = int(summary_df["closed_trades"].sum())
    total_wins = int(summary_df["winning_trades"].sum())
    aggregate_hit = None if total_closed == 0 else total_wins / total_closed * 100.0

    print("Mag 7 YTD W-bottom audit complete")
    print(f"Date window: {start_date.isoformat()} to before {end_date_exclusive.isoformat()}")
    print(f"Symbols tested: {', '.join(MAG7)}")
    print(f"Patterns detected: {len(patterns_df)}")
    print(f"Closed trades: {total_closed}")
    print(f"Winning trades: {total_wins}")
    print(f"Aggregate hit rate: {'N/A' if aggregate_hit is None else f'{aggregate_hit:.1f}%'}")
    print("Wrote mag7_ytd_w_bottom_summary.csv, mag7_ytd_w_bottom_trades.csv, and mag7_ytd_w_bottom_results.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
