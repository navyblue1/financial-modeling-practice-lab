#!/usr/bin/env python3
"""
Trend-filter analysis for the best broad W-model trades.

This takes a completed broad_w_model_* run, annotates each trade using only
pre-entry information, tests trend filters, and simulates the same 20% portfolio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from broad_w_model_optimizer import OUTPUT_DIR, simulate_portfolio


def load_prices_for_trades(trades: pd.DataFrame, interval: str, start_date: str, end_date: str) -> dict[tuple[str, str], pd.DataFrame]:
    cache_dir = OUTPUT_DIR / "broad_w_prices"
    frames = {}
    for symbol in sorted(trades["symbol"].unique()):
        path = cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        if path.exists():
            frames[(symbol, interval)] = pd.read_csv(path)
    return frames


def annotate_trades(trades: pd.DataFrame, price_frames: dict[tuple[str, str], pd.DataFrame], interval: str) -> pd.DataFrame:
    rows = []
    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"])
        frame = price_frames.get((symbol, interval))
        if frame is None or frame.empty:
            continue

        df = frame.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["typical"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["session_vwap"] = (df["typical"] * df["Volume"]).groupby(df["session"]).cumsum() / df["Volume"].groupby(df["session"]).cumsum()
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["vol_sma20"] = df["Volume"].rolling(20).mean()
        entry_dt = pd.to_datetime(trade["entry_date"])
        matches = df.index[df["Date"] == entry_dt].tolist()
        if not matches:
            continue
        idx = matches[0]
        prev_idx = idx - 1
        if prev_idx < 50:
            continue

        prev = df.iloc[prev_idx]
        def prior_return(bars: int) -> float:
            if prev_idx - bars < 0:
                return 0.0
            base = float(df.iloc[prev_idx - bars]["Close"])
            return float(prev["Close"]) / base - 1.0 if base > 0 else 0.0

        annotated = trade.to_dict()
        annotated.update(
            {
                "trend_30m_pct": prior_return(6) * 100.0,
                "trend_60m_pct": prior_return(12) * 100.0,
                "trend_120m_pct": prior_return(24) * 100.0,
                "above_ema20": bool(float(prev["Close"]) > float(prev["ema20"])),
                "ema20_rising": bool(float(prev["ema20"]) > float(df.iloc[prev_idx - 5]["ema20"])),
                "ema20_above_ema50": bool(float(prev["ema20"]) > float(prev["ema50"])),
                "above_vwap": bool(float(prev["Close"]) > float(prev["session_vwap"])),
                "volume_ratio": float(prev["Volume"]) / float(prev["vol_sma20"]) if float(prev["vol_sma20"]) > 0 else 1.0,
            }
        )
        rows.append(annotated)

    return pd.DataFrame(rows)


def filter_definitions() -> list[tuple[str, callable]]:
    return [
        ("all_trades", lambda df: pd.Series(True, index=df.index)),
        ("prior_30m_positive", lambda df: df["trend_30m_pct"] > 0.0),
        ("prior_60m_positive", lambda df: df["trend_60m_pct"] > 0.0),
        ("prior_120m_positive", lambda df: df["trend_120m_pct"] > 0.0),
        ("prior_60m_gt_1pct", lambda df: df["trend_60m_pct"] > 1.0),
        ("prior_120m_gt_2pct", lambda df: df["trend_120m_pct"] > 2.0),
        ("above_ema20", lambda df: df["above_ema20"]),
        ("ema20_rising", lambda df: df["ema20_rising"]),
        ("above_vwap", lambda df: df["above_vwap"]),
        ("volume_ratio_gt_1", lambda df: df["volume_ratio"] > 1.0),
        ("ema20_above_ema50", lambda df: df["ema20_above_ema50"]),
        ("prior_60m_positive_and_above_ema20", lambda df: (df["trend_60m_pct"] > 0.0) & df["above_ema20"]),
        ("prior_60m_positive_and_above_vwap", lambda df: (df["trend_60m_pct"] > 0.0) & df["above_vwap"]),
        ("above_ema20_and_vwap", lambda df: df["above_ema20"] & df["above_vwap"]),
        ("trend_ema_vwap", lambda df: (df["trend_60m_pct"] > 0.0) & df["above_ema20"] & df["above_vwap"]),
        ("strong_trend_ema", lambda df: (df["trend_60m_pct"] > 1.0) & df["above_ema20"] & df["ema20_rising"]),
        ("clean_uptrend_stack", lambda df: df["above_ema20"] & df["ema20_rising"] & df["ema20_above_ema50"] & df["above_vwap"]),
    ]


def summarize_filtered(name: str, trades: pd.DataFrame, price_frames, interval: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    portfolio_df, events_df = simulate_portfolio(trades, price_frames, interval)
    returns = pd.to_numeric(trades["return_pct"], errors="coerce").fillna(0.0)
    r_values = pd.to_numeric(trades["r_multiple"], errors="coerce").fillna(0.0)
    quality = returns >= 0.01
    p = portfolio_df.iloc[0].to_dict()
    row = {
        "filter": name,
        "trades": int(len(trades)),
        "wins": int((returns > 0).sum()),
        "quality_wins": int(quality.sum()),
        "quality_win_rate_pct": round(float(quality.mean()) * 100.0, 4) if len(trades) else 0.0,
        "avg_return_pct": round(float(returns.mean()) * 100.0, 4) if len(trades) else 0.0,
        "total_trade_return_pct": round(float(returns.sum()) * 100.0, 4) if len(trades) else 0.0,
        "avg_r_multiple": round(float(r_values.mean()), 4) if len(trades) else 0.0,
        **p,
    }
    return row, portfolio_df, events_df


def save_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        section_font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        section_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((60, 28), "Trend filter test for best W model", fill="#111827", font=title_font)
    draw.text((60, 66), "Filters use only pre-entry trend information. Ranking is by 20% portfolio return.", fill="#4b5563", font=label_font)

    top = summary_df.sort_values("portfolio_return_pct", ascending=True).tail(12).reset_index(drop=True)
    chart_left, chart_right = 420, 1410
    chart_top, chart_bottom = 130, 760
    vals = list(top["portfolio_return_pct"]) + [0.0]
    min_v, max_v = min(vals), max(vals)
    pad = max((max_v - min_v) * 0.15, 1.0)
    min_v -= pad
    max_v += pad

    def value_to_x(value: float) -> float:
        return chart_left + (value - min_v) / (max_v - min_v) * (chart_right - chart_left)

    zero_x = value_to_x(0.0)
    draw.line((zero_x, chart_top, zero_x, chart_bottom), fill="#6b7280", width=2)
    row_gap = (chart_bottom - chart_top) / max(len(top), 1)
    for idx, row in top.iterrows():
        yy = chart_top + row_gap * (idx + 0.5)
        draw.text((60, yy - 8), str(row["filter"]), fill="#111827", font=label_font)
        value = float(row["portfolio_return_pct"])
        end_x = value_to_x(value)
        x0, x1 = sorted([zero_x, end_x])
        draw.rounded_rectangle((x0, yy - 9, x1, yy + 9), radius=3, fill="#16a34a" if value >= 0 else "#dc2626")
        draw.text((end_x + (6 if value >= 0 else -58), yy - 8), f"{value:.2f}%", fill="#111827", font=small_font)
        draw.text((chart_right - 180, yy - 8), f"QW {row['quality_win_rate_pct']:.1f}% / {int(row['trades'])} trades", fill="#4b5563", font=small_font)

    best = summary_df.sort_values("portfolio_return_pct", ascending=False).iloc[0]
    draw.text((60, 820), f"Best filter: {best['filter']} | return {best['portfolio_return_pct']:.2f}% | max drawdown {best['max_drawdown_pct']:.2f}% | ending ${best['ending_equity']:,.2f}", fill="#374151", font=section_font)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trend filters for a broad W-model run.")
    parser.add_argument("--prefix", default="broad_w_model_80")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start-date", default="2026-05-01")
    parser.add_argument("--end-date", default="2026-06-01")
    parser.add_argument("--output-prefix", default="broad_w_trend_filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trades_path = OUTPUT_DIR / f"{args.prefix}_best_trades.csv"
    trades = pd.read_csv(trades_path)
    if trades.empty:
        raise RuntimeError(f"No trades in {trades_path}")

    price_frames = load_prices_for_trades(trades, args.interval, args.start_date, args.end_date)
    annotated = annotate_trades(trades, price_frames, args.interval)

    rows = []
    best_filtered = pd.DataFrame()
    best_events = pd.DataFrame()
    best_return = -10**9
    for name, predicate in filter_definitions():
        mask = predicate(annotated)
        filtered = annotated[mask].copy()
        row, _portfolio_df, events_df = summarize_filtered(name, filtered, price_frames, args.interval)
        rows.append(row)
        if row["portfolio_return_pct"] > best_return and row["trades"] >= 20:
            best_return = float(row["portfolio_return_pct"])
            best_filtered = filtered
            best_events = events_df

    summary_df = pd.DataFrame(rows).sort_values("portfolio_return_pct", ascending=False)
    output_prefix = args.output_prefix
    annotated.to_csv(OUTPUT_DIR / f"{output_prefix}_annotated_trades.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / f"{output_prefix}_summary.csv", index=False)
    best_filtered.to_csv(OUTPUT_DIR / f"{output_prefix}_best_trades.csv", index=False)
    best_events.to_csv(OUTPUT_DIR / f"{output_prefix}_best_portfolio_events.csv", index=False)
    save_chart(summary_df, OUTPUT_DIR / f"{output_prefix}_results.png")

    best = summary_df.iloc[0]
    print("Trend filter analysis complete")
    print(f"Best filter: {best['filter']}")
    print(f"Portfolio return: {best['portfolio_return_pct']:.4f}%")
    print(f"Quality win rate: {best['quality_win_rate_pct']:.2f}% on {int(best['trades'])} trades")
    print(f"Ending equity: ${best['ending_equity']:,.2f}")


if __name__ == "__main__":
    main()
