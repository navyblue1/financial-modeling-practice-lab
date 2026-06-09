#!/usr/bin/env python3
"""
Find the best time-based sell point for the Mag 7 W-bottom setup.

For each confirmed W entry, this script tests exits after k bars and reports
the average R-multiple. It also compares a pure timed exit with a risk-managed
exit where the stop or 2R target can trigger before the timed exit.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from mag7_intraday_w_bottom_optimizer import MAG7, detect_intraday_w_bottoms
from w_bottom_demo import StrategyConfig


OUTPUT_DIR = Path(__file__).resolve().parent
PRICE_DIR = OUTPUT_DIR / "mag7_intraday_prices"
WINDOW_LABEL = "2026-05-01_2026-06-01"
INTERVAL = "5m"
MAX_K_BARS = 60
MIN_TRADES_FOR_BEST = 50


def load_selected_config() -> StrategyConfig:
    config_path = OUTPUT_DIR / "mag7_intraday_selected_config.json"
    fields = json.loads(config_path.read_text(encoding="utf-8"))
    return StrategyConfig(**fields)


def load_price_data(symbol: str) -> pd.DataFrame:
    path = PRICE_DIR / f"{symbol}_{INTERVAL}_{WINDOW_LABEL}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached price file: {path}")
    return pd.read_csv(path)


def build_entries(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> list[dict[str, object]]:
    patterns = detect_intraday_w_bottoms(df, symbol, cfg)
    entries: list[dict[str, object]] = []

    for pattern in patterns:
        entry_idx = pattern.breakout_idx + 1
        if entry_idx >= len(df):
            continue

        entry_row = df.iloc[entry_idx]
        row_time = datetime.strptime(str(entry_row["time"]), "%H:%M").time()
        if row_time >= time(15, 45):
            continue

        entry_price = float(entry_row["Open"])
        stop_price = float(pattern.stop_price)
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            continue

        target_price = entry_price + cfg.reward_risk * risk_per_share
        entries.append(
            {
                "pattern_id": pattern.pattern_id,
                "symbol": symbol,
                "entry_idx": entry_idx,
                "entry_date": str(entry_row["Date"]),
                "session": str(entry_row["session"]),
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_per_share": risk_per_share,
            }
        )

    return entries


def simulate_exits(
    df: pd.DataFrame,
    entries: list[dict[str, object]],
    k_bars: int,
    mode: str,
) -> list[dict[str, object]]:
    last_idx_by_session = df.groupby("session").apply(lambda frame: int(frame.index.max())).to_dict()
    trades: list[dict[str, object]] = []
    next_available_idx = -1

    for entry in entries:
        entry_idx = int(entry["entry_idx"])
        if entry_idx <= next_available_idx:
            continue

        last_session_idx = last_idx_by_session[str(entry["session"])]
        exit_deadline_idx = entry_idx + k_bars
        if exit_deadline_idx > last_session_idx:
            continue

        entry_price = float(entry["entry_price"])
        stop_price = float(entry["stop_price"])
        target_price = float(entry["target_price"])
        risk_per_share = float(entry["risk_per_share"])

        exit_idx = exit_deadline_idx
        exit_price = float(df.iloc[exit_idx]["Close"])
        exit_reason = "time_k"

        if mode == "risk_managed":
            for scan_idx in range(entry_idx + 1, exit_deadline_idx + 1):
                row = df.iloc[scan_idx]
                if float(row["Low"]) <= stop_price:
                    exit_idx = scan_idx
                    exit_price = stop_price
                    exit_reason = "stop"
                    break
                if float(row["High"]) >= target_price:
                    exit_idx = scan_idx
                    exit_price = target_price
                    exit_reason = "target"
                    break

        r_multiple = (exit_price - entry_price) / risk_per_share
        trades.append(
            {
                **entry,
                "k_bars": k_bars,
                "mode": mode,
                "exit_idx": exit_idx,
                "exit_date": str(df.iloc[exit_idx]["Date"]),
                "exit_price": round(exit_price, 4),
                "exit_reason": exit_reason,
                "r_multiple": round(r_multiple, 4),
                "return_pct": round(exit_price / entry_price - 1.0, 5),
                "win": bool(exit_price > entry_price),
            }
        )
        next_available_idx = exit_idx

    return trades


def summarize(trades: list[dict[str, object]], k_bars: int, mode: str) -> dict[str, object]:
    if not trades:
        return {
            "mode": mode,
            "k_bars": k_bars,
            "minutes_after_entry": k_bars * 5,
            "trades": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "avg_r_multiple": 0.0,
            "median_r_multiple": 0.0,
            "total_r_multiple": 0.0,
            "avg_return_pct": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    frame = pd.DataFrame(trades)
    r_values = pd.to_numeric(frame["r_multiple"], errors="coerce")
    returns = pd.to_numeric(frame["return_pct"], errors="coerce")
    wins = int((r_values > 0).sum())
    return {
        "mode": mode,
        "k_bars": k_bars,
        "minutes_after_entry": k_bars * 5,
        "trades": int(len(frame)),
        "wins": wins,
        "win_rate_pct": round(wins / len(frame) * 100.0, 2),
        "avg_r_multiple": round(float(r_values.mean()), 4),
        "median_r_multiple": round(float(r_values.median()), 4),
        "total_r_multiple": round(float(r_values.sum()), 4),
        "avg_return_pct": round(float(returns.mean()) * 100.0, 4),
        "target_exits": int((frame["exit_reason"] == "target").sum()),
        "stop_exits": int((frame["exit_reason"] == "stop").sum()),
        "time_exits": int((frame["exit_reason"] == "time_k").sum()),
    }


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_selected_config()
    entries_by_symbol: dict[str, list[dict[str, object]]] = {}
    dfs: dict[str, pd.DataFrame] = {}

    for symbol in MAG7:
        df = load_price_data(symbol)
        dfs[symbol] = df
        entries_by_symbol[symbol] = build_entries(df, symbol, cfg)

    summary_rows: list[dict[str, object]] = []
    all_trades: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []

    for symbol, entries in entries_by_symbol.items():
        entry_rows.extend(entries)

    for mode in ["pure_time", "risk_managed"]:
        for k_bars in range(1, MAX_K_BARS + 1):
            k_trades: list[dict[str, object]] = []
            for symbol, entries in entries_by_symbol.items():
                trades = simulate_exits(dfs[symbol], entries, k_bars, mode)
                k_trades.extend(trades)
            summary_rows.append(summarize(k_trades, k_bars, mode))
            all_trades.extend(k_trades)

    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(all_trades)
    entries_df = pd.DataFrame(entry_rows)
    return summary_df, trades_df, entries_df


def best_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    eligible = summary_df[summary_df["trades"] >= MIN_TRADES_FOR_BEST].copy()
    if eligible.empty:
        eligible = summary_df.copy()
    return (
        eligible.sort_values(["mode", "avg_r_multiple", "win_rate_pct", "trades"], ascending=[True, False, False, False])
        .groupby("mode", as_index=False)
        .head(1)
        .sort_values("mode")
    )


def save_chart(summary_df: pd.DataFrame, best_df: pd.DataFrame, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 860
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        label_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((60, 28), "Mag 7 W-bottom sell-time R analysis", fill="#111827", font=title_font)
    draw.text(
        (60, 66),
        "Full May 2026, 5-minute bars. R is measured from actual entry price to initial stop.",
        fill="#4b5563",
        font=label_font,
    )

    chart_left, chart_right = 100, 1415
    chart_top, chart_bottom = 135, 650
    x_min, x_max = 1, MAX_K_BARS
    y_min = min(float(summary_df["avg_r_multiple"].min()), 0.0)
    y_max = max(float(summary_df["avg_r_multiple"].max()), 0.0)
    pad = max((y_max - y_min) * 0.14, 0.05)
    y_min -= pad
    y_max += pad

    def x_pos(k_bars: float) -> float:
        return chart_left + (k_bars - x_min) / (x_max - x_min) * (chart_right - chart_left)

    def y_pos(value: float) -> float:
        return chart_bottom - (value - y_min) / (y_max - y_min) * (chart_bottom - chart_top)

    for tick in range(0, 61, 10):
        if tick < x_min:
            continue
        x = x_pos(tick)
        draw.line((x, chart_top, x, chart_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 18, chart_bottom + 10), f"{tick * 5}m", fill="#4b5563", font=small_font)

    y_tick_start = int(y_min * 10) - 1
    y_tick_end = int(y_max * 10) + 1
    for tick_i in range(y_tick_start, y_tick_end + 1):
        value = tick_i / 10.0
        if y_min <= value <= y_max:
            y = y_pos(value)
            draw.line((chart_left, y, chart_right, y), fill="#e5e7eb", width=1)
            draw.text((42, y - 8), f"{value:.1f}R", fill="#4b5563", font=small_font)

    zero_y = y_pos(0.0)
    draw.line((chart_left, zero_y, chart_right, zero_y), fill="#6b7280", width=2)

    colors = {"pure_time": "#2563eb", "risk_managed": "#16a34a"}
    labels = {"pure_time": "Pure timed exit", "risk_managed": "Stop/2R target before timed exit"}

    for mode, mode_df in summary_df.groupby("mode"):
        ordered = mode_df.sort_values("k_bars")
        points = [(x_pos(float(row["k_bars"])), y_pos(float(row["avg_r_multiple"]))) for _, row in ordered.iterrows()]
        if len(points) > 1:
            draw.line(points, fill=colors[mode], width=3)

    for _, row in best_df.iterrows():
        mode = str(row["mode"])
        x = x_pos(float(row["k_bars"]))
        y = y_pos(float(row["avg_r_multiple"]))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[mode])
        draw.text(
            (x + 10, y - 18),
            f"{int(row['minutes_after_entry'])}m: {row['avg_r_multiple']:.3f}R",
            fill="#111827",
            font=small_font,
        )

    draw.rectangle((980, 96, 1000, 110), fill=colors["pure_time"])
    draw.text((1008, 92), labels["pure_time"], fill="#374151", font=small_font)
    draw.rectangle((1135, 96, 1155, 110), fill=colors["risk_managed"])
    draw.text((1163, 92), labels["risk_managed"], fill="#374151", font=small_font)

    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="#9ca3af", width=1)
    draw.text((60, 705), "Best eligible rows", fill="#111827", font=label_font)

    y = 735
    for _, row in best_df.iterrows():
        line = (
            f"{labels[str(row['mode'])]}: sell after {int(row['k_bars'])} bars "
            f"({int(row['minutes_after_entry'])} minutes), avg {row['avg_r_multiple']:.4f}R, "
            f"win rate {row['win_rate_pct']:.1f}%, trades {int(row['trades'])}"
        )
        draw.text((60, y), line, fill="#374151", font=label_font)
        y += 28

    image.save(output_path)


def main() -> None:
    summary_df, trades_df, entries_df = run_analysis()
    best_df = best_rows(summary_df)

    summary_df.to_csv(OUTPUT_DIR / "mag7_sell_time_r_summary.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "mag7_sell_time_r_trades.csv", index=False)
    entries_df.to_csv(OUTPUT_DIR / "mag7_sell_time_r_entries.csv", index=False)
    best_df.to_csv(OUTPUT_DIR / "mag7_sell_time_r_best.csv", index=False)
    save_chart(summary_df, best_df, OUTPUT_DIR / "mag7_sell_time_r_chart.png")

    print("Sell-time R analysis complete")
    for _, row in best_df.iterrows():
        print(
            f"{row['mode']}: best k={int(row['k_bars'])} bars "
            f"({int(row['minutes_after_entry'])} minutes), avg R={row['avg_r_multiple']:.4f}, "
            f"win rate={row['win_rate_pct']:.1f}%, trades={int(row['trades'])}"
        )


if __name__ == "__main__":
    main()
