#!/usr/bin/env python3
"""
Educational W-bottom strategy demo.

This is not investment advice and it does not connect to any broker.
It detects a simple double-bottom / W-bottom setup, runs a long-only
backtest, and exports manual order tickets for review.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    swing_window: int = 4
    low_tolerance_pct: float = 0.04
    max_second_low_undercut_pct: float = 0.025
    min_rebound_pct: float = 0.06
    breakout_lookahead: int = 22
    breakout_buffer_pct: float = 0.004
    volume_sma_window: int = 20
    require_volume_confirmation: bool = False
    volume_multiplier: float = 1.10
    max_first_low_to_neckline_bars: int = 45
    max_neckline_to_second_low_bars: int = 45
    stop_buffer_pct: float = 0.02
    reward_risk: float = 2.0
    max_hold_days: int = 25
    starting_cash: float = 100_000.00
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.25
    commission_per_order: float = 9.95


@dataclass(frozen=True)
class WBottomPattern:
    pattern_id: str
    symbol: str
    first_low_idx: int
    neckline_idx: int
    second_low_idx: int
    breakout_idx: int
    first_low_date: str
    neckline_date: str
    second_low_date: str
    breakout_date: str
    first_low_price: float
    neckline_price: float
    second_low_price: float
    breakout_close: float
    stop_price: float
    target_price: float
    low_similarity_pct: float
    rebound_pct: float
    breakout_volume_ratio: float
    score: float


def generate_sample_data(rows: int = 260, seed: int = 11) -> pd.DataFrame:
    """Create deterministic sample OHLCV data with a few W-bottom areas."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-06-02", periods=rows)
    close = np.zeros(rows, dtype=float)

    anchors = {
        0: 100.0,
        28: 106.0,
        48: 90.0,
        64: 101.5,
        82: 91.2,
        94: 103.4,
        122: 111.0,
        150: 117.0,
        171: 104.0,
        187: 116.3,
        203: 105.3,
        216: 119.5,
        259: 126.0,
    }

    anchor_items = sorted(anchors.items())
    for (left_idx, left_price), (right_idx, right_price) in zip(anchor_items, anchor_items[1:]):
        segment_len = right_idx - left_idx
        baseline = np.linspace(left_price, right_price, segment_len + 1)
        noise = rng.normal(0.0, 0.45, segment_len + 1)
        noise[0] = 0.0
        noise[-1] = 0.0
        close[left_idx : right_idx + 1] = baseline + noise

    open_ = np.empty(rows)
    open_[0] = close[0] * (1.0 + rng.normal(0, 0.002))
    open_[1:] = close[:-1] * (1.0 + rng.normal(0, 0.004, rows - 1))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.002, 0.014, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.002, 0.014, rows))
    volume = rng.integers(180_000, 420_000, rows)

    for idx in (94, 216):
        if idx < rows:
            volume[idx] = int(volume[idx] * 1.9)

    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )


def load_prices(csv_path: Path | None, output_dir: Path) -> pd.DataFrame:
    if csv_path is None:
        df = generate_sample_data()
        sample_path = output_dir / "sample_market_data.csv"
        df.to_csv(sample_path, index=False)
        return df

    df = pd.read_csv(csv_path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.sort_values("Date").reset_index(drop=True)


def find_swing_points(df: pd.DataFrame, window: int) -> tuple[list[int], list[int]]:
    lows: list[int] = []
    highs: list[int] = []

    for i in range(window, len(df) - window):
        low_slice = df["Low"].iloc[i - window : i + window + 1]
        high_slice = df["High"].iloc[i - window : i + window + 1]
        if df["Low"].iloc[i] <= low_slice.min():
            lows.append(i)
        if df["High"].iloc[i] >= high_slice.max():
            highs.append(i)

    return lows, highs


def _first_breakout(
    df: pd.DataFrame,
    start_idx: int,
    neckline_price: float,
    cfg: StrategyConfig,
) -> tuple[int | None, float]:
    volume_sma = df["Volume"].rolling(cfg.volume_sma_window).mean()
    breakout_level = neckline_price * (1.0 + cfg.breakout_buffer_pct)
    end_idx = min(len(df) - 2, start_idx + cfg.breakout_lookahead)

    for idx in range(start_idx, end_idx + 1):
        if df["Close"].iloc[idx] < breakout_level:
            continue

        avg_volume = volume_sma.iloc[idx]
        volume_ratio = 1.0 if not avg_volume or math.isnan(avg_volume) else df["Volume"].iloc[idx] / avg_volume
        if cfg.require_volume_confirmation and volume_ratio < cfg.volume_multiplier:
            continue
        return idx, float(volume_ratio)

    return None, 0.0


def detect_w_bottoms(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> list[WBottomPattern]:
    swing_lows, swing_highs = find_swing_points(df, cfg.swing_window)
    raw_patterns: list[WBottomPattern] = []

    for first_low_idx in swing_lows:
        possible_necklines = [
            idx
            for idx in swing_highs
            if first_low_idx < idx <= first_low_idx + cfg.max_first_low_to_neckline_bars
        ]
        for neckline_idx in possible_necklines:
            possible_second_lows = [
                idx
                for idx in swing_lows
                if neckline_idx < idx <= neckline_idx + cfg.max_neckline_to_second_low_bars
            ]
            for second_low_idx in possible_second_lows:
                first_low = float(df["Low"].iloc[first_low_idx])
                neckline = float(df["High"].iloc[neckline_idx])
                second_low = float(df["Low"].iloc[second_low_idx])
                avg_low = (first_low + second_low) / 2.0
                low_similarity_pct = abs(first_low - second_low) / avg_low

                if low_similarity_pct > cfg.low_tolerance_pct:
                    continue
                if second_low < first_low * (1.0 - cfg.max_second_low_undercut_pct):
                    continue

                rebound_from_first = neckline / first_low - 1.0
                rebound_from_second = neckline / second_low - 1.0
                rebound_pct = min(rebound_from_first, rebound_from_second)
                if rebound_pct < cfg.min_rebound_pct:
                    continue

                breakout_idx, volume_ratio = _first_breakout(df, second_low_idx + 1, neckline, cfg)
                if breakout_idx is None:
                    continue

                breakout_close = float(df["Close"].iloc[breakout_idx])
                pattern_low = min(first_low, second_low)
                stop_price = pattern_low * (1.0 - cfg.stop_buffer_pct)
                entry_proxy = breakout_close
                risk_per_share = max(entry_proxy - stop_price, 0.01)
                target_price = entry_proxy + cfg.reward_risk * risk_per_share
                score = rebound_pct * 100.0 - low_similarity_pct * 100.0 + min(volume_ratio, 3.0)

                raw_patterns.append(
                    WBottomPattern(
                        pattern_id="",
                        symbol=symbol,
                        first_low_idx=first_low_idx,
                        neckline_idx=neckline_idx,
                        second_low_idx=second_low_idx,
                        breakout_idx=breakout_idx,
                        first_low_date=str(df["Date"].iloc[first_low_idx]),
                        neckline_date=str(df["Date"].iloc[neckline_idx]),
                        second_low_date=str(df["Date"].iloc[second_low_idx]),
                        breakout_date=str(df["Date"].iloc[breakout_idx]),
                        first_low_price=round(first_low, 2),
                        neckline_price=round(neckline, 2),
                        second_low_price=round(second_low, 2),
                        breakout_close=round(breakout_close, 2),
                        stop_price=round(stop_price, 2),
                        target_price=round(target_price, 2),
                        low_similarity_pct=round(low_similarity_pct, 4),
                        rebound_pct=round(rebound_pct, 4),
                        breakout_volume_ratio=round(volume_ratio, 2),
                        score=round(score, 2),
                    )
                )

    return dedupe_patterns(raw_patterns)


def dedupe_patterns(patterns: Iterable[WBottomPattern]) -> list[WBottomPattern]:
    best_by_breakout: dict[int, WBottomPattern] = {}
    for pattern in patterns:
        current = best_by_breakout.get(pattern.breakout_idx)
        if current is None or pattern.score > current.score:
            best_by_breakout[pattern.breakout_idx] = pattern

    selected: list[WBottomPattern] = []
    last_second_low_idx = -1
    for pattern in sorted(best_by_breakout.values(), key=lambda item: (item.breakout_idx, -item.score)):
        if pattern.first_low_idx <= last_second_low_idx:
            continue
        selected.append(pattern)
        last_second_low_idx = pattern.second_low_idx

    numbered: list[WBottomPattern] = []
    for number, pattern in enumerate(selected, start=1):
        data = asdict(pattern)
        data["pattern_id"] = f"W{number:03d}"
        numbered.append(WBottomPattern(**data))
    return numbered


def backtest(
    df: pd.DataFrame,
    patterns: list[WBottomPattern],
    cfg: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cash = cfg.starting_cash
    position: dict[str, float | int | str] | None = None
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    pattern_by_entry_idx = {pattern.breakout_idx + 1: pattern for pattern in patterns if pattern.breakout_idx + 1 < len(df)}

    for idx, row in df.iterrows():
        date = str(row["Date"])

        if position is not None:
            exit_price = None
            exit_reason = ""
            if float(row["Low"]) <= float(position["stop_price"]):
                exit_price = float(position["stop_price"])
                exit_reason = "stop"
            elif float(row["High"]) >= float(position["target_price"]):
                exit_price = float(position["target_price"])
                exit_reason = "target"
            elif idx - int(position["entry_idx"]) >= cfg.max_hold_days:
                exit_price = float(row["Close"])
                exit_reason = "time_exit"

            if exit_price is not None:
                shares = int(position["shares"])
                entry_price = float(position["entry_price"])
                gross_exit = shares * exit_price
                cash += gross_exit - cfg.commission_per_order
                pnl = (exit_price - entry_price) * shares - 2.0 * cfg.commission_per_order
                risk_per_share = max(entry_price - float(position["stop_price"]), 0.01)
                trades[-1].update(
                    {
                        "exit_date": date,
                        "exit_price": round(exit_price, 2),
                        "exit_reason": exit_reason,
                        "bars_held": idx - int(position["entry_idx"]),
                        "pnl": round(pnl, 2),
                        "return_pct": round(exit_price / entry_price - 1.0, 4),
                        "r_multiple": round((exit_price - entry_price) / risk_per_share, 2),
                    }
                )
                position = None

        if position is None and idx in pattern_by_entry_idx:
            pattern = pattern_by_entry_idx[idx]
            entry_price = float(row["Open"])
            stop_price = float(pattern.stop_price)
            risk_per_share = entry_price - stop_price
            if risk_per_share > 0:
                equity_now = cash
                risk_budget = equity_now * cfg.risk_per_trade_pct
                max_position_value = equity_now * cfg.max_position_pct
                risk_shares = math.floor(risk_budget / risk_per_share)
                cap_shares = math.floor(max_position_value / entry_price)
                shares = max(min(risk_shares, cap_shares), 0)

                if shares > 0:
                    cost = shares * entry_price + cfg.commission_per_order
                    if cost <= cash:
                        cash -= cost
                        position = {
                            "pattern_id": pattern.pattern_id,
                            "entry_idx": idx,
                            "entry_date": date,
                            "entry_price": entry_price,
                            "shares": shares,
                            "stop_price": stop_price,
                            "target_price": float(pattern.target_price),
                        }
                        trades.append(
                            {
                                "pattern_id": pattern.pattern_id,
                                "symbol": pattern.symbol,
                                "entry_date": date,
                                "entry_price": round(entry_price, 2),
                                "shares": shares,
                                "stop_price": round(stop_price, 2),
                                "target_price": round(float(pattern.target_price), 2),
                                "exit_date": "",
                                "exit_price": "",
                                "exit_reason": "",
                                "bars_held": "",
                                "pnl": "",
                                "return_pct": "",
                                "r_multiple": "",
                            }
                        )

        mark_to_market = 0.0 if position is None else int(position["shares"]) * float(row["Close"])
        equity = cash + mark_to_market
        equity_rows.append(
            {
                "Date": date,
                "cash": round(cash, 2),
                "position_value": round(mark_to_market, 2),
                "equity": round(equity, 2),
            }
        )

    if position is not None:
        final_row = df.iloc[-1]
        final_price = float(final_row["Close"])
        shares = int(position["shares"])
        entry_price = float(position["entry_price"])
        pnl = (final_price - entry_price) * shares - 2.0 * cfg.commission_per_order
        trades[-1].update(
            {
                "exit_date": str(final_row["Date"]),
                "exit_price": round(final_price, 2),
                "exit_reason": "end_of_data",
                "bars_held": len(df) - 1 - int(position["entry_idx"]),
                "pnl": round(pnl, 2),
                "return_pct": round(final_price / entry_price - 1.0, 4),
                "r_multiple": round((final_price - entry_price) / max(entry_price - float(position["stop_price"]), 0.01), 2),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["peak_equity"] = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = (equity_df["equity"] / equity_df["peak_equity"] - 1.0).round(4)

    trades_df = pd.DataFrame(trades)
    tickets_df = build_manual_tickets(trades_df, patterns)
    return trades_df, equity_df, tickets_df


def build_manual_tickets(trades_df: pd.DataFrame, patterns: list[WBottomPattern]) -> pd.DataFrame:
    pattern_map = {pattern.pattern_id: pattern for pattern in patterns}
    tickets: list[dict[str, object]] = []

    for _, trade in trades_df.iterrows():
        pattern = pattern_map.get(str(trade["pattern_id"]))
        if pattern is None:
            continue
        tickets.append(
            {
                "review_date": trade["entry_date"],
                "symbol": trade["symbol"],
                "setup": "W-bottom breakout",
                "action": "BUY",
                "quantity": trade["shares"],
                "entry_plan": "Review next-session entry after breakout; demo backtest uses next open.",
                "reference_entry_price": trade["entry_price"],
                "protective_stop": trade["stop_price"],
                "profit_target": trade["target_price"],
                "neckline_price": pattern.neckline_price,
                "breakout_date": pattern.breakout_date,
                "notes": "Manual review only. Do not auto-submit to RBC from this demo.",
            }
        )

    return pd.DataFrame(tickets)


def save_chart(
    df: pd.DataFrame,
    patterns: list[WBottomPattern],
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return save_chart_with_pillow(df, patterns, trades_df, equity_df, output_path)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(df))

    axes[0].plot(x, df["Close"], color="#1f2937", linewidth=1.6, label="Close")
    axes[0].set_title("W-bottom pattern demo")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.22)

    for pattern in patterns:
        axes[0].scatter(pattern.first_low_idx, pattern.first_low_price, color="#dc2626", marker="v", s=70)
        axes[0].scatter(pattern.second_low_idx, pattern.second_low_price, color="#dc2626", marker="v", s=70)
        axes[0].scatter(pattern.neckline_idx, pattern.neckline_price, color="#2563eb", marker="^", s=70)
        axes[0].scatter(pattern.breakout_idx, pattern.breakout_close, color="#16a34a", marker="o", s=70)
        axes[0].hlines(
            pattern.neckline_price,
            pattern.first_low_idx,
            pattern.breakout_idx,
            colors="#2563eb",
            linestyles="dashed",
            linewidth=1.0,
            alpha=0.7,
        )
        axes[0].annotate(pattern.pattern_id, (pattern.breakout_idx, pattern.breakout_close), xytext=(4, 8), textcoords="offset points")

    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            entry_idx = df.index[df["Date"] == trade["entry_date"]]
            if len(entry_idx):
                axes[0].scatter(entry_idx[0], trade["entry_price"], color="#0891b2", marker=">", s=85)
            if trade["exit_date"]:
                exit_idx = df.index[df["Date"] == trade["exit_date"]]
                if len(exit_idx):
                    axes[0].scatter(exit_idx[0], trade["exit_price"], color="#9333ea", marker="x", s=85)

    axes[0].legend(loc="best")

    if not equity_df.empty:
        axes[1].plot(x[: len(equity_df)], equity_df["equity"], color="#0f766e", linewidth=1.6)
        axes[1].set_ylabel("Equity")
        axes[1].set_xlabel("Trading day")
        axes[1].grid(True, alpha=0.22)

    tick_step = max(len(df) // 8, 1)
    ticks = x[::tick_step]
    labels = df["Date"].iloc[::tick_step]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def save_chart_with_pillow(
    df: pd.DataFrame,
    patterns: list[WBottomPattern],
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    output_path: Path,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    width, height = 1400, 850
    left, right = 90, 45
    price_top, price_bottom = 70, 540
    equity_top, equity_bottom = 625, 795
    plot_width = width - left - right

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 26)
        label_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    def x_pos(idx: int) -> float:
        if len(df) <= 1:
            return left
        return left + idx / (len(df) - 1) * plot_width

    def y_pos(value: float, low: float, high: float, top: int, bottom: int) -> float:
        if high <= low:
            return (top + bottom) / 2
        return bottom - (value - low) / (high - low) * (bottom - top)

    close_values = [float(value) for value in df["Close"]]
    price_min = min(close_values + [pattern.stop_price for pattern in patterns])
    price_max = max(close_values + [pattern.target_price for pattern in patterns])
    pad = max((price_max - price_min) * 0.08, 1.0)
    price_min -= pad
    price_max += pad

    draw.text((left, 22), "W-bottom pattern demo", fill="#111827", font=title_font)
    draw.text((left + 360, 31), "Red: W lows  Blue: neckline  Green: breakout  Teal: entry  Purple: exit", fill="#374151", font=small_font)
    draw.text((left, 565), "Backtested equity", fill="#111827", font=label_font)

    for y in np.linspace(price_min, price_max, 6):
        yy = y_pos(float(y), price_min, price_max, price_top, price_bottom)
        draw.line((left, yy, width - right, yy), fill="#e5e7eb", width=1)
        draw.text((15, yy - 7), f"{y:,.0f}", fill="#4b5563", font=small_font)

    price_points = [(x_pos(idx), y_pos(value, price_min, price_max, price_top, price_bottom)) for idx, value in enumerate(close_values)]
    draw.line(price_points, fill="#1f2937", width=3)

    def draw_triangle(idx: int, price: float, color: str, up: bool) -> None:
        x = x_pos(idx)
        y = y_pos(price, price_min, price_max, price_top, price_bottom)
        if up:
            points = [(x, y - 8), (x - 8, y + 8), (x + 8, y + 8)]
        else:
            points = [(x, y + 8), (x - 8, y - 8), (x + 8, y - 8)]
        draw.polygon(points, fill=color)

    def draw_circle(idx: int, price: float, color: str) -> None:
        x = x_pos(idx)
        y = y_pos(price, price_min, price_max, price_top, price_bottom)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)

    for pattern in patterns:
        y_neckline = y_pos(pattern.neckline_price, price_min, price_max, price_top, price_bottom)
        draw.line(
            (x_pos(pattern.first_low_idx), y_neckline, x_pos(pattern.breakout_idx), y_neckline),
            fill="#2563eb",
            width=2,
        )
        draw_triangle(pattern.first_low_idx, pattern.first_low_price, "#dc2626", up=False)
        draw_triangle(pattern.second_low_idx, pattern.second_low_price, "#dc2626", up=False)
        draw_triangle(pattern.neckline_idx, pattern.neckline_price, "#2563eb", up=True)
        draw_circle(pattern.breakout_idx, pattern.breakout_close, "#16a34a")
        draw.text(
            (x_pos(pattern.breakout_idx) + 8, y_pos(pattern.breakout_close, price_min, price_max, price_top, price_bottom) - 16),
            pattern.pattern_id,
            fill="#111827",
            font=small_font,
        )

    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            entry_matches = df.index[df["Date"] == trade["entry_date"]]
            if len(entry_matches):
                draw_circle(int(entry_matches[0]), float(trade["entry_price"]), "#0891b2")
            if trade["exit_date"]:
                exit_matches = df.index[df["Date"] == trade["exit_date"]]
                if len(exit_matches):
                    x = x_pos(int(exit_matches[0]))
                    y = y_pos(float(trade["exit_price"]), price_min, price_max, price_top, price_bottom)
                    draw.line((x - 8, y - 8, x + 8, y + 8), fill="#9333ea", width=3)
                    draw.line((x - 8, y + 8, x + 8, y - 8), fill="#9333ea", width=3)

    draw.rectangle((left, price_top, width - right, price_bottom), outline="#9ca3af", width=1)

    if not equity_df.empty:
        equity_values = [float(value) for value in equity_df["equity"]]
        eq_min, eq_max = min(equity_values), max(equity_values)
        eq_pad = max((eq_max - eq_min) * 0.1, 100.0)
        eq_min -= eq_pad
        eq_max += eq_pad
        for y in np.linspace(eq_min, eq_max, 4):
            yy = y_pos(float(y), eq_min, eq_max, equity_top, equity_bottom)
            draw.line((left, yy, width - right, yy), fill="#e5e7eb", width=1)
            draw.text((15, yy - 7), f"{y:,.0f}", fill="#4b5563", font=small_font)
        equity_points = [
            (x_pos(idx), y_pos(value, eq_min, eq_max, equity_top, equity_bottom))
            for idx, value in enumerate(equity_values)
        ]
        draw.line(equity_points, fill="#0f766e", width=3)
        draw.rectangle((left, equity_top, width - right, equity_bottom), outline="#9ca3af", width=1)

    tick_step = max(len(df) // 8, 1)
    for idx in range(0, len(df), tick_step):
        x = x_pos(idx)
        draw.line((x, price_bottom, x, price_bottom + 5), fill="#6b7280", width=1)
        draw.text((x - 38, height - 34), str(df["Date"].iloc[idx]), fill="#4b5563", font=small_font)

    image.save(output_path)
    return True


def write_outputs(
    output_dir: Path,
    patterns: list[WBottomPattern],
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    pattern_rows = [asdict(pattern) for pattern in patterns]
    pd.DataFrame(pattern_rows).to_csv(output_dir / "w_bottom_patterns.csv", index=False)
    trades_df.to_csv(output_dir / "w_bottom_trades.csv", index=False)
    equity_df.to_csv(output_dir / "w_bottom_equity_curve.csv", index=False)
    tickets_df.to_csv(output_dir / "w_bottom_manual_order_tickets.csv", index=False)
    chart_saved = save_chart(df, patterns, trades_df, equity_df, output_dir / "w_bottom_demo_chart.png")
    if not chart_saved:
        (output_dir / "chart_not_created.txt").write_text(
            "matplotlib was not available, so the PNG chart was not created.\n",
            encoding="utf-8",
        )
    else:
        missing_chart_note = output_dir / "chart_not_created.txt"
        if missing_chart_note.exists():
            missing_chart_note.unlink()


def summarize(trades_df: pd.DataFrame, equity_df: pd.DataFrame, patterns: list[WBottomPattern], cfg: StrategyConfig) -> str:
    ending_equity = cfg.starting_cash if equity_df.empty else float(equity_df["equity"].iloc[-1])
    total_return = ending_equity / cfg.starting_cash - 1.0
    max_drawdown = 0.0 if equity_df.empty else float(equity_df["drawdown_pct"].min())

    closed_trades = trades_df[trades_df["exit_date"].astype(str) != ""] if not trades_df.empty else trades_df
    wins = 0
    if not closed_trades.empty:
        wins = int((pd.to_numeric(closed_trades["pnl"]) > 0).sum())
    win_rate = 0.0 if closed_trades.empty else wins / len(closed_trades)

    lines = [
        "W-bottom demo complete",
        f"Patterns detected: {len(patterns)}",
        f"Trades taken: {len(trades_df)}",
        f"Closed trades: {len(closed_trades)}",
        f"Win rate: {win_rate:.1%}",
        f"Ending equity: ${ending_equity:,.2f}",
        f"Total return: {total_return:.2%}",
        f"Max drawdown: {max_drawdown:.2%}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and backtest a simple W-bottom trading setup.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional OHLCV CSV. Required columns: Date, Open, High, Low, Close, Volume.")
    parser.add_argument("--symbol", default="DEMO", help="Symbol label used in reports.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_and_testing",
        help="Where CSV and chart outputs are written.",
    )
    parser.add_argument("--volume-confirmation", action="store_true", help="Require breakout volume above the rolling average.")
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.01, help="Fraction of equity risked per trade. Default: 0.01.")
    parser.add_argument("--max-position-pct", type=float, default=0.25, help="Maximum equity allocated to one position. Default: 0.25.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = StrategyConfig(
        require_volume_confirmation=args.volume_confirmation,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_position_pct=args.max_position_pct,
    )
    df = load_prices(args.input_csv, output_dir)
    patterns = detect_w_bottoms(df, args.symbol, cfg)
    trades_df, equity_df, tickets_df = backtest(df, patterns, cfg)
    write_outputs(output_dir, patterns, trades_df, equity_df, tickets_df, df)
    print(summarize(trades_df, equity_df, patterns, cfg))


if __name__ == "__main__":
    main()
