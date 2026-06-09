#!/usr/bin/env python3
"""
Optimize and test the W-bottom detector on Magnificent Seven 1-minute bars.

This script is for research/demo use only. It downloads recent intraday OHLCV
data from Yahoo Finance, tests several W-bottom parameter sets, chooses the
highest closed-trade hit rate with basic anti-overfit guardrails, and exports
CSV files plus a chart.
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
from itertools import product
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from w_bottom_demo import StrategyConfig, WBottomPattern, dedupe_patterns, detect_w_bottoms, find_swing_points


MAG7 = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
}


NY_TZ = ZoneInfo("America/New_York")


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def unix_midnight_utc(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())


def yahoo_intraday_url(
    symbol: str,
    interval: str,
    range_: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> str:
    params = {
        "interval": interval,
        "includePrePost": "false",
        "events": "history",
    }
    if start_date is not None and end_date is not None:
        params["period1"] = str(unix_midnight_utc(start_date))
        params["period2"] = str(unix_midnight_utc(end_date))
    else:
        params["range"] = range_ or "5d"
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{urllib.parse.urlencode(params)}"


def fetch_yahoo_intraday(
    symbol: str,
    interval: str,
    range_: str | None,
    start_date: date | None,
    end_date: date | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    url = yahoo_intraday_url(symbol, interval, range_, start_date, end_date)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 WBottomIntradayAudit/1.0",
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
    meta = data.get("meta", {})

    rows = []
    for i, ts in enumerate(timestamps):
        try:
            dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY_TZ)
            row = {
                "Date": dt.strftime("%Y-%m-%d %H:%M"),
                "session": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M"),
                "Open": quote["open"][i],
                "High": quote["high"][i],
                "Low": quote["low"][i],
                "Close": quote["close"][i],
                "Volume": quote["volume"][i],
            }
        except (KeyError, IndexError):
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No usable rows returned for {symbol}")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    df["Volume"] = df["Volume"].astype("int64")

    session_start = time(9, 30)
    session_end = time(16, 0)
    bar_times = pd.to_datetime(df["time"], format="%H:%M").dt.time
    df = df[(bar_times >= session_start) & (bar_times <= session_end)].reset_index(drop=True)
    return df, meta


def cfg_from_params(params: dict[str, float | int | bool]) -> StrategyConfig:
    return StrategyConfig(
        swing_window=int(params["swing_window"]),
        low_tolerance_pct=float(params["low_tolerance_pct"]),
        max_second_low_undercut_pct=float(params["max_second_low_undercut_pct"]),
        min_rebound_pct=float(params["min_rebound_pct"]),
        breakout_lookahead=int(params["breakout_lookahead"]),
        breakout_buffer_pct=float(params["breakout_buffer_pct"]),
        volume_sma_window=20,
        require_volume_confirmation=bool(params["require_volume_confirmation"]),
        volume_multiplier=float(params["volume_multiplier"]),
        max_first_low_to_neckline_bars=int(params["max_first_low_to_neckline_bars"]),
        max_neckline_to_second_low_bars=int(params["max_neckline_to_second_low_bars"]),
        stop_buffer_pct=float(params["stop_buffer_pct"]),
        reward_risk=float(params["reward_risk"]),
        max_hold_days=int(params["max_hold_bars"]),
        starting_cash=100_000.0,
        risk_per_trade_pct=0.003,
        max_position_pct=0.20,
        commission_per_order=0.0,
    )


def parameter_grid() -> list[dict[str, float | int | bool]]:
    grid = []
    for (
        swing_window,
        low_tolerance_pct,
        min_rebound_pct,
        breakout_buffer_pct,
        stop_buffer_pct,
        reward_risk,
        max_hold_bars,
        w_width,
        require_volume_confirmation,
    ) in product(
        [2],
        [0.006, 0.012],
        [0.0025, 0.006],
        [0.0, 0.0006],
        [0.0025, 0.005],
        [2.0, 2.5, 3.0],
        [12, 20, 40],
        [24],
        [False],
    ):
        grid.append(
            {
                "swing_window": swing_window,
                "low_tolerance_pct": low_tolerance_pct,
                "max_second_low_undercut_pct": min(low_tolerance_pct, 0.008),
                "min_rebound_pct": min_rebound_pct,
                "breakout_lookahead": max(10, int(w_width / 2)),
                "breakout_buffer_pct": breakout_buffer_pct,
                "require_volume_confirmation": require_volume_confirmation,
                "volume_multiplier": 1.05,
                "max_first_low_to_neckline_bars": w_width,
                "max_neckline_to_second_low_bars": w_width,
                "stop_buffer_pct": stop_buffer_pct,
                "reward_risk": reward_risk,
                "max_hold_bars": max_hold_bars,
            }
        )
    return grid


def _starting_w_idx(df: pd.DataFrame, first_low_idx: int, cfg: StrategyConfig) -> int | None:
    start = max(0, first_low_idx - cfg.max_first_low_to_neckline_bars)
    if start >= first_low_idx:
        return None
    return int(df["High"].iloc[start:first_low_idx].idxmax())


def detect_early_higher_low_w_bottoms(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> list[WBottomPattern]:
    swing_lows, swing_highs = find_swing_points(df, cfg.swing_window)
    raw_patterns: list[WBottomPattern] = []

    for first_low_idx in swing_lows:
        start_idx = _starting_w_idx(df, first_low_idx, cfg)
        if start_idx is None:
            continue
        starting_w_price = float(df["High"].iloc[start_idx])

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

                if second_low <= first_low:
                    continue
                if neckline <= starting_w_price:
                    continue

                avg_low = (first_low + second_low) / 2.0
                low_similarity_pct = abs(first_low - second_low) / avg_low
                if low_similarity_pct > cfg.low_tolerance_pct:
                    continue

                rebound_from_first = neckline / first_low - 1.0
                rebound_from_second = neckline / second_low - 1.0
                rebound_pct = min(rebound_from_first, rebound_from_second)
                if rebound_pct < cfg.min_rebound_pct:
                    continue

                signal_idx = second_low_idx + cfg.swing_window
                if signal_idx >= len(df) - 1:
                    continue

                signal_close = float(df["Close"].iloc[signal_idx])
                stop_price = first_low * (1.0 - cfg.stop_buffer_pct)
                risk_per_share = max(signal_close - stop_price, 0.01)
                target_price = signal_close + cfg.reward_risk * risk_per_share
                score = (
                    (second_low / first_low - 1.0) * 100.0
                    + (neckline / starting_w_price - 1.0) * 100.0
                    + rebound_pct * 100.0
                    - low_similarity_pct * 100.0
                )

                raw_patterns.append(
                    WBottomPattern(
                        pattern_id="",
                        symbol=symbol,
                        first_low_idx=first_low_idx,
                        neckline_idx=neckline_idx,
                        second_low_idx=second_low_idx,
                        breakout_idx=signal_idx,
                        first_low_date=str(df["Date"].iloc[first_low_idx]),
                        neckline_date=str(df["Date"].iloc[neckline_idx]),
                        second_low_date=str(df["Date"].iloc[second_low_idx]),
                        breakout_date=str(df["Date"].iloc[signal_idx]),
                        first_low_price=round(first_low, 2),
                        neckline_price=round(neckline, 2),
                        second_low_price=round(second_low, 2),
                        breakout_close=round(signal_close, 2),
                        stop_price=round(stop_price, 2),
                        target_price=round(target_price, 2),
                        low_similarity_pct=round(low_similarity_pct, 4),
                        rebound_pct=round(rebound_pct, 4),
                        breakout_volume_ratio=0.0,
                        score=round(score, 2),
                    )
                )

    return dedupe_patterns(raw_patterns)


def detect_intraday_w_bottoms(
    df: pd.DataFrame,
    symbol: str,
    cfg: StrategyConfig,
    entry_mode: str = "breakout",
) -> list[WBottomPattern]:
    patterns: list[WBottomPattern] = []
    pattern_number = 1

    for _session, session_df in df.groupby("session", sort=True):
        minimum_session_bars = max(20, cfg.swing_window * 2 + 6)
        if len(session_df) < minimum_session_bars:
            continue

        offset = int(session_df.index[0])
        local_df = session_df.reset_index(drop=True)
        if entry_mode == "early-higher-low":
            session_patterns = detect_early_higher_low_w_bottoms(local_df, symbol, cfg)
        else:
            session_patterns = detect_w_bottoms(local_df, symbol, cfg)

        for pattern in session_patterns:
            data = asdict(pattern)
            for field in ["first_low_idx", "neckline_idx", "second_low_idx", "breakout_idx"]:
                data[field] = int(data[field]) + offset
            data["pattern_id"] = f"W{pattern_number:03d}"
            patterns.append(WBottomPattern(**data))
            pattern_number += 1

    return patterns


def split_by_session(dfs: dict[str, pd.DataFrame], train_fraction: float = 0.65) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    sessions = sorted({session for df in dfs.values() for session in df["session"].unique()})
    if len(sessions) < 3:
        split_idx = max(1, int(len(sessions) * train_fraction))
    else:
        split_idx = min(max(1, int(len(sessions) * train_fraction)), len(sessions) - 1)
    train_sessions = set(sessions[:split_idx])
    test_sessions = set(sessions[split_idx:])

    train = {symbol: df[df["session"].isin(train_sessions)].reset_index(drop=True) for symbol, df in dfs.items()}
    test = {symbol: df[df["session"].isin(test_sessions)].reset_index(drop=True) for symbol, df in dfs.items()}
    return train, test


def intraday_backtest(df: pd.DataFrame, patterns: list[WBottomPattern], cfg: StrategyConfig) -> pd.DataFrame:
    pattern_by_entry_idx = {
        pattern.breakout_idx + 1: pattern
        for pattern in patterns
        if pattern.breakout_idx + 1 < len(df)
    }
    position: dict[str, object] | None = None
    trades: list[dict[str, object]] = []

    for idx, row in df.iterrows():
        session = str(row["session"])
        if position is not None:
            exit_price = None
            exit_reason = ""
            new_session = session != str(position["entry_session"])

            if new_session:
                previous_row = df.iloc[idx - 1]
                exit_price = float(previous_row["Close"])
                exit_reason = "session_close"
                exit_date = str(previous_row["Date"])
                bars_held = idx - 1 - int(position["entry_idx"])
            elif float(row["Low"]) <= float(position["stop_price"]):
                exit_price = float(position["stop_price"])
                exit_reason = "stop"
                exit_date = str(row["Date"])
                bars_held = idx - int(position["entry_idx"])
            elif float(row["High"]) >= float(position["target_price"]):
                exit_price = float(position["target_price"])
                exit_reason = "target"
                exit_date = str(row["Date"])
                bars_held = idx - int(position["entry_idx"])
            elif idx - int(position["entry_idx"]) >= cfg.max_hold_days:
                exit_price = float(row["Close"])
                exit_reason = "time_exit"
                exit_date = str(row["Date"])
                bars_held = idx - int(position["entry_idx"])

            if exit_price is not None:
                entry_price = float(position["entry_price"])
                stop_price = float(position["stop_price"])
                risk_per_share = max(entry_price - stop_price, 0.01)
                return_pct = exit_price / entry_price - 1.0
                trades[-1].update(
                    {
                        "exit_date": exit_date,
                        "exit_price": round(exit_price, 4),
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                        "pnl_per_share": round(exit_price - entry_price, 4),
                        "return_pct": round(return_pct, 5),
                        "r_multiple": round((exit_price - entry_price) / risk_per_share, 2),
                        "win": bool(exit_price > entry_price),
                    }
                )
                position = None

        if position is None and idx in pattern_by_entry_idx:
            pattern = pattern_by_entry_idx[idx]
            row_time = datetime.strptime(str(row["time"]), "%H:%M").time()
            if row_time >= time(15, 45):
                continue

            entry_price = float(row["Open"])
            stop_price = float(pattern.stop_price)
            target_price = entry_price + cfg.reward_risk * (entry_price - stop_price)
            if stop_price >= entry_price or target_price <= entry_price:
                continue

            position = {
                "pattern_id": pattern.pattern_id,
                "entry_idx": idx,
                "entry_session": session,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
            }
            trades.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "symbol": pattern.symbol,
                    "entry_date": str(row["Date"]),
                    "entry_price": round(entry_price, 4),
                    "stop_price": round(stop_price, 4),
                    "target_price": round(target_price, 4),
                    "exit_date": "",
                    "exit_price": "",
                    "exit_reason": "",
                    "bars_held": "",
                    "pnl_per_share": "",
                    "return_pct": "",
                    "r_multiple": "",
                    "win": "",
                }
            )

    if position is not None:
        row = df.iloc[-1]
        entry_price = float(position["entry_price"])
        exit_price = float(row["Close"])
        risk_per_share = max(entry_price - float(position["stop_price"]), 0.01)
        trades[-1].update(
            {
                "exit_date": str(row["Date"]),
                "exit_price": round(exit_price, 4),
                "exit_reason": "end_of_data",
                "bars_held": len(df) - 1 - int(position["entry_idx"]),
                "pnl_per_share": round(exit_price - entry_price, 4),
                "return_pct": round(exit_price / entry_price - 1.0, 5),
                "r_multiple": round((exit_price - entry_price) / risk_per_share, 2),
                "win": bool(exit_price > entry_price),
            }
        )

    return pd.DataFrame(trades)


def evaluate_config(
    dfs: dict[str, pd.DataFrame],
    cfg: StrategyConfig,
    entry_mode: str = "breakout",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trades = []
    all_patterns = []
    for symbol, df in dfs.items():
        if len(df) < 100:
            continue
        patterns = detect_intraday_w_bottoms(df, symbol, cfg, entry_mode)
        for pattern in patterns:
            all_patterns.append(asdict(pattern))
        trades = intraday_backtest(df, patterns, cfg)
        if not trades.empty:
            all_trades.extend(trades.to_dict("records"))
    return pd.DataFrame(all_trades), pd.DataFrame(all_patterns)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "closed_trades": 0,
            "wins": 0,
            "hit_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "avg_r_multiple": 0.0,
        }

    closed = trades[trades["exit_date"].astype(str) != ""].copy()
    if closed.empty:
        return {
            "closed_trades": 0,
            "wins": 0,
            "hit_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "avg_r_multiple": 0.0,
        }

    returns = pd.to_numeric(closed["return_pct"], errors="coerce").fillna(0.0)
    r_values = pd.to_numeric(closed["r_multiple"], errors="coerce").fillna(0.0)
    wins = int((returns > 0).sum())
    return {
        "closed_trades": int(len(closed)),
        "wins": wins,
        "hit_rate_pct": round(wins / len(closed) * 100.0, 2),
        "avg_return_pct": round(float(returns.mean()) * 100.0, 3),
        "total_return_pct": round(float(returns.sum()) * 100.0, 3),
        "avg_r_multiple": round(float(r_values.mean()), 3),
    }


def optimize(
    train_dfs: dict[str, pd.DataFrame],
    min_train_trades: int,
    entry_mode: str = "breakout",
) -> tuple[StrategyConfig, pd.DataFrame]:
    rows = []
    best_cfg: StrategyConfig | None = None
    best_key: tuple[float, float, float, int] | None = None

    for params in parameter_grid():
        cfg = cfg_from_params(params)
        trades, _patterns = evaluate_config(train_dfs, cfg, entry_mode)
        stats = summarize_trades(trades)
        row = {"entry_mode": entry_mode, **params, **stats}
        rows.append(row)

        if stats["closed_trades"] < min_train_trades:
            continue

        key = (
            float(stats["hit_rate_pct"]),
            float(stats["avg_r_multiple"]),
            float(stats["total_return_pct"]),
            int(stats["closed_trades"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg

    results = pd.DataFrame(rows).sort_values(
        ["closed_trades", "hit_rate_pct", "avg_r_multiple", "total_return_pct"],
        ascending=[False, False, False, False],
    )

    if best_cfg is None:
        eligible = results[results["closed_trades"] > 0].copy()
        if eligible.empty:
            best_cfg = cfg_from_params(parameter_grid()[0])
        else:
            best_row = eligible.sort_values(
                ["hit_rate_pct", "avg_r_multiple", "total_return_pct", "closed_trades"],
                ascending=[False, False, False, False],
            ).iloc[0]
            best_cfg = cfg_from_params(best_row.to_dict())

    return best_cfg, results


def per_symbol_summary(
    dfs: dict[str, pd.DataFrame],
    cfg: StrategyConfig,
    entry_mode: str = "breakout",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    all_trades = []
    all_patterns = []

    for symbol, df in dfs.items():
        patterns = detect_intraday_w_bottoms(df, symbol, cfg, entry_mode)
        trades = intraday_backtest(df, patterns, cfg)
        stats = summarize_trades(trades)
        stock_return = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0 if len(df) else 0.0

        for pattern in patterns:
            pattern_row = asdict(pattern)
            pattern_row["company"] = MAG7.get(symbol, symbol)
            pattern_row["entry_mode"] = entry_mode
            all_patterns.append(pattern_row)

        if not trades.empty:
            trades = trades.copy()
            trades.insert(1, "company", MAG7.get(symbol, symbol))
            trades.insert(2, "entry_mode", entry_mode)
            all_trades.extend(trades.to_dict("records"))

        rows.append(
            {
                "symbol": symbol,
                "company": MAG7.get(symbol, symbol),
                "entry_mode": entry_mode,
                "first_bar": df["Date"].iloc[0] if len(df) else "",
                "last_bar": df["Date"].iloc[-1] if len(df) else "",
                "bars": len(df),
                "stock_return_pct": round(stock_return, 3),
                "patterns_detected": len(patterns),
                "closed_trades": stats["closed_trades"],
                "wins": stats["wins"],
                "hit_rate_pct": stats["hit_rate_pct"] if stats["closed_trades"] else None,
                "avg_return_pct": stats["avg_return_pct"],
                "total_trade_return_pct": stats["total_return_pct"],
                "avg_r_multiple": stats["avg_r_multiple"],
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(all_trades), pd.DataFrame(all_patterns)


def save_chart(
    full_summary: pd.DataFrame,
    test_summary: dict[str, float | int],
    train_summary: dict[str, float | int],
    selected_cfg: StrategyConfig,
    interval_label: str,
    window_label: str,
    entry_mode: str,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 940
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

    entry_label = "early higher-low entry" if entry_mode == "early-higher-low" else "neckline breakout entry"
    draw.text((60, 26), f"Mag 7 {interval_label} W-bottom optimization", fill="#111827", font=title_font)
    draw.text(
        (60, 64),
        f"Window: {window_label}. Entry mode: {entry_label}.",
        fill="#4b5563",
        font=label_font,
    )

    cfg_text = (
        f"Selected params: swing={selected_cfg.swing_window}, low tolerance={selected_cfg.low_tolerance_pct:.2%}, "
        f"min rebound={selected_cfg.min_rebound_pct:.2%}, breakout buffer={selected_cfg.breakout_buffer_pct:.2%}, "
        f"stop buffer={selected_cfg.stop_buffer_pct:.2%}, R/R={selected_cfg.reward_risk:.1f}, hold={selected_cfg.max_hold_days} bars"
    )
    draw.text((60, 94), cfg_text, fill="#374151", font=small_font)
    draw.text(
        (60, 118),
        f"Train hit rate: {train_summary['hit_rate_pct']:.1f}% on {train_summary['closed_trades']} trades | "
        f"Holdout hit rate: {test_summary['hit_rate_pct']:.1f}% on {test_summary['closed_trades']} trades",
        fill="#374151",
        font=small_font,
    )

    ordered = full_summary.sort_values("hit_rate_pct", ascending=True, na_position="first").reset_index(drop=True)
    chart_left, chart_right = 230, 1410
    top_a, bottom_a = 175, 500
    top_b, bottom_b = 595, 840

    def value_to_x(value: float, min_value: float, max_value: float) -> float:
        if max_value <= min_value:
            return (chart_left + chart_right) / 2
        return chart_left + (value - min_value) / (max_value - min_value) * (chart_right - chart_left)

    def draw_percent_axis(top: int, bottom: int, max_value: float = 100.0) -> None:
        draw.line((chart_left, bottom, chart_right, bottom), fill="#9ca3af", width=1)
        for tick in [0, 20, 40, 60, 80, 100]:
            x = value_to_x(float(tick), 0.0, max_value)
            draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
            draw.text((x - 10, bottom + 8), f"{tick}%", fill="#4b5563", font=small_font)

    draw.text((60, 145), "Full-window closed-trade hit rate", fill="#111827", font=section_font)
    draw_percent_axis(top_a, bottom_a)
    row_gap = (bottom_a - top_a) / len(ordered)
    for idx, row in ordered.iterrows():
        y_center = top_a + row_gap * (idx + 0.5)
        draw.text((60, y_center - 9), f"{row['symbol']}  {row['company']}", fill="#111827", font=label_font)
        if pd.isna(row["hit_rate_pct"]):
            draw.text((chart_left, y_center - 9), "No closed W-bottom trades", fill="#6b7280", font=small_font)
            continue
        hit = float(row["hit_rate_pct"])
        trades = int(row["closed_trades"])
        x0 = value_to_x(0.0, 0.0, 100.0)
        x1 = value_to_x(hit, 0.0, 100.0)
        draw.rounded_rectangle((x0, y_center - 10, x1, y_center + 10), radius=3, fill="#f59e0b")
        label = f"{hit:.0f}% ({trades})"
        label_width = draw.textbbox((0, 0), label, font=small_font)[2]
        label_x = x1 + 8
        fill = "#111827"
        if label_x + label_width > width - 60:
            label_x = x1 - label_width - 8
            fill = "white"
        draw.text((label_x, y_center - 8), label, fill=fill, font=small_font)

    draw.text((60, 560), "Stock return vs summed trade return", fill="#111827", font=section_font)
    return_values = (
        list(full_summary["stock_return_pct"].fillna(0.0))
        + list(full_summary["total_trade_return_pct"].fillna(0.0))
        + [0.0]
    )
    min_return = min(return_values)
    max_return = max(return_values)
    pad = max((max_return - min_return) * 0.15, 1.0)
    min_return -= pad
    max_return += pad
    zero_x = value_to_x(0.0, min_return, max_return)
    draw.line((chart_left, bottom_b, chart_right, bottom_b), fill="#9ca3af", width=1)
    draw.line((zero_x, top_b, zero_x, bottom_b), fill="#6b7280", width=2)

    ticks = [-10, -5, 0, 5, 10]
    for tick in ticks:
        if min_return <= tick <= max_return:
            x = value_to_x(float(tick), min_return, max_return)
            draw.line((x, top_b, x, bottom_b), fill="#e5e7eb", width=1)
            draw.text((x - 12, bottom_b + 8), f"{tick}%", fill="#4b5563", font=small_font)

    draw.rectangle((1040, 560, 1060, 574), fill="#2563eb")
    draw.text((1068, 556), "Stock", fill="#374151", font=small_font)
    draw.rectangle((1130, 560, 1150, 574), fill="#16a34a")
    draw.text((1158, 556), "W trades", fill="#374151", font=small_font)

    row_gap_b = (bottom_b - top_b) / len(ordered)
    bar_h = min(18, row_gap_b * 0.28)
    for idx, row in ordered.iterrows():
        y_center = top_b + row_gap_b * (idx + 0.5)
        draw.text((60, y_center - 9), str(row["symbol"]), fill="#111827", font=label_font)
        for offset, field, color in [
            (-bar_h * 0.75, "stock_return_pct", "#2563eb"),
            (bar_h * 0.75, "total_trade_return_pct", "#16a34a"),
        ]:
            value = float(row[field] or 0.0)
            x_end = value_to_x(value, min_return, max_return)
            x0, x1 = sorted([zero_x, x_end])
            draw.rounded_rectangle((x0, y_center + offset - bar_h / 2, x1, y_center + offset + bar_h / 2), radius=3, fill=color)
            text_x = x_end + (6 if value >= 0 else -52)
            draw.text((text_x, y_center + offset - 8), f"{value:.1f}%", fill="#111827", font=small_font)

    total_trades = int(full_summary["closed_trades"].sum())
    total_wins = int(full_summary["wins"].sum())
    full_hit = 0.0 if total_trades == 0 else total_wins / total_trades * 100.0
    draw.text(
        (60, 890),
        f"Full-window aggregate: {total_wins}/{total_trades} wins = {full_hit:.1f}% | Source: Yahoo Finance chart endpoint",
        fill="#374151",
        font=label_font,
    )
    image.save(output_path)


def write_selected_config(cfg: StrategyConfig, output_path: Path) -> None:
    fields = asdict(cfg)
    output_path.write_text(json.dumps(fields, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize W-bottom parameters on Mag 7 intraday bars.")
    parser.add_argument("--interval", default="1m", help="Yahoo Finance interval. Default: 1m.")
    parser.add_argument("--range", default="5d", help="Yahoo Finance range. Default: 5d.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional exclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--entry-mode",
        choices=["breakout", "early-higher-low"],
        default="breakout",
        help="Entry rule. breakout waits for neckline breakout; early-higher-low enters after confirmed low 2 with low2 > low1 and neckline > W start.",
    )
    parser.add_argument("--min-train-trades", type=int, default=5, help="Minimum train trades needed for parameter selection.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent, help="Output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prices_dir = output_dir / "mag7_intraday_prices"
    prices_dir.mkdir(exist_ok=True)
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    if (start_date is None) != (end_date is None):
        raise ValueError("Use both --start-date and --end-date, or neither.")
    if start_date is not None and end_date is not None and end_date <= start_date:
        raise ValueError("--end-date must be after --start-date.")
    window_label = f"{start_date}_{end_date}" if start_date and end_date else args.range

    intraday_dfs: dict[str, pd.DataFrame] = {}
    metadata_rows = []
    for symbol in MAG7:
        df, meta = fetch_yahoo_intraday(symbol, args.interval, args.range, start_date, end_date)
        df.to_csv(prices_dir / f"{symbol}_{args.interval}_{window_label}.csv", index=False)
        intraday_dfs[symbol] = df
        metadata_rows.append(
            {
                "symbol": symbol,
                "first_bar": df["Date"].iloc[0],
                "last_bar": df["Date"].iloc[-1],
                "bars": len(df),
                "data_granularity": meta.get("dataGranularity", args.interval),
                "range": meta.get("range", args.range),
                "requested_start_date": "" if start_date is None else start_date.isoformat(),
                "requested_end_date": "" if end_date is None else end_date.isoformat(),
            }
        )

    train_dfs, test_dfs = split_by_session(intraday_dfs)
    selected_cfg, optimization_results = optimize(train_dfs, args.min_train_trades, args.entry_mode)
    train_trades, _train_patterns = evaluate_config(train_dfs, selected_cfg, args.entry_mode)
    test_trades, _test_patterns = evaluate_config(test_dfs, selected_cfg, args.entry_mode)
    train_stats = summarize_trades(train_trades)
    test_stats = summarize_trades(test_trades)
    full_summary, full_trades, full_patterns = per_symbol_summary(intraday_dfs, selected_cfg, args.entry_mode)

    pd.DataFrame(metadata_rows).to_csv(output_dir / "mag7_intraday_data_metadata.csv", index=False)
    optimization_results.to_csv(output_dir / "mag7_intraday_optimization_grid.csv", index=False)
    full_summary.to_csv(output_dir / "mag7_intraday_w_bottom_summary.csv", index=False)
    full_trades.to_csv(output_dir / "mag7_intraday_w_bottom_trades.csv", index=False)
    full_patterns.to_csv(output_dir / "mag7_intraday_w_bottom_patterns.csv", index=False)
    write_selected_config(selected_cfg, output_dir / "mag7_intraday_selected_config.json")
    save_chart(
        full_summary,
        test_stats,
        train_stats,
        selected_cfg,
        args.interval,
        window_label,
        args.entry_mode,
        output_dir / "mag7_intraday_w_bottom_results.png",
    )

    total_trades = int(full_summary["closed_trades"].sum())
    total_wins = int(full_summary["wins"].sum())
    full_hit = 0.0 if total_trades == 0 else total_wins / total_trades * 100.0

    print(f"Mag 7 {args.interval} W-bottom optimization complete")
    print(f"Entry mode: {args.entry_mode}")
    if start_date and end_date:
        print(f"Interval: {args.interval}; date window: {start_date.isoformat()} to before {end_date.isoformat()}")
    else:
        print(f"Interval: {args.interval}; range: {args.range}")
    print(f"Symbols tested: {', '.join(MAG7)}")
    print(f"Selected train hit rate: {train_stats['hit_rate_pct']:.1f}% on {train_stats['closed_trades']} trades")
    print(f"Holdout hit rate: {test_stats['hit_rate_pct']:.1f}% on {test_stats['closed_trades']} trades")
    print(f"Full-window hit rate: {full_hit:.1f}% on {total_trades} trades")
    print("Wrote mag7_intraday_w_bottom_results.png and Mag 7 intraday CSV outputs")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
