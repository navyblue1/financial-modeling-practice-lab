#!/usr/bin/env python3
"""
Run the fixed W-bottom breakout algorithm on a custom stock universe.

Defaults match the latest breakout setup:
- 5-minute bars
- full May 2026 window
- confirmed neckline breakout entry
- 2R target family config loaded from mag7_breakout_w_selected_config.json
- 20% portfolio allocation per signal, capped at five concurrent positions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mag7_intraday_w_bottom_optimizer import (
    detect_intraday_w_bottoms,
    fetch_yahoo_intraday,
    parse_date,
    summarize_trades,
)
from w_bottom_demo import StrategyConfig


DEFAULT_SYMBOLS = ["MU", "SNDK", "GLW", "OLKO", "RGTI", "IONQ", "CSCO", "AMKR", "ASTS", "RKLB"]
OUTPUT_DIR = Path(__file__).resolve().parent / "data_and_testing"


def load_breakout_config() -> StrategyConfig:
    path = OUTPUT_DIR / "mag7_breakout_w_selected_config.json"
    if path.exists():
        return StrategyConfig(**json.loads(path.read_text(encoding="utf-8")))

    return StrategyConfig(
        swing_window=2,
        low_tolerance_pct=0.012,
        max_second_low_undercut_pct=0.008,
        min_rebound_pct=0.0025,
        breakout_lookahead=12,
        breakout_buffer_pct=0.0006,
        max_first_low_to_neckline_bars=24,
        max_neckline_to_second_low_bars=24,
        stop_buffer_pct=0.005,
        reward_risk=2.0,
        max_hold_days=40,
        risk_per_trade_pct=0.003,
        max_position_pct=0.2,
        commission_per_order=0.0,
    )


def intraday_backtest_with_actual_target(df: pd.DataFrame, patterns, cfg: StrategyConfig) -> pd.DataFrame:
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
                risk_per_share = max(entry_price - float(position["stop_price"]), 0.01)
                trades[-1].update(
                    {
                        "exit_date": exit_date,
                        "exit_price": round(exit_price, 4),
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                        "pnl_per_share": round(exit_price - entry_price, 4),
                        "return_pct": round(exit_price / entry_price - 1.0, 5),
                        "r_multiple": round((exit_price - entry_price) / risk_per_share, 2),
                        "win": bool(exit_price > entry_price),
                    }
                )
                position = None

        if position is None and idx in pattern_by_entry_idx:
            pattern = pattern_by_entry_idx[idx]
            if str(row["time"]) >= "15:45":
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


def run_universe(symbols: list[str], interval: str, start_date: str, end_date: str, output_prefix: str):
    cfg = load_breakout_config()
    start = parse_date(start_date)
    end = parse_date(end_date)
    prices_dir = OUTPUT_DIR / f"{output_prefix}_prices"
    prices_dir.mkdir(exist_ok=True)

    summary_rows = []
    all_trades = []
    all_patterns = []
    failed_rows = []
    price_frames: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            df, meta = fetch_yahoo_intraday(symbol, interval, None, start, end)
        except Exception as exc:
            failed_rows.append({"symbol": symbol, "error": str(exc)})
            continue

        df.to_csv(prices_dir / f"{symbol}_{interval}_{start}_{end}.csv", index=False)
        price_frames[symbol] = df

        patterns = detect_intraday_w_bottoms(df, symbol, cfg, "breakout")
        trades = intraday_backtest_with_actual_target(df, patterns, cfg)
        stats = summarize_trades(trades)
        stock_return = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0

        for pattern in patterns:
            row = {
                "symbol": symbol,
                "pattern_id": pattern.pattern_id,
                "first_low_date": pattern.first_low_date,
                "neckline_date": pattern.neckline_date,
                "second_low_date": pattern.second_low_date,
                "breakout_date": pattern.breakout_date,
                "first_low_price": pattern.first_low_price,
                "neckline_price": pattern.neckline_price,
                "second_low_price": pattern.second_low_price,
                "breakout_close": pattern.breakout_close,
                "stop_price": pattern.stop_price,
                "target_price": pattern.target_price,
                "score": pattern.score,
            }
            all_patterns.append(row)

        if not trades.empty:
            trades = trades.copy()
            trades.insert(0, "symbol", trades.pop("symbol"))
            all_trades.extend(trades.to_dict("records"))

        summary_rows.append(
            {
                "symbol": symbol,
                "first_bar": df["Date"].iloc[0],
                "last_bar": df["Date"].iloc[-1],
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

    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(all_trades)
    patterns_df = pd.DataFrame(all_patterns)
    failed_df = pd.DataFrame(failed_rows)

    portfolio_summary, events_df = simulate_20pct_portfolio(trades_df, price_frames)

    summary_df.to_csv(OUTPUT_DIR / f"{output_prefix}_summary.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / f"{output_prefix}_trades.csv", index=False)
    patterns_df.to_csv(OUTPUT_DIR / f"{output_prefix}_patterns.csv", index=False)
    failed_df.to_csv(OUTPUT_DIR / f"{output_prefix}_failed_symbols.csv", index=False)
    portfolio_summary.to_csv(OUTPUT_DIR / f"{output_prefix}_20pct_portfolio_summary.csv", index=False)
    events_df.to_csv(OUTPUT_DIR / f"{output_prefix}_20pct_portfolio_events.csv", index=False)
    save_chart(summary_df, portfolio_summary, failed_df, OUTPUT_DIR / f"{output_prefix}_results.png")
    return summary_df, trades_df, failed_df, portfolio_summary


def simulate_20pct_portfolio(trades_df: pd.DataFrame, price_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial = 100_000.0
    if trades_df.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "initial_equity": initial,
                        "ending_equity": initial,
                        "portfolio_return_pct": 0.0,
                        "trades_available": 0,
                        "trades_taken": 0,
                        "trades_skipped_max_positions": 0,
                        "trades_skipped_no_cash": 0,
                    }
                ]
            ),
            pd.DataFrame(),
        )

    trades = trades_df.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades = trades.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    indexed_prices = {
        symbol: df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date")
        for symbol, df in price_frames.items()
    }

    cash = initial
    positions: dict[int, dict[str, float | str]] = {}
    events: list[dict[str, object]] = []
    taken = 0
    skipped_slots = 0
    skipped_cash = 0

    def mark_equity(ts) -> float:
        total = cash
        for pos in positions.values():
            df = indexed_prices[str(pos["symbol"])]
            idx = df.index.searchsorted(ts, side="right") - 1
            px = float(pos["entry_price"]) if idx < 0 else float(df.iloc[idx]["Close"])
            total += float(pos["shares"]) * px
        return total

    timeline = []
    for i, trade in trades.iterrows():
        timeline.append((trade["exit_date"], 0, i))
        timeline.append((trade["entry_date"], 1, i))
    timeline.sort(key=lambda item: (item[0], item[1]))

    for ts, event_type, trade_id in timeline:
        trade = trades.loc[trade_id]
        symbol = str(trade["symbol"])

        if event_type == 0:
            if trade_id not in positions:
                continue
            pos = positions.pop(trade_id)
            proceeds = float(pos["shares"]) * float(trade["exit_price"])
            cash += proceeds
            events.append(
                {
                    "timestamp": ts,
                    "event": "exit",
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "cash": round(cash, 2),
                    "equity": round(mark_equity(ts), 2),
                    "open_positions": len(positions),
                    "return_pct": trade["return_pct"],
                    "pnl": round(proceeds - float(pos["cost"]), 2),
                }
            )
            continue

        equity = mark_equity(ts)
        allocation = 0.20 * equity
        if len(positions) >= 5:
            skipped_slots += 1
            events.append(
                {
                    "timestamp": ts,
                    "event": "skip_max_positions",
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "cash": round(cash, 2),
                    "equity": round(equity, 2),
                    "open_positions": len(positions),
                    "return_pct": "",
                    "pnl": "",
                }
            )
            continue
        if cash < allocation:
            skipped_cash += 1
            events.append(
                {
                    "timestamp": ts,
                    "event": "skip_no_cash",
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "cash": round(cash, 2),
                    "equity": round(equity, 2),
                    "open_positions": len(positions),
                    "return_pct": "",
                    "pnl": "",
                }
            )
            continue

        shares = allocation / float(trade["entry_price"])
        cash -= allocation
        positions[trade_id] = {
            "symbol": symbol,
            "shares": shares,
            "entry_price": float(trade["entry_price"]),
            "cost": allocation,
        }
        taken += 1
        events.append(
            {
                "timestamp": ts,
                "event": "entry",
                "symbol": symbol,
                "trade_id": trade_id,
                "cash": round(cash, 2),
                "equity": round(equity, 2),
                "open_positions": len(positions),
                "return_pct": "",
                "pnl": "",
            }
        )

    final_equity = mark_equity(timeline[-1][0])
    quick_weighted_return = 0.20 * pd.to_numeric(trades["return_pct"]).sum() * 100.0
    portfolio_summary = pd.DataFrame(
        [
            {
                "initial_equity": initial,
                "ending_equity": round(final_equity, 2),
                "portfolio_return_pct": round((final_equity / initial - 1.0) * 100.0, 4),
                "trades_available": len(trades),
                "trades_taken": taken,
                "trades_skipped_max_positions": skipped_slots,
                "trades_skipped_no_cash": skipped_cash,
                "quick_20pct_weighted_return_pct": round(quick_weighted_return, 4),
            }
        ]
    )
    return portfolio_summary, pd.DataFrame(events)


def save_chart(summary_df: pd.DataFrame, portfolio_summary: pd.DataFrame, failed_df: pd.DataFrame, output_path: Path) -> None:
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

    draw.text((60, 28), "Custom universe W-breakout test", fill="#111827", font=title_font)
    draw.text((60, 66), "Full May 2026, 5-minute bars, neckline breakout entry, 2R target, 20% portfolio allocation.", fill="#4b5563", font=label_font)

    if summary_df.empty:
        draw.text((60, 140), "No valid symbols returned data.", fill="#b91c1c", font=section_font)
        image.save(output_path)
        return

    ordered = summary_df.sort_values("total_trade_return_pct", ascending=True).reset_index(drop=True)
    chart_left, chart_right = 230, 1410
    top, bottom = 140, 610

    values = list(ordered["stock_return_pct"].fillna(0.0)) + list(ordered["total_trade_return_pct"].fillna(0.0)) + [0.0]
    min_value = min(values)
    max_value = max(values)
    pad = max((max_value - min_value) * 0.15, 2.0)
    min_value -= pad
    max_value += pad

    def value_to_x(value: float) -> float:
        return chart_left + (value - min_value) / (max_value - min_value) * (chart_right - chart_left)

    zero_x = value_to_x(0.0)
    draw.text((60, 105), "Stock return vs summed W-trade return", fill="#111827", font=section_font)
    draw.line((zero_x, top, zero_x, bottom), fill="#6b7280", width=2)
    draw.line((chart_left, bottom, chart_right, bottom), fill="#9ca3af", width=1)

    for tick in [-30, -20, -10, 0, 10, 20, 30, 40]:
        if min_value <= tick <= max_value:
            x = value_to_x(float(tick))
            draw.line((x, top, x, bottom), fill="#e5e7eb", width=1)
            draw.text((x - 14, bottom + 8), f"{tick}%", fill="#4b5563", font=small_font)

    row_gap = (bottom - top) / len(ordered)
    bar_h = min(16, row_gap * 0.25)
    for idx, row in ordered.iterrows():
        y = top + row_gap * (idx + 0.5)
        draw.text((60, y - 9), str(row["symbol"]), fill="#111827", font=label_font)
        for offset, field, color in [
            (-bar_h * 0.75, "stock_return_pct", "#2563eb"),
            (bar_h * 0.75, "total_trade_return_pct", "#16a34a"),
        ]:
            value = float(row[field] or 0.0)
            end_x = value_to_x(value)
            x0, x1 = sorted([zero_x, end_x])
            draw.rounded_rectangle((x0, y + offset - bar_h / 2, x1, y + offset + bar_h / 2), radius=3, fill=color)
            text_x = end_x + (6 if value >= 0 else -54)
            draw.text((text_x, y + offset - 8), f"{value:.1f}%", fill="#111827", font=small_font)

    draw.rectangle((1040, 105, 1060, 119), fill="#2563eb")
    draw.text((1068, 101), "Stock", fill="#374151", font=small_font)
    draw.rectangle((1130, 105, 1150, 119), fill="#16a34a")
    draw.text((1158, 101), "W trades", fill="#374151", font=small_font)

    p = portfolio_summary.iloc[0]
    footer = (
        f"20% portfolio result: ${p['initial_equity']:,.0f} -> ${p['ending_equity']:,.2f} "
        f"({p['portfolio_return_pct']:.2f}%). Trades taken {int(p['trades_taken'])}/{int(p['trades_available'])}."
    )
    draw.text((60, 675), footer, fill="#111827", font=label_font)

    if not failed_df.empty:
        failed = ", ".join(failed_df["symbol"].astype(str).tolist())
        draw.text((60, 715), f"No data/error for: {failed}", fill="#b91c1c", font=label_font)

    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed W-breakout test on custom tickers.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Tickers to test.")
    parser.add_argument("--interval", default="5m", help="Yahoo interval. Default: 5m.")
    parser.add_argument("--start-date", default="2026-05-01", help="Inclusive start date.")
    parser.add_argument("--end-date", default="2026-06-01", help="Exclusive end date.")
    parser.add_argument("--output-prefix", default="custom_growth_w_breakout", help="Output filename prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.upper().strip() for symbol in args.symbols if symbol.strip()]
    summary_df, trades_df, failed_df, portfolio_summary = run_universe(
        symbols,
        args.interval,
        args.start_date,
        args.end_date,
        args.output_prefix,
    )

    total_trades = int(summary_df["closed_trades"].sum()) if not summary_df.empty else 0
    total_wins = int(summary_df["wins"].sum()) if not summary_df.empty else 0
    hit_rate = 0.0 if total_trades == 0 else total_wins / total_trades * 100.0
    portfolio = portfolio_summary.iloc[0]

    print("Custom W-breakout universe test complete")
    print(f"Symbols requested: {', '.join(symbols)}")
    if not failed_df.empty:
        print(f"Failed symbols: {', '.join(failed_df['symbol'].astype(str))}")
    print(f"Closed trades: {total_trades}; wins: {total_wins}; hit rate: {hit_rate:.1f}%")
    print(f"20% portfolio return: {portfolio['portfolio_return_pct']:.4f}%")
    print(f"Ending equity: ${portfolio['ending_equity']:,.2f}")


if __name__ == "__main__":
    main()
