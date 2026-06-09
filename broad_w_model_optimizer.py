#!/usr/bin/env python3
"""
Broad W-model optimizer for growth/volatile US stocks.

Research/demo only. It scans a configurable stock universe over 5m, 15m, and
30m candles, tests several W-pattern variants, requires targets to be at least
1% above entry, and counts a quality win only if realized return is >= 1%.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mag7_intraday_w_bottom_optimizer import (
    detect_intraday_w_bottoms,
    fetch_yahoo_intraday,
    parse_date,
)
from w_bottom_demo import StrategyConfig


OUTPUT_DIR = Path(__file__).resolve().parent / "data_and_testing"
DEFAULT_START_DATE = "2026-05-01"
DEFAULT_END_DATE = "2026-06-01"
MIN_PROFIT_RETURN = 0.01


DEFAULT_UNIVERSE = [
    "MU", "SNDK", "GLW", "OKLO", "RGTI", "IONQ", "CSCO", "AMKR", "ASTS", "RKLB",
    "AMD", "AVGO", "ARM", "SMCI", "TSM", "ASML", "MRVL", "ON", "LRCX", "KLAC",
    "QCOM", "ADI", "MCHP", "MPWR", "LSCC", "ALAB", "CRDO", "COHR", "WOLF", "AEHR",
    "PLTR", "APP", "SOUN", "BBAI", "AI", "PATH", "UPST", "AFRM", "HOOD", "COIN",
    "MARA", "RIOT", "CLSK", "IREN", "CORZ", "CIFR", "WULF", "HIVE", "BTDR", "HUT",
    "QBTS", "QUBT", "ARQQ", "LAES", "QS", "ENVX", "JOBY", "ACHR", "LUNR", "SPCE",
    "NNE", "SMR", "LEU", "CCJ", "UEC", "DNN", "RUN", "ENPH", "SEDG", "FSLR",
    "BE", "PLUG", "BLDP", "LAC", "ALB", "RIVN", "LCID", "NIO", "XPEV", "LI",
    "CHPT", "BLNK", "SERV", "TEM", "RXRX", "SDGR", "CRSP", "EDIT", "BEAM", "NTLA",
    "PACB", "DNA", "TMDX", "GH", "CELH", "ELF", "CAVA", "HIMS", "DDOG", "NET",
    "CRWD", "SNOW", "MDB", "SHOP", "SE", "MELI", "NU", "TOST", "RBLX", "U",
    "DKNG", "RDDT", "CART", "SOFI", "UWM", "OPEN", "RDFN", "CVNA", "W", "ETSY",
    "FUBO", "ROKU", "TTD", "BILL", "TWLO", "DOCN", "FROG", "ESTC", "GTLB", "CFLT",
]


@dataclass(frozen=True)
class Variant:
    name: str
    entry_mode: str
    swing_window: int
    low_tolerance_pct: float
    min_rebound_pct: float
    breakout_buffer_pct: float
    stop_buffer_pct: float
    reward_risk: float
    hold_minutes: int
    w_width_bars: int


VARIANTS = [
    Variant("breakout_base_2r", "breakout", 2, 0.012, 0.0025, 0.0006, 0.005, 2.0, 200, 24),
    Variant("breakout_wide_2r", "breakout", 2, 0.020, 0.0040, 0.0000, 0.0075, 2.0, 240, 30),
    Variant("breakout_tight_2p5r", "breakout", 2, 0.010, 0.0060, 0.0008, 0.0050, 2.5, 240, 24),
    Variant("breakout_wide_3r", "breakout", 2, 0.020, 0.0060, 0.0006, 0.0100, 3.0, 300, 30),
    Variant("early_hl_base_2r", "early-higher-low", 2, 0.012, 0.0060, 0.0000, 0.0025, 2.0, 100, 24),
    Variant("early_hl_wide_2r", "early-higher-low", 2, 0.020, 0.0060, 0.0000, 0.0050, 2.0, 160, 30),
    Variant("early_hl_tight_2p5r", "early-higher-low", 2, 0.010, 0.0080, 0.0000, 0.0025, 2.5, 160, 24),
    Variant("early_hl_wide_3r", "early-higher-low", 2, 0.020, 0.0100, 0.0000, 0.0050, 3.0, 240, 30),
]


def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    raise ValueError(f"Unsupported interval: {interval}")


def config_from_variant(variant: Variant, interval: str) -> StrategyConfig:
    bars = max(1, math.ceil(variant.hold_minutes / interval_to_minutes(interval)))
    return StrategyConfig(
        swing_window=variant.swing_window,
        low_tolerance_pct=variant.low_tolerance_pct,
        max_second_low_undercut_pct=min(variant.low_tolerance_pct, 0.01),
        min_rebound_pct=variant.min_rebound_pct,
        breakout_lookahead=max(6, int(variant.w_width_bars / 2)),
        breakout_buffer_pct=variant.breakout_buffer_pct,
        volume_sma_window=20,
        require_volume_confirmation=False,
        volume_multiplier=1.05,
        max_first_low_to_neckline_bars=variant.w_width_bars,
        max_neckline_to_second_low_bars=variant.w_width_bars,
        stop_buffer_pct=variant.stop_buffer_pct,
        reward_risk=variant.reward_risk,
        max_hold_days=bars,
        starting_cash=100_000.0,
        risk_per_trade_pct=0.003,
        max_position_pct=0.20,
        commission_per_order=0.0,
    )


def load_or_fetch_prices(symbol: str, interval: str, start_date: str, end_date: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.csv"
    if path.exists():
        return pd.read_csv(path)

    df, _meta = fetch_yahoo_intraday(symbol, interval, None, parse_date(start_date), parse_date(end_date))
    df.to_csv(path, index=False)
    return df


def backtest_patterns(df: pd.DataFrame, patterns, cfg: StrategyConfig, min_profit_return: float) -> pd.DataFrame:
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
                        "quality_win": bool(return_pct >= min_profit_return),
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
            target_return = target_price / entry_price - 1.0
            if stop_price >= entry_price or target_return < min_profit_return:
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
                    "target_return_pct": round(target_return * 100.0, 4),
                    "exit_date": "",
                    "exit_price": "",
                    "exit_reason": "",
                    "bars_held": "",
                    "pnl_per_share": "",
                    "return_pct": "",
                    "r_multiple": "",
                    "win": "",
                    "quality_win": "",
                }
            )

    if position is not None:
        row = df.iloc[-1]
        entry_price = float(position["entry_price"])
        exit_price = float(row["Close"])
        risk_per_share = max(entry_price - float(position["stop_price"]), 0.01)
        return_pct = exit_price / entry_price - 1.0
        trades[-1].update(
            {
                "exit_date": str(row["Date"]),
                "exit_price": round(exit_price, 4),
                "exit_reason": "end_of_data",
                "bars_held": len(df) - 1 - int(position["entry_idx"]),
                "pnl_per_share": round(exit_price - entry_price, 4),
                "return_pct": round(return_pct, 5),
                "r_multiple": round((exit_price - entry_price) / risk_per_share, 2),
                "win": bool(exit_price > entry_price),
                "quality_win": bool(return_pct >= min_profit_return),
            }
        )

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "quality_wins": 0,
            "hit_rate_pct": 0.0,
            "quality_win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "total_trade_return_pct": 0.0,
            "avg_r_multiple": 0.0,
        }

    returns = pd.to_numeric(trades["return_pct"], errors="coerce").fillna(0.0)
    r_values = pd.to_numeric(trades["r_multiple"], errors="coerce").fillna(0.0)
    wins = int((returns > 0).sum())
    quality_wins = int((returns >= MIN_PROFIT_RETURN).sum())
    trade_count = len(trades)
    return {
        "trades": trade_count,
        "wins": wins,
        "quality_wins": quality_wins,
        "hit_rate_pct": round(wins / trade_count * 100.0, 2),
        "quality_win_rate_pct": round(quality_wins / trade_count * 100.0, 2),
        "avg_return_pct": round(float(returns.mean()) * 100.0, 4),
        "total_trade_return_pct": round(float(returns.sum()) * 100.0, 4),
        "avg_r_multiple": round(float(r_values.mean()), 4),
    }


def simulate_portfolio(trades_df: pd.DataFrame, price_frames: dict[tuple[str, str], pd.DataFrame], interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial = 100_000.0
    if trades_df.empty:
        return pd.DataFrame([{"initial_equity": initial, "ending_equity": initial, "portfolio_return_pct": 0.0}]), pd.DataFrame()

    trades = trades_df.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades = trades.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    indexed_prices = {
        key: frame.assign(Date=pd.to_datetime(frame["Date"])).set_index("Date")
        for key, frame in price_frames.items()
        if key[1] == interval
    }

    cash = initial
    positions: dict[int, dict[str, object]] = {}
    events: list[dict[str, object]] = []
    taken = 0
    skipped_slots = 0
    skipped_cash = 0

    def mark_equity(ts) -> float:
        total = cash
        for pos in positions.values():
            key = (str(pos["symbol"]), interval)
            frame = indexed_prices[key]
            idx = frame.index.searchsorted(ts, side="right") - 1
            px = float(pos["entry_price"]) if idx < 0 else float(frame.iloc[idx]["Close"])
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
            equity = mark_equity(ts)
            events.append(
                {
                    "timestamp": ts,
                    "event": "exit",
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "cash": round(cash, 2),
                    "equity": round(equity, 2),
                    "open_positions": len(positions),
                    "return_pct": trade["return_pct"],
                    "quality_win": trade["quality_win"],
                }
            )
            continue

        equity = mark_equity(ts)
        allocation = 0.20 * equity
        if len(positions) >= 5:
            skipped_slots += 1
            events.append({"timestamp": ts, "event": "skip_max_positions", "symbol": symbol, "trade_id": trade_id, "cash": round(cash, 2), "equity": round(equity, 2), "open_positions": len(positions), "return_pct": "", "quality_win": ""})
            continue
        if cash < allocation:
            skipped_cash += 1
            events.append({"timestamp": ts, "event": "skip_no_cash", "symbol": symbol, "trade_id": trade_id, "cash": round(cash, 2), "equity": round(equity, 2), "open_positions": len(positions), "return_pct": "", "quality_win": ""})
            continue

        shares = allocation / float(trade["entry_price"])
        cash -= allocation
        positions[trade_id] = {"symbol": symbol, "shares": shares, "entry_price": float(trade["entry_price"])}
        taken += 1
        events.append({"timestamp": ts, "event": "entry", "symbol": symbol, "trade_id": trade_id, "cash": round(cash, 2), "equity": round(equity, 2), "open_positions": len(positions), "return_pct": "", "quality_win": ""})

    final_equity = mark_equity(timeline[-1][0])
    events_df = pd.DataFrame(events)
    if not events_df.empty and "equity" in events_df:
        equity_series = pd.to_numeric(events_df["equity"], errors="coerce").dropna()
        max_drawdown = 0.0 if equity_series.empty else float((equity_series / equity_series.cummax() - 1.0).min() * 100.0)
    else:
        max_drawdown = 0.0

    portfolio_df = pd.DataFrame(
        [
            {
                "initial_equity": initial,
                "ending_equity": round(final_equity, 2),
                "portfolio_return_pct": round((final_equity / initial - 1.0) * 100.0, 4),
                "max_drawdown_pct": round(max_drawdown, 4),
                "trades_available": len(trades),
                "trades_taken": taken,
                "trades_skipped_max_positions": skipped_slots,
                "trades_skipped_no_cash": skipped_cash,
                "quick_20pct_weighted_return_pct": round(0.20 * pd.to_numeric(trades["return_pct"]).sum() * 100.0, 4),
            }
        ]
    )
    return portfolio_df, events_df


def run_scan(symbols: list[str], intervals: list[str], start_date: str, end_date: str, max_symbols: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_symbols = symbols[:max_symbols] if max_symbols else symbols
    cache_dir = OUTPUT_DIR / "broad_w_prices"
    cache_dir.mkdir(exist_ok=True)

    price_frames: dict[tuple[str, str], pd.DataFrame] = {}
    failures: list[dict[str, object]] = []

    for interval in intervals:
        for symbol in selected_symbols:
            try:
                price_frames[(symbol, interval)] = load_or_fetch_prices(symbol, interval, start_date, end_date, cache_dir)
            except Exception as exc:
                failures.append({"symbol": symbol, "interval": interval, "error": str(exc)})

    algorithm_rows = []
    all_trades = []
    symbol_rows = []
    best_events = pd.DataFrame()
    best_key: tuple[float, float, float, int] | None = None
    best_trades = pd.DataFrame()
    best_portfolio = pd.DataFrame()

    for interval in intervals:
        for variant in VARIANTS:
            cfg = config_from_variant(variant, interval)
            variant_trades = []
            variant_symbol_rows = []

            for symbol in selected_symbols:
                df = price_frames.get((symbol, interval))
                if df is None or df.empty:
                    continue
                patterns = detect_intraday_w_bottoms(df, symbol, cfg, variant.entry_mode)
                trades = backtest_patterns(df, patterns, cfg, MIN_PROFIT_RETURN)
                stats = summarize_trades(trades)
                stock_return = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0

                variant_symbol_rows.append(
                    {
                        "interval": interval,
                        "variant": variant.name,
                        "entry_mode": variant.entry_mode,
                        "symbol": symbol,
                        "stock_return_pct": round(stock_return, 4),
                        "patterns": len(patterns),
                        "trades": stats["trades"],
                        "wins": stats["wins"],
                        "quality_wins": stats["quality_wins"],
                        "hit_rate_pct": stats["hit_rate_pct"],
                        "quality_win_rate_pct": stats["quality_win_rate_pct"],
                        "avg_return_pct": stats["avg_return_pct"],
                        "total_trade_return_pct": stats["total_trade_return_pct"],
                        "avg_r_multiple": stats["avg_r_multiple"],
                    }
                )

                if not trades.empty:
                    trades = trades.copy()
                    trades.insert(0, "interval", interval)
                    trades.insert(1, "variant", variant.name)
                    trades.insert(2, "entry_mode", variant.entry_mode)
                    variant_trades.extend(trades.to_dict("records"))

            trades_df = pd.DataFrame(variant_trades)
            portfolio_df, events_df = simulate_portfolio(trades_df, price_frames, interval)
            stats = summarize_trades(trades_df)
            portfolio = portfolio_df.iloc[0].to_dict()
            row = {
                "interval": interval,
                "variant": variant.name,
                "entry_mode": variant.entry_mode,
                "symbols_tested": sum((symbol, interval) in price_frames for symbol in selected_symbols),
                "hold_minutes": variant.hold_minutes,
                "reward_risk": variant.reward_risk,
                "low_tolerance_pct": variant.low_tolerance_pct,
                "min_rebound_pct": variant.min_rebound_pct,
                "stop_buffer_pct": variant.stop_buffer_pct,
                **stats,
                **portfolio,
            }
            algorithm_rows.append(row)
            symbol_rows.extend(variant_symbol_rows)
            all_trades.extend(variant_trades)

            key = (
                float(portfolio["portfolio_return_pct"]),
                float(stats["quality_win_rate_pct"]),
                float(stats["avg_r_multiple"]),
                int(stats["trades"]),
            )
            if stats["trades"] >= 20 and (best_key is None or key > best_key):
                best_key = key
                best_trades = trades_df
                best_portfolio = portfolio_df
                best_events = events_df

    return (
        pd.DataFrame(algorithm_rows),
        pd.DataFrame(symbol_rows),
        pd.DataFrame(all_trades),
        pd.DataFrame(failures),
        best_trades,
        best_portfolio,
        best_events,
    )


def save_chart(algorithm_df: pd.DataFrame, symbol_df: pd.DataFrame, best_trades: pd.DataFrame, best_portfolio: pd.DataFrame, output_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 1000
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

    draw.text((60, 28), "Broad W-model optimizer", fill="#111827", font=title_font)
    draw.text((60, 66), "5m/15m/30m candles, target >= 1%, quality win = realized return >= 1%.", fill="#4b5563", font=label_font)

    top_algos = algorithm_df.sort_values("portfolio_return_pct", ascending=False).head(10).reset_index(drop=True)
    draw.text((60, 110), "Top algorithm recipes by 20% portfolio return", fill="#111827", font=section_font)
    x0, y0 = 60, 145
    headers = ["interval", "variant", "portfolio", "quality win", "trades", "avg R", "drawdown"]
    widths = [80, 245, 110, 110, 70, 80, 90]
    x = x0
    for header, width_col in zip(headers, widths):
        draw.text((x, y0), header, fill="#374151", font=small_font)
        x += width_col
    y = y0 + 24
    for _, row in top_algos.iterrows():
        values = [
            row["interval"],
            row["variant"],
            f"{row['portfolio_return_pct']:.2f}%",
            f"{row['quality_win_rate_pct']:.1f}%",
            str(int(row["trades"])),
            f"{row['avg_r_multiple']:.3f}",
            f"{row['max_drawdown_pct']:.2f}%",
        ]
        x = x0
        for value, width_col in zip(values, widths):
            draw.text((x, y), str(value), fill="#111827", font=small_font)
            x += width_col
        y += 22

    if not best_trades.empty:
        best_interval = str(best_trades["interval"].iloc[0])
        best_variant = str(best_trades["variant"].iloc[0])
        top_symbols = (
            best_trades.assign(return_pct=pd.to_numeric(best_trades["return_pct"], errors="coerce"))
            .groupby("symbol", as_index=False)
            .agg(trades=("return_pct", "count"), total_return_pct=("return_pct", lambda s: float(s.sum()) * 100.0), quality_wins=("quality_win", lambda s: int(pd.Series(s).astype(str).str.lower().isin(["true", "1"]).sum())))
            .sort_values("total_return_pct", ascending=True)
            .tail(15)
        )
        draw.text((60, 420), f"Best recipe symbol contributors: {best_interval} {best_variant}", fill="#111827", font=section_font)
        chart_left, chart_right = 220, 1500
        chart_top, chart_bottom = 470, 880
        vals = list(top_symbols["total_return_pct"]) + [0.0]
        min_v, max_v = min(vals), max(vals)
        pad = max((max_v - min_v) * 0.15, 2.0)
        min_v -= pad
        max_v += pad

        def value_to_x(value: float) -> float:
            return chart_left + (value - min_v) / (max_v - min_v) * (chart_right - chart_left)

        zero_x = value_to_x(0.0)
        draw.line((zero_x, chart_top, zero_x, chart_bottom), fill="#6b7280", width=2)
        row_gap = (chart_bottom - chart_top) / max(len(top_symbols), 1)
        for idx, row in top_symbols.reset_index(drop=True).iterrows():
            yy = chart_top + row_gap * (idx + 0.5)
            draw.text((60, yy - 8), f"{row['symbol']} ({int(row['trades'])})", fill="#111827", font=label_font)
            value = float(row["total_return_pct"])
            end_x = value_to_x(value)
            xa, xb = sorted([zero_x, end_x])
            draw.rounded_rectangle((xa, yy - 8, xb, yy + 8), radius=3, fill="#16a34a" if value >= 0 else "#dc2626")
            draw.text((end_x + (6 if value >= 0 else -58), yy - 8), f"{value:.1f}%", fill="#111827", font=small_font)

    if not best_portfolio.empty:
        p = best_portfolio.iloc[0]
        draw.text((60, 930), f"Best portfolio: ${p['initial_equity']:,.0f} -> ${p['ending_equity']:,.2f} ({p['portfolio_return_pct']:.2f}%), max drawdown {p['max_drawdown_pct']:.2f}%.", fill="#374151", font=label_font)

    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broad W-model optimizer.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE, help="Symbols to scan.")
    parser.add_argument("--max-symbols", type=int, default=80, help="Limit symbols for this run. Default: 80.")
    parser.add_argument("--intervals", nargs="*", default=["5m", "15m", "30m"], help="Intervals to test.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output-prefix", default="broad_w_model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.upper().strip() for symbol in args.symbols if symbol.strip()]
    algorithm_df, symbol_df, all_trades, failures_df, best_trades, best_portfolio, best_events = run_scan(
        symbols,
        args.intervals,
        args.start_date,
        args.end_date,
        args.max_symbols,
    )

    prefix = args.output_prefix
    algorithm_df.to_csv(OUTPUT_DIR / f"{prefix}_algorithm_summary.csv", index=False)
    symbol_df.to_csv(OUTPUT_DIR / f"{prefix}_symbol_summary.csv", index=False)
    all_trades.to_csv(OUTPUT_DIR / f"{prefix}_all_trades.csv", index=False)
    failures_df.to_csv(OUTPUT_DIR / f"{prefix}_failed_downloads.csv", index=False)
    best_trades.to_csv(OUTPUT_DIR / f"{prefix}_best_trades.csv", index=False)
    best_portfolio.to_csv(OUTPUT_DIR / f"{prefix}_best_portfolio_summary.csv", index=False)
    best_events.to_csv(OUTPUT_DIR / f"{prefix}_best_portfolio_events.csv", index=False)
    save_chart(algorithm_df, symbol_df, best_trades, best_portfolio, OUTPUT_DIR / f"{prefix}_results.png")

    best_algo = algorithm_df.sort_values("portfolio_return_pct", ascending=False).iloc[0]
    print("Broad W-model optimization complete")
    print(f"Symbols requested: {len(symbols)}; max scanned: {args.max_symbols}; intervals: {', '.join(args.intervals)}")
    print(f"Best recipe: {best_algo['interval']} {best_algo['variant']} ({best_algo['entry_mode']})")
    print(f"Portfolio return: {best_algo['portfolio_return_pct']:.4f}%")
    print(f"Quality win rate: {best_algo['quality_win_rate_pct']:.2f}% on {int(best_algo['trades'])} trades")
    print(f"Average R: {best_algo['avg_r_multiple']:.4f}; max drawdown: {best_algo['max_drawdown_pct']:.4f}%")


if __name__ == "__main__":
    main()
