#!/usr/bin/env python3
"""
Scan today's intraday bars for the final W-model signals.

Research/demo only. This does not place trades. It reports model signals:
- ACTIVE: entered today and still open at the latest available bar
- PENDING_NEXT_BAR: W signal formed on the latest bar; model entry would be next bar
- CLOSED_TODAY: entered and exited today by stop/target/time/session rule
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from broad_w_model_optimizer import DEFAULT_UNIVERSE, Variant, config_from_variant
from mag7_intraday_w_bottom_optimizer import detect_intraday_w_bottoms


OUTPUT_DIR = Path(__file__).resolve().parent / "data_and_testing"
INTERVAL = "5m"
TREND_BARS = 12
MIN_TREND_RETURN = 0.01
MIN_TARGET_RETURN = 0.01

FINAL_VARIANT = Variant(
    name="final_early_hl_wide_3r_trend60",
    entry_mode="early-higher-low",
    swing_window=2,
    low_tolerance_pct=0.020,
    min_rebound_pct=0.010,
    breakout_buffer_pct=0.0,
    stop_buffer_pct=0.005,
    reward_risk=3.0,
    hold_minutes=240,
    w_width_bars=30,
)


def yahoo_range_url(symbol: str, interval: str, range_: str) -> str:
    params = {
        "range": range_,
        "interval": interval,
        "includePrePost": "false",
        "events": "history",
    }
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{urllib.parse.urlencode(params)}"


def fetch_today(symbol: str, interval: str = INTERVAL, range_: str = "1d") -> tuple[pd.DataFrame, dict[str, object]]:
    url = yahoo_range_url(symbol, interval, range_)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 TodayWScanner/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])
    result = chart.get("result") or []
    if not result:
        raise RuntimeError("No chart result")

    data = result[0]
    timestamps = data.get("timestamp") or []
    quote = (data.get("indicators", {}).get("quote") or [{}])[0]
    meta = data.get("meta", {})
    timezone_name = meta.get("exchangeTimezoneName", "America/New_York")

    rows = []
    for i, ts in enumerate(timestamps):
        try:
            dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert(timezone_name)
            rows.append(
                {
                    "Date": dt.strftime("%Y-%m-%d %H:%M"),
                    "session": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M"),
                    "Open": quote["open"][i],
                    "High": quote["high"][i],
                    "Low": quote["low"][i],
                    "Close": quote["close"][i],
                    "Volume": quote["volume"][i],
                }
            )
        except (KeyError, IndexError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable bars")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    df["Volume"] = df["Volume"].astype("int64")
    return df, meta


def trend_60m_ok(df: pd.DataFrame, entry_idx: int) -> tuple[bool, float]:
    prev_idx = entry_idx - 1
    base_idx = prev_idx - TREND_BARS
    if base_idx < 0:
        return False, 0.0
    base = float(df.iloc[base_idx]["Close"])
    latest = float(df.iloc[prev_idx]["Close"])
    trend_return = latest / base - 1.0 if base > 0 else 0.0
    return trend_return > MIN_TREND_RETURN, trend_return


def simulate_symbol(symbol: str, df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cfg = config_from_variant(FINAL_VARIANT, INTERVAL)
    patterns = detect_intraday_w_bottoms(df, symbol, cfg, FINAL_VARIANT.entry_mode)
    pattern_by_entry_idx = {
        pattern.breakout_idx + 1: pattern
        for pattern in patterns
    }

    records: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    position: dict[str, object] | None = None
    latest_idx = len(df) - 1
    latest_time = str(df.iloc[-1]["Date"])

    for pattern in patterns:
        if pattern.breakout_idx == latest_idx and pattern.breakout_idx + 1 >= len(df):
            pending.append(
                {
                    "symbol": symbol,
                    "status": "PENDING_NEXT_BAR",
                    "signal_time": pattern.breakout_date,
                    "latest_bar": latest_time,
                    "first_low": pattern.first_low_price,
                    "second_low": pattern.second_low_price,
                    "neckline": pattern.neckline_price,
                    "note": "Signal formed on latest bar; model entry requires next bar open.",
                }
            )

    for idx, row in df.iterrows():
        if position is not None:
            exit_price = None
            exit_reason = ""
            if str(row["session"]) != str(position["entry_session"]):
                prev = df.iloc[idx - 1]
                exit_price = float(prev["Close"])
                exit_reason = "session_close"
                exit_time = str(prev["Date"])
                bars_held = idx - 1 - int(position["entry_idx"])
            elif float(row["Low"]) <= float(position["stop_price"]):
                exit_price = float(position["stop_price"])
                exit_reason = "stop"
                exit_time = str(row["Date"])
                bars_held = idx - int(position["entry_idx"])
            elif float(row["High"]) >= float(position["target_price"]):
                exit_price = float(position["target_price"])
                exit_reason = "target"
                exit_time = str(row["Date"])
                bars_held = idx - int(position["entry_idx"])
            elif idx - int(position["entry_idx"]) >= cfg.max_hold_days:
                exit_price = float(row["Close"])
                exit_reason = "time_exit"
                exit_time = str(row["Date"])
                bars_held = idx - int(position["entry_idx"])

            if exit_price is not None:
                entry_price = float(position["entry_price"])
                return_pct = exit_price / entry_price - 1.0
                records.append(
                    {
                        **position,
                        "status": "CLOSED_TODAY",
                        "exit_time": exit_time,
                        "exit_price": round(exit_price, 4),
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                        "return_pct": round(return_pct * 100.0, 4),
                        "quality_win": bool(return_pct >= MIN_TARGET_RETURN),
                    }
                )
                position = None

        if position is None and idx in pattern_by_entry_idx:
            pattern = pattern_by_entry_idx[idx]
            if str(row["time"]) >= "15:45":
                continue
            ok, trend_return = trend_60m_ok(df, idx)
            if not ok:
                continue

            entry_price = float(row["Open"])
            stop_price = float(pattern.stop_price)
            target_price = entry_price + cfg.reward_risk * (entry_price - stop_price)
            target_return = target_price / entry_price - 1.0
            if stop_price >= entry_price or target_return < MIN_TARGET_RETURN:
                continue

            position = {
                "symbol": symbol,
                "entry_time": str(row["Date"]),
                "entry_idx": idx,
                "entry_session": str(row["session"]),
                "entry_price": round(entry_price, 4),
                "stop_price": round(stop_price, 4),
                "target_price": round(target_price, 4),
                "target_return_pct": round(target_return * 100.0, 4),
                "trend_60m_pct": round(trend_return * 100.0, 4),
                "first_low_time": pattern.first_low_date,
                "second_low_time": pattern.second_low_date,
                "neckline_time": pattern.neckline_date,
                "first_low": pattern.first_low_price,
                "second_low": pattern.second_low_price,
                "neckline": pattern.neckline_price,
            }

    if position is not None:
        latest = df.iloc[-1]
        entry_price = float(position["entry_price"])
        latest_price = float(latest["Close"])
        return_pct = latest_price / entry_price - 1.0
        records.append(
            {
                **position,
                "status": "ACTIVE",
                "latest_time": str(latest["Date"]),
                "latest_price": round(latest_price, 4),
                "unrealized_return_pct": round(return_pct * 100.0, 4),
                "bars_open": latest_idx - int(position["entry_idx"]),
                "distance_to_stop_pct": round((latest_price / float(position["stop_price"]) - 1.0) * 100.0, 4),
                "distance_to_target_pct": round((float(position["target_price"]) / latest_price - 1.0) * 100.0, 4),
            }
        )

    return records, pending


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan today's current W-model signals.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE, help="Symbols to scan.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional cap for symbols.")
    parser.add_argument("--output-prefix", default="today_w_signals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [symbol.upper().strip() for symbol in args.symbols if symbol.strip()]
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    signal_rows = []
    pending_rows = []
    failure_rows = []
    metadata_rows = []

    for symbol in symbols:
        try:
            df, meta = fetch_today(symbol)
            metadata_rows.append(
                {
                    "symbol": symbol,
                    "bars": len(df),
                    "first_bar": df["Date"].iloc[0],
                    "last_bar": df["Date"].iloc[-1],
                    "regular_market_price": meta.get("regularMarketPrice"),
                    "exchange": meta.get("exchangeName"),
                }
            )
            records, pending = simulate_symbol(symbol, df)
            signal_rows.extend(records)
            pending_rows.extend(pending)
        except Exception as exc:
            failure_rows.append({"symbol": symbol, "error": str(exc)})

    signals = pd.DataFrame(signal_rows)
    pending_df = pd.DataFrame(pending_rows)
    failures = pd.DataFrame(failure_rows)
    metadata = pd.DataFrame(metadata_rows)

    if not signals.empty:
        signals = signals.sort_values(["status", "entry_time", "symbol"], ascending=[True, False, True])
    signals.to_csv(OUTPUT_DIR / f"{args.output_prefix}.csv", index=False)
    pending_df.to_csv(OUTPUT_DIR / f"{args.output_prefix}_pending.csv", index=False)
    failures.to_csv(OUTPUT_DIR / f"{args.output_prefix}_failures.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / f"{args.output_prefix}_metadata.csv", index=False)

    active = signals[signals["status"].eq("ACTIVE")] if not signals.empty else pd.DataFrame()
    closed = signals[signals["status"].eq("CLOSED_TODAY")] if not signals.empty else pd.DataFrame()
    print("Today W-signal scan complete")
    print(f"Symbols scanned: {len(metadata)} / requested {len(symbols)}")
    print(f"Active signals now: {len(active)}")
    print(f"Pending next-bar signals: {len(pending_df)}")
    print(f"Closed model trades today: {len(closed)}")
    if not active.empty:
        cols = ["symbol", "entry_time", "entry_price", "latest_price", "unrealized_return_pct", "stop_price", "target_price", "bars_open", "trend_60m_pct"]
        print(active[cols].to_csv(index=False))
    if not pending_df.empty:
        print("Pending:")
        print(pending_df.to_csv(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
