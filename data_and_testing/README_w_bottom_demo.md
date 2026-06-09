# W-Bottom Trading Demo

This demo shows how AI-assisted trading code can detect a simple bottom W pattern, backtest it, and export manual order tickets for review. It does not connect to RBC Direct Investing or any other broker.

## Pattern Logic

The strategy treats a bottom W as a double-bottom breakout:

1. First swing low forms.
2. Price rebounds to a middle peak, also called the neckline.
3. Price pulls back to a second swing low near the first low.
4. Price closes above the neckline by a small buffer.
5. The demo enters on the next session open, with a protective stop below the lower low and a target based on reward/risk.

## Run The Demo

From this workspace, run:

```powershell
C:\Users\jeffr\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe C:\Users\jeffr\Documents\Codex\2026-06-07\i-am-a-quantitative-trader-and\outputs\w_bottom_demo.py
```

The default run creates synthetic sample market data, then writes:

- `sample_market_data.csv`
- `w_bottom_patterns.csv`
- `w_bottom_trades.csv`
- `w_bottom_equity_curve.csv`
- `w_bottom_manual_order_tickets.csv`
- `w_bottom_demo_chart.png`

## Use Your Own Data

Provide a CSV with these columns:

```text
Date,Open,High,Low,Close,Volume
```

Run:

```powershell
C:\Users\jeffr\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe C:\Users\jeffr\Documents\Codex\2026-06-07\i-am-a-quantitative-trader-and\outputs\w_bottom_demo.py --input-csv C:\path\to\your_prices.csv --symbol RY.TO
```

Optional stricter breakout volume filter:

```powershell
C:\Users\jeffr\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe C:\Users\jeffr\Documents\Codex\2026-06-07\i-am-a-quantitative-trader-and\outputs\w_bottom_demo.py --volume-confirmation
```

## Risk Controls

The default model risks 1% of equity per trade, caps one position at 25% of equity, uses a stop 2% below the lower W low, and exits after 25 bars if neither stop nor target is reached.

For a brokerage demo with RBC Direct Investing, use `w_bottom_manual_order_tickets.csv` as a review artifact only. Do not automate RBC login or order entry from this demo.
