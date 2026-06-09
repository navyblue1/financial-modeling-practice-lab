# W-Model Trading Research Demo

This repository contains a research/demo workflow for detecting and backtesting a simple W-bottom trading model on intraday stock candles.

The final demo model uses:

- 5-minute candles
- early higher-low W entry
- second low above first low
- W middle/neckline above the W start
- prior 60-minute return greater than 1%
- target at 3R
- minimum target return of 1%
- stop, 240-minute time exit, and regular-session close logic

This code is for quantitative research and demonstration only. It does not place trades automatically and is not financial advice.

## Main Files

- `outputs/w_bottom_demo.py` - original standalone W-bottom demo with manual ticket output.
- `outputs/mag7_ytd_w_bottom_test.py` - daily YTD Mag 7 test.
- `outputs/mag7_intraday_w_bottom_optimizer.py` - intraday W model optimizer with breakout and early higher-low modes.
- `outputs/mag7_sell_time_r_analysis.py` - sell-time / R-multiple analysis.
- `outputs/custom_w_bottom_universe_test.py` - custom small/growth stock universe test.
- `outputs/broad_w_model_optimizer.py` - broader-market model optimizer.
- `outputs/broad_w_trend_filter_analysis.py` - trend-filter comparison and final-model selection.
- `outputs/today_w_signal_scanner.py` - current-day 5-minute scanner for active/pending final-model signals.

## Key Result Artifacts

- `outputs/broad_w_trend_filter_130_5m_summary.csv`
- `outputs/broad_w_trend_filter_130_5m_results.png`
- `outputs/broad_w_trend_filter_130_5m_best_trades.csv`
- `outputs/broad_w_trend_filter_130_5m_symbol_contributors.csv`
- `outputs/today_w_signals.csv`

## Example Commands

Run the final current-day scanner:

```powershell
python outputs/today_w_signal_scanner.py --output-prefix today_w_signals
```

Run the broad optimizer:

```powershell
python outputs/broad_w_model_optimizer.py --output-prefix outputs/broad_w_model
```

Run the trend-filter analysis:

```powershell
python outputs/broad_w_trend_filter_analysis.py
```

## Notes

The scripts use Yahoo Finance chart data for demo/research backtests. Broker integration is intentionally not included because many retail broker accounts, including RBC Direct Investing, do not provide a public automated-trading API for this type of workflow.
