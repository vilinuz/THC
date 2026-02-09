
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backtesting.strategy_backtester import BacktestResult, MultiLayerBacktester, Trade
from db.duckdb_manager import DuckDBManager


def run_wfo():
    print("Starting Walk-Forward Optimization Backtest (NO SMC)...")

    # Configuration
    TRAIN_SIZE = 400  # Bars for HMM training
    TEST_SIZE = 168  # Bars for testing (e.g. 1 week of 1h)
    STEP_SIZE = 72  # Step forward (e.g. 3 days)

    # Load Data
    config_db = {"path": "data/market.duckdb"}
    db_path = str(project_root / config_db["path"])
    parquet_dir = str(project_root / "data/parquet")
    db_manager = DuckDBManager(db_path, parquet_dir)

    symbol = "BTCUSDT"
    # Ensure enough data: Sep 2025 has ~720 bars.
    # With 400 train, we have 320 remaining.
    start_date = "2025-09-01 00:00:00"
    end_date = "2025-09-30 23:59:59"

    try:
        df = db_manager.get_aggregated_ohlcv(
            symbol, start_date, end_date, timeframe="1h"
        )
    finally:
        db_manager.close()

    if len(df) < TRAIN_SIZE + 50:
        print(f"Insufficient data for WFO. Got {len(df)}, need > {TRAIN_SIZE}")
        return

    print(f"Data loaded: {len(df)} bars. Period: {df.index[0]} - {df.index[-1]}")

    # Initialize Backtester
    backtester = MultiLayerBacktester(
        config={
            "initial_capital": 10000.0,
            "strategy_config": {
                "use_smc": False,  # DISABLED SMC
                "stable_trend_threshold": 0.6,
                "hmm_lookback": 100,  # Use smaller lookback for classification
            },
        }
    )

    # WFO Loop
    current_idx = TRAIN_SIZE
    total_trades = []
    
    window_count = 0

    while current_idx < len(df):
        # Define Windows
        train_start_idx = current_idx - TRAIN_SIZE
        train_end_idx = current_idx

        test_start_idx = current_idx
        # Ensure we don't go out of bounds
        test_end_idx = min(current_idx + TEST_SIZE, len(df))

        if test_start_idx >= test_end_idx:
            break

        print(f"\nWindow {window_count + 1}:")
        print(f"  Train: {df.index[train_start_idx]} -> {df.index[train_end_idx-1]}")
        print(f"  Test:  {df.index[test_start_idx]} -> {df.index[test_end_idx-1]}")

        train_df = df.iloc[train_start_idx:train_end_idx]
        test_df = df.iloc[test_start_idx:test_end_idx]

        # 1. Train Strategy on History
        # Reset seed for determinism in this window
        np.random.seed(42 + window_count)

        print(f"  Training on {len(train_df)} bars...")
        backtester.strategy.train(train_df)

        # 2. Run Backtest on Test Window
        # Note: 'train_ratio=0' because we already trained manually
        print(f"  Testing on {len(test_df)} bars...")
        
        # Skip if test window is too small for backtester
        if len(test_df) < 100:
            print(f"  Skipping window (Insufficient data: {len(test_df)} < 100)")
            break

        result = backtester.run(test_df, train_ratio=0.0)

        # 3. Collect Results
        if result.trades:
            print(
                f"  Trades: {len(result.trades)} | Return: ${result.total_return:.2f}"
            )
            total_trades.extend(result.trades)
        else:
            print("  No trades.")

        # Move Forward
        current_idx += STEP_SIZE
        window_count += 1

    # === Aggregation ===
    print("\n" + "=" * 50)
    print("WALK-FORWARD OPTIMIZATION RESULTS (NO SMC)")
    print("=" * 50)

    if not total_trades:
        print("No trades generated across all windows.")
        return

    # Calculate aggregate metrics
    initial_cap = 10000.0
    total_pnl = sum([t.pnl for t in total_trades if t.pnl])
    final_cap = initial_cap + total_pnl
    return_pct = total_pnl / initial_cap

    wins = [t for t in total_trades if t.pnl > 0]
    losses = [t for t in total_trades if t.pnl <= 0]
    win_rate = len(wins) / len(total_trades) if total_trades else 0

    print(f"Total Trades:    {len(total_trades)}")
    print(f"Win Rate:        {win_rate:.2%}")
    print(f"Initial Capital: ${initial_cap:,.2f}")
    print(f"Final Capital:   ${final_cap:,.2f}")
    print(f"Total Return:    ${total_pnl:,.2f} ({return_pct:.2%})")

    # Profit Factor
    gross_profit = sum([t.pnl for t in wins])
    gross_loss = abs(sum([t.pnl for t in losses]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"Profit Factor:   {pf:.2f}")


if __name__ == "__main__":
    run_wfo()
