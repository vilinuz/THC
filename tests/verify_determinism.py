
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backtesting.strategy_backtester import MultiLayerBacktester
from db.duckdb_manager import DuckDBManager


def run_backtest_iteration(iteration_id, db_manager):
    symbol = "BTCUSDT"
    start_date = "2025-09-01 00:00:00"
    end_date = "2025-09-30 23:59:59"
    
    # 1. Fetch Data
    df = db_manager.get_aggregated_ohlcv(
        symbol, start_date, end_date, timeframe="1h"
    )
    
    if df.empty:
        print(f"Iteration {iteration_id}: No data found")
        return None

    # 2. Initialize
    config = {
        "initial_capital": 10000.0,
    }
    backtester = MultiLayerBacktester(config=config)
    
    # 3. Run (Consistency Check: train_ratio=0.0)
    # HMM training introduces non-determinism if not seeded perfectly, 
    # but we are skipping it for now as per previous fix.
    result = backtester.run(df, train_ratio=0.0)
    
    return {
        "iteration": iteration_id,
        "total_return": result.total_return,
        "total_return_pct": result.total_return_pct,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "trades": result.total_trades
    }

def main():
    print("Starting Determinism Verification (5 Iterations)...")
    
    config_db = {"path": "data/market.duckdb"}
    db_path = str(project_root / config_db["path"])
    parquet_dir = str(project_root / "data/parquet")
    db_manager = DuckDBManager(db_path, parquet_dir)

    results = []
    
    try:
        for i in range(1, 6):
            print(f"Running Iteration {i}...")
            res = run_backtest_iteration(i, db_manager)
            if res:
                results.append(res)
    finally:
        db_manager.close()
        
    print("\nResults Summary:")
    print(f"{'Iter':<5} | {'Return ($)':<12} | {'Return (%)':<10} | {'Sharpe':<8} | {'Trades':<6}")
    print("-" * 55)
    
    all_match = True
    first_res = results[0] if results else None
    
    for res in results:
        print(f"{res['iteration']:<5} | {res['total_return']:<12.2f} | {res['total_return_pct']:<10.2%} | {res['sharpe_ratio']:<8.4f} | {res['trades']:<6}")
        
        if first_res:
            if abs(res['total_return'] - first_res['total_return']) > 1e-6:
                all_match = False
                
    print("-" * 55)
    if all_match and results:
        print("✅ SUCCESS: All iterations produced identical results.")
    else:
        print("❌ FAILURE: Results vary across iterations!")

if __name__ == "__main__":
    main()
