
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from db.duckdb_manager import DuckDBManager
from backtesting.strategy_backtester import MultiLayerBacktester
from strategy.multi_layer_strategy import MultiLayerStrategy

def run_test():
    print("=" * 60)
    print("FULL YEAR COMPREHENSIVE BACKTEST (ALL LAYERS)")
    print("=" * 60)
    
    # 1. Load Data
    config_db = {"path": "data/market.duckdb"}
    db_path = str(project_root / config_db["path"])
    parquet_dir = str(project_root / "data/parquet")
    db_manager = DuckDBManager(db_path, parquet_dir)
    
    symbol = "BTCUSDT"
    # End date: roughly now (Jan 2026 or as far as we have)
    # Start date: Feb 2025
    start_date = "2025-05-01 00:00:00"
    end_date = "2026-02-01 00:00:00"
    
    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    df = db_manager.get_aggregated_ohlcv(symbol, start_date, end_date, timeframe="1h")
    db_manager.close()
    
    if len(df) == 0:
        print("CRITICAL: No data found! Ensure ingestion script finished successfully.")
        return

    print(f"Loaded {len(df)} bars.")
    
    # 2. Configure Strategy with EVERYTHING enabled
    strategy_config = {
        "use_smc": True,
        "use_bcpd": True,
        "use_causal": True,
        "hmm_lookback": 100,  # Lookback for regime classification
        "initial_capital": 10000.0,
        "causal_lookback": 252,
        # Risk settings for profitability
        "account_risk_pct": 0.02, # 2% risk
        "atr_stop_multiplier": 2.5
    }
    
    backtester = MultiLayerBacktester(
        config={
            "initial_capital": 10000.0,
            "strategy_config": strategy_config,
        }
    )
    
    # 3. Validation of Components
    print("\nVerifying Components:")
    strat = backtester.strategy
    print(f"- HMM Available: {strat.regime_classifier.model is not None}")
    print(f"- Kalman Available: {strat.kalman_filter is not None}")
    print(f"- SMC Available: {strat.use_smc} (Wrapper: {strat.smc_wrapper is not None})")
    print(f"- BCPD Available: {strat.use_bcpd} (Instance: {strat.bcpd is not None})")
    print(f"- Causal Available: {strat.use_causal} (Instance: {strat.causal_model is not None})")
    
    # Check Causal Leader Context
    if strat.causal_context is None:
        print("NOTE: No Causal Leader context set (requires 2nd asset like ETH). Causal checks will be passive.")

    # 5. Run Backtest
    print("\nRunning Backtest Simulation...")
    # Use 20% for initial training
    result = backtester.run(df, train_ratio=0.2)
    
    # 6. Report Results
    print("\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    print(f"Total Return: ${result.total_return:.2f} ({result.total_return_pct*100:.2f}%)")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct*100:.2f}%")
    print(f"Win Rate:     {result.win_rate*100:.2f}%")
    print(f"Trades:       {result.total_trades} (W: {result.winning_trades}, L: {result.losing_trades})")
    
    if result.total_return > 0:
        print("\nSUCCESS: Strategy is profitable.")
    else:
        print("\nFAILURE: Strategy is generating losses.")

if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\nTest interrupted.")
