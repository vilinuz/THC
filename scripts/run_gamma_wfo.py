import sys
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_causal_discovery import BinanceVisionDownloader
from backtesting.wfo_engine import WalkForwardOptimizationEngine
from strategy.strategy_gamma import StrategyGamma

def main():
    downloader = BinanceVisionDownloader()
    
    symbol = 'BTCUSDT'
    interval = '1h'
    # Use last 12 months of data for a robust WFO test
    months = 12
    start_date = datetime.now() - timedelta(days=30*months)
    end_date = datetime.now()
    
    print(f"--- Fetching {months} months of {symbol} {interval} Data ---")
    df = downloader.get_data(symbol, interval, start_date, end_date)
    
    if df.empty or len(df) < 500:
        print("Not enough data fetched to perform optimization.")
        return
        
    print(f"Data loaded: {len(df)} bars")
    
    # Define Parameter Space for Strategy Gamma
    # This represents a vast, complex space where Grid Search would fail.
    # SA excels here.
    param_grid = {
        'adx_period': [10, 14, 20],
        'adx_trend_threshold': [20, 25, 30],
        'chop_period': [10, 14, 20],
        'chop_trend_threshold': [50.0, 61.8, 65.0],
        't3_fast_len': [5, 8, 12],
        't3_slow_len': [15, 21, 30],
        't3_volume_factor': [0.6, 0.7, 0.8],
        'fisher_period': [9, 14, 20],
        'fisher_overbought': [1.2, 1.5, 2.0],
        'fisher_oversold': [-2.0, -1.5, -1.2]
    }
    
    print("\n--- Initializing Walk-Forward Optimization Engine with Simulated Annealing ---")
    
    wfo_engine = WalkForwardOptimizationEngine(
        data=df,
        strategy_class=StrategyGamma,
        param_grid=param_grid,
        l_train_days=180, # 6 months training
        l_test_days=30,   # 1 month testing
        optimization_method="simulated_annealing",
        sa_initial_temp=15.0, # Higher starting temp for larger grid space
        sa_cooling_rate=0.95,
        sa_iterations=40 
    )
    
    print("Running WFO on Strategy Gamma using Simulated Annealing...")
    wfe, oos_returns, summary_df = wfo_engine.run()
    
    print("\n=== Walk-Forward Optimization (SA) Results: Strategy Gamma ===")
    
    # Calculate stats
    total_return = (1 + oos_returns).prod() - 1
    annualized_return = oos_returns.mean() * 24 * 365 # 1h bars
    daily_sharpe = oos_returns.mean() / (oos_returns.std() + 1e-9) * np.sqrt(24 * 365)
    
    print(f"Cumulative OOS Return: {total_return:.2%}")
    print(f"Annualized OOS Return: {annualized_return:.2%}")
    print(f"Annualized OOS Sharpe: {daily_sharpe:.2f}")
    print(f"Walk-Forward Efficiency (WFE): {wfe:.2f}%")
    
    print("\nOptimization History (Windows):")
    for idx, row in summary_df.iterrows():
        print(f"Window {idx + 1}:")
        print(f"  Train: {row['train_start'].strftime('%Y-%m-%d')} to {row['train_end'].strftime('%Y-%m-%d')}")
        print(f"  Test:  {row['test_start'].strftime('%Y-%m-%d')} to {row['test_end'].strftime('%Y-%m-%d')}")
        print(f"  Optimal Params Found (via SA): \n    {row['optimal_params']}")
        print(f"  In-Sample Score (Sharpe): {row['is_score']:.2f}")
        print(f"  OOS Score (Sharpe):       {row['oos_score']:.2f}\n")
        
    # Save Report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/sa_wfo_gamma_results.csv'
    summary_df.to_csv(report_path, index=False)
    print(f"Detailed window summary saved to {report_path}")

if __name__ == "__main__":
    main()
