
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.multi_layer_strategy import MultiLayerStrategy, CausalContext
from backtesting.backtest_engine import BacktestEngine
# Reuse the downloader from the discovery script (or move it to a shared module, but import is easier for now)
from scripts.test_causal_discovery import BinanceVisionDownloader

def run_backtest_comparison(leader_symbol='BTCUSDT', follower_symbol='LTCUSDT', lag=1):
    print(f"--- Starting Causal Backtest Comparison: {leader_symbol} -> {follower_symbol} (Lag: {lag}) ---")
    
    # 1. Fetch Data
    downloader = BinanceVisionDownloader()
    start_date = datetime.now() - timedelta(days=180) # 6 Months
    end_date = datetime.now()
    
    print("Fetching data...")
    leader_df = downloader.get_data(leader_symbol, '1h', start_date, end_date)
    follower_df = downloader.get_data(follower_symbol, '1h', start_date, end_date)
    
    # Align Data
    # We need to ensure timestamps match exactly
    aligned_leader, aligned_follower = leader_df.align(follower_df, join='inner', axis=0)
    
    if len(aligned_follower) < 500:
        print("Error: Insufficient aligned data.")
        return

    print(f"Data aligned: {len(aligned_follower)} bars.")

    # 2. Configure Strategies
    
    # Base Config
    config = {
        'initial_capital': 10000, 
        'commission': 0.001,
        'use_smc': False, # Disable SMC if lib not avail, or keep simple
        'use_fisher_timing': True 
    }
    
    # Strategy A: Standard (No Causal)
    strat_standard = MultiLayerStrategy(config)
    
    # Strategy B: Causal Augmented
    strat_causal = MultiLayerStrategy(config)
    # Inject Context
    # We assume 'ete_rising' is True for the backtest period purely based on our "Discovery" finding that ETE is positive.
    # In a real live run, this would be re-calculated dynamicall window by window. 
    # For this backtest, we simulate the "Structural Edge" found in the report.
    causal_ctx = CausalContext(
        leader_df=aligned_leader,
        optimal_lag=lag,
        ete_rising=True, 
        leader_name=leader_symbol,
        is_valid=True
    )
    strat_causal.set_leader_context(causal_ctx)
    
    # 3. Generate Signals
    print("Generating Standard Signals...")
    signals_std = strat_standard.generate_signals(aligned_follower)
    
    print("Generating Causal Signals...")
    signals_causal = strat_causal.generate_signals(aligned_follower)
    
    # 4. Run Backtest Engine
    engine = BacktestEngine(config)
    
    print("Running Standard Backtest...")
    res_std = engine.run(aligned_follower, {'standard': signals_std})
    
    print("Running Causal Backtest...")
    res_causal = engine.run(aligned_follower, {'causal': signals_causal})
    
    # 5. Compare Metrics
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<20} | {'Standard':<15} | {'Causal (Phase 6)':<15} | {'Diff':<10}")
    print("-" * 70)
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    for m in metrics:
        val_std = res_std.get(m, 0)
        val_causal = res_causal.get(m, 0)
        diff = val_causal - val_std
        print(f"{m:<20} | {val_std:<15.4f} | {val_causal:<15.4f} | {diff:>+10.4f}")
        
    # 6. Visualization
    plot_results(res_std, res_causal, follower_symbol)

def plot_results(res_std, res_causal, symbol):
    """Visualize Equity Curves"""
    df_std = res_std['equity_curve']
    df_causal = res_causal['equity_curve']
    
    plt.figure(figsize=(12, 8))
    
    # Plot Portfolio Values
    plt.plot(df_std.index, df_std['portfolio_value'], label='Standard Strategy', alpha=0.7)
    plt.plot(df_causal.index, df_causal['portfolio_value'], label='Causal Strategy', linewidth=2)
    
    plt.title(f"Causal Strategy Backtest: {symbol} (Using BTC Leader)")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save Plot
    os.makedirs('reports', exist_ok=True)
    filename = f"reports/causal_backtest_{symbol}.png"
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")

if __name__ == "__main__":
    # Using LTCUSDT as it had High TE and High Corr in our report
    run_backtest_comparison(leader_symbol='BTCUSDT', follower_symbol='LTCUSDT', lag=1)
