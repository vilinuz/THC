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
from strategy.strategy_beta import StrategyBeta

def mock_hmm_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mock Continuous Regime Inference for demonstration.
    Adds simple rolling volatility proxy to infer regimes.
    """
    df_out = df.copy()
    
    # Calculate rolling volatility
    vol = df_out['close'].pct_change().rolling(24).std()
    
    # Normalized volatility approx [0, 1]
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    vol_norm = vol_norm.fillna(0.5)
    
    # Simple heuristic to assign probabilities
    # Low vol -> Trending (State 1) or Choppy (State 2)
    # High vol -> Stress (State 3)
    
    # Let's create a more noisy but structured proxy
    state_1, state_2, state_3 = [], [], []
    for v in vol_norm:
        if v > 0.7:
            s1, s2, s3 = 0.1, 0.1, 0.8
        elif v < 0.3:
            s1, s2, s3 = 0.2, 0.7, 0.1
        else:
            s1, s2, s3 = 0.7, 0.2, 0.1
            
        # Add some noise
        s1 += random.uniform(-0.1, 0.1)
        s2 += random.uniform(-0.1, 0.1)
        s3 += random.uniform(-0.1, 0.1)
        
        # Normalize to 1.0
        s1, s2, s3 = max(0, s1), max(0, s2), max(0, s3)
        total = s1 + s2 + s3
        
        state_1.append(s1/total)
        state_2.append(s2/total)
        state_3.append(s3/total)
        
    df_out['hmm_state_1_prob'] = state_1
    df_out['hmm_state_2_prob'] = state_2
    df_out['hmm_state_3_prob'] = state_3
    
    return df_out

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
    
    # Enrich with mock HMM features
    print("Enriching data with mock HMM state probabilities...")
    df = mock_hmm_regimes(df)
    
    # Define Parameter Space for Strategy Beta
    param_grid = {
        'ema_fast': [5, 9, 12, 15],
        'ema_slow': [20, 21, 26, 30],
        'bb_lookback': [15, 20, 25, 30],
        'bb_std': [1.5, 2.0, 2.5],
        'rsi_period': [10, 14, 21]
    }
    
    print("\n--- Initializing Walk-Forward Optimization Engine with Simulated Annealing ---")
    
    wfo_engine = WalkForwardOptimizationEngine(
        data=df,
        strategy_class=StrategyBeta,
        param_grid=param_grid,
        l_train_days=180, # 6 months training
        l_test_days=30,   # 1 month testing
        optimization_method="simulated_annealing",
        sa_initial_temp=10.0,
        sa_cooling_rate=0.92,
        sa_iterations=30 # Keep iterations moderate for runtime
    )
    
    print("Running WFO using Simulated Annealing. This might take a moment as it evaluates strategies...")
    wfe, oos_returns, summary_df = wfo_engine.run()
    
    print("\n=== Walk-Forward Optimization (SA) Results ===")
    
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
        print(f"  Optimal Params Found (via SA): {row['optimal_params']}")
        print(f"  In-Sample Score (Sharpe): {row['is_score']:.2f}")
        print(f"  OOS Score (Sharpe):       {row['oos_score']:.2f}\n")
        
    # Save Report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/sa_wfo_beta_results.csv'
    summary_df.to_csv(report_path, index=False)
    print(f"Detailed window summary saved to {report_path}")

if __name__ == "__main__":
    main()
