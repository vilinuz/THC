
import sys
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.walk_forward import WalkForwardOptimizer
from strategy.multi_layer_strategy import MultiLayerStrategy, CausalContext
from backtesting.backtest_engine import BacktestEngine
from scripts.test_causal_discovery import BinanceVisionDownloader

# Global cache for data to avoid passing large DFs around if not needed, 
# but WFO class handles slicing. We just need to make sure we slice BOTH leader and follower.

def load_data(leader_symbol='BTCUSDT', follower_symbol='LTCUSDT', months=6):
    """Load and align data"""
    downloader = BinanceVisionDownloader()
    start_date = datetime.now() - timedelta(days=30*months)
    end_date = datetime.now()
    
    print(f"Fetching {months} months of data for {leader_symbol} and {follower_symbol}...")
    leader_df = downloader.get_data(leader_symbol, '1h', start_date, end_date)
    follower_df = downloader.get_data(follower_symbol, '1h', start_date, end_date)
    
    # Align
    aligned_leader, aligned_follower = leader_df.align(follower_df, join='inner', axis=0)
    print(f"Data aligned: {len(aligned_follower)} bars")
    
    return aligned_leader, aligned_follower

def causal_optimize_func(train_df: pd.DataFrame, param_bounds: Dict, extra_context: Dict = None) -> Dict:
    """
    Optimization function for WalkForwardOptimizer.
    Performs Random Search to find best params on train_df.
    
    extra_context must contain 'leader_df' corresponding to train_df index.
    """
    leader_df = extra_context.get('leader_df')
    if leader_df is None:
        raise ValueError("Leader data missing in optimization context")
        
    best_score = -float('inf')
    best_params = {}
    
    # 20 Iterations of Random Search (Keep it fast)
    for i in range(20):
        # Sample params
        params = {}
        for k, v in param_bounds.items():
            if isinstance(v, list):
                if isinstance(v[0], int):
                    params[k] = random.randint(v[0], v[1])
                else:
                    params[k] = random.uniform(v[0], v[1])
                    
        # Setup Strategy
        # Fix: Ensure non-optimized params are set defaults
        full_params = {
            'initial_capital': 10000,
            'commission': 0.001,
            'use_smc': False,
            'use_fisher_timing': True,
            **params
        }
        
        strategy = MultiLayerStrategy(full_params)
        
        # Inject Causal Context
        ctx = CausalContext(
            leader_df=leader_df,
            optimal_lag=1, # Fixed at 1 based on research
            ete_rising=True,
            leader_name="BTC",
            is_valid=True
        )
        strategy.set_leader_context(ctx)
        
        # Generate Signals
        try:
            signals = strategy.generate_signals(train_df)
        except Exception as e:
            # print(f"Strategy fail: {e}")
            continue
            
        # Fast Scoring (Vectorized approx of Sharpe)
        # We don't run full BacktestEngine here for speed, just simplified returns
        returns = train_df['close'].pct_change().fillna(0)
        strategy_returns = returns * signals.shift(1).fillna(0)
        
        if len(strategy_returns[strategy_returns != 0]) < 10:
            score = -100 # Penalize no trades
        else:
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()
            if std_ret == 0:
                score = 0
            else:
                score = (mean_ret / std_ret) * np.sqrt(24*365) # Annualized Sharpe approx
                
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params

def causal_backtest_func(test_df: pd.DataFrame, params: Dict, extra_context: Dict = None) -> Dict:
    """
    Backtest function for WalkForwardOptimizer.
    Runs full backtest on test_df using params.
    """
    leader_df = extra_context.get('leader_df')
    
    # Setup
    full_params = {
        'initial_capital': 10000,
        'commission': 0.001, 
        'use_smc': False,
        'use_fisher_timing': True,
        **params
    }
    
    strategy = MultiLayerStrategy(full_params)
    
    if leader_df is not None:
        ctx = CausalContext(
            leader_df=leader_df,
            optimal_lag=1,
            ete_rising=True,
            leader_name="BTC",
            is_valid=True
        )
        strategy.set_leader_context(ctx)
        
    signals = strategy.generate_signals(test_df)
    
    engine = BacktestEngine(full_params)
    results = engine.run(test_df, {'causal_opt': signals})
    
    return results

class CausalWalkForwardOptimizer(WalkForwardOptimizer):
    """
    Extended Optimizer to handle Leader Data passing
    """
    def run_causal_walk_forward(self, 
                               follower_df: pd.DataFrame, 
                               leader_df: pd.DataFrame,
                               optimize_func, 
                               backtest_func, 
                               param_bounds: Dict) -> pd.DataFrame:
        
        results = []
        windows = self.generate_windows(follower_df.index[0], follower_df.index[-1])
        
        print(f"Generated {len(windows)} Walk-Forward Windows.")
        
        for i, window in enumerate(windows):
            print(f"Processing Window {i+1}/{len(windows)}...", end='\r')
            
            # Slice Data
            train_start, train_end = window['train_start'], window['train_end']
            test_start, test_end = window['test_start'], window['test_end']
            
            train_follower = follower_df[train_start:train_end]
            test_follower = follower_df[test_start:test_end]
            
            train_leader = leader_df[train_start:train_end]
            test_leader = leader_df[test_start:test_end]
            
            if len(train_follower) < 100 or len(test_follower) < 50:
                continue
                
            # Optimize (Train)
            best_params = optimize_func(
                train_follower, 
                param_bounds, 
                extra_context={'leader_df': train_leader}
            )
            
            # Test (Backtest)
            window_results = backtest_func(
                test_follower, 
                best_params, 
                extra_context={'leader_df': test_leader}
            )
            
            # Store
            record = {
                'window_index': i,
                'train_start': train_start,
                'test_start': test_start,
                'test_end': test_end,
                **best_params,
                'total_return': window_results['total_return'],
                'sharpe_ratio': window_results['sharpe_ratio'],
                'max_drawdown': window_results['max_drawdown'],
                'trades': len(window_results['trades']),
                'win_rate': window_results['win_rate']
            }
            results.append(record)
            
        print("\nOptimization Complete.")
        return pd.DataFrame(results)

def main():
    # 1. Load Data
    # Using 6 months to ensure we have enough data for a few windows
    # Window settings: Train 60 days, Test 20 days, Step 20 days
    leader_df, follower_df = load_data(leader_symbol='BTCUSDT', follower_symbol='LTCUSDT', months=6)
    
    if len(follower_df) < 1000:
        print("Not enough data for Walk-Forward.")
        return

    # 2. Define Parameter Space
    # Optimizing Kinetic (KAMA), Tactical (ADX), and Risk (ATR)
    param_bounds = {
        'kama_period': [10, 30],       # KAMA Baseline
        'kama_fast': [2, 5],           # KAMA Fast SC
        'kama_slow': [20, 50],         # KAMA Slow SC
        'adx_threshold': [15, 35],     # Trend Strength
        'aroon_threshold': [50, 80],   # Momentum Strength
        'atr_stop_multiplier': [1.5, 4.0], # Stop Loss width
        'rsi_period': [10, 20] 
    }
    
    # 3. Initialize Optimizer
    # Train 45 days, Test 15 days, Step 15 days -> ~3 test periods / month -> ~18 windows for 6 months
    optimizer = CausalWalkForwardOptimizer(
        train_period_days=45,
        test_period_days=15,
        step_days=15
    )
    
    # 4. Run
    results_df = optimizer.run_causal_walk_forward(
        follower_df,
        leader_df,
        causal_optimize_func,
        causal_backtest_func,
        param_bounds
    )
    
    # 5. Analysis
    if not results_df.empty:
        # Save raw
        os.makedirs('reports', exist_ok=True)
        results_df.to_csv('reports/causal_wfo_results.csv', index=False)
        
        # Calculate Aggregates
        avg_return = results_df['total_return'].mean()
        cum_return = (1 + results_df['total_return']).prod() - 1
        avg_sharpe = results_df['sharpe_ratio'].mean()
        avg_dd = results_df['max_drawdown'].mean()
        
        print("\n=== Walk-Forward Optimization Results (Aggregated) ===")
        print(f"Cumulative Return (Out-of-Sample): {cum_return:.2%}")
        print(f"Average Period Return: {avg_return:.2%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Max Drawdown: {avg_dd:.2%}")
        
        # Best Params Frequency
        print("\nTop Parameter Configurations:")
        print(results_df[list(param_bounds.keys())].mode().head(1))
        
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
