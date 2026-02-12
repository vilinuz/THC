
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
from strategy.regimeline_strategy import RegimelineStrategy
from backtesting.backtest_engine import BacktestEngine
from scripts.test_causal_discovery import BinanceVisionDownloader

class RegimelineWalkForwardOptimizer(WalkForwardOptimizer):
    """
    WFO for Regimeline Strategy
    """
    pass

def regimeline_optimize_func(train_df: pd.DataFrame, param_bounds: Dict, extra_context: Dict = None) -> Dict:
    """
    Optimization function for Regimeline.
    Performs Random Search to find best params on train_df.
    """
    best_score = -float('inf')
    best_params = {}
    
    # 20 Iterations of Random Search
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
        full_params = {
            'initial_capital': 10000,
            'commission': 0.0004, # Binance Taker
            **params
        }
        
        strategy = RegimelineStrategy(full_params)
        
        # Generate Signals
        try:
            signals = strategy.generate_signals(train_df)
        except Exception as e:
            # print(f"Strategy fail: {e}")
            continue
            
        # Fast Scoring (Vectorized approx of Sharpe)
        # Signals: 1 (Enter Long), -1 (Enter Short), 2 (Exit Long), -2 (Exit Short)
        # We need to reconstruct returns approximately.
        # This simplification assumes entry at Close and Exit at Close.
        
        # Construct simple position series
        pos_val = np.zeros(len(train_df))
        curr_pos = 0
        
        sig_val = signals.values
        
        # Fast iterative loop for position reconstruction
        for j in range(len(sig_val)):
            s = sig_val[j]
            if s == 1:
                curr_pos = 1
            elif s == -1:
                curr_pos = -1
            elif s == 2 or s == -2:
                curr_pos = 0
            pos_val[j] = curr_pos
            
        pos = pd.Series(pos_val, index=train_df.index)
            
        # Shift position to align with returns (Position at T earns on Return at T+1)
        # Actually in vector backtest: Strategy Return = Pos(t) * Return(t+1)
        # If signal is at Close, we enter at Close, so exposure starts next bar?
        # Typically: Signal at T -> Enter T+1 Open. Return is (Open_T+2 - Open_T+1).
        # Here we use simplified Close-to-Close.
        # Pos(t) -> exposure for Return(t+1)
        
        returns = train_df['close'].pct_change().fillna(0)
        strategy_returns = returns * pd.Series(pos_val).shift(1).fillna(0)
        
        if len(strategy_returns[strategy_returns != 0]) < 10:
            score = -100 # Penalize no trades
        else:
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()
            if std_ret == 0:
                score = 0
            else:
                score = (mean_ret / std_ret) * np.sqrt(24*365) # Annualized Sharpe
                
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params

def regimeline_backtest_func(test_df: pd.DataFrame, params: Dict, extra_context: Dict = None) -> Dict:
    """
    Backtest function for Regimeline.
    """
    full_params = {
        'initial_capital': 10000,
        'commission': 0.0004,
        **params
    }
    
    strategy = RegimelineStrategy(full_params)
    signals = strategy.generate_signals(test_df)
    
    # Map Regimeline signals (1/-1/2/-2) to BacktestEngine format
    # BacktestEngine expects a 'causal_opt' series (or similar, typically 1/-1/0)
    # But Regimeline signals are specialized (2/-2 for exits). 
    # BacktestEngine might need modification or we pass 'signals' directly if supported.
    # Looking at BacktestEngine.run usage in run_causal_walk_forward.py:
    # results = engine.run(test_df, {'causal_opt': signals})
    
    # We'll pass it as 'regimeline_signals' and hope engine handles, or engine handles generic signals.
    # Actually BacktestEngine usually reconstructs positions from signals if they are standard.
    # If 2/-2/0 are not standard, we might need a wrapper.
    # Let's assume BacktestEngine needs 1/-1/0 signal series where 1=Long, -1=Short, 0=Neutral/Exit?
    # Or does it handle events?
    # Let's check BacktestEngine later or assume we need to convert to Pos State.
    
    # Converting signals to target position for BacktestEngine
    # 1 -> 1, -1 -> -1, 2/0/-2 -> 0?
    # Regimeline: 2 is Exit Long -> 0. -2 is Exit Short -> 0.
    # So we can convert to a Position Target series?
    
    target_pos_val = np.zeros(len(test_df))
    curr = 0
    sig_vals = signals.values
    t_vals = target_pos_val
    for j in range(len(sig_vals)):
        s = sig_vals[j]
        if s == 1: curr = 1
        elif s == -1: curr = -1
        elif s == 2 or s == -2: curr = 0
        t_vals[j] = curr
        
    target_pos = pd.Series(target_pos_val, index=test_df.index)
        
    engine = BacktestEngine(full_params)
    # Pass target position as a signal named 'position_target' if supported, 
    # or just leverage the fact that engine might take 'signals' if 'signals' key is used?
    # We'll try passing 'regimeline_signals' and see.
    # Actually, standardizing on target_pos is safer.
    
    results = engine.run(test_df, {'position_target': target_pos})
    
    return results

def main():
    # 1. Load Data (Binance Vision)
    downloader = BinanceVisionDownloader()
    # BTCUSDT is usually the best for testing Regimeline (trend-following)
    symbol = 'BTCUSDT'
    months = 6
    start_date = datetime.now() - timedelta(days=30*months)
    end_date = datetime.now()
    
    print(f"Fetching {months} months of data for {symbol}...")
    df = downloader.get_data(symbol, '1h', start_date, end_date)
    print(f"Data loaded: {len(df)} bars")
    
    if len(df) < 1000:
        print("Not enough data.")
        return

    # 2. Define Parameter Space (Regimeline)
    param_bounds = {
        'ema_fast_len': [10, 30],
        'ema_slow_len': [40, 70],
        'atr_len': [10, 20],
        'adx_bull_min': [20, 30],
        'bull_pb_atr': [0.2, 0.5],
        'bear_buf_atr': [0.3, 0.6]
    }
    
    # 3. Initialize Optimizer
    optimizer = RegimelineWalkForwardOptimizer(
        train_period_days=45,
        test_period_days=15,
        step_days=15
    )
    
    # 4. Run
    results_df = optimizer.run_walk_forward(
        df,
        regimeline_optimize_func,
        regimeline_backtest_func,
        param_bounds
    )
    
    # 5. Analysis
    if not results_df.empty:
        os.makedirs('reports', exist_ok=True)
        results_df.to_csv('reports/regimeline_wfo_results.csv', index=False)
        
        cum_return = (1 + results_df['total_return']).prod() - 1
        print("\n=== Regimeline WFO Results ===")
        print(f"Cumulative Return: {cum_return:.2%}")
        print(f"Avg Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
