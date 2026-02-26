import os
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_causal_discovery import BinanceVisionDownloader
from strategy.regimeline_strategy import RegimelineStrategy
from backtesting.backtest_engine import BacktestEngine
from ml.walk_forward import WalkForwardOptimizer

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    # BinanceVisionDownloader already formats the index to DatetimeIndex('timestamp') 
    # and keeps the ['open', 'high', 'low', 'close', 'volume'] columns.
    
    # Just ensure they are floats
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def optimize_func_memory(train_df: pd.DataFrame, param_bounds: dict, extra_context: dict = None) -> dict:
    best_score = -float('inf')
    best_params = {}
    
    # 20 Iterations of Random Search (Speed)
    for i in range(20):
        params = {}
        for k, v in param_bounds.items():
            if isinstance(v, list):
                if isinstance(v[0], int):
                    params[k] = random.randint(v[0], v[1])
                else:
                    params[k] = random.uniform(v[0], v[1])
                    
        full_params = {
            'initial_capital': 10000,
            'commission': 0.0004,
            **params
        }
        
        strategy = RegimelineStrategy(full_params)
        
        try:
            signals = strategy.generate_signals(train_df)
        except Exception:
            continue
            
        # Fast Scoring
        pos_val = np.zeros(len(train_df))
        curr_pos = 0
        sig_val = signals.values
        
        for j in range(len(sig_val)):
            s = sig_val[j]
            if s == 1: curr_pos = 1
            elif s == -1: curr_pos = -1
            elif s == 2 or s == -2: curr_pos = 0
            pos_val[j] = curr_pos
            
        pos = pd.Series(pos_val, index=train_df.index)
        returns = train_df['close'].pct_change().fillna(0)
        strategy_returns = returns * pos.shift(1).fillna(0)
        
        if len(strategy_returns[strategy_returns != 0]) < 5:
            score = -100
        else:
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()
            if std_ret == 0:
                score = 0
            else:
                score = (mean_ret / std_ret) * np.sqrt(24*365)
                
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params

def backtest_func_memory(test_df: pd.DataFrame, params: dict, extra_context: dict = None) -> dict:
    full_params = {
        'initial_capital': 10000,
        'commission': 0.0004,
        **params
    }
    
    strategy = RegimelineStrategy(full_params)
    signals = strategy.generate_signals(test_df)
    
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
    results = engine.run(test_df, {'position_target': target_pos})
    
    return results

def main():
    symbol = "ETHUSDT"
    interval = "1h"
    months = 9
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*months)
    
    print(f"Downloading {symbol} {interval} data from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    downloader = BinanceVisionDownloader(cache_dir="data/binance_vision")
    df = downloader.get_data(symbol, interval, start_date, end_date)
    
    if df.empty:
        print("No data downloaded.")
        return
        
    df = prep_data(df)
    print(f"Data loaded: {len(df)} rows. {df.index.min()} to {df.index.max()}")
    
    print("Starting In-Memory Walk-Forward Optimization for Regimeline v7...")
    
    param_bounds = {
        'ema_fast_len': [15, 25], 
        'ema_slow_len': [50, 60], 
        'atr_len': [12, 16],      
        'bull_pb_atr': [0.20, 0.40],
        'bear_buf_atr': [0.40, 0.55]
    }
    
    optimizer = WalkForwardOptimizer(
        train_period_days=60,
        test_period_days=20,
        step_days=20
    )
    
    results = optimizer.run_walk_forward(
        df,
        optimize_func_memory,
        backtest_func_memory,
        param_bounds
    )
    
    if results is not None and not results.empty:
        avg_ret = results['total_return'].mean()
        cum_ret = (1 + results['total_return']).prod() - 1
        
        print("\n" + "="*50)
        print("WFO RESULTS (Regimeline v7 - In Memory)")
        print("="*50)
        print(f"Cumulative Return: {cum_ret*100:.2f}%")
        print(f"Avg Period Return: {avg_ret*100:.2f}%")
        print(f"Total Periods: {len(results)}")
        print("="*50)
    else:
        print("WFO Failed or returned no results.")
        
if __name__ == "__main__":
    main()
