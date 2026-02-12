
import sys
import os
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.walk_forward import WalkForwardOptimizer
from strategy.regimeline_strategy import RegimelineStrategy
from backtesting.backtest_engine import BacktestEngine
from data_fetchers.full_data_fetcher import BinanceHistoryFetcher
from db.duckdb_manager import DuckDBManager

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FineTuning")

# Re-use optimization/backtest functions from run_regimeline_wfo.py, 
# but defined here to be self-contained or importable.
# To ensure consistency, we'll redefine them.

def regimeline_optimize_func(train_df: pd.DataFrame, param_bounds: Dict, extra_context: Dict = None) -> Dict:
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
        
        if len(strategy_returns[strategy_returns != 0]) < 10:
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

def regimeline_backtest_func(test_df: pd.DataFrame, params: Dict, extra_context: Dict = None) -> Dict:
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

def run_asset_wfo(symbol: str, months: int = 9, db_manager: DuckDBManager = None):
    logger.info(f"Starting Fine-Tuning for {symbol}...")
    
    # 1. Load Data
    # 9 months back
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*months)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Loading data from DuckDB for {symbol} ({start_str} to {end_str})")
    df = db_manager.get_aggregated_ohlcv(symbol, start_str, end_str, '1h')
    
    if len(df) < 1000:
        logger.error(f"Insufficient data for {symbol}: {len(df)} bars")
        return None
        
    # 2. Parameters (Fine-Tuning around known good defaults or wider exploration?)
    # User asked for fine-tuning. Let's use slightly tighter bounds or centered around previous winners if known.
    # But since we don't have previous winning params hardcoded, we keep the space broad but reasonable.
    param_bounds = {
        'ema_fast_len': [15, 25],      # Centered around 21
        'ema_slow_len': [50, 60],      # Centered around 55
        'atr_len': [12, 16],           # Centered around 14
        'adx_bull_min': [20, 25],
        'bull_pb_atr': [0.25, 0.40],
        'bear_buf_atr': [0.40, 0.55]
    }
    
    # 3. Optimizer
    optimizer = WalkForwardOptimizer(
        train_period_days=60, # Increase train period for robustness
        test_period_days=20,
        step_days=20
    )
    
    results = optimizer.run_walk_forward(
        df,
        regimeline_optimize_func,
        regimeline_backtest_func,
        param_bounds
    )
    
    return results

def main():
    assets = ['ETHUSDT', 'SOLUSDT', 'ADAUSDT']
    months = 9
    
    # Setup Data Infrastructure
    fetcher = BinanceHistoryFetcher()
    db_manager = fetcher.db_manager
    
    try:
        # 1. Data Ingestion Phase
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        
        # Adjust start_date to beginning of month for full monthly download
        start_download = start_date.replace(day=1) 
        start_str = start_download.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Ensuring data exists for period: {start_str} to {end_str}")
        
        for symbol in assets:
            fetcher.download_range(symbol, start_str, end_str)

        # 2. Optimization Phase
        all_results = []
        
        for symbol in assets:
            res = run_asset_wfo(symbol, months, db_manager)
            if res is not None and not res.empty:
                res['symbol'] = symbol
                all_results.append(res)
                
                # Immediate Report
                avg_ret = res['total_return'].mean()
                cum_ret = (1 + res['total_return']).prod() - 1
                logger.info(f"[{symbol}] Cumulative Return: {cum_ret:.2%}, Avg Period Return: {avg_ret:.2%}")
                
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            os.makedirs('reports', exist_ok=True)
            final_df.to_csv('reports/fine_tuning_results.csv', index=False)
            logger.info("Saved aggregated results to reports/fine_tuning_results.csv")
        else:
            logger.warning("No results generated.")
            
    finally:
        fetcher.close()

if __name__ == "__main__":
    main()
