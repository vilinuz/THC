
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import itertools
import random
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.multi_layer_strategy import MultiLayerStrategy
from backtesting.backtest_engine import BacktestEngine
from ml.walk_forward import WalkForwardOptimizer

def load_data(symbol='BTC-USD', period='2y', interval='1h'):
    """Load data from yfinance"""
    print(f"Fetching {symbol} data...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    # Ensure columns are lowercase
    df.columns = df.columns.astype(str).str.lower()
    
    # Map yfinance columns to standard names if needed
    # yfinance returns: Open, High, Low, Close, Adj Close, Volume
    # We need: open, high, low, close, volume
    if 'adj close' in df.columns:
        df = df.drop('adj close', axis=1)
        
    return df

def optimize_func(train_df, param_bounds):
    """
    optimize_func: Function to optimize parameters on train set
    Iterates through random combinations of parameters and returns the best set based on Sharpe Ratio.
    """
    # Generate random combinations
    keys = list(param_bounds.keys())
    values = list(param_bounds.values())
    
    # Create all combinations -> too slow?
    # Use Random Search: sample N combinations
    n_iter = 20
    best_score = -float('inf')
    best_params = {}
    
    # Fix for empty data
    if len(train_df) < 100:
        return {k: v[0] for k,v in param_bounds.items()}

    for _ in range(n_iter):
        # Sample random params
        params = {k: random.choice(v) for k, v in param_bounds.items()}
        
        # Run simplified backtest (just signal generation + rough sharp)
        # For speed, we can use the Strategy class but lightweight
        try:
            strategy = MultiLayerStrategy(config=params)
             # Generate signals
            signals = strategy.generate_signals(train_df)
            
            # Simple vector backtest for speed
            returns = train_df['close'].pct_change().shift(-1)
            strategy_returns = returns * signals
            sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(24*365) # Annualized
            
            if sharpe > best_score:
                best_score = sharpe
                best_params = params
                
        except Exception as e:
            continue
            
    return best_params if best_params else {k: v[0] for k, v in param_bounds.items()}

def backtest_func(test_df, params):
    """
    backtest_func: Function to backtest with parameters on test set
    """
    if len(test_df) < 50:
        return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

    strategy = MultiLayerStrategy(config=params)
    signals = strategy.generate_signals(test_df)
    
    # Backtest Engine expects dict of signals
    engine = BacktestEngine({'initial_capital': 10000, 'commission': 0.001})
    results = engine.run(test_df, {'multi_layer': signals})
    
    return {
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'win_rate': results['win_rate']
    }

def main():
    # 1. Load Data
    symbol = 'BTC-USD'
    print("Generating synthetic data for walk-forward test...")
    dates = pd.date_range(end=datetime.now(), periods=24*365*2, freq='h')
    
    # Create random walk with some trend and volatility
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, size=len(dates))
    price_path = 10000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price_path,
        'high': price_path * (1 + np.abs(np.random.normal(0, 0.005, size=len(dates)))),
        'low': price_path * (1 - np.abs(np.random.normal(0, 0.005, size=len(dates)))),
        'close': price_path * (1 + np.random.normal(0, 0.001, size=len(dates))),
        'volume': np.random.randint(100, 10000, size=len(dates))
    }, index=dates)
    
    # Ensure High is highest and Low is lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    print(f"Data loaded: {len(df)} candles")

    # 2. Define Parameter Space
    param_bounds = {
         # Regime (Phase 1)
        'hmm_lookback': [50, 100, 200],
        
        # Kinetic (Phase 2)
        'kama_period': [10, 20, 30],
        
        # Tactical (Phase 4)
        'adx_threshold': [20, 25, 30],
        'aroon_threshold': [70, 75, 80],
        
        # Risk (Phase 5)
        'atr_stop_multiplier': [1.5, 2.0, 3.0]
    }

    # 3. Initialize Optimizer
    # Train 90 days, Test 30 days, Step 30 days
    optimizer = WalkForwardOptimizer(
        train_period_days=90,
        test_period_days=30,
        step_days=30
    )

    # 4. Run optimization
    print("Starting Walk-Forward Optimization...")
    results_df = optimizer.run_walk_forward(
        df,
        optimize_func,
        backtest_func,
        param_bounds
    )

    # 5. Analyze Results
    print("\noptimization Complete!")
    print("-" * 50)
    print(results_df.tail())
    
    # Save results
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/walk_forward_results.csv')
    print("Results saved to reports/walk_forward_results.csv")
    
    metrics = optimizer.calculate_metrics(results_df)
    print("\nAggregate Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
