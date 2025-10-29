"""
Walk-forward optimization and analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple
from datetime import timedelta

class WalkForwardOptimizer:
    """Walk-forward optimization for trading strategies"""
    
    def __init__(
        self,
        train_period_days: int = 365,
        test_period_days: int = 90,
        step_days: int = 30
    ):
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.step = timedelta(days=step_days)
        
    def generate_windows(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[Dict]:
        """
        Generate train/test windows for walk-forward analysis
        
        Returns:
            List of dicts with train_start, train_end, test_start, test_end
        """
        windows = []
        current_start = start_date
        
        while current_start + self.train_period + self.test_period <= end_date:
            train_start = current_start
            train_end = current_start + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            current_start += self.step

        return windows

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        optimize_func: Callable,
        backtest_func: Callable,
        param_bounds: Dict
    ) -> pd.DataFrame:
        """
        Run complete walk-forward optimization

        Args:
            df: Full dataset
            optimize_func: Function to optimize parameters on train set
            backtest_func: Function to backtest with parameters on test set
            param_bounds: Parameter boundaries for optimization

        Returns:
            DataFrame with results for each window
        """
        windows = self.generate_windows(df.index[0], df.index[-1])
        results = []

        for i, window in enumerate(windows):
            print(f"Processing window {i+1}/{len(windows)}")

            # Split data
            train_data = df[window['train_start']:window['train_end']]
            test_data = df[window['test_start']:window['test_end']]

            # Optimize on train set
            best_params = optimize_func(train_data, param_bounds)

            # Backtest on test set
            test_results = backtest_func(test_data, best_params)

            # Store results
            results.append({
                'window': i + 1,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'params': best_params,
                **test_results
            })

        return pd.DataFrame(results)

    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate aggregate walk-forward metrics"""
        metrics = {
            'avg_return': results_df['total_return'].mean(),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'win_rate': (results_df['total_return'] > 0).mean(),
            'max_drawdown': results_df['max_drawdown'].min(),
            'std_return': results_df['total_return'].std(),
            'num_windows': len(results_df)
        }

        return metrics
