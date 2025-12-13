"""
Performance metrics calculation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class PerformanceMetrics:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_all(equity_df: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate all performance metrics"""
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        metrics = {
            'total_return': (equity_df['portfolio_value'].iloc[-1] / initial_capital) - 1,
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(equity_df['portfolio_value']),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(equity_df['portfolio_value'])
        }
        
        return metrics
        
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.std() == 0:
            return 0.0
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown.min()
        
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Calculate win rate"""
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        return winning_trades / total_trades if total_trades > 0 else 0.0
        
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses != 0 else np.inf
        
    @staticmethod
    def calmar_ratio(equity: pd.Series) -> float:
        """Calculate Calmar ratio"""
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        max_dd = PerformanceMetrics.max_drawdown(equity)
        return total_return / abs(max_dd) if max_dd != 0 else 0.0
'''

# Logger Utility
code_files['utils/logger.py'] = '''"""
Logging configuration
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
    return logger
