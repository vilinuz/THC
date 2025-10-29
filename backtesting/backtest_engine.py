"""
Backtesting engine for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from .performance_metrics import PerformanceMetrics

class BacktestEngine:
    """Backtest trading strategies"""
    
    def __init__(self, config: Dict):
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        
    def run(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> Dict:
        """
        Run backtest
        
        Args:
            df: OHLCV data
            signals: Dict of signal sources
        """
        # Combine signals (simple average for now)
        combined_signal = pd.DataFrame(signals).mean(axis=1)
        combined_signal = combined_signal.apply(
            lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)
        )
        
        # Initialize tracking
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            signal = combined_signal.iloc[i]
            
            # Execute trades
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short
                    pnl = position * (df['close'].iloc[trades[-1]['entry_idx']] - current_price)
                    cash += abs(position) * current_price + pnl
                    position = 0
                    
                # Open long
                position_value = cash * 0.95  # Use 95% of cash
                position = position_value / current_price
                cash -= position * current_price * (1 + self.commission)
                
                trades.append({
                    'entry_idx': i,
                    'entry_price': current_price,
                    'side': 'long',
                    'size': position
                })
                
            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long
                    pnl = position * (current_price - df['close'].iloc[trades[-1]['entry_idx']])
                    cash += position * current_price + pnl
                    position = 0
                    
            # Update portfolio value
            portfolio_value = cash + abs(position) * current_price
            equity_curve.append({
                'timestamp': df.index[i],
                'portfolio_value': portfolio_value,
                'position': position
            })
            
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        metrics = PerformanceMetrics.calculate_all(equity_df, self.initial_capital)
        
        return {
            'equity_curve': equity_df,
            'trades': trades,
            **metrics
        }

