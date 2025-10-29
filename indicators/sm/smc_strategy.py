"""
Complete Smart Money Concepts Trading Strategy
Combines all SMC elements into a unified strategy
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class SMCStrategy:
    """
    Comprehensive Smart Money Concepts strategy combining:
    - Order Blocks
    - Fair Value Gaps
    - Liquidity Sweeps
    - Market Structure (BOS/CHoCH)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_confluence': 2,  # Minimum signals needed
            'ob_weight': 1.5,
            'fvg_weight': 1.0,
            'liquidity_weight': 1.2,
            'structure_weight': 2.0
        }
        
    def analyze(self, df: pd.DataFrame, smc_results: Dict) -> pd.DataFrame:
        """
        Comprehensive SMC analysis
        
        Args:
            df: OHLCV DataFrame
            smc_results: Dict from SMCWrapper.comprehensive_analysis()
            
        Returns:
            DataFrame with combined signals and confluence
        """
        from .order_blocks import OrderBlocks
        from .fvg import FairValueGap
        from .liquidity import Liquidity
        from .market_structure import MarketStructure
        
        # Extract individual signals
        ob_signals = OrderBlocks.signal(df, smc_results['order_blocks'])
        fvg_signals = FairValueGap.signal(df, smc_results['fvg'])
        
        # Liquidity sweeps
        sweeps = Liquidity.detect_sweeps(df, smc_results['liquidity'])
        liq_signals = Liquidity.signal(df, sweeps)
        
        structure_signals = MarketStructure.signal(smc_results['structure'])
        
        # Combine all signals
        combined = pd.DataFrame(index=df.index)
        combined['ob_signal'] = ob_signals
        combined['fvg_signal'] = fvg_signals
        combined['liq_signal'] = liq_signals
        combined['structure_signal'] = structure_signals
        
        # Calculate weighted score
        combined['weighted_score'] = (
            ob_signals * self.config['ob_weight'] +
            fvg_signals * self.config['fvg_weight'] +
            liq_signals * self.config['liquidity_weight'] +
            structure_signals * self.config['structure_weight']
        )
        
        # Calculate confluence (number of agreeing signals)
        combined['bullish_confluence'] = (
            (ob_signals == 1).astype(int) +
            (fvg_signals == 1).astype(int) +
            (liq_signals == 1).astype(int) +
            (structure_signals == 1).astype(int)
        )
        
        combined['bearish_confluence'] = (
            (ob_signals == -1).astype(int) +
            (fvg_signals == -1).astype(int) +
            (liq_signals == -1).astype(int) +
            (structure_signals == -1).astype(int)
        )
        
        return combined
        
    def generate_signals(self, combined_df: pd.DataFrame) -> pd.Series:
        """
        Generate final trading signals based on confluence
        
        Returns:
            Series with 1 (buy), -1 (sell), 0 (neutral)
        """
        signals = pd.Series(0, index=combined_df.index)
        min_confluence = self.config['min_confluence']
        
        # Buy signals
        buy_condition = combined_df['bullish_confluence'] >= min_confluence
        signals[buy_condition] = 1
        
        # Sell signals
        sell_condition = combined_df['bearish_confluence'] >= min_confluence
        signals[sell_condition] = -1
        
        return signals
        
    def get_signal_strength(self, combined_df: pd.DataFrame, idx: int) -> Dict:
        """
        Get detailed signal strength at a specific index
        
        Returns:
            Dict with signal breakdown and strength
        """
        row = combined_df.iloc[idx]
        
        return {
            'weighted_score': row['weighted_score'],
            'bullish_confluence': int(row['bullish_confluence']),
            'bearish_confluence': int(row['bearish_confluence']),
            'signals': {
                'order_blocks': int(row['ob_signal']),
                'fair_value_gap': int(row['fvg_signal']),
                'liquidity': int(row['liq_signal']),
                'structure': int(row['structure_signal'])
            }
        }
        
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float = 10000
    ) -> Dict:
        """
        Backtest the SMC strategy
        
        Returns:
            Dict with performance metrics
        """
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        portfolio_value = initial_capital
        position = 0
        trades = []
        equity = []
        
        for i in range(len(df_copy)):
            current_price = df_copy['close'].iloc[i]
            signal = signals.iloc[i]
            
            if signal == 1 and position <= 0:  # Buy
                if position < 0:  # Close short
                    pnl = position * (trades[-1]['entry_price'] - current_price)
                    portfolio_value += abs(position) * current_price + pnl
                    position = 0
                    
                # Open long
                position = (portfolio_value * 0.95) / current_price
                portfolio_value -= position * current_price
                
                trades.append({
                    'index': i,
                    'entry_price': current_price,
                    'side': 'long',
                    'size': position
                })
                
            elif signal == -1 and position >= 0:  # Sell
                if position > 0:  # Close long
                    pnl = position * (current_price - trades[-1]['entry_price'])
                    portfolio_value += position * current_price + pnl
                    position = 0
                    
            # Track equity
            current_equity = portfolio_value + abs(position) * current_price
            equity.append({
                'index': i,
                'equity': current_equity
            })
            
        # Calculate metrics
        equity_df = pd.DataFrame(equity)
        returns = equity_df['equity'].pct_change().dropna()
        
        total_return = (equity_df['equity'].iloc[-1] / initial_capital) - 1
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'num_trades': len(trades),
            'equity_curve': equity_df
        }
