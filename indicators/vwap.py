import pandas as pd
import numpy as np

class VWAP:
    """Volume Weighted Average Price indicator"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, anchor: str = 'D') -> pd.Series:
        """
        Calculate VWAP
        
        Args:
            df: DataFrame with OHLCV data
            anchor: Time anchor for VWAP reset ('D' for daily, 'W' for weekly)
        """
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative TPV and volume grouped by anchor
        df_copy = df.copy()
        df_copy['typical_price'] = typical_price
        df_copy['tpv'] = typical_price * df['volume']
        
        # Group by anchor period
        df_copy['period'] = df_copy.index.to_period(anchor)
        
        # Calculate VWAP
        vwap = df_copy.groupby('period').apply(
            lambda x: (x['tpv'].cumsum() / x['volume'].cumsum())
        ).reset_index(level=0, drop=True)
        
        return vwap
        
    @staticmethod
    def signal(df: pd.DataFrame, vwap: pd.Series) -> pd.Series:
        """
        Generate VWAP-based signals
        
        Returns:
            Series with 1 (bullish), -1 (bearish), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        # Bullish: Price crosses above VWAP
        signals[(df['close'] > vwap) & (df['close'].shift(1) <= vwap.shift(1))] = 1

        # Bearish: Price crosses below VWAP
        signals[(df['close'] < vwap) & (df['close'].shift(1) >= vwap.shift(1))] = -1

        return signals
