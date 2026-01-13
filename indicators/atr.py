import pandas as pd
import numpy as np


class ATR:
    """
    Average True Range (ATR) Indicator
    
    Measures market volatility by decomposing the entire range of an asset price
    for that period. Used for:
    - Stop-loss placement
    - Position sizing (volatility-based)
    - Volatility regime detection
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ATR using Wilder's Smoothing Method.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period (default 14)
            
        Returns:
            pd.Series: ATR values
        """
        df = df.copy()
        high = pd.to_numeric(df['high'])
        low = pd.to_numeric(df['low'])
        close = pd.to_numeric(df['close'])
        
        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        # True Range = max of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's Smoothing (EMA with alpha = 1/period)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def calculate_with_details(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate ATR with additional details for analysis.
        
        Returns:
            DataFrame with columns: 'tr', 'atr', 'atr_pct' (ATR as % of close)
        """
        df = df.copy()
        high = pd.to_numeric(df['high'])
        low = pd.to_numeric(df['low'])
        close = pd.to_numeric(df['close'])
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        # ATR as percentage of close (useful for position sizing)
        atr_pct = (atr / close) * 100
        
        return pd.DataFrame({
            'tr': true_range,
            'atr': atr,
            'atr_pct': atr_pct
        }, index=df.index)
