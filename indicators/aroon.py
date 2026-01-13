import pandas as pd
import numpy as np


class Aroon:
    """
    Aroon Indicator
    
    Measures the time since the highest high and lowest low within a lookback period.
    Used for:
    - Trend identification
    - Timing entries near recent highs/lows
    - Trend strength confirmation
    
    Aroon Up > 70: Strong uptrend, price near recent highs
    Aroon Down > 70: Strong downtrend, price near recent lows
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """
        Calculate Aroon Up, Aroon Down, and Aroon Oscillator.
        
        Args:
            df: DataFrame with 'high', 'low' columns
            period: Lookback period (default 25)
            
        Returns:
            DataFrame with 'aroon_up', 'aroon_down', 'aroon_osc' columns
        """
        high = pd.to_numeric(df['high'])
        low = pd.to_numeric(df['low'])
        
        # Days since highest high in period
        # rolling().apply() with argmax gives position of max (0-indexed from start of window)
        def days_since_high(x):
            return period - x.argmax()
        
        def days_since_low(x):
            return period - x.argmin()
        
        # Calculate using rolling window
        aroon_up_raw = high.rolling(window=period + 1, min_periods=period + 1).apply(
            days_since_high, raw=True
        )
        aroon_down_raw = low.rolling(window=period + 1, min_periods=period + 1).apply(
            days_since_low, raw=True
        )
        
        # Convert to percentage (0-100)
        aroon_up = ((period - aroon_up_raw) / period) * 100
        aroon_down = ((period - aroon_down_raw) / period) * 100
        
        # Aroon Oscillator = Aroon Up - Aroon Down
        aroon_osc = aroon_up - aroon_down
        
        return pd.DataFrame({
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_osc': aroon_osc
        }, index=df.index)
    
    @staticmethod
    def calculate_up(df: pd.DataFrame, period: int = 25) -> pd.Series:
        """
        Calculate only Aroon Up (for efficiency when only uptrend is needed).
        
        Args:
            df: DataFrame with 'high' column
            period: Lookback period (default 25)
            
        Returns:
            pd.Series: Aroon Up values (0-100)
        """
        high = pd.to_numeric(df['high'])
        
        def days_since_high(x):
            return period - x.argmax()
        
        aroon_up_raw = high.rolling(window=period + 1, min_periods=period + 1).apply(
            days_since_high, raw=True
        )
        
        aroon_up = ((period - aroon_up_raw) / period) * 100
        
        return aroon_up
    
    @staticmethod
    def calculate_down(df: pd.DataFrame, period: int = 25) -> pd.Series:
        """
        Calculate only Aroon Down (for efficiency when only downtrend is needed).
        
        Args:
            df: DataFrame with 'low' column
            period: Lookback period (default 25)
            
        Returns:
            pd.Series: Aroon Down values (0-100)
        """
        low = pd.to_numeric(df['low'])
        
        def days_since_low(x):
            return period - x.argmin()
        
        aroon_down_raw = low.rolling(window=period + 1, min_periods=period + 1).apply(
            days_since_low, raw=True
        )
        
        aroon_down = ((period - aroon_down_raw) / period) * 100
        
        return aroon_down
