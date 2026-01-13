"""
Kaufman Adaptive Moving Average (KAMA)

An adaptive moving average that adjusts its smoothing based on market volatility.
- Fast in trending markets (low noise)
- Slow in choppy markets (high noise)
"""

import pandas as pd
import numpy as np


class KAMA:
    """
    Kaufman Adaptive Moving Average
    
    KAMA adapts to price volatility by adjusting its smoothing constant:
    - Trending market -> faster response (closer to fast EMA)
    - Choppy market -> slower response (closer to slow EMA)
    
    Efficiency Ratio (ER) = Direction / Volatility
    - ER close to 1: Strong trend
    - ER close to 0: No trend (choppy)
    
    Best used for:
    - Trend confirmation (price above/below KAMA)
    - Entry timing in conjunction with other signals
    - Filtering choppy conditions
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 10, 
                  fast_period: int = 2, slow_period: int = 30) -> pd.Series:
        """
        Calculate KAMA.
        
        Args:
            df: DataFrame with 'close' column
            period: Efficiency ratio lookback period (default 10)
            fast_period: Fast EMA period for trending (default 2)
            slow_period: Slow EMA period for choppy (default 30)
            
        Returns:
            pd.Series: KAMA values
        """
        close = pd.to_numeric(df['close'])
        
        # Calculate Efficiency Ratio (ER)
        # Direction = abs(close - close[period ago])
        # Volatility = sum of abs(close - close[1 ago]) over period
        direction = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(window=period).sum()
        
        # Avoid division by zero
        er = direction / volatility.replace(0, 0.0001)
        er = er.clip(0, 1)  # Bound ER between 0 and 1
        
        # Smoothing Constants
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        
        # Adaptive Smoothing Constant
        # SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA iteratively
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period] = close.iloc[period]  # Initialize with first valid close
        
        for i in range(period + 1, len(close)):
            if pd.isna(kama.iloc[i-1]):
                kama.iloc[i] = close.iloc[i]
            else:
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    @staticmethod
    def calculate_with_details(df: pd.DataFrame, period: int = 10,
                                fast_period: int = 2, slow_period: int = 30) -> pd.DataFrame:
        """
        Calculate KAMA with additional analysis metrics.
        
        Returns:
            DataFrame with columns:
            - kama: The KAMA values
            - er: Efficiency Ratio (trend strength)
            - sc: Smoothing Constant
            - kama_slope: KAMA rate of change
            - price_vs_kama: Price position relative to KAMA
        """
        close = pd.to_numeric(df['close'])
        
        # Efficiency Ratio
        direction = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(window=period).sum()
        er = direction / volatility.replace(0, 0.0001)
        er = er.clip(0, 1)
        
        # Smoothing Constants
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period] = close.iloc[period]
        
        for i in range(period + 1, len(close)):
            if pd.isna(kama.iloc[i-1]):
                kama.iloc[i] = close.iloc[i]
            else:
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        # Derived metrics
        kama_slope = kama.diff()
        price_vs_kama = close - kama
        
        return pd.DataFrame({
            'kama': kama,
            'er': er,
            'sc': sc,
            'kama_slope': kama_slope,
            'price_vs_kama': price_vs_kama
        }, index=df.index)
    
    @staticmethod
    def trend_signal(df: pd.DataFrame, period: int = 10,
                     er_threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate trend signals based on KAMA.
        
        Args:
            df: DataFrame with 'close' column
            period: KAMA period
            er_threshold: Minimum ER for trend confirmation
            
        Returns:
            DataFrame with:
            - kama: KAMA values
            - trend: 1 (up), -1 (down), 0 (neutral)
            - trend_strength: Efficiency Ratio
            - is_trending: True if ER > threshold
        """
        details = KAMA.calculate_with_details(df, period)
        close = pd.to_numeric(df['close'])
        
        # Trend based on price vs KAMA and KAMA slope
        trend = pd.Series(0, index=df.index)
        
        # Uptrend: Price above KAMA AND KAMA rising
        uptrend = (close > details['kama']) & (details['kama_slope'] > 0)
        # Downtrend: Price below KAMA AND KAMA falling
        downtrend = (close < details['kama']) & (details['kama_slope'] < 0)
        
        trend[uptrend] = 1
        trend[downtrend] = -1
        
        # Is market trending? (ER above threshold)
        is_trending = details['er'] > er_threshold
        
        return pd.DataFrame({
            'kama': details['kama'],
            'trend': trend,
            'trend_strength': details['er'],
            'is_trending': is_trending
        }, index=df.index)
