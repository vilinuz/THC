import pandas as pd

class EMA:
    """Exponential Moving Average indicator"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 21, column: str = 'close') -> pd.Series:
        """Calculate EMA"""
        return df[column].ewm(span=period, adjust=False).mean()
        
    @staticmethod
    def multi_ema(df: pd.DataFrame, periods: list = [9, 21, 50, 200]) -> pd.DataFrame:
        """Calculate multiple EMAs"""
        result = pd.DataFrame(index=df.index)
        for period in periods:
            result[f'ema_{period}'] = EMA.calculate(df, period)
        return result
        
    @staticmethod
    def crossover_signal(fast_ema: pd.Series, slow_ema: pd.Series) -> pd.Series:
        """
        Generate crossover signals
        
        Returns:
            Series with 1 (bullish cross), -1 (bearish cross), 0 (no cross)
        """
        signals = pd.Series(0, index=fast_ema.index)
        
        # Bullish crossover
        signals[(fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))] = 1
        
        # Bearish crossover
        signals[(fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))] = -1
        
        return signals
        
    @staticmethod
    def trend_signal(df: pd.DataFrame, ema: pd.Series, lookback: int = 5) -> pd.Series:
        """
        Generate trend signals based on price position relative to EMA
        
        Args:
            lookback: Number of candles to confirm trend
        """
        signals = pd.Series(0, index=df.index)
        
        # Count consecutive candles above/below EMA
        above_ema = (df['close'] > ema).astype(int)
        below_ema = (df['close'] < ema).astype(int)
        
        consecutive_above = above_ema.rolling(window=lookback).sum()
        consecutive_below = below_ema.rolling(window=lookback).sum()
        
        # Strong uptrend
        signals[consecutive_above == lookback] = 1
        
        # Strong downtrend
        signals[consecutive_below == lookback] = -1
        
        return signals 
