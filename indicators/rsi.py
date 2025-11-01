import pandas as pd
import numpy as np

class RSI:
    """Relative Strength Index indicator"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate RSI"""
        delta = df[column].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    @staticmethod
    def signal(rsi: pd.Series, overbought: int = 70, oversold: int = 30) -> pd.Series:
        """
        Generate RSI signals
        
        Returns:
            Series with 1 (oversold/buy), -1 (overbought/sell), 0 (neutral)
        """
        signals = pd.Series(0, index=rsi.index)
        
        # Buy signal: RSI crosses above oversold level
        signals[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
        
        # Sell signal: RSI crosses below overbought level
        signals[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1
        
        return signals
    
    #not sure if Divergence Window works well with value of 14  
    @staticmethod
    def divergence(df: pd.DataFrame, rsi: pd.Series, window: int = 14) -> pd.DataFrame:
        """
        Detect RSI divergences
        
        Returns DataFrame with bullish_div and bearish_div columns
        """
        result = pd.DataFrame(index=df.index)
        result['bullish_div'] = 0
        result['bearish_div'] = 0
        
        # Find price and RSI extremes
        price_lows = df['low'].rolling(window=window, center=True).min() == df['low']
        price_highs = df['high'].rolling(window=window, center=True).max() == df['high']
        
        rsi_lows = rsi.rolling(window=window, center=True).min() == rsi
        rsi_highs = rsi.rolling(window=window, center=True).max() == rsi
        
        # Bullish divergence: Price makes lower low but RSI makes higher low
        for i in range(window, len(df) - window):
            if price_lows.iloc[i]:
                prev_low_idx = price_lows.iloc[:i][::-1].idxmax()
                if (df['low'].iloc[i] < df['low'].loc[prev_low_idx] and 
                    rsi.iloc[i] > rsi.loc[prev_low_idx]):
                    result['bullish_div'].iloc[i] = 1
                    
            # Bearish divergence: Price makes higher high but RSI makes lower high
            if price_highs.iloc[i]:
                prev_high_idx = price_highs.iloc[:i][::-1].idxmax()
                if (df['high'].iloc[i] > df['high'].loc[prev_high_idx] and 
                    rsi.iloc[i] < rsi.loc[prev_high_idx]):
                    result['bearish_div'].iloc[i] = 1
                    
        return result
