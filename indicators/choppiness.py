import pandas as pd
import numpy as np
from indicators.atr import ATR

class ChoppinessIndex:
    """
    Choppiness Index (Log Formula)
    
    Formula:
    100 * LOG10( SUM(ATR(1), n) / ( MaxHigh(n) - MinLow(n) ) ) / LOG10(n)
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Choppiness Index
        """
        df = df.copy()
        high = pd.to_numeric(df['high'])
        low = pd.to_numeric(df['low'])
        close = pd.to_numeric(df['close'])
        
        # Calculate ATR(1) - essentially True Range
        # Pine: math.sum(ta.atr(1), len)
        # ta.atr(1) is the TR of the current bar (smoothed by 1? No, usually just TR)
        # In Pine, atr(1) is rma(tr, 1) which is just tr.
        
        tr1 = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        sum_atr = tr1.rolling(window=period).sum()
        
        hh = high.rolling(window=period).max()
        ll = low.rolling(window=period).min()
        
        # Avoid division by zero
        range_hl = hh - ll
        range_hl = range_hl.replace(0, np.nan) 
        
        num = np.log10(sum_atr / range_hl)
        den = np.log10(period)
        
        chop = 100 * num / den
        
        return chop
