import pandas as pd
import numpy as np

class HurstProxy:
    """
    Hurst Exponent Proxy
    
    Simplified estimation using Efficiency Ratio (ER).
    Pine Script: H ~ 0.5 + ER/2
    ER = Change / Volatility
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 100, source_col: str = 'close') -> pd.Series:
        src = pd.to_numeric(df[source_col])
        
        # Change: abs(src - src[len])
        change = abs(src - src.shift(period))
        
        # Volatility: sum(abs(src - src[1]), len)
        # Sum of absolute 1-bar changes over 'period'
        one_bar_change = abs(src - src.shift(1))
        volatility = one_bar_change.rolling(window=period).sum()
        
        # ER calculation
        # Avoid division by zero
        er = change / volatility.replace(0, np.nan)
        er = er.fillna(0)
        
        # Proxy H
        hurst = 0.5 + (er / 2.0)
        
        return hurst
