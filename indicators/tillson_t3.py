import pandas as pd
import numpy as np

class TillsonT3:
    """
    Tillson T3 Indicator
    Formula: T3 = GD(GD(GD(x, v), v), v)
    where GD is Generalized DEMA
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame, length: int = 10, volume_factor: float = 1.7, column: str = 'close') -> pd.Series:
        """
        Calculate T3 Moving Average
        
        Args:
            df: DataFrame containing price data
            length: Period for the moving average (default 10)
            volume_factor: Volume Factor 'v' (default 1.7)
            column: Column to calculate on (default 'close')
            
        Returns:
            pd.Series: T3 values
        """
        source = df[column]
        
        # Generalized DEMA factor
        # GD(x, v) = EMA(x) * (1 + v) - EMA(EMA(x)) * v
        # But commonly T3 is calculated by chaining 6 EMAs in a specific way or using the GD formula iteratively.
        # The standard formula is:
        # e1 = EMA(x, len)
        # e2 = EMA(e1, len)
        # e3 = EMA(e2, len)
        # e4 = EMA(e3, len)
        # e5 = EMA(e4, len)
        # e6 = EMA(e5, len)
        # c1 = -v^3
        # c2 = 3v^2 + 3v^3
        # c3 = -6v^2 - 3v - 3v^3
        # c4 = 1 + 3v + v^3 + 3v^2
        # T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
        
        # Let's verify the GD chaining method which matches "GD(GD(GD(x)))"
        # GD(x, v) = EMA(x) * (1+v) - EMA(EMA(x)) * v
        
        # We will use the standard 6-EMA coefficients method which is mathematically equivalent to the triple GD 
        # provided the EMA period is effective period. 
        # However, Tillson's original T3 often simply chains the GD function 3 times.
        
        # Let's implement the recursive GD method as it's more readable and correct to definition:
        # T3 = GD(GD(GD(x)))
        
        def gd(series: pd.Series, period: int, v: float) -> pd.Series:
            ema1 = series.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            return ema1 * (1 + v) - ema2 * v

        t3_1 = gd(source, length, volume_factor)
        t3_2 = gd(t3_1, length, volume_factor)
        t3_3 = gd(t3_2, length, volume_factor)
        
        return t3_3
