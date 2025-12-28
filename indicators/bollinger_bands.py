import pandas as pd
import numpy as np

class BollingerBands:
    """Bollinger Bands"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and Band Width
        """
        result = pd.DataFrame(index=df.index)
        
        # Middle Band = SMA
        result['middle'] = df['close'].rolling(window=period).mean()
        
        # Std Dev
        std = df['close'].rolling(window=period).std()
        
        # Upper/Lower
        result['upper'] = result['middle'] + (std * std_dev)
        result['lower'] = result['middle'] - (std * std_dev)
        
        # Band Width (BBW) = (Upper - Lower) / Middle
        result['bandwidth'] = (result['upper'] - result['lower']) / result['middle']
        
        # %B = (Price - Lower) / (Upper - Lower)
        result['percent_b'] = (df['close'] - result['lower']) / (result['upper'] - result['lower'])
        
        return result
        
    @staticmethod
    def is_squeeze(df_bb: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
        """Return boolean series if bandwidth is below threshold (Squeeze)"""
        return df_bb['bandwidth'] < threshold
