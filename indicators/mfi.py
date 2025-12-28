import pandas as pd
import numpy as np

class MFI:
    """Money Flow Index"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate MFI
        """
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Raw Money Flow
        rmf = tp * df['volume']
        
        # Money Flow Ratio
        # Positive Flow: tp > prev_tp
        # Negative Flow: tp < prev_tp
        
        diff = tp.diff()
        
        positive_flow = np.where(diff > 0, rmf, 0)
        negative_flow = np.where(diff < 0, rmf, 0)
        
        # Rolling Sums
        pos_sum = pd.Series(positive_flow, index=df.index).rolling(window=period).sum()
        neg_sum = pd.Series(negative_flow, index=df.index).rolling(window=period).sum()
        
        # MFI = 100 - (100 / (1 + Ratio))
        mfi_ratio = pos_sum / neg_sum.replace(0, 0.001)
        mfi = 100 - (100 / (1 + mfi_ratio))
        
        return mfi
