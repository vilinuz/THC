import pandas as pd
import numpy as np

class FisherTransform:
    """Fisher Transform Indicator"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 9) -> pd.Series:
        """
        Calculate Fisher Transform
        Expects 'high' and 'low' columns to determine range.
        Typically uses (High + Low) / 2
        """
        # 1. Midpoint
        hl2 = (df['high'] + df['low']) / 2
        
        # 2. Highest High and Lowest Low over period
        # Rolling max/min
        hh = hl2.rolling(window=period).max()
        ll = hl2.rolling(window=period).min()
        
        # 3. Normalize to -1..1
        # val = 2 * ((hl2 - ll) / (hh - ll)) - 1
        # Handle div/0
        denom = (hh - ll).replace(0, 0.001)
        val = 2 * ((hl2 - ll) / denom) - 1
        
        # Smooth and Clamp
        # Usually smoothed with previous value:
        # val = 0.33 * new_val + 0.67 * prev_val
        # But we need iterative calculation for that.
        # Alternatively, we can use ewm? 
        # Standard code ref:
        # Value1 = 0.33*2*((Price-Low)/(High-Low)-0.5) + 0.67*Value1[1]
        # if Value1 > 0.99, Value1=.999, if < -0.99, ...
        # Fisher = 0.5 * ln((1+Value1)/(1-Value1)) + 0.5 * Fisher[1]
        
        # Let's do iterative for precision to match "God Mode"
        
        fisher = np.zeros(len(df))
        signal = np.zeros(len(df)) # Fisher[1] essentially
        
        vals = val.fillna(0).values
        
        # State vars
        prev_val = 0.0
        prev_fisher = 0.0
        
        # Pre-calculated normalized series 'val' is not exactly Value1 yet because smoothing 
        # is applied TO the calc.
        
        # Let's reimplement strict loop
        hl2_vals = hl2.fillna(0).values
        hh_vals = hh.fillna(0).values
        ll_vals = ll.fillna(0).values
        
        processed_fishers = []
        
        # Need to iterate.
        # Initialize
        value1 = 0.0
        fisher_val = 0.0
        
        for i in range(len(df)):
            if np.isnan(hh_vals[i]) or np.isnan(ll_vals[i]):
                processed_fishers.append(0.0)
                continue
                
            current_hl2 = hl2_vals[i]
            current_hh = hh_vals[i]
            current_ll = ll_vals[i]
            
            denom = current_hh - current_ll
            if denom == 0:
                raw = 0
            else:
                raw = 2 * ((current_hl2 - current_ll) / denom) - 1
                
            # Smoothing Value1
            value1 = 0.33 * raw + 0.67 * value1
            
            # Clamp
            if value1 > 0.99: value1 = 0.999
            if value1 < -0.99: value1 = -0.999
            
            # Fisher
            # 0.5 * ln((1+v)/(1-v)) + 0.5 * prev_fisher
            fisher_val = 0.5 * np.log((1 + value1) / (1 - value1)) + 0.5 * fisher_val
            
            processed_fishers.append(fisher_val)
            
        return pd.Series(processed_fishers, index=df.index)
