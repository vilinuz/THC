import pandas as pd
import numpy as np

class MarkovProxy:
    """
    Markov Chain Proxy (Simplified 3-State)
    
    Pine Script logic:
    State 1 (Up > 0.1%), -1 (Down < -0.1%), 0 (Flat)
    
    P(Up):
    - If last 2 states were 1 (Up, Up), P(Up) = 0.75
    - If last 2 states were -1 (Down, Down), P(Up) = 0.25
    - Else 0.5
    """
    
    @staticmethod
    def calculate_prob_up(df: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(df['close'])
        
        # Calculate Returns %
        # ret = (close - close[1]) / close[1] * 100
        ret = close.pct_change() * 100
        
        # Determine State
        # 1: Up > 0.1%
        # -1: Down < -0.1%
        # 0: Flat
        state = np.zeros(len(close), dtype=int)
        
        # Use numpy where for speed
        vals = ret.values
        state = np.where(vals > 0.1, 1, np.where(vals < -0.1, -1, 0))
        
        # Convert state to Series for shifting
        state_s = pd.Series(state, index=df.index)
        
        # Calculate Prob Up
        # Default 0.5
        p_up = pd.Series(0.5, index=df.index)
        
        s1 = state_s.shift(1)
        s2 = state_s.shift(2)
        
        # Conditions
        # if state[1] == 1 and state[2] == 1 -> 0.75
        mask_bull = (s1 == 1) & (s2 == 1)
        p_up[mask_bull] = 0.75
        
        # if state[1] == -1 and state[2] == -1 -> 0.25
        mask_bear = (s1 == -1) & (s2 == -1)
        p_up[mask_bear] = 0.25
        
        return p_up
