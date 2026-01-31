import pandas as pd
import numpy as np

class SmartMoneyIndicator:
    """ 
    Quantifies SMC: Specifically 'Volume-Price Divergence' (CVD Proxy).
    If Price makes a Lower Low but Volume/Flow makes a Higher Low -> Buy Signal.
    """
    def __init__(self, window=14):
        self.window = window
        self.price_history = []
        self.vol_history = []

    def update(self, price, volume):
        self.price_history.append(price)
        self.vol_history.append(volume)
        
        if len(self.price_history) > self.window:
            self.price_history.pop(0)
            self.vol_history.pop(0)
        
        if len(self.price_history) < self.window:
            return 0 # Neutral

        # Simple Correlation Logic
        # If Price and Volume are negatively correlated during a drop, it's absorption.
        # We return a score: -1 (Distribution) to +1 (Accumulation)
        
        price_series = pd.Series(self.price_history)
        vol_series = pd.Series(self.vol_history)
        
        # Calculate Force Index (Price Change * Volume)
        force = (price_series.diff() * vol_series).fillna(0)
        
        # Smooth it
        smc_signal = force.ewm(span=3).mean().iloc[-1]
        
        # Normalize (simplified for demo)
        return np.tanh(smc_signal / vol_series.mean()) 

