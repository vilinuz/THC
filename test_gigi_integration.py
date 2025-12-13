import sys
import os
import logging
# Add current directory to path
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from main import CryptoTradingPlatform
from indicators.gigi_strategy import GigiStrategy

# Setup logger
logging.basicConfig(level=logging.INFO)

# Mock data
dates = pd.date_range(start='2024-01-01', periods=300, freq='h')
np.random.seed(42)
df = pd.DataFrame({
    'open': np.random.rand(300) * 100 + 10000,
    'high': np.random.rand(300) * 100 + 10100,
    'low': np.random.rand(300) * 100 + 9900,
    'close': np.random.rand(300) * 100 + 10050,
    'volume': np.random.randint(100, 1000, 300)
}, index=dates)

# Create some artificial trends/patterns to trigger signals
# Create a dip (buy scenario)
df.iloc[50:60, 3] = df.iloc[50:60, 3] * 0.95 # Drop close
df.iloc[50:60, 2] = df.iloc[50:60, 2] * 0.95 # Drop low
# Create a spike (sell scenario)
df.iloc[70:80, 3] = df.iloc[70:80, 3] * 1.05
df.iloc[70:80, 1] = df.iloc[70:80, 1] * 1.05

class MockDB:
    def save_ohlcv(self, *args, **kwargs): pass
    def load_ohlcv(self, *args, **kwargs): return df
    def close(self): pass

class MockPlatform(CryptoTradingPlatform):
    def __init__(self):
        self.config = {
            'indicators': {
                'vwap': {'enabled': False}, 
                'ema': {'enabled': False}, 
                'rsi': {'enabled': False}, 
                'ichimoku': {'enabled': False}
            },
            'smart_money': {'enabled': False},
            'gigi_strategy': {
                'enabled': True,
                'rsi_len': 14,
                'rsi_buy_level': 30,
                'rsi_sell_level': 60,
                'wick_ratio_min': 0.1, # Relaxed for test
                'atr_len': 14,
                'use_bounce_filter': False # Disable for easier testing of raw signals
            },
            'ml_models': {
                'xgboost': {'enabled': False}
            },
            'backtesting': {'initial_capital': 10000}
        }
        self.setup_components()

    def setup_components(self):
        self.db = MockDB()
        self.cache = type('MockCache', (), {'cache_dataframe': lambda *a: None, 'close': lambda: None})()

    async def fetch_and_store_data(self, *args, **kwargs):
        return df

async def main():
    print("Starting Gigi Strategy Test...")
    p = MockPlatform()
    
    print("Calculating indicators...")
    try:
        inds = p.calculate_indicators(df)
        gigi_keys = [k for k in inds.keys() if 'gigi' in k]
        print(f"Gigi Keys Found: {gigi_keys}")
        
        if 'gigi_gigi_signal' in inds:
            signals = inds['gigi_gigi_signal']
            buys = signals[signals == 1].count()
            sells = signals[signals == -1].count()
            print(f"Signals generated - Buys: {buys}, Sells: {sells}")
            
            if len(gigi_keys) > 0:
                print("SUCCESS: Gigi indicators were calculated and integrated.")
            else:
                print("FAILURE: Gigi keys missing.")
        else:
            print("FAILURE: 'gigi_gigi_signal' not found in indicators.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
