import sys
import os
import logging
# Add current directory to path
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from main import CryptoTradingPlatform
from ml.xgboost_model import XGBoostModel

# Setup logger
logging.basicConfig(level=logging.INFO)

# Mock data
dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
df = pd.DataFrame({
    'close': np.array([100] * 90 + [105] * 10) # Jump of 5% at end
}, index=dates)

class MockDB:
    def save_ohlcv(self, *args, **kwargs): pass
    def load_ohlcv(self, *args, **kwargs): return df
    def close(self): pass

class MockPlatform(CryptoTradingPlatform):
    def __init__(self):
        # Config mimicking the real one + pump detection
        self.config = {
            'indicators': {
                'vwap': {'enabled': False}, 
                'ema': {'enabled': False}, 
                'rsi': {'enabled': False}, 
                'ichimoku': {'enabled': False}
            },
            'smart_money': {'enabled': False},
            'ml_models': {
                'xgboost': {'enabled': True, 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 10},
                'pump_detection': {
                    'enabled': True,
                    'horizon': 5,
                    'threshold': 0.02 # 2%
                }
            },
            'backtesting': {'initial_capital': 10000}
        }
        self.setup_components()

    def setup_components(self):
        self.db = MockDB()
        self.cache = type('MockCache', (), {'cache_dataframe': lambda *a: None, 'close': lambda: None})()
        
    def run_test_backtest(self):
        # This mirrors run_backtest logic but stripped down for unit testing config usage
        print("Running test backtest logic...")
        
        xgb_model = XGBoostModel(self.config['ml_models']['xgboost'])
        
        # Test if logic matches main.py
        pump_config = self.config['ml_models'].get('pump_detection', {'enabled': False, 'horizon': 5, 'threshold': 0.001})
        horizon = pump_config['horizon']
        threshold = pump_config['threshold']
        
        print(f"Configured Horizon: {horizon}")
        print(f"Configured Threshold: {threshold}")
        
        labels = xgb_model.prepare_labels(df, horizon=horizon, threshold=threshold)
        
        # We expect positive labels where price jumps > 2% in next 5 bars
        # Price jumps at index 90 from 100 to 105 (5%). 
        # So at index 85 (looking 5 ahead), price is 100, future is 105. Return = 0.05 > 0.02. Label = 1.
        
        positive_labels = labels.sum()
        print(f"Positive Labels Found: {positive_labels}")
        
        if positive_labels > 0:
            print("SUCCESS: Pump detection logic generated positive labels.")
        else:
            print("FAILURE: No positive labels found despite valid pump scenario.")

async def main():
    print("Starting Pump Detection Test...")
    p = MockPlatform()
    p.run_test_backtest()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
