import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('/home/vilivom/src/THC')

from strategy.multi_layer_strategy_2 import MultiLayerStrategy2

def generate_synthetic_data(n=500):
    np.random.seed(42)
    # Generate a trend + accumulation + trend
    indices = pd.date_range(start='2024-01-01', periods=n, freq='15T')
    
    # Random walk with drift
    # Phase 1: Uptrend (Start to 200)
    # Phase 2: Chop (200 to 300)
    # Phase 3: Downtrend (300 to 500)
    
    close = np.zeros(n)
    close[0] = 10000
    
    for i in range(1, n):
        noise = np.random.normal(0, 10)
        drift = 0
        if i < 200:
             drift = 5 # Up
        elif i < 300:
             drift = 0 # Chop
        else:
             drift = -5 # Down
             
        close[i] = close[i-1] + drift + noise
        
    # High/Low/Open derivation
    high = close + np.random.rand(n) * 20
    low = close - np.random.rand(n) * 20
    open_p = close + np.random.normal(0, 5) # Rough approx
    
    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    }, index=indices)
    
    return df

def test_strategy():
    print("Generaring synthetic data...")
    df = generate_synthetic_data()
    print(f"Data generated: {len(df)} bars.")
    
    print("Initializing MultiLayerStrategy2...")
    strategy = MultiLayerStrategy2()
    
    print("Generating signals...")
    signals = strategy.generate_signals(df)
    
    print("\n=== Verification Results ===")
    print(f"Signal Series Length: {len(signals)}")
    print(f"Columns: {signals.columns.tolist()}")
    
    # Check for NaNs (some expected at start due to warmpup)
    total_nans = signals['signal'].isna().sum()
    print(f"NaNs in signal: {total_nans}")
    
    # Show last 10 rows
    print("\nLast 10 bars:")
    combined = pd.concat([df['close'], signals], axis=1).tail(10)
    print(combined[['close', 'regime_status', 'votes_bull', 'votes_bear', 'signal']])
    
    # Stats
    counts = signals['regime_status'].value_counts()
    print("\nRegime Counts:")
    print(counts)
    
    if len(signals) == len(df):
        print("\nSUCCESS: Signal generation complete.")
    else:
        print("\nFAILURE: Length mismatch.")

if __name__ == "__main__":
    test_strategy()
