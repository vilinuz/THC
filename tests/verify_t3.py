import pandas as pd
import numpy as np
import sys
import os

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indicators.tillson_t3 import TillsonT3
from strategy.b_sniper_strategy import BSniperStrategy

def test_t3_calculation():
    print("Testing T3 Calculation...")
    # Generate dummy data
    data = {'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    df = pd.DataFrame(data)
    
    t3 = TillsonT3.calculate(df, length=3, volume_factor=0.7)
    print(f"T3 Output:\n{t3}")
    
    assert len(t3) == len(df), "T3 length mismatch"
    print("T3 Calculation OK")

def test_b_sniper_logic():
    print("\nTesting B Sniper Logic...")
    
    # Create a scenario
    # 0. Initial
    # 1. T3 Rising
    # 2. Case A: Green Candle, Gap Up -> Valid
    # 3. Case B: Green Candle, Touches T3 -> Invalid
    # 4. Case C: Green Marubozu, Touches T3 -> Valid
    
    # We need enough initial data to stabilize T3 or manipulate it
    # Ideally mock T3, but integration test is better
    
    # Let's verify manual logic by inspecting the BSniperStrategy code logic with mocked DF if T3 was separate,
    # but BSniperStrategy calculates T3 internally.
    # To test strictly, we can construct price data that guarantees T3 behavior or mock the T3 calculate method.
    
    # For this smoke test, let's create a scenario where Price >> T3 to ensure rising T3 and Gap.
    
    # Scenario: Flat then Jump
    prices = [100] * 10 + [150 + i*2 for i in range(10)] 
    # T3 will lag significantly on the jump
    
    opens =  [p - 2 for p in prices]    # Green candles
    highs =  [p + 5 for p in prices]
    lows =   [p - 2 for p in prices]
    
    df = pd.DataFrame({
        'close': prices,
        'open': opens,
        'high': highs,
        'low': lows,
        'volume': [1000] * 20
    })
    
    strat = BSniperStrategy(config={'t3_length': 5})
    signals = strat.generate_signals(df)
    
    t3_vals = TillsonT3.calculate(df, 5, 1.7)
    df['t3'] = t3_vals
    print(df[['close', 'low', 't3']].tail())
    
    print(f"Signals (Standard Uptrend):\n{signals.tail()}")
    
    # Expect signals = 1 in uptrend with gap
    assert signals.iloc[-1] == 1, "Expected Buy Signal in clear uptrend gap"
    
    # Test Invalid Touch
    # Create a candle that dips to touch predicted T3
    # We need to know T3 value.
    t3_vals = TillsonT3.calculate(df, 5, 1.7)
    last_t3 = t3_vals.iloc[-1]
    
    print(f"Last T3: {last_t3}")
    
    # Create new row where low < last_t3 < high (Touching)
    new_row = {
        'close': last_t3 + 5,
        'open': last_t3 + 1,
        'high': last_t3 + 6,
        'low': last_t3 - 1, # Touches/Crosses T3
        'volume': 1000
    } # Not Marubozu (Range=7, Body=4, Ratio ~0.57)
    
    df2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    signals2 = strat.generate_signals(df2)
    
    # Should be 0 (Invalid due to touch)
    print(f"Signal for Touching Non-Marubozu: {signals2.iloc[-1]}")
    assert signals2.iloc[-1] == 0, "Expected NO Signal for touching non-marubozu"
    
    # Test Marubozu Touch
    # Marubozu > 0.9 body/range
    # Let's make a big green candle crossing T3
    maru_row = {
        'close': last_t3 + 10,
        'open': last_t3 - 1,
        'high': last_t3 + 10,
        'low': last_t3 - 1,
        'volume': 1000
    } # Body=11, Range=11 => Ratio=1.0 (Marubozu)
    
    df3 = pd.concat([df, pd.DataFrame([maru_row])], ignore_index=True)
    signals3 = strat.generate_signals(df3)
    
    print(f"Signal for Touching Marubozu: {signals3.iloc[-1]}")
    assert signals3.iloc[-1] == 1, "Expected Buy Signal for touching Marubozu"
    
    print("B Sniper Logic OK")

if __name__ == "__main__":
    test_t3_calculation()
    test_b_sniper_logic()
