from typing import Dict
import pandas as pd
import numpy as np
from .base_strateg import BaseStrategy
from indicators.tillson_t3 import TillsonT3

class BSniperStrategy(BaseStrategy):
    """
    B Sniper Entry Strategy based on Tillson T3
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.t3_length = self.config.get('t3_length', 10)
        self.volume_factor = self.config.get('volume_factor', 1.7)
        # Fibo values exposed but optionally used if user changes config
        self.t3_length_fibo = self.config.get('t3_length_fibo', 6) 
        self.volume_factor_fibo = self.config.get('volume_factor_fibo', 5.618)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Buy Signals for B Sniper Setup
        """
        if df.empty:
            return pd.Series()
            
        # Calculate T3
        t3 = TillsonT3.calculate(df, self.t3_length, self.volume_factor)
        
        # Identify Green T3 (Rising)
        t3_green = t3 > t3.shift(1)
        
        # Identify Green Candle
        green_candle = df['close'] > df['open']
        
        # Body and Wick calculations
        body_top = df[['open', 'close']].max(axis=1)
        body_bottom = df[['open', 'close']].min(axis=1)
        can_high = df['high']
        can_low = df['low']
        
        body_size = body_bottom - body_top # wait, abs dif
        body_abs = (df['close'] - df['open']).abs()
        range_len = df['high'] - df['low']
        
        # Marubozu Check: High body to range ratio (e.g., > 90%)
        # Guard against zero division
        marubozu = (body_abs / range_len.replace(0, 0.0001)) > 0.9
        
        # "touching" logic:
        # Candle touches T3 if High >= T3 and Low <= T3
        # But specifically the requirements say: "The green candle should NOT touch the Tillson T3 (body + wick)"
        # So Low > T3 (Gap Up)
        
        # Wait, if T3 is above candle, it's not a buy anyway? 
        # Typically B Sniper is for trends. "T3 should be green". That means uptrend.
        # Usually price is above T3 in uptrend.
        # So "Not Touch" means the candle is completely ABOVE the T3 line (float).
        
        # Condition: Low > T3
        candle_above_t3 = df['low'] > t3
        
        # Marubozu Exception: "If the Green candle is a marubozu, rule number 4 is not applicable"
        # Rule 4 is "Green candle should NOT touch T3".
        # So if Marubozu, it CAN touch T3.
        # But presumably it must still be a Green candle and T3 Green.
        
        # Does Marubozu imply it allows crossing? Or just touching?
        # "Entry on the next candle"
        
        signals = pd.Series(0, index=df.index)
        
        # Helper for "Touch"
        # Touches = (Low <= T3 <= High)
        touches_t3 = (df['low'] <= t3) & (df['high'] >= t3)
        
        # Final Logic
        # 1. T3 Green
        # 2. Green Candle
        # 3. (Not Touching) OR (Marubozu)
        # Note: If it's NOT a Marubozu, it MUST NOT TOUCH. 
        # Meaning: if !Marubozu -> Low > T3.
        # If Marubozu -> It can be anywhere? 
        # Likely still needs to be generally "Valid".
        # Assume "Marubozu rule" means specific exception to the gap rule.
        # User said: "Suppose the candle touches T3 -> INVALID BUY. Skip immediately. If... marubozu, rule 4 not applicable"
        
        valid_separation = candle_above_t3 | marubozu
        
        # But wait, if Marubozu touches T3, is it valid? Yes.
        # If Non-Marubozu touches T3, is it valid? No.
        
        # Also need to check if Marubozu is BELOW T3? 
        # "B label must be under a GREEN candle" -> Usually implies candle is somewhat up.
        # Let's assume the signal requires Close > T3 at least? 
        # User didn't specify Close > T3, but "B Sniper" implies breakout/trend.
        # Let's stick to the strict rules provided:
        # Rule 1: T3 Green
        # Rule 2: Green Candle
        # Rule 3: !Touch OR Marubozu
        
        # Let's refine Rule 3 based on "Suppose candle touches T3 -> INVALID".
        # Exceptions logic:
        # if marubozu: Valid even if touches.
        # if not marubozu: Invalid if touches. (Must be above? Or just not touching? Logic implies Low > T3 for trend following).
        
        # Let's codify:
        # Is Valid = T3_Green AND Candle_Green AND ( (Low > T3) OR (Marubozu) )
        
        buy_signal = t3_green & green_candle & (candle_above_t3 | marubozu)
        
        signals[buy_signal] = 1
        
        return signals

    def calculate_position_size(self, signal: int, price: float, portfolio_value: float, risk_params: Dict) -> float:
        return portfolio_value * 0.1 / price # Dummy implementation
