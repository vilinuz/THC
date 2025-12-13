"""
Market Structure - Break of Structure (BOS) and Change of Character (CHoCH)
Core concepts in Smart Money analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class MarketStructure:
    """
    Identifies market structure breaks and changes in character.
    BOS = Break of Structure (continuation)
    CHoCH = Change of Character (reversal)
    """
    
    @staticmethod
    def identify_swing_points(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
        """
        Identify swing highs and swing lows
        
        Args:
            df: DataFrame with OHLC data
            swing_length: Lookback period for swings
            
        Returns:
            DataFrame with swing points marked
        """
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        result = pd.DataFrame(index=df_copy.index)
        result['swing_high'] = False
        result['swing_low'] = False
        result['swing_high_price'] = np.nan
        result['swing_low_price'] = np.nan
        
        # Identify swing highs
        for i in range(swing_length, len(df_copy) - swing_length):
            window = df_copy['high'].iloc[i-swing_length:i+swing_length+1]
            if df_copy['high'].iloc[i] == window.max():
                result.loc[result.index[i], 'swing_high'] = True
                result.loc[result.index[i], 'swing_high_price'] = df_copy['high'].iloc[i]
                
        # Identify swing lows
        for i in range(swing_length, len(df_copy) - swing_length):
            window = df_copy['low'].iloc[i-swing_length:i+swing_length+1]
            if df_copy['low'].iloc[i] == window.min():
                result.loc[result.index[i], 'swing_low'] = True
                result.loc[result.index[i], 'swing_low_price'] = df_copy['low'].iloc[i]
                
        return result
        
    @staticmethod
    def detect_bos_choch(df: pd.DataFrame, swing_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH)
        
        Returns:
            DataFrame with BOS and CHoCH markers
        """
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        result = pd.DataFrame(index=df_copy.index)
        result['bullish_bos'] = False
        result['bearish_bos'] = False
        result['bullish_choch'] = False
        result['bearish_choch'] = False
        result['trend'] = 0  # 1 = bullish, -1 = bearish, 0 = ranging
        
        # Track previous swing points
        prev_swing_high = None
        prev_swing_low = None
        current_trend = 0
        
        for i in range(len(df_copy)):
            # Update swing highs
            if swing_df['swing_high'].iloc[i]:
                new_high = swing_df['swing_high_price'].iloc[i]
                
                if prev_swing_high is not None:
                    if new_high > prev_swing_high:
                        # Higher high
                        if current_trend == 1:
                            # Continuation - BOS
                            result.loc[result.index[i], 'bullish_bos'] = True
                        else:
                            # Reversal - CHoCH
                            result.loc[result.index[i], 'bullish_choch'] = True
                            current_trend = 1
                    else:
                        # Lower high - potential reversal
                        if current_trend == 1:
                            result.loc[result.index[i], 'bearish_choch'] = True
                            current_trend = -1
                            
                prev_swing_high = new_high
                
            # Update swing lows
            if swing_df['swing_low'].iloc[i]:
                new_low = swing_df['swing_low_price'].iloc[i]
                
                if prev_swing_low is not None:
                    if new_low < prev_swing_low:
                        # Lower low
                        if current_trend == -1:
                            # Continuation - BOS
                            result.loc[result.index[i], 'bearish_bos'] = True
                        else:
                            # Reversal - CHoCH
                            result.loc[result.index[i], 'bearish_choch'] = True
                            current_trend = -1
                    else:
                        # Higher low - potential reversal
                        if current_trend == -1:
                            result.loc[result.index[i], 'bullish_choch'] = True
                            current_trend = 1
                            
                prev_swing_low = new_low
                
            result.loc[result.index[i], 'trend'] = current_trend
            
        return result
        
    @staticmethod
    def signal(structure_df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on market structure
        
        Returns:
            Series with 1 (buy), -1 (sell), 0 (neutral)
        """
        signals = pd.Series(0, index=structure_df.index)
        
        for i in range(len(structure_df)):
            # Strong buy signal: Bullish BOS in uptrend
            if structure_df['bullish_bos'].iloc[i] and structure_df['trend'].iloc[i] == 1:
                signals.iloc[i] = 1
                
            # Strong sell signal: Bearish BOS in downtrend
            if structure_df['bearish_bos'].iloc[i] and structure_df['trend'].iloc[i] == -1:
                signals.iloc[i] = -1
                
            # Reversal buy signal: Bullish CHoCH
            if structure_df['bullish_choch'].iloc[i]:
                signals.iloc[i] = 1
                
            # Reversal sell signal: Bearish CHoCH
            if structure_df['bearish_choch'].iloc[i]:
                signals.iloc[i] = -1
                
        return signals
        
    @staticmethod
    def get_current_trend(structure_df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Get current market trend and recent structure breaks
        
        Returns:
            Dict with trend information
        """
        current_trend = structure_df['trend'].iloc[current_idx]
        
        # Find recent BOS and CHoCH
        recent = structure_df.iloc[max(0, current_idx-50):current_idx]
        
        recent_bos = {
            'bullish': recent[recent['bullish_bos']].index.tolist(),
            'bearish': recent[recent['bearish_bos']].index.tolist()
        }
        
        recent_choch = {
            'bullish': recent[recent['bullish_choch']].index.tolist(),
            'bearish': recent[recent['bearish_choch']].index.tolist()
        }
        
        return {
            'trend': 'bullish' if current_trend == 1 else ('bearish' if current_trend == -1 else 'ranging'),
            'recent_bos': recent_bos,
            'recent_choch': recent_choch
        }

