"""
Order Blocks (OB) - Smart Money Concepts
Identifies institutional buying/selling zones
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class OrderBlocks:
    """
    Order Blocks represent the last opposing candle before a strong impulsive move.
    They indicate zones where institutions placed significant orders.
    """
    
    @staticmethod
    def detect(df: pd.DataFrame, swing_length: int = 10) -> pd.DataFrame:
        """
        Detect bullish and bearish order blocks
        
        Args:
            df: DataFrame with OHLC data (lowercase columns)
            swing_length: Number of candles to identify swing points
            
        Returns:
            DataFrame with order block zones
        """
        # Prepare lowercase columns
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        result = pd.DataFrame(index=df_copy.index)
        result['bullish_ob'] = False
        result['bearish_ob'] = False
        result['ob_top'] = np.nan
        result['ob_bottom'] = np.nan
        result['ob_strength'] = 0.0
        
        # Identify swing highs and lows
        highs = df_copy['high'].rolling(window=swing_length*2+1, center=True).max()
        lows = df_copy['low'].rolling(window=swing_length*2+1, center=True).min()
        
        swing_highs = df_copy['high'] == highs
        swing_lows = df_copy['low'] == lows
        
        # Detect bullish order blocks
        for i in range(swing_length, len(df_copy) - swing_length):
            # Look for swing low followed by bullish impulse
            if swing_lows.iloc[i]:
                # Find the last bearish candle before the move up
                for j in range(i-1, max(0, i-swing_length), -1):
                    if df_copy['close'].iloc[j] < df_copy['open'].iloc[j]:
                        # Check if there's an impulse move after
                        impulse_strength = (
                            df_copy['high'].iloc[i:i+swing_length].max() - 
                            df_copy['low'].iloc[j]
                        ) / df_copy['low'].iloc[j]
                        
                        if impulse_strength > 0.02:  # 2% minimum impulse
                            result.loc[result.index[j], 'bullish_ob'] = True
                            result.loc[result.index[j], 'ob_top'] = df_copy['high'].iloc[j]
                            result.loc[result.index[j], 'ob_bottom'] = df_copy['low'].iloc[j]
                            result.loc[result.index[j], 'ob_strength'] = impulse_strength
                            break
                            
        # Detect bearish order blocks
        for i in range(swing_length, len(df_copy) - swing_length):
            # Look for swing high followed by bearish impulse
            if swing_highs.iloc[i]:
                # Find the last bullish candle before the move down
                for j in range(i-1, max(0, i-swing_length), -1):
                    if df_copy['close'].iloc[j] > df_copy['open'].iloc[j]:
                        # Check if there's an impulse move after
                        impulse_strength = (
                            df_copy['high'].iloc[j] - 
                            df_copy['low'].iloc[i:i+swing_length].min()
                        ) / df_copy['high'].iloc[j]
                        
                        if impulse_strength > 0.02:  # 2% minimum impulse
                            result.loc[result.index[j], 'bearish_ob'] = True
                            result.loc[result.index[j], 'ob_top'] = df_copy['high'].iloc[j]
                            result.loc[result.index[j], 'ob_bottom'] = df_copy['low'].iloc[j]
                            result.loc[result.index[j], 'ob_strength'] = impulse_strength
                            break
                            
        return result
        
    @staticmethod
    def signal(df: pd.DataFrame, ob_df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on order blocks
        
        Returns:
            Series with 1 (buy at bullish OB), -1 (sell at bearish OB), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
        
        # Check for price touching order blocks
        for i in range(1, len(df_copy)):
            # Bullish OB signal: price touches and bounces
            bullish_obs = ob_df[ob_df['bullish_ob']].index
            for ob_idx in bullish_obs:
                if ob_idx < i:
                    ob_bottom = ob_df.loc[ob_idx, 'ob_bottom']
                    ob_top = ob_df.loc[ob_idx, 'ob_top']
                    
                    # Check if current candle touches the OB zone
                    if (df_copy['low'].iloc[i] <= ob_top and 
                        df_copy['low'].iloc[i] >= ob_bottom):
                        # Confirm bounce
                        if df_copy['close'].iloc[i] > df_copy['open'].iloc[i]:
                            signals.iloc[i] = 1
                            
            # Bearish OB signal: price touches and rejects
            bearish_obs = ob_df[ob_df['bearish_ob']].index
            for ob_idx in bearish_obs:
                if ob_idx < i:
                    ob_bottom = ob_df.loc[ob_idx, 'ob_bottom']
                    ob_top = ob_df.loc[ob_idx, 'ob_top']
                    
                    # Check if current candle touches the OB zone
                    if (df_copy['high'].iloc[i] >= ob_bottom and 
                        df_copy['high'].iloc[i] <= ob_top):
                        # Confirm rejection
                        if df_copy['close'].iloc[i] < df_copy['open'].iloc[i]:
                            signals.iloc[i] = -1
                            
        return signals
        
    @staticmethod
    def get_active_zones(ob_df: pd.DataFrame, current_idx: int, lookback: int = 50) -> Dict:
        """
        Get currently active order block zones
        
        Args:
            ob_df: Order blocks DataFrame
            current_idx: Current index position
            lookback: How far back to look for active zones
            
        Returns:
            Dict with active bullish and bearish zones
        """
        start_idx = max(0, current_idx - lookback)
        recent_obs = ob_df.iloc[start_idx:current_idx]
        
        bullish_zones = []
        bearish_zones = []
        
        for idx, row in recent_obs.iterrows():
            if row['bullish_ob']:
                bullish_zones.append({
                    'index': idx,
                    'top': row['ob_top'],
                    'bottom': row['ob_bottom'],
                    'strength': row['ob_strength']
                })
            if row['bearish_ob']:
                bearish_zones.append({
                    'index': idx,
                    'top': row['ob_top'],
                    'bottom': row['ob_bottom'],
                    'strength': row['ob_strength']
                })
                
        return {
            'bullish': bullish_zones,
            'bearish': bearish_zones
        }
