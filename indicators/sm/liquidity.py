"""
Liquidity Zones - Smart Money Concepts
Identifies areas where stop losses accumulate (liquidity pools)
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class Liquidity:
    """
    Liquidity zones represent areas where many stop losses are placed.
    Institutions often target these zones to fill large orders.
    """
    
    @staticmethod
    def detect(
        df: pd.DataFrame, 
        range_percent: float = 0.01,
        swing_length: int = 5
    ) -> pd.DataFrame:
        """
        Detect liquidity zones (equal highs/lows and swing points)
        
        Args:
            df: DataFrame with OHLC data
            range_percent: Percentage range for equal highs/lows (0.01 = 1%)
            swing_length: Lookback for swing points
            
        Returns:
            DataFrame with liquidity zones
        """
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        result = pd.DataFrame(index=df_copy.index)
        result['buy_side_liquidity'] = False  # Above current price (sell stops)
        result['sell_side_liquidity'] = False  # Below current price (buy stops)
        result['liquidity_level'] = np.nan
        result['liquidity_strength'] = 0
        
        # Detect equal highs (buy-side liquidity)
        for i in range(swing_length, len(df_copy)):
            recent_highs = df_copy['high'].iloc[i-swing_length:i]
            current_high = df_copy['high'].iloc[i]
            
            # Find highs within range_percent of current high
            similar_highs = recent_highs[
                (recent_highs >= current_high * (1 - range_percent)) &
                (recent_highs <= current_high * (1 + range_percent))
            ]
            
            if len(similar_highs) >= 2:  # At least 2 equal highs
                result.loc[result.index[i], 'buy_side_liquidity'] = True
                result.loc[result.index[i], 'liquidity_level'] = current_high
                result.loc[result.index[i], 'liquidity_strength'] = len(similar_highs)
                
        # Detect equal lows (sell-side liquidity)
        for i in range(swing_length, len(df_copy)):
            recent_lows = df_copy['low'].iloc[i-swing_length:i]
            current_low = df_copy['low'].iloc[i]
            
            # Find lows within range_percent of current low
            similar_lows = recent_lows[
                (recent_lows >= current_low * (1 - range_percent)) &
                (recent_lows <= current_low * (1 + range_percent))
            ]
            
            if len(similar_lows) >= 2:  # At least 2 equal lows
                result.loc[result.index[i], 'sell_side_liquidity'] = True
                result.loc[result.index[i], 'liquidity_level'] = current_low
                result.loc[result.index[i], 'liquidity_strength'] = len(similar_lows)
                
        return result
        
    @staticmethod
    def detect_sweeps(df: pd.DataFrame, liq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity sweeps (when price takes out liquidity zones)
        
        Returns:
            DataFrame with sweep information
        """
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        sweeps = pd.DataFrame(index=df_copy.index)
        sweeps['buy_side_sweep'] = False
        sweeps['sell_side_sweep'] = False
        sweeps['sweep_strength'] = 0.0
        
        for i in range(1, len(df_copy)):
            current_candle = df_copy.iloc[i]
            
            # Check for buy-side liquidity sweeps (price goes above then reverses)
            buy_liq = liq_df[liq_df['buy_side_liquidity']].iloc[:i]
            
            for idx, liq in buy_liq.iterrows():
                liq_level = liq['liquidity_level']
                
                # Price must sweep above the level
                if current_candle['high'] > liq_level:
                    # Check for reversal (close back below)
                    if current_candle['close'] < liq_level:
                        sweeps.loc[sweeps.index[i], 'buy_side_sweep'] = True
                        sweeps.loc[sweeps.index[i], 'sweep_strength'] = liq['liquidity_strength']
                        break
                        
            # Check for sell-side liquidity sweeps
            sell_liq = liq_df[liq_df['sell_side_liquidity']].iloc[:i]
            
            for idx, liq in sell_liq.iterrows():
                liq_level = liq['liquidity_level']
                
                # Price must sweep below the level
                if current_candle['low'] < liq_level:
                    # Check for reversal (close back above)
                    if current_candle['close'] > liq_level:
                        sweeps.loc[sweeps.index[i], 'sell_side_sweep'] = True
                        sweeps.loc[sweeps.index[i], 'sweep_strength'] = liq['liquidity_strength']
                        break
                        
        return sweeps
        
    @staticmethod
    def signal(df: pd.DataFrame, sweeps_df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on liquidity sweeps
        
        Returns:
            Series with 1 (buy after sell-side sweep), -1 (sell after buy-side sweep), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
        
        for i in range(1, len(df_copy)):
            # Buy signal: After sell-side liquidity sweep (stop hunts below support)
            if sweeps_df['sell_side_sweep'].iloc[i]:
                # Confirm with bullish candle
                if df_copy['close'].iloc[i] > df_copy['open'].iloc[i]:
                    signals.iloc[i] = 1
                    
            # Sell signal: After buy-side liquidity sweep (stop hunts above resistance)
            if sweeps_df['buy_side_sweep'].iloc[i]:
                # Confirm with bearish candle
                if df_copy['close'].iloc[i] < df_copy['open'].iloc[i]:
                    signals.iloc[i] = -1
                    
        return signals
        
    @staticmethod
    def get_active_liquidity(liq_df: pd.DataFrame, current_price: float) -> Dict:
        """
        Get active liquidity zones relative to current price
        
        Returns:
            Dict with liquidity above and below current price
        """
        above_price = liq_df[
            (liq_df['buy_side_liquidity']) & 
            (liq_df['liquidity_level'] > current_price)
        ]
        
        below_price = liq_df[
            (liq_df['sell_side_liquidity']) & 
            (liq_df['liquidity_level'] < current_price)
        ]
        
        return {
            'above': above_price.to_dict('records'),
            'below': below_price.to_dict('records')
        }
