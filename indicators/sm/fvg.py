"""
Fair Value Gaps (FVG) - Smart Money Concepts
Identifies price imbalances that institutions may fill
"""
import pandas as pd
import numpy as np
from typing import Dict

class FairValueGap:
    """
    Fair Value Gaps represent inefficiencies in price where the market moved too quickly,
    leaving unfilled orders. Price often returns to fill these gaps.
    """
    
    @staticmethod
    def detect(df: pd.DataFrame, min_gap_size: float = 0.001) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG)
        
        Args:
            df: DataFrame with OHLC data
            min_gap_size: Minimum gap size as percentage (0.001 = 0.1%)
            
        Returns:
            DataFrame with FVG zones
        """
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
            
        result = pd.DataFrame(index=df_copy.index)
        result['bullish_fvg'] = False
        result['bearish_fvg'] = False
        result['fvg_top'] = np.nan
        result['fvg_bottom'] = np.nan
        result['fvg_size'] = 0.0
        result['fvg_filled'] = False
        
        # FVG requires 3 consecutive candles
        for i in range(2, len(df_copy)):
            candle_1 = df_copy.iloc[i-2]
            candle_2 = df_copy.iloc[i-1]  # Middle candle
            candle_3 = df_copy.iloc[i]
            
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            # Middle candle must not fill the gap
            gap_low = candle_1['high']
            gap_high = candle_3['low']
            
            if gap_high > gap_low:  # There's a gap up
                gap_size = (gap_high - gap_low) / gap_low
                
                if gap_size >= min_gap_size:
                    # Check if middle candle doesn't fill it
                    if candle_2['low'] > gap_low:
                        result['bullish_fvg'].iloc[i] = True
                        result['fvg_bottom'].iloc[i] = gap_low
                        result['fvg_top'].iloc[i] = gap_high
                        result['fvg_size'].iloc[i] = gap_size
                        
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            gap_low = candle_3['high']
            gap_high = candle_1['low']
            
            if gap_high > gap_low:  # There's a gap down
                gap_size = (gap_high - gap_low) / gap_high
                
                if gap_size >= min_gap_size:
                    # Check if middle candle doesn't fill it
                    if candle_2['high'] < gap_high:
                        result['bearish_fvg'].iloc[i] = True
                        result['fvg_bottom'].iloc[i] = gap_low
                        result['fvg_top'].iloc[i] = gap_high
                        result['fvg_size'].iloc[i] = gap_size
                        
        # Mark filled FVGs
        for i in range(len(result)):
            if result['bullish_fvg'].iloc[i]:
                fvg_bottom = result['fvg_bottom'].iloc[i]
                # Check if any future candle fills it
                for j in range(i+1, len(df_copy)):
                    if df_copy['low'].iloc[j] <= fvg_bottom:
                        result['fvg_filled'].iloc[i] = True
                        break
                        
            if result['bearish_fvg'].iloc[i]:
                fvg_top = result['fvg_top'].iloc[i]
                # Check if any future candle fills it
                for j in range(i+1, len(df_copy)):
                    if df_copy['high'].iloc[j] >= fvg_top:
                        result['fvg_filled'].iloc[i] = True
                        break
                        
        return result
        
    @staticmethod
    def signal(df: pd.DataFrame, fvg_df: pd.DataFrame, mitigation_threshold: float = 0.5) -> pd.Series:
        """
        Generate trading signals based on FVG mitigation
        
        Args:
            mitigation_threshold: How much of the gap should be filled (0.5 = 50%)
            
        Returns:
            Series with 1 (buy), -1 (sell), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        df_copy = df.copy()
        if 'Open' in df_copy.columns:
            df_copy.columns = df_copy.columns.str.lower()
        
        for i in range(len(df_copy)):
            current_price = df_copy['close'].iloc[i]
            
            # Look for unfilled bullish FVGs
            bullish_fvgs = fvg_df[(fvg_df['bullish_fvg']) & (~fvg_df['fvg_filled'])].iloc[:i]
            
            for idx, fvg in bullish_fvgs.iterrows():
                fvg_bottom = fvg['fvg_bottom']
                fvg_top = fvg['fvg_top']
                gap_size = fvg_top - fvg_bottom
                
                # Check if price is mitigating the FVG
                if fvg_bottom <= current_price <= fvg_top:
                    mitigation = (fvg_top - current_price) / gap_size
                    
                    if mitigation >= mitigation_threshold:
                        # Price has filled enough of the gap
                        if df_copy['close'].iloc[i] > df_copy['open'].iloc[i]:
                            signals.iloc[i] = 1  # Buy signal
                            
            # Look for unfilled bearish FVGs
            bearish_fvgs = fvg_df[(fvg_df['bearish_fvg']) & (~fvg_df['fvg_filled'])].iloc[:i]
            
            for idx, fvg in bearish_fvgs.iterrows():
                fvg_bottom = fvg['fvg_bottom']
                fvg_top = fvg['fvg_top']
                gap_size = fvg_top - fvg_bottom
                
                # Check if price is mitigating the FVG
                if fvg_bottom <= current_price <= fvg_top:
                    mitigation = (current_price - fvg_bottom) / gap_size
                    
                    if mitigation >= mitigation_threshold:
                        # Price has filled enough of the gap
                        if df_copy['close'].iloc[i] < df_copy['open'].iloc[i]:
                            signals.iloc[i] = -1  # Sell signal
                            
        return signals
        
    @staticmethod
    def get_unfilled_gaps(fvg_df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Get currently unfilled Fair Value Gaps
        
        Returns:
            Dict with unfilled bullish and bearish gaps
        """
        recent_fvgs = fvg_df.iloc[:current_idx]
        
        bullish_gaps = []
        bearish_gaps = []
        
        for idx, row in recent_fvgs.iterrows():
            if row['bullish_fvg'] and not row['fvg_filled']:
                bullish_gaps.append({
                    'index': idx,
                    'top': row['fvg_top'],
                    'bottom': row['fvg_bottom'],
                    'size': row['fvg_size']
                })
            if row['bearish_fvg'] and not row['fvg_filled']:
                bearish_gaps.append({
                    'index': idx,
                    'top': row['fvg_top'],
                    'bottom': row['fvg_bottom'],
                    'size': row['fvg_size']
                })
                
        return {
            'bullish': bullish_gaps,
            'bearish': bearish_gaps
        }
