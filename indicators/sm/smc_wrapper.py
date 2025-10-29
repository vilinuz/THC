"""
Wrapper for smartmoneyconcepts library integration
Provides unified interface to external SMC library
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

class SMCWrapper:
    """
    Wrapper for the smartmoneyconcepts library with fallback to custom implementations
    """
    
    def __init__(self, use_library: bool = True):
        self.use_library = use_library
        self.smc = None
        
        if use_library:
            try:
                from smartmoneyconcepts import smc
                self.smc = smc
            except ImportError:
                print("Warning: smartmoneyconcepts library not found. Using custom implementation.")
                self.use_library = False
                
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for SMC library (requires lowercase columns)
        """
        df_prep = df.copy()
        
        # Convert column names to lowercase
        df_prep.columns = df_prep.columns.str.lower()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df_prep.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df_prep
        
    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps using library or custom implementation
        """
        df_prep = self.prepare_data(df)
        
        if self.use_library and self.smc:
            try:
                result = self.smc.fvg(df_prep)
                return result
            except Exception as e:
                print(f"Library FVG detection failed: {e}. Using custom implementation.")
                
        # Fallback to custom implementation
        from .fvg import FairValueGap
        return FairValueGap.detect(df_prep)
        
    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks using library or custom implementation
        """
        df_prep = self.prepare_data(df)
        
        if self.use_library and self.smc:
            try:
                result = self.smc.ob(df_prep)
                return result
            except Exception as e:
                print(f"Library OB detection failed: {e}. Using custom implementation.")
                
        # Fallback to custom implementation
        from .order_blocks import OrderBlocks
        return OrderBlocks.detect(df_prep)
        
    def detect_liquidity(
        self, 
        df: pd.DataFrame, 
        range_percent: float = 0.01,
        up_thresh: float = 0.05,
        down_thresh: float = -0.05
    ) -> pd.DataFrame:
        """
        Detect Liquidity zones using library or custom implementation
        """
        df_prep = self.prepare_data(df)
        
        if self.use_library and self.smc:
            try:
                result = self.smc.liquidity(
                    df_prep,
                    range_percent=range_percent,
                    up_thresh=up_thresh,
                    down_thresh=down_thresh
                )
                return result
            except Exception as e:
                print(f"Library liquidity detection failed: {e}. Using custom implementation.")
                
        # Fallback to custom implementation
        from .liquidity import Liquidity
        return Liquidity.detect(df_prep, range_percent=range_percent)
        
    def detect_swing_points(
        self,
        df: pd.DataFrame,
        up_thresh: float = 0.05,
        down_thresh: float = -0.05
    ) -> pd.DataFrame:
        """
        Detect swing highs and lows
        """
        df_prep = self.prepare_data(df)
        
        if self.use_library and self.smc:
            try:
                result = self.smc.highs_lows(
                    df_prep,
                    up_thresh=up_thresh,
                    down_thresh=down_thresh
                )
                return result
            except Exception as e:
                print(f"Library swing detection failed: {e}. Using custom implementation.")
                
        # Fallback to custom implementation
        from .market_structure import MarketStructure
        return MarketStructure.identify_swing_points(df_prep)
        
    def comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all SMC analyses and return comprehensive results
        
        Returns:
            Dict with all SMC indicators
        """
        results = {}
        
        # Fair Value Gaps
        results['fvg'] = self.detect_fvg(df)
        
        # Order Blocks
        results['order_blocks'] = self.detect_order_blocks(df)
        
        # Liquidity
        results['liquidity'] = self.detect_liquidity(df)
        
        # Swing Points
        results['swings'] = self.detect_swing_points(df)
        
        # Market Structure (custom only)
        from .market_structure import MarketStructure
        df_prep = self.prepare_data(df)
        swing_df = MarketStructure.identify_swing_points(df_prep)
        results['structure'] = MarketStructure.detect_bos_choch(df_prep, swing_df)
        
        return results
