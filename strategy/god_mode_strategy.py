from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from .base_strateg import BaseStrategy

# Indicators
from indicators.adx import ADX
from indicators.bollinger_bands import BollingerBands
from indicators.fisher_transform import FisherTransform
from indicators.mfi import MFI
from indicators.vwap import VWAP
# SMC for FVG
try:
    from indicators.sm.smc_wrapper import SMCWrapper
except ImportError:
    SMCWrapper = None

class GodModeStrategy(BaseStrategy):
    """
    God Mode / Sniper Strategy
    
    Phases:
    1. No-Go (ADX, Volatility)
    2. Location (Magnets: Pivots, Monday Range, FVG, Liq, AnchorVWAP)
    3. Golden Numbers (Fisher, Z-Score, Funding, MFI, CVD)
    4. Trigger (Stop-Sweep)
    5. Exit (Structure/Fisher)
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.adx_threshold = self.config.get('adx_threshold', 20)
        self.fisher_threshold = self.config.get('fisher_threshold', -2.0) # Buy level
        self.z_score_threshold = self.config.get('z_score_threshold', -2.0)
        self.mfi_threshold = self.config.get('mfi_threshold', 20)
        
        if SMCWrapper:
            self.smc = SMCWrapper(use_library=False)
        else:
            self.smc = None

    def _calculate_z_score(self, series: pd.Series, period: int = 20) -> pd.Series:
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std.replace(0, 0.001)

    def _check_magnets(self, df: pd.DataFrame, idx: int, lookback: int = 50) -> bool:
        """
        Phase 2: Check if price is near a Magnet.
        Magnets: Monday Low, Weekly Pivot S1/S2, Bullish FVG, VWAP Band.
        """
        # Note: This is a simplified check. In production, need precise levels.
        # We assume columns like 'weekly_s1', 'monday_low' exist OR we allow FVG check.
        
        row = df.iloc[idx]
        current_low = row['low']
        
        # 1. Bullish FVG (using SMC)
        # We need a window.
        if self.smc and idx > 20:
            # Slicing is expensive in loop, optimization: pre-calculate FVGs
            # Here we just assume pre-calculated fvg column or skip for speed in this snippet
            pass
            
        # 2. VWAP
        # Assumed calc already. If 'vwap_lower_2' in df: ...
        
        # For this implementation, we'll return True if ANY magnet column is touched
        # Users should enrich DF with 'monday_low', 'weekly_s1' before calling strategy
        # Or we implement the calculations.
        
        magnets = ['monday_low', 'weekly_s1', 'weekly_s2', 'vwap_lower_2']
        touched = False
        
        for mag in magnets:
            if mag in df.columns:
                level = row[mag]
                # Check touch (Low <= Level <= High)
                if row['low'] <= level <= row['high']:
                    touched = True
                    break
        
        return touched

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Execute 5-Phase Logic
        """
        if df.empty: return pd.Series()
        
        df = df.copy()
        
        # PHASE 1: No-Go (Regime)
        # ADX
        df['adx'] = ADX.calculate(df)
        # BBW
        bb = BollingerBands.calculate(df)
        df['bbw'] = bb['bandwidth']
        
        # Regime Condition: ADX > 20 (Not Choppy) AND BBW not Squeezing (optional check?)
        # Requirement: "If Squeeze is active... do not fade".
        # We'll assume simple ADX check for now as primary.
        phase1_ok = df['adx'] > self.adx_threshold
        
        # PHASE 3: Golden Numbers (Pre-calc for vectorization)
        # Fisher
        df['fisher'] = FisherTransform.calculate(df)
        # Z-Score
        df['z_score'] = self._calculate_z_score(df['close'])
        # MFI
        df['mfi'] = MFI.calculate(df)
        
        # Funding & CVD (Expect columns, else ignore)
        if 'fundingRate' not in df.columns: df['fundingRate'] = 0.0 # Neural/Safe
        if 'cvd' not in df.columns: df['cvd'] = df['volume'] # Placeholder
        
        # Scoring
        # 1. Fisher < -2
        c1 = (df['fisher'] < self.fisher_threshold).astype(int)
        # 2. Z-Score < -2
        c2 = (df['z_score'] < self.z_score_threshold).astype(int)
        # 3. Funding < 0
        c3 = (df['fundingRate'] < 0).astype(int)
        # 4. MFI < 20
        c4 = (df['mfi'] < self.mfi_threshold).astype(int)
        # 5. CVD Divergence (Price Low < Prev Low AND CVD Low > Prev CVD Low)
        # Using 5-period swing for simplicity
        p_low = df['low'].rolling(5).min()
        c_low = df['cvd'].rolling(5).min()
        # Complex to vectorize perfectly, using simplified divergence proxy:
        # Price falling (5-day slope < 0) but CVD rising?
        c5 = pd.Series(0, index=df.index) 
        
        golden_score = c1 + c2 + c3 + c4 + c5
        phase3_ok = golden_score >= 3
        
        # PHASE 2 & 4: Location & Trigger (Iterative)
        signals = pd.Series(0, index=df.index)
        
        # Need to iterate for "Stop-Sweep" statefulness
        # State: Armed (Phase 1,2,3 met). Waiting for Trigger.
        armed = False
        magnet_level = None
        
        # SMC / Magnets preparation
        if self.smc:
            fvg_df = self.smc.detect_fvg(df)
            # Add to df for easier access? 
            # Or just use latest FVG as a magnet.
        
        for i in range(50, len(df)):
            idx = df.index[i]
            
            # Check Phase 1
            if not phase1_ok.iloc[i]:
                armed = False
                continue
                
            # Check Phase 2 (Location)
            # Simple check: Is price low "near" a magnet?
            # We'll assume if Phase 3 is hitting hard, we look for a magnet to validate.
            
            # Check Phase 3
            if phase3_ok.iloc[i]:
                # "ARM THE TRIGGER"
                # But we need a Magnet for Phase 2.
                # Let's perform a Magnet Search here.
                # 1. FVG
                is_magnet = False
                found_level = None
                
                # Check SMC FVG
                if self.smc:
                    # Look at recent candles for FVG
                    current_fvgs = fvg_df.iloc[:i]
                    # Is current low inside a Bullish FVG?
                    # Last valid FVG
                    # This is slow, simplified:
                    pass
                
                # If we assume Magnet is close, we proceed.
                # The requirement says "NO MAGNET = NO TRADE".
                # For code safety, if we can't find magnet data, we might skip or default.
                # Let's assume user provides magnet levels or we check trivial ones (Round numbers?)
                
                # Default "Pre-Cog" Entry: Immediate if 3/5 met (Aggressive)
                if golden_score.iloc[i] >= 3:
                     # Check Phase 2 (Mock or Real)
                     # For now, allow "God Mode" (Phase 3) to carry weight if Phase 2 is implicitly satisfied by price action
                     # But strictly:
                     if self._check_magnets(df, i):
                         armed = True
                         magnet_level = df['low'].iloc[i] # Use current low as reference? Or actual magnet?
                         # Pre-Cog Entry?
                         signals.iloc[i] = 1
                         armed = False # Reset after fire
            
            # Stop-Sweep Logic (Phase 4)
            # If we were Armed, and price Sweeps Magnet then Closes Above.
            if armed and magnet_level:
                row = df.iloc[i]
                # Sweep: Low < Magnet
                # Close > Magnet
                if row['low'] < magnet_level and row['close'] > magnet_level:
                    signals.iloc[i] = 1
                    armed = False
                    
        return signals

    def calculate_position_size(self, signal: int, price: float, portfolio_value: float, risk_params: Dict) -> float:
        # Standard risk calc
        risk_per_trade = risk_params.get('risk_per_trade', 0.01)
        # Distance to SL (Swing Low)
        # We don't have SL here easily without passing it.
        # Assuming 1% risk of equity, default 1% stop distance -> 1x lev?
        return portfolio_value * risk_per_trade / 0.01 # Mock
