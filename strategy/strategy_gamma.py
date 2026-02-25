import numpy as np
import pandas as pd
from typing import Dict
import logging

from strategy.base_strategy import BaseStrategy
from indicators.adx import ADX
from indicators.choppiness import ChoppinessIndex
from indicators.tillson_t3 import TillsonT3
from indicators.fisher_transform import FisherTransform

logger = logging.getLogger(__name__)

class StrategyGamma(BaseStrategy):
    """
    Strategy Gamma: Indicator-Fused Adaptive Strategy
    Dynamically identifies regimes using ADX and Choppiness Index.
    Unleashes Tillson T3 moving average crossovers for Trend following.
    Unleashes Fisher Transform extremes for Choppy/Mean-Reverting regimes.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Regime Detection Parameters
        self.adx_period = self.config.get("adx_period", 14)
        self.adx_trend_threshold = self.config.get("adx_trend_threshold", 25)
        self.chop_period = self.config.get("chop_period", 14)
        self.chop_trend_threshold = self.config.get("chop_trend_threshold", 61.8) # Standard Fib
        
        # Trending Sub-Strategy Parameters (Tillson T3)
        self.t3_fast_len = self.config.get("t3_fast_len", 8)
        self.t3_slow_len = self.config.get("t3_slow_len", 21)
        self.t3_volume_factor = self.config.get("t3_volume_factor", 0.7)
        
        # Choppy Sub-Strategy Parameters (Fisher Transform)
        self.fisher_period = self.config.get("fisher_period", 9)
        self.fisher_overbought = self.config.get("fisher_overbought", 1.5)
        self.fisher_oversold = self.config.get("fisher_oversold", -1.5)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates all indicators dynamically based on hyperparams and evaluates logic tick-by-tick.
        """
        signals = pd.Series(0, index=df.index)

        # 1. Pre-calculate Required Indicators
        work_df = df.copy()
        
        # Regime Indicators
        # ADX.calculate currently returns just the ADX series in this codebase
        adx_series = ADX.calculate(work_df, period=self.adx_period)
        chop_series = ChoppinessIndex.calculate(work_df, period=self.chop_period)
        
        # Trending Indicators (Tillson T3)
        t3_fast_series = TillsonT3.calculate(work_df, length=self.t3_fast_len, volume_factor=self.t3_volume_factor)
        t3_slow_series = TillsonT3.calculate(work_df, length=self.t3_slow_len, volume_factor=self.t3_volume_factor)
        
        # Choppy Indicators (Fisher Transform)
        fisher_series = FisherTransform.calculate(work_df, period=self.fisher_period)

        # To prevent lookahead and NA issues, find the valid starting index
        # which is the max lookback period required
        max_lookback = max(self.adx_period * 2, self.chop_period, self.t3_slow_len, self.fisher_period)
        
        # Extract numpy arrays for fast iteration
        adx_vals = adx_series.values
        chop_vals = chop_series.values
        t3f_vals = t3_fast_series.values
        t3s_vals = t3_slow_series.values
        fish_vals = fisher_series.values
        
        for i in range(max_lookback, len(work_df)):
            
            # --- Dynamic Regime Detection ---
            adx_val = adx_vals[i]
            chop_val = chop_vals[i]
            
            # If indicators are NaNs, continue
            if np.isnan(adx_val) or np.isnan(chop_val):
                continue
                
            is_trending = (adx_val > self.adx_trend_threshold) and (chop_val < self.chop_trend_threshold)
            is_choppy = (adx_val <= self.adx_trend_threshold) and (chop_val >= self.chop_trend_threshold)
            
            # --- Sub-Strategy Allocation ---
            if is_trending:
                # Trending Regime: Tillson T3 Crossover logic
                t3_fast_curr, t3_fast_prev = t3f_vals[i], t3f_vals[i-1]
                t3_slow_curr, t3_slow_prev = t3s_vals[i], t3s_vals[i-1]
                
                # Check for NaNs
                if np.isnan(t3_fast_curr) or np.isnan(t3_slow_curr):
                    continue
                    
                # Bullish Crossover (Fast crosses above Slow)
                if t3_fast_prev <= t3_slow_prev and t3_fast_curr > t3_slow_curr:
                    signals.iloc[i] = 1
                # Bearish Crossover (Fast crosses below Slow)
                elif t3_fast_prev >= t3_slow_prev and t3_fast_curr < t3_slow_curr:
                    signals.iloc[i] = -1
                    
            elif is_choppy:
                # Choppy Regime: Fisher Transform Extremes Mean Reversion
                fisher_curr, fisher_prev = fish_vals[i], fish_vals[i-1]
                
                # Fisher calculates Gaussian probability hooks
                if np.isnan(fisher_curr):
                    continue
                    
                # Buy when Fisher crosses from extremely oversold (-1.5) back upwards
                if fisher_prev <= self.fisher_oversold and fisher_curr > self.fisher_oversold:
                    signals.iloc[i] = 1
                # Sell when Fisher crosses from extremely overbought (+1.5) back downwards
                elif fisher_prev >= self.fisher_overbought and fisher_curr < self.fisher_overbought:
                    signals.iloc[i] = -1
                    
            else:
                # Mixed Regime: Sit out or scale down. Here we enforce no new signal (holds previous position).
                pass

        return signals

    def calculate_position_size(
        self, signal: int, price: float, portfolio_value: float, risk_params: Dict
    ) -> float:
        """
        Simple position sizing.
        """
        risk_per_trade = risk_params.get("risk_per_trade", 0.02)
        return (portfolio_value * risk_per_trade) / price
