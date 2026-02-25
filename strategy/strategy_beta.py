import numpy as np
import pandas as pd
from typing import Dict
import logging

from strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class StrategyBeta(BaseStrategy):
    """
    Strategy Beta: HMM-Gated Adaptive Volatility Arbitrage
    A meta-strategy that dynamically alters its core execution logic and parameter sets 
    based on real-time probabilistic regime output (HMM).
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hmm_trend_threshold = self.config.get("hmm_trend_threshold", 0.70)
        self.hmm_chop_threshold = self.config.get("hmm_chop_threshold", 0.70)
        
        # Sub-strategy parameters
        # Breakout parameters (Trending)
        self.breakout_lookback = self.config.get("breakout_lookback", 20)
        self.ema_fast = self.config.get("ema_fast", 9)
        self.ema_slow = self.config.get("ema_slow", 21)
        
        # Mean-reversion parameters (Choppy)
        self.bb_lookback = self.config.get("bb_lookback", 20)
        self.bb_std = self.config.get("bb_std", 2.0)
        self.rsi_period = self.config.get("rsi_period", 14)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Sub-Strategy Allocation based on HMM state probabilities.
        """
        signals = pd.Series(0, index=df.index)
        
        # Working copy to add indicators dynamically
        work_df = df.copy()
        
        # Calculate dynamic indicators based on current params
        work_df['ema_fast'] = work_df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        work_df['ema_slow'] = work_df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Bollinger Bands
        rolling_mean = work_df['close'].rolling(window=self.bb_lookback).mean()
        rolling_std = work_df['close'].rolling(window=self.bb_lookback).std()
        work_df['bb_upper'] = rolling_mean + (rolling_std * self.bb_std)
        work_df['bb_lower'] = rolling_mean - (rolling_std * self.bb_std)
        
        # RSI
        delta = work_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        work_df['rsi'] = 100 - (100 / (1 + rs))

        # Iterate through dataframe
        for i in range(max(self.breakout_lookback, self.bb_lookback, self.ema_slow), len(work_df)):
            current_row = work_df.iloc[i]
            
            # Continuous Regime Inference
            state_1_prob = current_row.get("hmm_state_1_prob", 0.33)
            state_2_prob = current_row.get("hmm_state_2_prob", 0.33)
            state_3_prob = current_row.get("hmm_state_3_prob", 0.34)
            
            # Dynamic Sub-Strategy Allocation
            if state_1_prob > self.hmm_trend_threshold:
                signals.iloc[i] = self._trending_sub_strategy(work_df, i)
            elif state_2_prob > self.hmm_chop_threshold:
                signals.iloc[i] = self._choppy_sub_strategy(work_df, i)
            else:
                signals.iloc[i] = self._stress_sub_strategy(work_df, i)

        return signals

    def _trending_sub_strategy(self, df: pd.DataFrame, idx: int) -> int:
        """
        State 1: Trending. Donchian Channel breakouts and EMA crossovers.
        Maximizes position sizing and uses loose trailing stops.
        """
        # Calculate trailing indicators (in real implementation these are precalculated)
        close = df['close'].iloc[idx]
        
        # Simple EMA Crossover mock
        # We assume ema_fast and ema_slow are pre-calculated in df
        ema_fast_current = df.get("ema_fast", pd.Series(0, index=df.index)).iloc[idx]
        ema_slow_current = df.get("ema_slow", pd.Series(0, index=df.index)).iloc[idx]
        
        if ema_fast_current > ema_slow_current:
            return 1 # Buy signal
        elif ema_fast_current < ema_slow_current:
            return -1 # Sell signal
            
        return 0

    def _choppy_sub_strategy(self, df: pd.DataFrame, idx: int) -> int:
        """
        State 2: Choppy / Ranging. Mean-reversion trading Bollinger Bands and RSI extremes.
        Aggressively targets Liquidity Sweeps.
        """
        close = df['close'].iloc[idx]
        
        # Bollinger Bands and RSI (assume pre-calculated)
        bb_upper = df.get("bb_upper", pd.Series(float('inf'), index=df.index)).iloc[idx]
        bb_lower = df.get("bb_lower", pd.Series(0, index=df.index)).iloc[idx]
        rsi = df.get("rsi", pd.Series(50, index=df.index)).iloc[idx]
        
        # Buy when price crosses below lower BB and RSI is oversold (< 30)
        if close < bb_lower and rsi < 30:
            return 1
            
        # Sell when price crosses above upper BB and RSI is overbought (> 70)
        elif close > bb_upper and rsi > 70:
            return -1
            
        return 0

    def _stress_sub_strategy(self, df: pd.DataFrame, idx: int) -> int:
        """
        State 3: Stress / High Volatility.
        Drastically scales down position inversely proportional to Shannon Entropy / LSTM-GARCH.
        Might transition entirely to delta-neutral (pairs trading).
        """
        # In this stub we output 0 (hold/sit out) to isolate from directional risk.
        return 0

    def calculate_position_size(
        self, signal: int, price: float, portfolio_value: float, risk_params: Dict
    ) -> float:
        """
        Dynamic Position Sizing based on HMM state and Shannon Entropy/LSTM-GARCH.
        """
        base_risk = risk_params.get("risk_per_trade", 0.01)
        
        # Depending on the active state (which we'd track), we adjust size
        # Assume state comes from risk_params or we store it
        current_state = risk_params.get("current_hmm_state", 1) # 1=Trend, 2=Chop, 3=Stress
        
        if current_state == 1:
            # Maximized sizing
            alloc_multiplier = 1.5
        elif current_state == 2:
            alloc_multiplier = 1.0
        elif current_state == 3:
            # Scaled down inversely proportional to volatility metrics
            shannon_entropy = risk_params.get("shannon_entropy", 1.0)
            volatility = risk_params.get("garch_volatility", 1.0)
            alloc_multiplier = 1.0 / (shannon_entropy * volatility + 1e-6)
        else:
            alloc_multiplier = 0.5
            
        pos_size = (portfolio_value * base_risk * alloc_multiplier) / price
        return pos_size
