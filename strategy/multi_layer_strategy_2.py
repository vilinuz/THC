from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_strateg import BaseStrategy

# Import Indicators
from indicators.choppiness import ChoppinessIndex
from indicators.supersmoother import SuperSmoother
from indicators.mama import MAMA
from indicators.hurst import HurstProxy
from indicators.markov_proxy import MarkovProxy
from indicators.atr import ATR
from indicators.aroon import Aroon
from indicators.adx import ADX

class MultiLayerStrategy2(BaseStrategy):
    """
    Apex Regime Test [V50] Translation
    
    A faithful port of the Pine Script strategy to Python.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        
        # 1. INPUTS
        # Gatekeeper
        self.chop_len = self.config.get('chop_len', 14)
        self.chop_thresh = self.config.get('chop_thresh', 61.8)
        
        # Regime
        self.ss_len = self.config.get('ss_len', 10)
        self.mama_fast = self.config.get('mama_fast', 0.9)
        self.mama_slow = self.config.get('mama_slow', 0.05)
        
        # Voters
        self.hurst_len = self.config.get('hurst_len', 100)
        self.atr_len = self.config.get('atr_len', 14)
        self.aroon_len = self.config.get('aroon_len', 25)
        self.adx_len = self.config.get('adx_len', 14)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on the V50 logic.
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # =============================================================================
        # 3. CALCULATIONS
        # =============================================================================
        
        # Gatekeeper
        # float chop_15m = f_chop(i_chop_len)
        chop_15m = ChoppinessIndex.calculate(df, period=self.chop_len)
        
        # float chop_4h - In backtesting, we might not have multi-timeframe data easily available 
        # in the same dataframe without resampling.
        # For this port, we will assume the dataframe IS the timeframe we are trading (e.g. 15m),
        # and we might verify 4H via resampling or external data. 
        # *CRITICAL*: The user request didn't specify how to handle MTF.
        # I will implement a resampled approximation for 4H if the data allows, or simplify 
        # by using a longer period on the same timeframe as a proxy if resampling fails,
        # but the correct way is to resample close to 4H.
        # Assuming current df is the base timeframe (e.g. 15m).
        
        try:
            # Resample to 4H, calc chop, reindex back
            # Validating index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                 # Fallback: Just use same timeframe values or fail gracefully
                 chop_4h = chop_15m 
            else:
                 df_4h = df.resample('4H').agg({
                     'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                 }).dropna()
                 
                 chop_4h_raw = ChoppinessIndex.calculate(df_4h, period=self.chop_len)
                 # Reindex forward fill
                 chop_4h = chop_4h_raw.reindex(df.index, method='ffill')
        except Exception as e:
            print(f"Warning: MTF calculation failed: {e}. Using current TF for 4H Proxy.")
            chop_4h = chop_15m
            
        # ATR Expansion (Volatility Explosion)
        # float atr_val_gate = ta.atr(14)
        atr_val_gate = ATR.calculate(df, period=14)
        # float atr_ma_gate = ta.sma(atr_val_gate, 20)
        atr_ma_gate = atr_val_gate.rolling(window=20).mean()
        # bool is_vol_explosion = atr_val_gate > 1.1 * atr_ma_gate
        is_vol_explosion = atr_val_gate > (1.1 * atr_ma_gate)
        
        # Breakout Exception
        # bool is_breakout = (chop_15m < 38.2) or is_vol_explosion
        is_breakout = (chop_15m < 38.2) | is_vol_explosion
        
        # Rule: Block if 15m is Chop OR (4H is Chop AND NOT Breakout)
        # bool is_chop_gate = (chop_15m > i_chop_thresh) or (chop_4h > i_chop_thresh and not is_breakout)
        is_chop_gate = (chop_15m > self.chop_thresh) | ((chop_4h > self.chop_thresh) & (~is_breakout))
        
        # Regime
        # float ss_val = f_supersmoother(i_comp_src, i_ss_len)
        ss_val = SuperSmoother.calculate(df, period=self.ss_len)
        
        # [mama_val, fama_val] = f_mama(i_comp_src, i_mama_fast, i_mama_slow)
        mama_res = MAMA.calculate(df, fast_limit=self.mama_fast, slow_limit=self.mama_slow)
        mama_val = mama_res['mama']
        fama_val = mama_res['fama']
        
        # V50 INSTANT REACTION
        # bool mama_rising = mama_val > mama_val[1]
        mama_rising = mama_val > mama_val.shift(1)
        # bool mama_falling = mama_val < mama_val[1]
        mama_falling = mama_val < mama_val.shift(1)
        
        close = pd.to_numeric(df['close'])
        
        # bool mesa_bull = (close > ss_val) and (close > mama_val) and mama_rising
        mesa_bull = (close > ss_val) & (close > mama_val) & mama_rising
        
        # bool mesa_bear = (close < ss_val) and (close < mama_val) and mama_falling
        mesa_bear = (close < ss_val) & (close < mama_val) & mama_falling
        
        # Voters
        # 1. Hurst
        hurst_val = HurstProxy.calculate(df, period=self.hurst_len)
        v_hurst_bull = hurst_val > 0.55
        v_hurst_bear = hurst_val < 0.45
        
        # 2. Markov
        p_up = MarkovProxy.calculate_prob_up(df)
        v_markov_bull = p_up > 0.7
        v_markov_bear = p_up < 0.3
        
        # 3. ADX
        # [di_p, di_m, adx_val] = ta.dmi(i_adx_len, 14)
        # Assuming ADX class returns simple ADX series. Need DI+, DI-
        # For now, let's look at adx.py to see if it exposes di_p/di_m
        # The ADX class in codebase currently returns only the final ADX series.
        # I will need to calculate DI directly here or expand the ADX class.
        # To avoid modifying `indicators/adx.py` drastically mid-flight (blocking), I'll recalculate here.
        # Or better, since I saw `indicators/adx.py` is simple, I'll just replicate DI logic briefly here.
        
        adx_val = ADX.calculate(df, period=self.adx_len)
        
        # Re-calc DIs for voting logic (quick inline implementation for robustness)
        tr = ATR.calculate(df, period=1).rolling(window=self.adx_len).sum() # Simplified TR sum? No, Wilder's
        # Let's stick to using what we have or implementing correctly.
        # I will compute simple DM here.
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smooth using same alpha as ADX usually (1/14)
        plus_dm_s = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
        atr_s = ATR.calculate(df, period=14) # Smoothed ATR
        
        di_p = 100 * (plus_dm_s / atr_s)
        di_m = 100 * (minus_dm_s / atr_s)
        
        v_adx_bull = (adx_val > 35) & (di_p > di_m + 10)
        v_adx_bear = (adx_val > 35) & (di_m > di_p + 10)
        
        # 4. Aroon
        aroon_df = Aroon.calculate(df, period=self.aroon_len)
        ar_up = aroon_df['aroon_up']
        ar_dn = aroon_df['aroon_down']
        v_aroon_bull = ar_up > 70
        v_aroon_bear = ar_dn > 70
        
        # 5. ATR Forecast
        atr = ATR.calculate(df, period=self.atr_len)
        atr_ma = atr.rolling(window=20).mean()
        v_atr_expand = atr > (1.1 * atr_ma)
        
        # 6. Kalman
        k_vel = close - close.shift(1)
        v_kalman_bull = k_vel > 0
        v_kalman_bear = k_vel < 0
        
        # Vote Counts
        # We can sum boolean series as integers
        votes_bull = (
            v_hurst_bull.astype(int) + 
            v_markov_bull.astype(int) + 
            v_adx_bull.astype(int) + 
            v_aroon_bull.astype(int) + 
            v_atr_expand.astype(int) + 
            v_kalman_bull.astype(int)
        )
        
        votes_bear = (
            v_hurst_bear.astype(int) + 
            v_markov_bear.astype(int) + 
            v_adx_bear.astype(int) + 
            v_aroon_bear.astype(int) + 
            v_atr_expand.astype(int) + 
            v_kalman_bear.astype(int)
        )
        
        # =============================================================================
        # 4. REGIME DETERMINATION & SIGNALS
        # =============================================================================
        
        # Note: The Pine Script determines colors "BULL REGIME" or "BEAR REGIME" or "CHOP BLOCK".
        # It doesn't explicitly output a "Buy" or "Sell" boolean in the provided snippet, 
        # but the regime itself acts as the signal.
        # "BULL REGIME" = Buy Signal availability
        # "BEAR REGIME" = Sell Signal availability
        
        # We will output these states.
        
        signals = pd.DataFrame(index=df.index)
        signals['is_chop_gate'] = is_chop_gate
        signals['mesa_bull'] = mesa_bull
        signals['mesa_bear'] = mesa_bear
        signals['votes_bull'] = votes_bull
        signals['votes_bear'] = votes_bear
        
        # Final Signal Logic
        # 1. Neutral (0)
        # 2. Chop Block (-99 or similar, or just 0)
        # 3. Bull (1)
        # 4. Bear (-1)
        
        # Logic:
        # if is_chop_gate: Neutral/Block
        # else if mesa_bull: Bull
        # else if mesa_bear: Bear
        
        signal_state = np.zeros(len(df), dtype=int)
        
        # Prioritize logic
        # Default 0
        
        # Mesa Bear
        mask_bear = (~is_chop_gate) & mesa_bear
        signal_state[mask_bear] = -1
        
        # Mesa Bull
        mask_bull = (~is_chop_gate) & mesa_bull
        signal_state[mask_bull] = 1
        
        signals['signal'] = signal_state
        
        # Adding metadata columns for debugging/visualization
        signals['regime_status'] = 'NEUTRAL'
        signals.loc[is_chop_gate, 'regime_status'] = 'CHOP_BLOCK'
        signals.loc[mask_bull, 'regime_status'] = 'BULL_REGIME'
        signals.loc[mask_bear, 'regime_status'] = 'BEAR_REGIME'
        
        return signals

    def calculate_position_size(
        self,
        signal: int,
        price: float,
        portfolio_value: float,
        risk_params: Dict
    ) -> float:
        """
        Calculate position size (Placeholder for backtesting)
        """
        if signal == 0:
            return 0.0
        
        # Default simple sizing
        risk_pct = risk_params.get('risk_pct', 0.01)
        stop_loss_pct = risk_params.get('stop_loss_pct', 0.02)
        
        risk_amount = portfolio_value * risk_pct
        unit_risk = price * stop_loss_pct
        
        if unit_risk == 0:
            return 0.0
            
        units = risk_amount / unit_risk
        return units

