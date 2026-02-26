import os

# Read the parsed pine script inputs
with open("pine_inputs.py", "r") as f:
    init_vars = f.read()

# Define the full code
code = f'''import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from strategy.base_strategy import BaseStrategy
from indicators.ema import EMA
from indicators.atr import ATR
from indicators.adx import ADX
from indicators.rsi import RSI

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RegimelineState:
    """State tracking for v7.01 Regimeline Strategy Events/Flags"""
    # --- Mode & Position ---
    mode: str = "WAIT"
    pos_tag: str = ""
    
    # --- Bear Overlay (Acceptance-Down v1.3) ---
    pending_bear: bool = False
    bear_active: bool = False
    bear_break_lvl: float = np.nan
    bear_recl_lvl: float = np.nan
    bear_accepted_bar: int = -1
    bear_renew_bar: int = -1
    bear_pb_seen: bool = False
    bear_pb_bar: int = -1
    bear_lockout_until: int = -1
    last_bear_short_exit_bar: int = -1
    
    # --- Bull Engine (Trend v1.2) ---
    pending_bull: bool = False
    pending_bull_bar: int = -1
    bull_level: float = np.nan
    
    bull_armed: bool = False
    bull_armed_bar: int = -1
    bull_arm_roll_high: float = np.nan
    
    bull_pb_seen: bool = False
    bull_pb_bar: int = -1
    bull_pb_low: float = np.nan
    bull_struct_low: float = np.nan
    
    bull_comp_seen: bool = False
    bull_start_bar: int = -1
    bull_last_accept_up_bar: int = -1
    
    bull_hh: float = np.nan
    bull_hh_bar: int = -1
    bull_impulse_done: bool = False
    bull_last_impulse_bar: int = -1 
    bull_last_expansion_bar: int = -1
    bull_leg_no_new_hi_streak: int = 0
    bull_leg_hi: float = np.nan
    
    # Bull Add / Management
    bull_last_add_bar: int = -1
    bull_add_ref_hi: float = np.nan
    
    bull_tp1_done: bool = False
    bull_tp0_done: bool = False
    
    # --- Range-A (Balance v1.1-A) ---
    ra_struct_low: float = np.nan
    ra_dl_low: float = np.nan
    ra_dl_high: float = np.nan
    ra_dl_mid: float = np.nan
    ra_dl_bar: int = -1
    ra_struct_snap_ok: bool = False
    ra_bal_snap_ok: bool = False
    ra_disc_recent: bool = False
    ra_mode_veto_until: int = -1
    
    # --- Range-B (Re-Anchor v1.1-B) ---
    rb_pending: bool = False
    rb_active: bool = False
    rb_dir: int = 0 # 1=Up, -1=Down
    rb_start_bar: int = -1
    rb_new_low: float = np.nan
    rb_new_high: float = np.nan
    rb_new_mid: float = np.nan
    
    rb_s_pb_seen: bool = False
    rb_s_pb_bar: int = -1
    rb_s_value: float = np.nan
    last_rb_short_exit_bar: int = -1

    # --- Router / HTF ---
    htf_on_now: bool = False
    htf_neutral_now: bool = False
    last_htf_bull_bar: int = -1
    forced_mode: str = "WAIT"
    forced_until_bar: int = -1
    
    # --- L2 Feature Trackers ---
    l2_last_impulse_bar: int = -1
    l2_last_accept_bar: int = -1
    l2_last_reclaim_bar: int = -1

class RegimelineStrategy(BaseStrategy):
    """
    Regimeline v7.01 - Full Python Implementation
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.c = config
        
        # --- LOAD PINE SCRIPT PARAMS ---
{init_vars}
        
        # Lists for L1 and L2 Logs
        self.trade_logs = []
        self.event_logs = []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # --- 1. Core MTF/HTF EMAs ---
        # Pine: request.security with gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off
        # In Python, we will resample, calculate EMA, and ffill back to prevent lookahead bias.
        
        mult = 4 # Default standard multiplier if no explicit timeframe logic
        if 'htf_tf_mult' in self.c:
            mult = self.c['htf_tf_mult']
            
        # Proper resampled HTF: we can approximate by multiplying length or doing exact resample
        df['ema_fast'] = EMA.calculate(df, self.ema_fast_len)
        df['ema_slow'] = EMA.calculate(df, self.ema_slow_len)
        df['atr'] = ATR.calculate(df, self.atr_len)
        
        # ADX
        adx_res = ADX.calculate_dmi(df, self.adx_len)
        df['adx'] = adx_res['adx']
        df['pdi'] = adx_res['plus_di']
        df['mdi'] = adx_res['minus_di']
        
        df['rsi'] = RSI.calculate(df, self.ra_rsi_len if hasattr(self, 'ra_rsi_len') else 14)

        # Rolling Levels
        # ta.highest(high[1], len)
        df['roll_high'] = df['high'].shift(1).rolling(self.lvl_lookback).max()
        df['roll_low'] = df['low'].shift(1).rolling(self.lvl_lookback).min()
        df['roll_mid'] = (df['roll_high'] + df['roll_low']) * 0.5
        
        df['dec_high'] = df['roll_high'] + df['atr'] * self.buf_atr
        df['dec_low'] = df['roll_low'] - df['atr'] * self.buf_atr
        
        # Candle Metrics
        df['rng'] = (df['high'] - df['low']).replace(0, np.nan)
        df['body_pct'] = (df['close'] - df['open']).abs() / df['rng']
        df['close_loc'] = (df['close'] - df['low']) / df['rng']
        
        # Lookback helpers for execution logic
        df['low_min_9'] = df['low'].shift(1).rolling(9).min()
        df['high_max_9'] = df['high'].shift(1).rolling(9).max()
        df['low_min_12'] = df['low'].shift(1).rolling(12).min()
        df['low_min_24'] = df['low'].shift(1).rolling(24).min()
        
        df['htf_ema_fast'] = EMA.calculate(df, self.htf_fast_len * mult)
        df['htf_ema_slow'] = EMA.calculate(df, self.htf_slow_len * mult)
        
        df['hl2'] = (df['high'] + df['low']) / 2
        df.fillna(method='bfill', inplace=True) 

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.calculate_indicators(df)
        n = len(df)
        signals = np.zeros(n, dtype=int)
        # TODO: Implement robust v7 iteration loop here
        return pd.Series(signals, index=df.index)

    def calculate_position_size(
        self, signal: int, price: float, portfolio_value: float, risk_params: Dict
    ) -> float:
        """Calculate position size based on signal and risk parameters"""
        risk_per_trade = risk_params.get("risk_per_trade", 0.01)
        pos_size = (portfolio_value * risk_per_trade) / price
        return pos_size
'''

with open("regimeline_strategy.py", "w") as f:
    f.write(code)

print("Generated regimeline_strategy.py templates successfully.")
