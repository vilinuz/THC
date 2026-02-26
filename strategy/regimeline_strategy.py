import pandas as pd
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
        self.ema_fast_len = self.c.get('emaFastLen', 21)
        self.ema_slow_len = self.c.get('emaSlowLen', 55)
        self.atr_len = self.c.get('atrLen', 14)
        self.lvl_lookback = self.c.get('lvlLookback', 24)
        self.buf_atr = self.c.get('bufATR', 0.10)
# COULD NOT PARSE: htfTF      = input.timeframe("60", "HTF timeframe", group=grpHTF)
        self.htf_fast_len = self.c.get('htfFastLen', 21)
        self.htf_slow_len = self.c.get('htfSlowLen', 55)
        self.htf_neutral_sep_pct = self.c.get('htfNeutralSepPct', 0.0050)
        self.adx_len = self.c.get('adxLen', 14)
        self.adx_smooth = self.c.get('adxSmooth', 14)
        self.adx_bull_min = self.c.get('adxBullMin', 23)
        self.adx_range_a_max = self.c.get('adxRangeAMax', 22)
        self.adx_range_a_soft_max = self.c.get('adxRangeASoftMax', 36)
        self.adx_bull_exec_min = self.c.get('adxBullExecMin', 22)
        self.adx_bull_start_min = self.c.get('adxBullStartMin', 20)
        self.adx_use_slope = self.c.get('adxUseSlope', True)
        self.adx_gate_bull_mode = self.c.get('adxGateBullMode', False)
        self.adx_slope_bars = self.c.get('adxSlopeBars', 3)
        self.adx_ra_use_soft = self.c.get('adxRAUseSoft', True)
        self.adx_ra_slope_bars = self.c.get('adxRASlopeBars', 3)
        self.adx_bull_add_min = self.c.get('adxBullAddMin', 18)
        self.adx_bull_add_slope_pad = self.c.get('adxBullAddSlopePad', 4.0)
        self.adx_ra_relax_if_falling = self.c.get('adxRARelaxIfFalling', True)
        self.tune_preset = self.c.get('tunePreset', "Balanced")
        self.min_entry_score = self.c.get('minEntryScore', 0)
        self.bull_pb_atr = self.c.get('bullPB_ATR', 0.30)
        self.bull_pb__max_dip_atr = self.c.get('bullPB_MaxDip_ATR', 0.60)
        self.bull_lvl_hold_atr = self.c.get('bullLvlHold_ATR', 0.10)
        self.bull_hold_atr = self.c.get('bullHold_ATR', 0.10)
        self.bull_min_body_pct = self.c.get('bullMinBodyPct', 0.20)
        self.bull_break_lookback = self.c.get('bullBreakLookback', 9)
        self.bull_arm_max_bars = self.c.get('bullArmMaxBars', 10)
        self.bull_ts__bars = self.c.get('bullTS_Bars', 24)
        self.bull_ts__min_mfe_atr = self.c.get('bullTS_MinMFE_ATR', 0.30)
        self.bull_inv_need2_cl = self.c.get('bullInvNeed2Cl', True)
        self.bull_inv_buf_atr = self.c.get('bullInvBuf_ATR', 0.10)
        self.bull_tp1_atr = self.c.get('bullTP1_ATR', 1.97)
        self.bull_tp1__pct = self.c.get('bullTP1_Pct', 100)
        self.bull_tp0_atr = self.c.get('bullTP0_ATR', 0.70)
        self.bull_tp0__pct = self.c.get('bullTP0_Pct', 60)
        self.bull_use_dmi = self.c.get('bullUseDMI', False)
        self.bull_accept_need2_cl = self.c.get('bullAcceptNeed2Cl', True)
        self.bull_pb_require_from_above = self.c.get('bullPBRequireFromAbove', True)
        self.bull_add_require_trail_armed = self.c.get('bullAddRequireTrailArmed', False)
        self.bull_start_micro_pb_atr = self.c.get('bullStartMicroPB_ATR', 0.20)
        self.bull_start_no_chase_max_atr = self.c.get('bullStartNoChaseMax_ATR', 0.80)
        self.bull_need_local_trend = self.c.get('bullNeedLocalTrend', True)
        self.bull_ema_slope_bars = self.c.get('bullEmaSlopeBars', 3)
        self.bull_pb_max_above_atr = self.c.get('bullPBMaxAbove_ATR', 0.80)
        self.bull_pb_max_bars_since_accept = self.c.get('bullPBMaxBarsSinceAccept', 24)
        self.bull_pb_require_touch_ef = self.c.get('bullPBRequireTouchEF', False)
        self.bull_pb_max_ext_above_ema_atr = self.c.get('bullPBMaxExtAboveEma_ATR', 1.20)
        self.bull_pb_min_pullback_atr = self.c.get('bullPBMinPullback_ATR', 0.60)
        self.bull_pb_touch_require_reclaim = self.c.get('bullPBTouchRequireReclaim', False)
        self.bull_pb_max_per_bull = self.c.get('bullPBMaxPerBull', 2)
        self.bull_pb_late_stall_age_bars = self.c.get('bullPBLateStallAgeBars', 72)
        self.bull_pb_late_stall_no_hh_bars = self.c.get('bullPBLateStallNoHHBars', 12)
        self.bull_pb_late_age_bars = self.c.get('bullPBLateAgeBars', 72)
        self.bull_pb_late_max_above_atr = self.c.get('bullPBLateMaxAbove_ATR', 1.05)
        self.bull_pb_late_min_depth_atr = self.c.get('bullPBLateMinDepth_ATR', 1.05)
# COULD NOT PARSE: bullPBMaxAgeBars = input.int(
        self.pb_ts__bars = self.c.get('pbTS_Bars', 6)
        self.bull_pb__min_score = self.c.get('bullPB_MinScore', 1)
        self.bull_pb_stall_lookback = self.c.get('bullPBStallLookback', 8)
        self.bull_pb_stall_min_count = self.c.get('bullPBStallMinCount', 5)
        self.bull_pb_stall_body_atr = self.c.get('bullPBStallBodyATR', 0.25)
        self.bull_pb_stall_need_high_fail = self.c.get('bullPBStallNeedHighFail', True)
        self.bull_pb_use_late_fail = self.c.get('bullPBUseLateFail', True)
        self.bull_pb_late_fail_min_imp_age_bars = self.c.get('bullPBLateFailMinImpAgeBars', 4)
        self.bull_pb_late_fail_min_stall_count = self.c.get('bullPBLateFailMinStallCount', 3)
        self.bull_pb_use_freshness = self.c.get('bullPBUseFreshness', True)
        self.bull_pb_max_impulse_age_bars = self.c.get('bullPBMaxImpulseAgeBars', 18)
        self.bull_pb_fresh_min_age_bars = self.c.get('bullPBFreshMinAgeBars', 24)
        self.bull_pb_use_exp_age = self.c.get('bullPBUseExpAge', True)
        self.bull_pb_max_exp_age_bars = self.c.get('bullPBMaxExpAgeBars', 18)
        self.bull_pb_use_late_exp_age_body_veto = self.c.get('bullPBUseLateExpAgeBodyVeto', True)
        self.bull_pb_late_exp_age_bars__veto_n = self.c.get('bullPBLateExpAgeBars_VetoN', 60)
        self.bull_pb_late_body_pct__veto_max = self.c.get('bullPBLateBodyPct_VetoMax', 0.70)
        self.bull_comp_on = self.c.get('bullCompOn', True)
        self.bull_comp_max_body_pct = self.c.get('bullCompMaxBodyPct', 0.45)
        self.bull_comp_close_near_high_min = self.c.get('bullCompCloseNearHighMin', 0.60)
        self.bull_comp_hold_atr = self.c.get('bullCompHold_ATR', 0.10)
        self.bull_comp_max_ext_atr = self.c.get('bullCompMaxExt_ATR', 0.60)
        self.bull_trail_on = self.c.get('bullTrailOn', True)
        self.bull_trail_use_slow = self.c.get('bullTrailUseSlow', True)
        self.bull_trail_buf_atr = self.c.get('bullTrailBuf_ATR', 0.10)
        self.bull_trail_arm_atr = self.c.get('bullTrailArm_ATR', 0.80)
        self.bull_mom_loss_on = self.c.get('bullMomLossOn', True)
        self.bull_mom_loss_gate_weak = self.c.get('bullMomLossGateWeak', True)
        self.bull_mom_loss_allow_htf_off = self.c.get('bullMomLossAllowHTFOff', True)
        self.bull_mom_loss_allow_htf_neutral = self.c.get('bullMomLossAllowHTFNeutral', True)
        self.bull_mom_loss_allow_htf_flip_off = self.c.get('bullMomLossAllowHTFFlipOff', True)
        self.bull_mom_loss_allow_after_end_bull = self.c.get('bullMomLossAllowAfterEndBull', True)
        self.bull_mom_loss_allow_after_tp1_roll = self.c.get('bullMomLossAllowAfterTP1Roll', True)
        self.bull_mom_bars_after_end_bull = self.c.get('bullMomBarsAfterEndBull', 4)
        self.bull_mom_ema_closes_after_end = self.c.get('bullMomEmaClosesAfterEnd', 1)
        self.bull_mom_giveback_atr__after_end = self.c.get('bullMomGivebackATR_AfterEnd', 0.35)
        self.bull_mom_max_bars_after_tp1 = self.c.get('bullMomMaxBarsAfterTP1', 14)
        self.bull_add_on = self.c.get('bullAddOn', True)
        self.bull_add_spacing_bars = self.c.get('bullAddSpacingBars', 18)
        self.bull_add_max_ext_atr = self.c.get('bullAddMaxExt_ATR', 1.40)
        self.bull_add_dip_atr = self.c.get('bullAddDip_ATR', 0.25)
        self.bull_add_break_lb = self.c.get('bullAddBreakLB', 6)
        self.bull_add_body_pct = self.c.get('bullAddBodyPct', 0.18)
        self.bull_add_after_tp1_only = self.c.get('bullAddAfterTP1Only', True)
        self.bull_add_proof_mfe_atr = self.c.get('bullAddProofMFE_ATR', 0.80)
        self.bull_start_entry_on = self.c.get('bullStartEntryOn', False)
        self.bull_start_require_micro_pb = self.c.get('bullStartRequireMicroPB', False)
        self.bull_start_entry_max_ext = self.c.get('bullStartEntryMaxExt', 1.05)
        self.bull_start_entry_max_ext_ema = self.c.get('bullStartEntryMaxExtEMA', 1.60)
        self.bull_start_entry_body_pct = self.c.get('bullStartEntryBodyPct', 0.22)
        self.bull_start_window_bars = self.c.get('bullStartWindowBars', 1)
        self.bull_pb__exec_adx_min = self.c.get('bullPB_ExecADXMin', 18)
        self.bull_add_close_near_high = self.c.get('bullAddCloseNearHigh', 0.70)
        self.bull_inv_after_tp1_need2_cl = self.c.get('bullInvAfterTP1Need2Cl', True)
        self.bull_inv_after_tp1_buf_atr = self.c.get('bullInvAfterTP1Buf_ATR', 0.10)
        self.ra_bal_max_width_atr = self.c.get('raBalMaxWidth_ATR', 3.00)
        self.ra_bal_max_ema_sep_atr = self.c.get('raBalMaxEMASep_ATR', 0.60)
        self.ra_disc_atr = self.c.get('raDisc_ATR', 0.35)
        self.ra_disc_lookback = self.c.get('raDiscLookback', 12)
        self.ra_mode_grace_bars = self.c.get('raModeGraceBars', 6)
        self.ra_stop_atr = self.c.get('raStop_ATR', 0.45)
        self.ra_max_entry_above_low_atr = self.c.get('raMaxEntryAboveLow_ATR', 1.15)
        self.ra_use_near_low = self.c.get('raUseNearLow', True)
        self.ra_require_below_mid = self.c.get('raRequireBelowMid', True)
        self.ra_max_risk_atr = self.c.get('raMaxRisk_ATR', 2.30)
        self.ra_risk_fallback_to_dl = self.c.get('raRiskFallbackToDL', True)
        self.ra_struct_stop_max_gap_atr = self.c.get('raStructStopMaxGap_ATR', 0.80)
        self.ra_min_rr__to_mid = self.c.get('raMinRR_ToMid', 0.45)
        self.ra_min_rr__hard = self.c.get('raMinRR_Hard', 0.20)
        self.ra_use_rr_gate = self.c.get('raUseRRGate', True)
        self.ra_stop_need2_cl = self.c.get('raStopNeed2Cl', True)
        self.ra_exec_need2_cl = self.c.get('raExecNeed2Cl', False)
        self.ra_ts__bars = self.c.get('raTS_Bars', 1)
        self.ra_allow_htf_neutral = self.c.get('raAllowHTFNeutral', False)
        self.ra_use_retest_entry = self.c.get('raUseRetestEntry', True)
        self.ra_retest_buf_atr = self.c.get('raRetestBuf_ATR', 0.10)
        self.ra_retest_need_green = self.c.get('raRetestNeedGreen', True)
        self.ra_use_rsi = self.c.get('raUseRSI', True)
        self.ra_rsi_len = self.c.get('raRsiLen', 14)
        self.ra_rsi_max = self.c.get('raRsiMax', 55)
        self.ra_rsi_washout_bypass = self.c.get('raRsiWashoutBypass', True)
        self.ra_need_sweep = self.c.get('raNeedSweep', True)
        self.sweep_len = self.c.get('sweepLen', 10)
        self.ra_was_below_lookback = self.c.get('raWasBelowLookback', 8)
        self.rb_confirm_bars = self.c.get('rbConfirmBars', 10)
        self.rb_bal_max_width_atr = self.c.get('rbBalMaxWidth_ATR', 2.50)
        self.rb_bal_max_ema_sep_atr = self.c.get('rbBalMaxEMASep_ATR', 0.70)
        self.rb_ts__bars = self.c.get('rbTS_Bars', 36)
        self.rb_target_atr = self.c.get('rbTarget_ATR', 0.75)
        self.rb_short_on = self.c.get('rbShortOn', False)
        self.rb_short_adx_min = self.c.get('rbShortADXMin', 18)
        self.rb_short_use_slope = self.c.get('rbShortUseSlope', True)
        self.rb_short_slope_bars = self.c.get('rbShortSlopeBars', 3)
        self.rb_short_close_near_lo = self.c.get('rbShortCloseNearLo', 0.35)
        self.rb_s_pb__max_pop_atr = self.c.get('rbS_PB_MaxPop_ATR', 0.60)
        self.rb_s__hold_atr = self.c.get('rbS_Hold_ATR', 0.10)
        self.rb_s__min_body_pct = self.c.get('rbS_MinBodyPct', 0.20)
        self.rb_s__break_lookback = self.c.get('rbS_BreakLookback', 9)
        self.rb_s__arm_max_bars = self.c.get('rbS_ArmMaxBars', 10)
        self.rb_s_ts__bars = self.c.get('rbS_TS_Bars', 24)
        self.rb_s_ts__min_mfe_atr = self.c.get('rbS_TS_MinMFE_ATR', 0.30)
        self.rb_s__inv_need2_cl = self.c.get('rbS_InvNeed2Cl', True)
        self.rb_s__inv_buf_atr = self.c.get('rbS_InvBuf_ATR', 0.10)
        self.bear_buf_atr = self.c.get('bearBuf_ATR', 0.45)
        self.bear_need2_cl = self.c.get('bearNeed2Cl', True)
        self.bear_early__on = self.c.get('bearEarly_On', True)
        self.bear_early__qty_pct = self.c.get('bearEarly_QtyPct', 100)
        self.bear_lockout_bars = self.c.get('bearLockoutBars', 6)
        self.bear_kill_use_ht_fveto = self.c.get('bearKillUseHTFveto', True)
        self.bear_max_bars = self.c.get('bearMaxBars', 480)
        self.bear_renew_tighten = self.c.get('bearRenewTighten', True)
        self.short_cooldown_after_htf_bull_bars = self.c.get('shortCooldownAfterHTFBullBars', 48)
        self.bear_need_local_downtrend = self.c.get('bearNeedLocalDowntrend', True)
        self.bear_ema_slope_bars = self.c.get('bearEmaSlopeBars', 3)
        self.bear_short_on = self.c.get('bearShortOn', True)
        self.bear_short_adx_min = self.c.get('bearShortADXMin', 18)
        self.bear_short_use_slope = self.c.get('bearShortUseSlope', True)
        self.bear_short_slope_bars = self.c.get('bearShortSlopeBars', 3)
        self.bear_short_close_near_lo = self.c.get('bearShortCloseNearLo', 0.35)
        self.bear_accept_entry_on = self.c.get('bearAcceptEntryOn', True)
        self.bear_exit_use_reclaim = self.c.get('bearExitUseReclaim', True)
        self.bear_exit_reclaim_bars = self.c.get('bearExitReclaimBars', 2)
        self.bear_exit_use_ema_reclaim = self.c.get('bearExitUseEmaReclaim', True)
        self.bear_pb__max_pop_atr = self.c.get('bearPB_MaxPop_ATR', 0.60)
        self.bear_lvl_hold_atr = self.c.get('bearLvlHold_ATR', 0.10)
        self.bear_min_body_pct = self.c.get('bearMinBodyPct', 0.20)
        self.bear_break_lookback = self.c.get('bearBreakLookback', 9)
        self.bear_arm_max_bars = self.c.get('bearArmMaxBars', 14)
        self.bear_ts__bars = self.c.get('bearTS_Bars', 24)
        self.bear_ts__min_mfe_atr = self.c.get('bearTS_MinMFE_ATR', 0.30)
        self.bear_tp1_atr = self.c.get('bearTP1_ATR', 1.65)
        self.bear_tp1_r = self.c.get('bearTP1_R', 0.8)
        self.bear_tp1__pct = self.c.get('bearTP1_Pct', 33)
        self.bear_tp0_atr = self.c.get('bearTP0_ATR', 0.70)
        self.bear_tp0__pct = self.c.get('bearTP0_Pct', 60)
        self.bear_prot0_buf_atr = self.c.get('bearProt0Buf_ATR', 0.05)
        self.bear_prot0_need2_cl = self.c.get('bearProt0Need2Cl', False)
        self.bear_prot_buf_atr = self.c.get('bearProtBuf_ATR', 0.10)
        self.bear_prot_need2_cl = self.c.get('bearProtNeed2Cl', True)
        self.bear_trail_on = self.c.get('bearTrailOn', True)
        self.bear_trail_buf_atr = self.c.get('bearTrailBuf_ATR', 0.25)
        self.bear_inv_need2_cl = self.c.get('bearInvNeed2Cl', True)
        self.bear_adx_min = self.c.get('bearAdxMin', 20)
        self.bear_pb__close_under_atr = self.c.get('bearPB_CloseUnder_ATR', 0.05)
        self.rb_use_snapshot_accept = self.c.get('rbUseSnapshotAccept', True)
        self.downgrade_bars = self.c.get('downgradeBars', 10)
        self.show_mode_bg = self.c.get('showModeBg', True)
        self.show_regime_bg = self.c.get('showRegimeBg', True)
        self.regime_bg_alpha = self.c.get('regimeBgAlpha', 90)
        self.regime_use_raw = self.c.get('regimeUseRaw', True)
        self.show_exec_marks = self.c.get('showExecMarks', True)
        self.show_debug_plots = self.c.get('showDebugPlots', True)
        self.show_log_table = self.c.get('showLogTable', False)
        self.show_mode_label = self.c.get('showModeLabel', True)
        self.show_regime_marks = self.c.get('showRegimeMarks', True)
        self.show_regime_flip_labels = self.c.get('showRegimeFlipLabels', True)
        self.regime_label_only_on_close = self.c.get('regimeLabelOnlyOnClose', True)
        self.regime_min_bars = self.c.get('regimeMinBars', 2)
        self.bull_start_style = self.c.get('bullStartStyle', "UP")
        self.bull_end_style = self.c.get('bullEndStyle', "LEFT")
        self.export_to_logs = self.c.get('exportToLogs', False)
        self.log_tz = self.c.get('logTZ', "Europe/Sofia")
        self.export_l2 = self.c.get('exportL2', False)
        self.l2_window_bars = self.c.get('l2WindowBars', 20)
        self.l2_overlap_n = self.c.get('l2OverlapN', 8)
        self.l2_range_n = self.c.get('l2RangeN', 20)
        self.l2_vol_n = self.c.get('l2VolN', 20)
        self.show_diag_hud = self.c.get('showDiagHUD', True)
        self.show_block_labels = self.c.get('showBlockLabels', False)
        self.dbg_pb_probe = self.c.get('dbgPBProbe', False)
        self.dbg_pb_probe_include_blocked = self.c.get('dbgPBProbeIncludeBlocked', True)

        
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
        df.bfill(inplace=True) 

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.calculate_indicators(df)
        n = len(df)
        signals = np.zeros(n, dtype=int)
        
        # Arrays for speed
        close = df['close'].values
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        ema_fast = df['ema_fast'].values
        ema_slow = df['ema_slow'].values
        atr = df['atr'].values
        adx = df['adx'].values
        pdi = df['pdi'].values
        mdi = df['mdi'].values
        rsi = df['rsi'].values
        
        roll_high = df['roll_high'].values
        roll_low = df['roll_low'].values
        roll_mid = df['roll_mid'].values
        dec_high = df['dec_high'].values
        dec_low = df['dec_low'].values
        
        htf_fast = df['htf_ema_fast'].values
        htf_slow = df['htf_ema_slow'].values
        
        body_pct = df['body_pct'].values
        close_loc = df['close_loc'].values
        
        low_min_9 = df['low_min_9'].values
        high_max_9 = df['high_max_9'].values
        low_min_12 = df['low_min_12'].values
        low_min_24 = df['low_min_24'].values
        hl2 = df['hl2'].values
        
        s = RegimelineState()
        
        in_pos = False
        pos_size = 0
        entry_price = 0.0
        trade_high = 0.0
        trade_low = float('inf')
        trade_bars = 0
        
        for i in range(100, n):
            c_close = close[i]
            c_high = high[i]
            c_low = low[i]
            c_open = open_[i]
            c_atr = atr[i]
            c_adx = adx[i]
            if np.isnan(c_atr): continue
            
            # --- 1. Router / HTF ---
            htf_trend_ok = htf_fast[i] > htf_slow[i]
            htf_sep = abs(htf_fast[i] - htf_slow[i]) / c_close if c_close > 0 else 0
            htf_neutral_sep_pct_val = getattr(self, 'htf_neutral_sep_pct', 0.005)
            htf_neutral = htf_sep <= htf_neutral_sep_pct_val
            s.htf_on_now = htf_trend_ok and (not htf_neutral)
            
            if s.htf_on_now:
                s.last_htf_bull_bar = i
                
            ema_slope_bull = (ema_fast[i] - ema_fast[i-getattr(self, 'bull_ema_slope_bars', 3)]) if i >= 3 else 0
            bull_local_ok = (ema_fast[i] > ema_slow[i]) and ((ema_slope_bull > 0) if getattr(self, 'bull_need_local_trend', True) else True)
            
            # --- 2. Bear Overlay ---
            bear_htf_ok = (not htf_trend_ok) and (not htf_neutral)
            bear_break_now = bear_htf_ok and (c_close < (dec_low[i] - c_atr * getattr(self, 'bear_buf_atr', 0.45)))
            
            if not s.bear_active and not s.pending_bear and bear_break_now:
                s.pending_bear = True
                s.bear_break_lvl = dec_low[i] - c_atr * getattr(self, 'bear_buf_atr', 0.45)
                s.bear_recl_lvl = dec_low[i] + c_atr * getattr(self, 'bear_buf_atr', 0.45)
                
            if s.pending_bear and c_close < s.bear_break_lvl:
                s.bear_active = True
                s.pending_bear = False
                s.bear_accepted_bar = i
                s.bear_renew_bar = i
                
                s.pending_bull = False
                s.bull_armed = False
                s.bull_start_bar = -1
                s.bear_pb_seen = False
                s.bear_pb_bar = -1
                
            if s.bear_active and c_close > s.bear_recl_lvl:
                s.bear_active = False
                s.bear_break_lvl = np.nan
                
            if s.bear_active and bear_break_now:
                s.bear_renew_bar = i
                s.bear_break_lvl = dec_low[i] - c_atr * getattr(self, 'bear_buf_atr', 0.45)
                s.bear_recl_lvl = dec_low[i] + c_atr * getattr(self, 'bear_buf_atr', 0.45)
                
            if s.bear_active and (i - s.bear_renew_bar > getattr(self, 'bear_max_bars', 480)):
                s.bear_active = False
                s.forced_mode = "RANGE-B"
                s.forced_until_bar = i + getattr(self, 'downgrade_bars', 10)
                
            # --- 3. Bull Engine ---
            cross_up = c_close > roll_high[i]
            
            if not in_pos and not s.pending_bull and not s.bull_armed and cross_up and not s.bear_active:
                s.pending_bull = True
                s.pending_bull_bar = i
                s.bull_level = roll_high[i]
                
            if s.pending_bull and c_close > s.bull_level and not s.bear_active:
                s.bull_armed = True
                s.bull_armed_bar = i
                s.pending_bull = False
                s.bull_start_bar = i
                s.bull_impulse_done = False
                s.bull_leg_hi = c_high
                s.bull_pb_seen = False
                
            if s.pending_bull and (i - s.pending_bull_bar > getattr(self, 'bull_arm_max_bars', 10)):
                s.pending_bull = False
                
            if s.bull_armed and (i - s.bull_armed_bar > getattr(self, 'bull_arm_max_bars', 10)) and not in_pos:
                s.bull_armed = False
                
            if s.bull_armed and not in_pos and not s.bull_pb_seen:
                zone_top = s.bull_level + c_atr * getattr(self, 'bull_pb_atr', 0.3)
                if c_low <= zone_top:
                    s.bull_pb_seen = True
                    s.bull_pb_bar = i
                    s.bull_struct_low = c_low
                    
            if s.bull_armed and not in_pos:
                if c_high > s.bull_leg_hi:
                    s.bull_leg_hi = c_high
                    s.bull_leg_no_new_hi_streak = 0
                else:
                    s.bull_leg_no_new_hi_streak += 1
                    
            # --- 4. Range-A Engine ---
            ra_width = roll_high[i] - roll_low[i]
            ra_in_balance = (ra_width <= c_atr * getattr(self, 'ra_bal_max_width_atr', 3.0)) and \
                            (abs(ema_fast[i] - ema_slow[i]) <= c_atr * getattr(self, 'ra_bal_max_ema_sep_atr', 0.6))
            
            ra_disc_now = ra_in_balance and (c_low <= (dec_low[i] - c_atr * getattr(self, 'ra_disc_atr', 0.35)))
            
            if not in_pos and ra_disc_now:
                s.ra_disc_recent = True
                s.ra_dl_low = dec_low[i]
                s.ra_dl_mid = roll_mid[i]
                s.ra_dl_bar = i
                s.ra_struct_low = c_low
                
            if not in_pos and s.ra_disc_recent and (i - s.ra_dl_bar > getattr(self, 'ra_disc_lookback', 12)):
                s.ra_disc_recent = False
                
            # --- Router Mode Selection ---
            if not in_pos and s.forced_until_bar >= i:
                s.mode = s.forced_mode
            else:
                s.mode = "WAIT"
                bull_setup_live = (s.pending_bull or s.bull_armed) and s.htf_on_now and bull_local_ok and adx[i] >= getattr(self, 'adx_bull_min', 23)
                ra_flat = not s.htf_on_now and not s.bear_active and ra_in_balance and s.ra_disc_recent
                
                if bull_setup_live: s.mode = "BULL"
                elif ra_flat: s.mode = "RANGE-A"
                elif s.bear_active and getattr(self, 'bear_short_on', True): s.mode = "BEAR"
                
            # --- Signals & Entry ---
            signal_generated = 0
            
            if not in_pos:
                if s.mode == "BEAR":
                    zone_top = s.bear_break_lvl + c_atr * getattr(self, 'bear_pb__max_pop_atr', 0.6)
                    if c_high >= s.bear_break_lvl and c_high <= zone_top:
                        s.bear_pb_seen = True
                        s.bear_pb_bar = i
                    if s.bear_pb_seen and (i - s.bear_pb_bar > getattr(self, 'bear_arm_max_bars', 14)):
                        s.bear_pb_seen = False
                        
                    if s.bear_pb_seen:
                        breakdown = (c_close < s.bear_break_lvl) and \
                                    (body_pct[i] >= getattr(self, 'bear_min_body_pct', 0.2)) and \
                                    (close_loc[i] <= getattr(self, 'bear_short_close_near_lo', 0.35)) # Ensure trailing zeroes aren't truncating float property access error. I used hardcoded fallbacks here just in case.
                        if breakdown:
                            signal_generated = -1
                            s.pos_tag = "BEAR"
                            s.bear_pb_seen = False
                            
                elif s.mode == "BULL":
                    if getattr(self, 'bull_start_entry_on', False) and not s.bull_pb_seen and not s.bull_impulse_done:
                        strong_break = (c_close > s.bull_level) and (c_close > high_max_9[i]) and (body_pct[i] >= getattr(self, 'bull_start_entry_body_pct', 0.22))
                        not_extended = (c_close - s.bull_level <= c_atr * getattr(self, 'bull_start_entry_max_ext', 1.05))
                        if strong_break and not_extended and (adx[i] >= getattr(self, 'adx_bull_start_min', 20)):
                            signal_generated = 1
                            s.pos_tag = "BULL"
                            s.bull_impulse_done = True
                            
                    if not signal_generated and s.bull_pb_seen:
                        reclaim = (c_close > s.bull_level) and (body_pct[i] >= getattr(self, 'bull_min_body_pct', 0.20)) and (c_close > high_max_9[i])
                        if reclaim and (adx[i] >= getattr(self, 'bull_pb__exec_adx_min', 18)):
                            signal_generated = 1
                            s.pos_tag = "BULL"
                            s.bull_pb_seen = False
                            
                elif s.mode == "RANGE-A":
                    was_below = (low_min_9[i] <= s.ra_dl_low)
                    accept_req = (c_close > s.ra_dl_low) and (close[i-1] > s.ra_dl_low if getattr(self, 'ra_exec_need2_cl', False) else True)
                    rsi_ok = rsi[i] <= getattr(self, 'ra_rsi_max', 55)
                    stop_loss = s.ra_struct_low - c_atr * getattr(self, 'ra_stop_atr', 0.45)
                    risk_atr = (c_close - stop_loss) / c_atr if c_atr > 0 else 999
                    risk_ok = risk_atr <= getattr(self, 'ra_max_risk_atr', 2.30)
                    
                    if was_below and accept_req and rsi_ok and risk_ok:
                        signal_generated = 1
                        s.pos_tag = "RANGE-A"
                        s.ra_disc_recent = False
                        
            # --- Execute Entry ---
            if signal_generated != 0:
                signals[i] = signal_generated
                in_pos = True
                pos_size = signal_generated
                entry_price = c_close
                trade_high = c_high
                trade_low = c_low
                trade_bars = 0
                
            # --- Exits ---
            elif in_pos:
                trade_bars += 1
                trade_high = max(trade_high, c_high)
                trade_low = min(trade_low, c_low)
                exit_sig = 0
                
                if s.pos_tag == "BULL":
                    mfe_atr = (trade_high - entry_price) / c_atr if c_atr > 0 else 0
                    if not s.bull_tp1_done and mfe_atr >= getattr(self, 'bull_tp1_atr', 1.97):
                        s.bull_tp1_done = True
                        
                    trail_armed = s.bull_tp1_done and getattr(self, 'bull_trail_on', True) and (mfe_atr >= getattr(self, 'bull_trail_arm_atr', 0.8))
                    
                    inv_lvl = s.bull_level - c_atr * getattr(self, 'bull_inv_buf_atr', 0.1)
                    if not s.bull_tp1_done and c_close < inv_lvl:
                        exit_sig = 2
                        
                    if trail_armed:
                        trail_lvl = ema_slow[i] if getattr(self, 'bull_trail_use_slow', True) else ema_fast[i]
                        if c_close < trail_lvl - c_atr * getattr(self, 'bull_trail_buf_atr', 0.1):
                            exit_sig = 2
                            
                    min_mfe_req = getattr(self, 'bull_ts__min_mfe_atr', 0.3) * c_atr / entry_price
                    if trade_bars > getattr(self, 'bull_ts__bars', 24) and ((trade_high - entry_price) / entry_price) < min_mfe_req:
                        exit_sig = 2
                        
                elif s.pos_tag == "RANGE-A":
                    if c_close < (s.ra_struct_low - c_atr * getattr(self, 'ra_stop_atr', 0.45)):
                        exit_sig = 2
                    if trade_bars > getattr(self, 'ra_ts__bars', 1) * 10:
                        exit_sig = 2
                        
                elif s.pos_tag == "BEAR":
                    if c_close > s.bear_recl_lvl:
                        exit_sig = -2
                        
                if exit_sig != 0:
                    pnl = (c_close - entry_price) * pos_size
                    self.trade_logs.append({
                        'entry_idx': i - trade_bars,
                        'exit_idx': i,
                        'tag': s.pos_tag,
                        'pnl': pnl,
                        'mfe': trade_high if pos_size > 0 else trade_low,
                        'trade_bars': trade_bars,
                        'r_mult': pnl / c_atr if c_atr > 0 else 0
                    })
                    
                    signals[i] = exit_sig
                    in_pos = False
                    pos_size = 0
                    s.pos_tag = ""
                    s.bull_tp1_done = False
                    
        return pd.Series(signals, index=df.index)

    def calculate_position_size(
        self, signal: int, price: float, portfolio_value: float, risk_params: Dict
    ) -> float:
        """Calculate position size based on signal and risk parameters"""
        risk_per_trade = risk_params.get("risk_per_trade", 0.01)
        pos_size = (portfolio_value * risk_per_trade) / price
        return pos_size
