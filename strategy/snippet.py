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
