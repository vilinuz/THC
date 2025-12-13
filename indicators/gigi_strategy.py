import pandas as pd
import numpy as np
from typing import Dict, Optional

class GigiStrategy:
    """
    Port of 'Gigi's Context RSI Buys + Bounce Filter + Auto-Pivot Scan' from Pine Script.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Defaults matching Pine Script
        self.rsi_len = self.config.get('rsi_len', 14)
        self.rsi_buy_level = self.config.get('rsi_buy_level', 30)
        self.rsi_sell_level = self.config.get('rsi_sell_level', 60)
        
        self.wick_ratio_min = self.config.get('wick_ratio_min', 0.3)
        self.sell_wick_ratio_min = self.config.get('sell_wick_ratio_min', 0.1)
        
        self.vol_len = self.config.get('vol_len', 20)
        self.use_vol_filter = self.config.get('use_vol_filter', False)
        
        self.atr_len = self.config.get('atr_len', 20)
        
        self.swing_buy_len = self.config.get('swing_buy_len', 3)
        self.swing_sell_len = self.config.get('swing_sell_len', 5)
        
        self.use_bounce_filter = self.config.get('use_bounce_filter', True)
        self.bounce_lookahead = self.config.get('bounce_lookahead', 20)
        self.bounce_atr_mult = self.config.get('bounce_atr_mult', 4.0)
        self.min_bars_between_buys = self.config.get('min_bars_between_buys', 40)
        
        self.enable_pullback_buys = self.config.get('enable_pullback_buys', True)
        self.rsi_pullback_level = self.config.get('rsi_pullback_level', 70.0)
        self.trend_ema_len = self.config.get('trend_ema_len', 50)
        self.require_trend_up = self.config.get('require_trend_up', True)
        
        self.use_pivot_sells = self.config.get('use_pivot_sells', True)
        self.sell_scan_bars = self.config.get('sell_scan_bars', 16)
        self.sell_require_profit = self.config.get('sell_require_profit', True)
        self.sell_rsi_soft_level = self.config.get('sell_rsi_soft_level', 55.0)
        
        self.buy_near_low_atr_mult = self.config.get('buy_near_low_atr_mult', 0.15)
        self.use_wick_atr_floor = self.config.get('use_wick_atr_floor', True)
        self.min_lower_wick_atr_mult = self.config.get('min_lower_wick_atr_mult', 0.05)
        
        self.use_sell_wick_atr_floor = self.config.get('use_sell_wick_atr_floor', True)
        self.min_upper_wick_atr_mult = self.config.get('min_upper_wick_atr_mult', 0.05)
        self.sell_near_high_atr_mult = self.config.get('sell_near_high_atr_mult', 0.15)
        
        self.near_peak_atr_mult = self.config.get('near_peak_atr_mult', 0.25)
        self.min_bars_in_trade = self.config.get('min_bars_in_trade', 10)
        self.peak_hold_bars = self.config.get('peak_hold_bars', 4)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy signals and return a DataFrame with indicators and signals.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Working copy
        data = df.copy()
        
        # ----------------------
        # Core Measures
        # ----------------------
        # RSI
        # Custom implementation to avoid dependency issues if pandas-ta not present
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_len).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(window=self.atr_len).mean()
        
        # Volume MA
        data['vol_ma'] = data['volume'].rolling(window=self.vol_len).mean()
        data['vol_ok'] = (not self.use_vol_filter) | (data['volume'] > data['vol_ma'])
        
        # Body and Wicks
        data['body_top'] = np.where(data['close'] > data['open'], data['close'], data['open'])
        data['body_bot'] = np.where(data['close'] > data['open'], data['open'], data['close'])
        data['body'] = data['body_top'] - data['body_bot']
        data['upper_w'] = data['high'] - data['body_top']
        data['lower_w'] = data['body_bot'] - data['low']
        
        # Swing High/Low
        data['is_swing_high'] = data['high'] == data['high'].rolling(window=self.swing_sell_len, center=False).max()
        data['is_swing_low'] = data['low'] == data['low'].rolling(window=self.swing_buy_len, center=False).min()
        
        # Context RSI (Lowest in lookback)
        # Pine: ta.lowest(rsi, swingBuyLen * 2)
        ctx_lookback = self.swing_buy_len * 2
        data['ctx_min_rsi'] = data['rsi'].rolling(window=ctx_lookback).min()
        data['context_oversold'] = data['ctx_min_rsi'] < self.rsi_buy_level
        
        # Trend EMA
        data['ema_trend'] = data['close'].ewm(span=self.trend_ema_len, adjust=False).mean()
        data['trend_up'] = (data['close'] > data['ema_trend']) & (data['ema_trend'] > data['ema_trend'].shift(1))
        
        # ----------------------
        # Gates and Wicks
        # ----------------------
        # Buy Low Gate
        swing_low_val = data['low'].rolling(window=self.swing_buy_len).min()
        is_near_swing_low = data['low'] <= swing_low_val + (data['atr'] * self.buy_near_low_atr_mult)
        data['low_gate'] = data['is_swing_low'] | is_near_swing_low
        
        # Buy Wick Checks
        buy_wick_ratio_ok = np.where(data['body'] > 0, 
                                     data['lower_w'] > (data['body'] * self.wick_ratio_min),
                                     data['lower_w'] > 0)
        buy_wick_atr_ok = (not self.use_wick_atr_floor) | (data['lower_w'] >= (data['atr'] * self.min_lower_wick_atr_mult))
        data['buy_wick_ok'] = buy_wick_ratio_ok & buy_wick_atr_ok
        
        # ----------------------
        # Buy Logic
        # ----------------------
        # Strict Buy
        buy_core_strict = (
            data['vol_ok'] & 
            data['low_gate'] & 
            data['context_oversold'] & 
            (data['rsi'] <= self.rsi_buy_level) & 
            data['buy_wick_ok']
        )
        
        # Pullback Buy
        pullback_allowed = (
            self.enable_pullback_buys & 
            data['vol_ok'] & 
            data['low_gate'] & 
            (data['rsi'] <= self.rsi_pullback_level) & 
            data['buy_wick_ok']
        )
        pullback_trend_ok = (not self.require_trend_up) | data['trend_up']
        buy_core_pullback = pullback_allowed & pullback_trend_ok
        
        # Combined Core Buy
        data['buy_core'] = buy_core_strict | buy_core_pullback
        
        # ----------------------
        # Scan & Bounce (Vectorized Approximation)
        # ----------------------
        # Note: Pine Script's `for` loops for scanning pivots are hard to vectorize perfectly efficiently.
        # We will iterate for signal generation or use rolling windows where possible.
        # For efficiency in ML pipelines, we iterate only where necessary.
        
        results = []
        in_long = False
        entry_price = float('nan')
        peak_since_entry = float('nan')
        bars_in_trade = 0
        peak_age = 0
        last_buy_idx = -999
        
        signals = np.zeros(len(data))
        
        # To avoid slow iteration over 100k rows, we can pre-calculate some rollings
        # But the state machine (in_long, bounce confirmation) is sequential.
        # We will iterate but skip calculation if not needed.
        
        # Convert to numpy for speed
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        opens = data['open'].values
        volumes = data['volume'].values
        atrs = data['atr'].values
        rsis = data['rsi'].values
        buy_cores = data['buy_core'].values
        vol_mas = data['vol_ma'].values
        
        # Pre-calc Sell Gates (High Gate)
        # High Gate
        swing_high_val = data['high'].rolling(window=self.swing_sell_len).max()
        is_near_swing_high = data['high'] >= swing_high_val - (data['atr'] * self.sell_near_high_atr_mult)
        data['high_gate'] = data['is_swing_high'] | is_near_swing_high
        high_gates = data['high_gate'].values
        
        upper_ws = data['upper_w'].values
        bodies = data['body'].values
        
        for i in range(len(data)):
            if i < max(self.bounce_lookahead, self.sell_scan_bars):
                continue
                
            # --- BUY LOGIC ---
            # Auto-Scan for Buy Pivot in lookback [i-lookahead : i]
            # We look for the 'best' low (lowest low) that satisfied buy_core
            best_buy_idx = -1
            min_low = float('inf')
            
            # Look back L bars
            start_scan = i - self.bounce_lookahead
            # Loop is small (e.g., 20), acceptable inside main loop
            for j in range(start_scan, i):
                if buy_cores[j]:
                    if lows[j] <= min_low:
                        min_low = lows[j]
                        best_buy_idx = j
            
            buy_signal_now = False
            if best_buy_idx != -1:
                # We found a potential pivot at best_buy_idx
                # Check Bounce: Highest high since pivot must exceed threshold
                pivot_low = lows[best_buy_idx]
                pivot_atr = atrs[best_buy_idx]
                threshold = pivot_low + (pivot_atr * self.bounce_atr_mult)
                
                # Max high from pivot+1 to current i
                bounce_high = np.max(highs[best_buy_idx+1 : i+1])
                
                bounce_ok = (not self.use_bounce_filter) or (bounce_high >= threshold)
                
                if bounce_ok:
                    # Check de-dup
                    if (best_buy_idx - last_buy_idx) >= self.min_bars_between_buys:
                        buy_signal_now = True
                        last_buy_idx = best_buy_idx # Mark the pivot as the last buy
                        
                        # Enter Trade Logic
                        in_long = True
                        entry_price = closes[best_buy_idx]
                        peak_since_entry = highs[i] # Current high
                        bars_in_trade = 0
                        peak_age = 0
                        signals[best_buy_idx] = 1 # Mark the PIVOT bar as the buy signal bar (like the label in Pine)

            # --- TRADE STATE UPDATE ---
            if in_long:
                if highs[i] > peak_since_entry:
                    peak_since_entry = highs[i]
                    peak_age = 0
                else:
                    peak_age += 1
                bars_in_trade += 1

                # --- SELL LOGIC ---
                # Scan scan window sL
                if self.use_pivot_sells:
                    best_sell_idx = -1
                    max_high = float('-inf')
                    
                    start_sell_scan = i - self.sell_scan_bars
                    
                    for j in range(start_sell_scan, i):
                        # Calculate conditions for bar j
                        # Wick
                        s_upper = upper_ws[j]
                        s_body = bodies[j]
                        s_atr = atrs[j]
                        
                        wick_ratio_ok = (s_upper > s_body * self.sell_wick_ratio_min) if s_body > 0 else (s_upper > 0)
                        wick_atr_ok = (not self.use_sell_wick_atr_floor) | (s_upper >= s_atr * self.min_upper_wick_atr_mult)
                        wick_ok = wick_ratio_ok and wick_atr_ok
                        
                        # RSI
                        rsi_thresh = self.sell_rsi_soft_level if in_long else self.rsi_sell_level
                        rsi_ok = rsis[j] >= rsi_thresh
                        
                        # Volume
                        vol_ok_sell = (not self.use_vol_filter) or (volumes[j] > vol_mas[j])
                        
                        # Profit
                        profit_ok = (not self.sell_require_profit) or (highs[j] > entry_price)
                        
                        # High Gate
                        hg_ok = high_gates[j]
                        
                        core_sell_ok = hg_ok & rsi_ok & wick_ok & vol_ok_sell & profit_ok
                        
                        if core_sell_ok:
                            if highs[j] >= max_high:
                                max_high = highs[j]
                                best_sell_idx = j
                    
                    if best_sell_idx != -1:
                        # Additional Checks on the best candidate
                        # Near Peak
                        near_peak_ok = highs[best_sell_idx] >= (peak_since_entry - atrs[best_sell_idx] * self.near_peak_atr_mult)
                        trade_age_ok = bars_in_trade >= self.min_bars_in_trade
                        peak_hold_ok = peak_age >= self.peak_hold_bars
                        
                        if near_peak_ok and trade_age_ok and peak_hold_ok:
                            # SELL CONFIRMED
                            signals[best_sell_idx] = -1 # Mark the SELL PIVOT bar
                            in_long = False
                            entry_price = float('nan')
                            peak_since_entry = float('nan')

        # Add signals to dataframe
        data['gigi_signal'] = signals
        
        # Return only the relevant columns for ML
        result_cols = [
            'gigi_signal', 
            'rsi', 
            'atr', 
            'buy_core', 
            'context_oversold', 
            'low_gate', 
            'high_gate'
        ]
        return data[result_cols]
