import pandas as pd
import numpy as np

class RsmStrategy:
    def __init__(self):
        self.kalman = NoiseCleaner()
        self.smc = SmartMoneyIndicator()
        self.bcpd = BayesianShockDetector(hazard=0.02) # Tuned for Crypto
        
        self.history = []
        self.min_history_for_hmm = 50
        
        self.position = 0 # 0 Cash, 1 Long
        self.equity = [1000] # Starting Capital
        
    def ingest(self, price, volume):
        """ The Main Loop: Runs once per candle """
        
        # 1. Clean the Noise
        smooth_price = self.kalman.update(price)
        
        # 2. Update History
        self.history.append({'price': price, 'smooth': smooth_price, 'volume': volume})
        if len(self.history) < 2: return "WAIT"
        
        # 3. Calculate Features
        prev_smooth = self.history[-2]['smooth']
        log_ret = np.log(smooth_price / prev_smooth) * 100
        
        # 4. Check for SHOCK (BCPD)
        shock_prob = self.bcpd.update(log_ret)
        
        # 5. Check for Smart Money (Accumulation)
        smc_score = self.smc.update(price, volume)
        
        # 6. Regime Check (HMM) - Only run if we have enough data
        regime_bull_prob = 0.5
        if len(self.history) >= self.min_history_for_hmm:
            # We retrain on a rolling window (e.g., last 100 bars)
            df = pd.DataFrame(self.history)[-100:]
            df['ret'] = np.log(df['smooth'] / df['smooth'].shift(1)) * 100
            df.dropna(inplace=True)
            
            try:
                # Simple switching variance model
                mod = MarkovAutoregression(df['ret'], k_regimes=2, order=0, switching_variance=True)
                res = mod.fit(disp=False)
                
                # Identify Bull Regime (Lower Variance)
                sigmas = res.params[res.params.index.str.contains('sigma2')]
                bull_idx = sigmas.idxmin() # The index of the low variance regime
                regime_bull_prob = res.filtered_marginal_probabilities.iloc[-1, int(bull_idx[-1])]
            except:
                pass # Fallback to 0.5 if fit fails
        
        # ==========================
        # 7. THE DECISION MATRIX
        # ==========================
        
        signal = "HOLD"
        
        # LOGIC:
        # If SHOCK is detected -> FORCE EXIT
        if shock_prob > 0.5:
            signal = "SELL_SHOCK"
            
        # If no shock, check REGIME
        elif regime_bull_prob > 0.7:
            # We are in Bull Regime.
            # Entry condition: SMC confirms accumulation OR momentum is strong
            if smc_score > 0.1 or log_ret > 0:
                signal = "BUY"
        
        elif regime_bull_prob < 0.3:
            # We are in Bear Regime.
            signal = "SELL"
            
        return signal, shock_prob, regime_bull_prob, smooth_price
