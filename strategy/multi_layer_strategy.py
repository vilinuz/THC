"""
Multi-Layer Trading Strategy

A 5-layer trading system:
1. Strategic Layer (Markov Chains): Defines "Rules of Engagement" - Is the market safe?
2. Kinetic Layer (Kalman Filter + KAMA): Defines "Direction" - Where is the ball moving?
3. Structure Layer (SMC): Defines "Location" - Are we at key institutional levels?
4. Tactical Layer (ADX & Aroon): Defines "Timing" - When do I strike?
5. Risk Layer (ATR): Defines "Protection" - How much do I bet?
"""

from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
    print("Warning: hmmlearn not found. HMM regime detection will be disabled.")

try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None
    print("Warning: pykalman not found. Kalman filtering will use fallback.")

from .base_strateg import BaseStrategy
from indicators.atr import ATR
from indicators.aroon import Aroon
from indicators.adx import ADX
from indicators.kama import KAMA
from indicators.fisher_transform import FisherTransform

# SMC imports with fallback
try:
    from indicators.sm.smc_wrapper import SMCWrapper
    from indicators.sm.market_structure import MarketStructure
    from indicators.sm.fvg import FairValueGap
    SMC_AVAILABLE = True
except ImportError:
    SMCWrapper = None
    MarketStructure = None
    FairValueGap = None
    SMC_AVAILABLE = False
    print("Warning: SMC modules not found. Smart Money analysis will be disabled.")

try:
    from ml.causal_inference import CausalVolatilityTrading
except ImportError:
    CausalVolatilityTrading = None
    print("Warning: CausalVolatilityTrading not found.")


@dataclass
class RegimeState:
    """Encapsulates the HMM regime classification result."""
    regime_id: int
    stable_trend_prob: float
    crash_prob: float
    is_tradeable: bool  # True if stable trend prob > threshold
    is_crash: bool      # True if crash regime detected


@dataclass
class KalmanState:
    """Encapsulates the Kalman filter state estimate."""
    price_estimate: float
    velocity: float
    trend_direction: str  # 'up', 'down', or 'neutral'


@dataclass
class TacticalSignal:
    """Encapsulates ADX + Aroon tactical conditions."""
    adx_value: float
    aroon_up: float
    aroon_down: float
    has_energy: bool      # ADX > threshold
    has_momentum: bool    # Aroon Up > 70 (for longs) or Aroon Down > 70 (for shorts)


@dataclass
class RiskParameters:
    """ATR-based risk parameters."""
    atr_value: float
    stop_loss_long: float
    stop_loss_short: float
    position_size_units: float


@dataclass
class KAMAState:
    """Encapsulates KAMA analysis results."""
    kama_value: float
    efficiency_ratio: float
    trend: int  # 1 = up, -1 = down, 0 = neutral
    is_trending: bool
    price_above_kama: bool


@dataclass
class SMCState:
    """Encapsulates Smart Money Concepts analysis."""
    has_bullish_fvg: bool
    has_bearish_fvg: bool
    has_bullish_bos: bool
    has_bearish_bos: bool
    has_bullish_choch: bool
    has_bearish_choch: bool
    market_trend: int  # 1 = bullish, -1 = bearish, 0 = ranging
    near_bullish_fvg: bool  # Price near a bullish FVG (potential long entry)
    near_bearish_fvg: bool  # Price near a bearish FVG (potential short entry)


@dataclass
class CausalContext:
    """Encapsulates Leader-Follower causal information."""
    leader_df: pd.DataFrame
    optimal_lag: int
    ete_rising: bool = True
    leader_name: str = "Unknown"
    is_valid: bool = False


class MarkovRegimeClassifier:
    """
    Phase 1: Regime Classification using Hidden Markov Model
    
    Classifies market into 3 latent regimes:
    - State 0: Low Volatility / Trending (Safe to trade)
    - State 1: High Volatility / Correction
    - State 2: Extreme Volatility / Crash (Hard close all positions)
    """
    
    def __init__(self, n_regimes: int = 3, n_iter: int = 200):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.model = None
        self.is_trained = False
        
        if hmm is not None:
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                random_state=42
            )
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare HMM input features: Log-Returns and ATR (Volatility).
        """
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # Volatility (ATR normalized by price)
        atr = ATR.calculate(df, period=14)
        atr_normalized = (atr / df['close']).fillna(0)
        
        # Stack features
        features = np.column_stack([
            log_returns.values,
            atr_normalized.values
        ])
        
        return np.nan_to_num(features)
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the HMM on historical data.
        """
        if self.model is None:
            print("HMM not available. Skipping training.")
            return
            
        features = self._prepare_features(df)
        
        print(f"Training HMM on {len(features)} samples...")
        self.model.fit(features)
        self.is_trained = True
        print(f"HMM trained. Transition matrix:\n{self.model.transmat_}")
    
    def classify(self, df: pd.DataFrame, stable_threshold: float = 0.6, 
                 crash_threshold: float = 0.3) -> RegimeState:
        """
        Classify current regime and return trading eligibility.
        
        Args:
            df: Recent price data (use last 50-100 bars for context)
            stable_threshold: Min probability for State 0 to allow trading
            crash_threshold: Min probability for State 2 to trigger hard-close
            
        Returns:
            RegimeState with classification and trading signals
        """
        if not self.is_trained or self.model is None:
            # Fallback: assume stable regime
            return RegimeState(
                regime_id=0,
                stable_trend_prob=0.7,
                crash_prob=0.0,
                is_tradeable=True,
                is_crash=False
            )
        
        features = self._prepare_features(df)
        
        # Get posterior probabilities
        try:
            probs = self.model.predict_proba(features)
            current_probs = probs[-1]  # Latest probabilities
            
            regime_id = np.argmax(current_probs)
            stable_prob = current_probs[0]  # State 0 = Stable Trend
            crash_prob = current_probs[2] if len(current_probs) > 2 else 0.0
            
            return RegimeState(
                regime_id=int(regime_id),
                stable_trend_prob=float(stable_prob),
                crash_prob=float(crash_prob),
                is_tradeable=stable_prob > stable_threshold,
                is_crash=crash_prob > crash_threshold
            )
        except Exception as e:
            print(f"HMM classification error: {e}")
            return RegimeState(
                regime_id=0,
                stable_trend_prob=0.5,
                crash_prob=0.0,
                is_tradeable=True,
                is_crash=False
            )


class VelocityKalmanFilter:
    """
    Phase 2: Signal Denoising using 1D Kalman Filter
    
    Models price as a particle moving with velocity:
    State Vector x = [price, velocity]^T
    
    Extracts velocity component for trend direction:
    - v > 0: Underlying trend is Up
    - v < 0: Underlying trend is Down
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State transition matrix (constant velocity model)
        # [p_t]   [1  dt] [p_{t-1}]
        # [v_t] = [0   1] [v_{t-1}]
        self.F = np.array([
            [1.0, 1.0],  # price = prev_price + velocity
            [0.0, 1.0]   # velocity persists
        ])
        
        # Observation matrix (we only observe price)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.Q = np.array([
            [self.process_noise, 0.0],
            [0.0, self.process_noise * 0.1]
        ])
        
        # Measurement noise covariance
        self.R = np.array([[self.measurement_noise]])
        
        # Current state estimate
        self.state_mean = None
        self.state_cov = None
    
    def _initialize_state(self, first_price: float):
        """Initialize state with first observation."""
        self.state_mean = np.array([first_price, 0.0])
        self.state_cov = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
    
    def update(self, price: float) -> KalmanState:
        """
        Process a new price observation and return filtered state.
        
        Args:
            price: Current close price
            
        Returns:
            KalmanState with price estimate and velocity
        """
        if self.state_mean is None:
            self._initialize_state(price)
            return KalmanState(
                price_estimate=price,
                velocity=0.0,
                trend_direction='neutral'
            )
        
        if KalmanFilter is not None:
            # Use pykalman for update
            kf = KalmanFilter(
                transition_matrices=self.F,
                observation_matrices=self.H,
                transition_covariance=self.Q,
                observation_covariance=self.R
            )
            
            new_mean, new_cov = kf.filter_update(
                self.state_mean,
                self.state_cov,
                observation=price
            )
        else:
            # Manual Kalman update (fallback)
            # Predict step
            pred_mean = self.F @ self.state_mean
            pred_cov = self.F @ self.state_cov @ self.F.T + self.Q
            
            # Update step
            y = price - (self.H @ pred_mean)[0]  # Innovation
            S = self.H @ pred_cov @ self.H.T + self.R  # Innovation covariance
            K = pred_cov @ self.H.T @ np.linalg.inv(S)  # Kalman gain
            
            new_mean = pred_mean + (K @ np.array([[y]])).flatten()
            new_cov = (np.eye(2) - K @ self.H) @ pred_cov
        
        self.state_mean = new_mean
        self.state_cov = new_cov
        
        # Extract estimates
        price_estimate = new_mean[0]
        velocity = new_mean[1]
        
        # Determine trend direction
        if velocity > 0.001:  # Small threshold to avoid noise
            trend = 'up'
        elif velocity < -0.001:
            trend = 'down'
        else:
            trend = 'neutral'
        
        return KalmanState(
            price_estimate=float(price_estimate),
            velocity=float(velocity),
            trend_direction=trend
        )
    
    def process_series(self, prices: pd.Series) -> pd.DataFrame:
        """
        Process an entire price series and return all states.
        
        Returns:
            DataFrame with 'kalman_price', 'kalman_velocity', 'trend_direction'
        """
        self.state_mean = None
        self.state_cov = None
        
        results = []
        for price in prices:
            state = self.update(price)
            results.append({
                'kalman_price': state.price_estimate,
                'kalman_velocity': state.velocity,
                'trend_direction': state.trend_direction
            })
        
        return pd.DataFrame(results, index=prices.index)


class MultiLayerStrategy(BaseStrategy):
    """
    Multi-Layer Trading Strategy
    
    Implements a 5-phase validation system:
    
    Phase 1: Regime Classification (Markov/HMM)
    - Trading only permitted if stable trend probability > 60%
    - Hard-close all positions if crash regime detected
    
    Phase 2: Signal Denoising (Kalman Filter + KAMA)
    - 1D Kalman Filter extracts price velocity
    - KAMA provides adaptive trend confirmation
    - Both must agree on direction for signal
    
    Phase 3: Structure Layer (Smart Money Concepts)
    - FVG (Fair Value Gaps): Identifies institutional entry zones
    - BOS/CHoCH: Confirms market structure alignment
    - Enhances signal quality with SMC confluence
    
    Phase 4: Filter & Trigger (ADX + Aroon)
    - ADX(14) > 20: Confirms trend is structurally significant
    - Aroon Up > 70: Ensures entry near recent high (for longs)
    - Aroon Down > 70: Ensures entry near recent low (for shorts)
    
    Phase 5: Volatility Sizing (ATR)
    - Stop Loss: Kalman_Price - (2 * ATR) for longs
    - Position Size: Account_Risk% / ATR
    
    Phase 6: Causal Augmentation (Ivan Lettery)
    - Checks Leader asset regime (Dual-Regime)
    - Projects Leader velocity (Signal Denoising)
    - Checks Transfer Entropy (Information Flow Gate)
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        
        self.causal_context: Optional[CausalContext] = None
        
        # Phase 1: HMM Configuration
        self.stable_trend_threshold = self.config.get('stable_trend_threshold', 0.6)
        self.crash_threshold = self.config.get('crash_threshold', 0.3)
        self.hmm_lookback = self.config.get('hmm_lookback', 100)
        
        # Phase 2: Kalman + KAMA + Z-Score Configuration
        self.kalman_process_noise = self.config.get('kalman_process_noise', 0.01)
        self.kalman_measurement_noise = self.config.get('kalman_measurement_noise', 1.0)
        self.kama_period = self.config.get('kama_period', 10)
        self.kama_fast = self.config.get('kama_fast', 2)
        self.kama_slow = self.config.get('kama_slow', 30)
        self.kama_er_threshold = self.config.get('kama_er_threshold', 0.4)  # Efficiency ratio threshold
        
        # Z-Score configuration for pullback entries (optional, disabled by default)
        self.zscore_period = self.config.get('zscore_period', 20)  # Lookback for std dev calculation
        self.zscore_entry_threshold = self.config.get('zscore_entry_threshold', -1.0)  # Enter on pullback (negative z-score)
        self.zscore_extreme_threshold = self.config.get('zscore_extreme_threshold', 2.0)  # Block if too extended
        self.use_zscore_timing = self.config.get('use_zscore_timing', False)  # Disabled by default
        
        # Phase 3: SMC Configuration
        self.use_smc = self.config.get('use_smc', True) and SMC_AVAILABLE
        self.smc_require_fvg = self.config.get('smc_require_fvg', False)  # If True, requires FVG for entry
        self.smc_require_structure = self.config.get('smc_require_structure', False)  # If True, requires BOS/CHoCH
        
        # Phase 4: ADX + Aroon + Fisher + RSI Configuration
        self.adx_period = self.config.get('adx_period', 14)
        self.adx_consolidation = self.config.get('adx_consolidation', 20)   # Below this = Consolidation
        self.adx_chop_max = self.config.get('adx_chop_max', 38)             # Chop Regime: 20-38
        self.aroon_period = self.config.get('aroon_period', 25)
        self.aroon_threshold = self.config.get('aroon_threshold', 70)
        self.fisher_period = self.config.get('fisher_period', 9)
        self.fisher_overbought = self.config.get('fisher_overbought', 2.0)
        self.fisher_oversold = self.config.get('fisher_oversold', -2.0)
        self.use_fisher_timing = self.config.get('use_fisher_timing', True)
        
        # RSI Configuration for exceptions
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_climax = self.config.get('rsi_climax', 28)           # Climax exception: RSI < 28
        self.rsi_strong_dip = self.config.get('rsi_strong_dip', 35)   # Strong dip exception: RSI < 35
        
        # Z-Score thresholds for exceptions
        self.zscore_climax = self.config.get('zscore_climax', -2.5)       # Climax: Z < -2.5
        self.zscore_strong_dip = self.config.get('zscore_strong_dip', -2.0)  # Strong dip: Z < -2.0
        
        # Phase 5: ATR Configuration
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_stop_multiplier = self.config.get('atr_stop_multiplier', 2.0)
        self.account_risk_pct = self.config.get('account_risk_pct', 0.01)  # 1% default
        
        # Initialize components
        self.regime_classifier = MarkovRegimeClassifier(n_regimes=3)
        self.kalman_filter = VelocityKalmanFilter(
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise
        )
        
        # Initialize SMC Wrapper
        self.smc_wrapper = None
        if self.use_smc and SMCWrapper is not None:
            self.smc_wrapper = SMCWrapper(use_library=False)
        
        # State tracking
        self._is_trained = False
        self._last_regime: Optional[RegimeState] = None
        self._last_kalman: Optional[KalmanState] = None
        self._last_kama: Optional[KAMAState] = None
        self._last_smc: Optional[SMCState] = None
        
    def set_leader_context(self, context: CausalContext) -> None:
        """Set the causal leader context for the next signal generation."""
        self.causal_context = context
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the HMM regime classifier on historical data.
        Call this before using generate_signals() for best results.
        """
        if len(df) < 200:
            print("Warning: Insufficient data for HMM training. Need 200+ bars.")
            return
        
        self.regime_classifier.train(df)
        self._is_trained = True
    
    def _phase1_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Phase 1: Classify market regime using HMM.
        Returns whether trading is permitted.
        """
        lookback_df = df.tail(self.hmm_lookback)
        regime = self.regime_classifier.classify(
            lookback_df,
            stable_threshold=self.stable_trend_threshold,
            crash_threshold=self.crash_threshold
        )
        self._last_regime = regime
        
        # === CAUSAL AUGMENTATION: DUAL-REGIME CHECK ===
        # If we have a valid leader context, check if the Leader is crashing.
        # If Leader is in State 2 (Crash), we override and block trading.
        if self.causal_context and self.causal_context.is_valid and self.causal_context.leader_df is not None:
             leader_regime_state = self.regime_classifier.classify(
                self.causal_context.leader_df.tail(self.hmm_lookback),
                stable_threshold=self.stable_trend_threshold,
                crash_threshold=self.crash_threshold
            )
             # If Leader is crashing, we are crashing (Lead-Lag effect)
             if leader_regime_state.is_crash:
                 print(f"Refusing trade: Leader {self.causal_context.leader_name} is in Crash Regime.")
                 regime.is_tradeable = False
                 regime.is_crash = True # Force crash logic
                 
        return regime
    
    def _phase2_kalman(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2a: Apply Kalman filter to extract velocity signals.
        Returns DataFrame with kalman_price, kalman_velocity, trend_direction.
        """
        kalman_df = self.kalman_filter.process_series(df['close'])
        
        # === CAUSAL AUGMENTATION: PROJECTED VELOCITY ===
        # Project Leader's velocity onto Follower using Optimal Lag.
        # If Projected Velocity direction opposes Follower's Kalman direction, VETO the signal.
        if self.causal_context and self.causal_context.is_valid:
            try:
                lag = self.causal_context.optimal_lag
                leader_close = self.causal_context.leader_df['close']
                
                # Get Leader's Kalman states
                leader_kalman = self.kalman_filter.process_series(leader_close)
                
                # Projected velocity at time t is Leader's velocity at time t-lag
                projected_velocity = leader_kalman['kalman_velocity'].shift(lag)
                
                # Add to dataframe for inspection
                kalman_df['projected_velocity'] = projected_velocity
                
                # Logic: If Follower says UP but Projected says DOWN (significant), downgrade/veto
                # We'll augment the 'trend_direction' column
                # Only apply if aligned with current index (which it is via assignment)
                
                # Check alignment of signs
                follower_vel = kalman_df['kalman_velocity']
                
                # Divergence: Follower Up vs Leader Projected Down
                divergence_mask = (follower_vel > 0) & (projected_velocity < -0.001)
                kalman_df.loc[divergence_mask, 'trend_direction'] = 'neutral' # Veto
                
                # Divergence: Follower Down vs Leader Projected Up
                divergence_mask_2 = (follower_vel < 0) & (projected_velocity > 0.001)
                kalman_df.loc[divergence_mask_2, 'trend_direction'] = 'neutral' # Veto
                
            except Exception as e:
                print(f"Causal velocity projection error: {e}")
                
        return kalman_df
    
    def _phase2_kama(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2b: Apply KAMA for adaptive trend confirmation with Z-Score pullback timing.
        
        Z-Score Logic:
        - Calculate how far price is from KAMA in standard deviations
        - For LONGS: Enter when z-score shows pullback toward KAMA (z-score < 0 or decreasing)
          in a higher-timeframe uptrend (KAMA rising)
        - For SHORTS: Enter when z-score shows pullback toward KAMA (z-score > 0 or decreasing magnitude)
          in a higher-timeframe downtrend (KAMA falling)
        """
        kama_result = KAMA.calculate_with_details(
            df, 
            period=self.kama_period,
            fast_period=self.kama_fast,
            slow_period=self.kama_slow
        )
        
        close = pd.to_numeric(df['close'])
        kama = kama_result['kama']
        
        # Calculate Z-Score: (Price - KAMA) / StdDev(Price - KAMA)
        price_deviation = close - kama
        deviation_std = price_deviation.rolling(window=self.zscore_period).std()
        zscore = price_deviation / deviation_std.replace(0, 0.0001)
        zscore = zscore.fillna(0)
        
        # Z-Score change (for detecting pullbacks)
        zscore_prev = zscore.shift(1)
        zscore_change = zscore - zscore_prev
        
        # Trend determination (higher-timeframe trend from KAMA)
        kama_trend = pd.Series(0, index=df.index)
        uptrend = (close > kama) & (kama_result['kama_slope'] > 0)
        downtrend = (close < kama) & (kama_result['kama_slope'] < 0)
        kama_trend[uptrend] = 1
        kama_trend[downtrend] = -1
        
        # Is market trending based on efficiency ratio?
        is_trending = kama_result['er'] > self.kama_er_threshold
        
        # Z-Score pullback signals
        # For LONGS: Pullback toward KAMA in uptrend
        # - KAMA is rising (uptrend)
        # - Z-score is negative (price below KAMA = pullback) OR decreasing (pulling back)
        # - NOT too extended to the downside (z-score not < -extreme)
        zscore_long_pullback = (
            (kama_result['kama_slope'] > 0) &  # Higher TF uptrend
            (
                (zscore < self.zscore_entry_threshold) |  # Significant pullback
                (zscore_change < 0)  # OR z-score decreasing (pulling back)
            ) &
            (zscore > -self.zscore_extreme_threshold)  # Not too extended down
        )
        
        # For SHORTS: Pullback toward KAMA in downtrend
        # - KAMA is falling (downtrend)
        # - Z-score is positive (price above KAMA = pullback) OR decreasing (pulling back)
        # - NOT too extended to the upside
        zscore_short_pullback = (
            (kama_result['kama_slope'] < 0) &  # Higher TF downtrend
            (
                (zscore > -self.zscore_entry_threshold) |  # Significant pullback (positive)
                (zscore_change > 0)  # OR z-score increasing (pulling back up in downtrend)
            ) &
            (zscore < self.zscore_extreme_threshold)  # Not too extended up
        )
        
        return pd.DataFrame({
            'kama': kama,
            'kama_er': kama_result['er'],
            'kama_trend': kama_trend,
            'kama_is_trending': is_trending,
            'price_above_kama': close > kama,
            'zscore': zscore,
            'zscore_change': zscore_change,
            'zscore_long_pullback': zscore_long_pullback,
            'zscore_short_pullback': zscore_short_pullback
        }, index=df.index)
    
    def _phase3_smc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 3: Smart Money Concepts analysis.
        Returns DataFrame with FVG, BOS, CHoCH, and market structure signals.
        """
        result = pd.DataFrame(index=df.index)
        
        # Initialize with defaults
        result['has_bullish_fvg'] = False
        result['has_bearish_fvg'] = False
        result['has_bullish_bos'] = False
        result['has_bearish_bos'] = False
        result['has_bullish_choch'] = False
        result['has_bearish_choch'] = False
        result['smc_trend'] = 0
        result['near_bullish_fvg'] = False
        result['near_bearish_fvg'] = False
        result['smc_long_confluence'] = False
        result['smc_short_confluence'] = False
        
        if self.smc_wrapper is None or len(df) < 20:
            return result
        
        try:
            # FVG Detection
            fvg_df = self.smc_wrapper.detect_fvg(df)
            result['has_bullish_fvg'] = fvg_df['bullish_fvg'].fillna(False)
            result['has_bearish_fvg'] = fvg_df['bearish_fvg'].fillna(False)
            
            # Check if price is near an unfilled FVG (within last 20 bars)
            close = pd.to_numeric(df['close'])
            for i in range(20, len(df)):
                # Look for unfilled bullish FVGs in last 20 bars
                recent_bull_fvg = fvg_df.iloc[max(0, i-20):i]
                bull_fvgs = recent_bull_fvg[
                    (recent_bull_fvg['bullish_fvg']) & 
                    (~recent_bull_fvg['fvg_filled'])
                ]
                if len(bull_fvgs) > 0:
                    for _, fvg in bull_fvgs.iterrows():
                        # Price entering the FVG zone
                        if fvg['fvg_bottom'] <= close.iloc[i] <= fvg['fvg_top']:
                            result.iloc[i, result.columns.get_loc('near_bullish_fvg')] = True
                            break
                
                # Look for unfilled bearish FVGs
                bear_fvgs = recent_bull_fvg[
                    (recent_bull_fvg['bearish_fvg']) & 
                    (~recent_bull_fvg['fvg_filled'])
                ]
                if len(bear_fvgs) > 0:
                    for _, fvg in bear_fvgs.iterrows():
                        if fvg['fvg_bottom'] <= close.iloc[i] <= fvg['fvg_top']:
                            result.iloc[i, result.columns.get_loc('near_bearish_fvg')] = True
                            break
            
            # Market Structure (BOS/CHoCH)
            swing_df = self.smc_wrapper.detect_swing_points(df)
            if MarketStructure is not None:
                structure_df = MarketStructure.detect_bos_choch(df, swing_df)
                result['has_bullish_bos'] = structure_df['bullish_bos'].fillna(False)
                result['has_bearish_bos'] = structure_df['bearish_bos'].fillna(False)
                result['has_bullish_choch'] = structure_df['bullish_choch'].fillna(False)
                result['has_bearish_choch'] = structure_df['bearish_choch'].fillna(False)
                result['smc_trend'] = structure_df['trend'].fillna(0)
            
            # SMC Confluence signals
            # Long confluence: Bullish structure (BOS or CHoCH) OR near bullish FVG
            result['smc_long_confluence'] = (
                result['has_bullish_bos'] | 
                result['has_bullish_choch'] | 
                result['near_bullish_fvg']
            )
            
            # Short confluence: Bearish structure OR near bearish FVG
            result['smc_short_confluence'] = (
                result['has_bearish_bos'] | 
                result['has_bearish_choch'] | 
                result['near_bearish_fvg']
            )
            
        except Exception as e:
            print(f"SMC analysis error: {e}")
        
        return result
    
    def _phase4_tactical(self, df: pd.DataFrame, zscore: pd.Series = None) -> pd.DataFrame:
        """
        Phase 4: Calculate ADX, Aroon, Fisher, RSI tactical indicators.
        
        ADX Regime Rules:
        - Trend Regime: ADX > 38 (Strong trend, normal trading)
        - Chop Regime: ADX between 20 and 38 (Choppy, cautious trading)
        - Consolidation: ADX < 20 (Dead market, BLOCKED unless exception applies)
        
        Exceptions (allow entry even in Consolidation/Dead Markets):
        - Climax Exception: Z-Score < -2.5 OR RSI < 28 (catches crashes)
        - Strong Dip Exception: Z-Score < -2.0 AND RSI < 35 (catches strong dips)
        
        RESTRICTION: If ADX < 20 and NOT a Climax/Strong Dip exception, entry is BLOCKED.
        """
        close = pd.to_numeric(df['close'])
        
        # ADX
        adx = ADX.calculate(df, period=self.adx_period)
        
        # Aroon
        aroon = Aroon.calculate(df, period=self.aroon_period)
        
        # Fisher Transform
        fisher = FisherTransform.calculate(df, period=self.fisher_period)
        fisher_prev = fisher.shift(1)
        
        # RSI Calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        
        # Use z-score from KAMA phase if provided, otherwise calculate simple z-score
        if zscore is None:
            # Simple z-score based on close price
            price_mean = close.rolling(window=20).mean()
            price_std = close.rolling(window=20).std()
            zscore = ((close - price_mean) / price_std.replace(0, 0.0001)).fillna(0)
        
        # === ADX REGIME CLASSIFICATION ===
        is_trend_regime = adx > self.adx_chop_max  # ADX > 38: Strong trend
        is_chop_regime = (adx >= self.adx_consolidation) & (adx <= self.adx_chop_max)  # ADX 20-38: Chop
        is_consolidation = adx < self.adx_consolidation  # ADX < 20: Dead market
        
        # === EXCEPTION CONDITIONS ===
        # Climax Exception: Z-Score < -2.5 OR RSI < 28 (catches crashes even in dead markets)
        is_climax = (zscore < self.zscore_climax) | (rsi < self.rsi_climax)
        
        # Strong Dip Exception: Z-Score < -2.0 AND RSI < 35 (catches strong dips in consolidation)
        is_strong_dip = (zscore < self.zscore_strong_dip) & (rsi < self.rsi_strong_dip)
        
        # Entry is allowed if:
        # 1. Trend Regime (ADX > 38): Always allowed
        # 2. Chop Regime (ADX 20-38): Allowed with caution
        # 3. Consolidation (ADX < 20): BLOCKED unless Climax or Strong Dip exception
        
        entry_allowed = is_trend_regime | is_chop_regime | is_climax | is_strong_dip

        # === CAUSAL AUGMENTATION: TRANSFER ENTROPY GATE ===
        # Only allow entry if Effective Transfer Entropy is rising/high (Information is flowing).
        # We assume `ete_rising` boolean in context captures this check (calculated externally or simply checked here).
        if self.causal_context and self.causal_context.is_valid:
            if not self.causal_context.ete_rising:
                # ETE Gate closed: Information flow suggests decoupling or noise
                # But we allow Climax/Strong Dip exceptions to override this (panic is panic)
                # Vectorized operation: Filter to only allow if exception logic holds
                entry_allowed = entry_allowed & (is_climax | is_strong_dip)
        
        # Momentum check (Aroon)
        has_up_momentum = aroon['aroon_up'] > self.aroon_threshold
        has_down_momentum = aroon['aroon_down'] > self.aroon_threshold
        
        # Fisher timing signals - SOFT filter approach
        fisher_long_timing = (
            (fisher < self.fisher_oversold) |  # Oversold = excellent entry
            ((fisher > fisher_prev) & (fisher_prev < 0)) |  # Crossing up = good
            (fisher < self.fisher_overbought * 0.8)  # Not overbought = acceptable
        )
        
        fisher_short_timing = (
            (fisher > self.fisher_overbought) |  # Overbought = excellent entry
            ((fisher < fisher_prev) & (fisher_prev > 0)) |  # Crossing down = good
            (fisher > self.fisher_oversold * 0.8)  # Not oversold = acceptable
        )
        
        # Combined tactical signals with ADX regime rules
        if self.use_fisher_timing:
            tactical_long = entry_allowed & has_up_momentum & fisher_long_timing
            tactical_short = entry_allowed & has_down_momentum & fisher_short_timing
        else:
            tactical_long = entry_allowed & has_up_momentum
            tactical_short = entry_allowed & has_down_momentum
        
        return pd.DataFrame({
            'adx': adx,
            'rsi': rsi,
            'aroon_up': aroon['aroon_up'],
            'aroon_down': aroon['aroon_down'],
            'fisher': fisher,
            'fisher_prev': fisher_prev,
            'is_trend_regime': is_trend_regime,
            'is_chop_regime': is_chop_regime,
            'is_consolidation': is_consolidation,
            'is_climax': is_climax,
            'is_strong_dip': is_strong_dip,
            'entry_allowed': entry_allowed,
            'has_up_momentum': has_up_momentum,
            'has_down_momentum': has_down_momentum,
            'fisher_long_timing': fisher_long_timing,
            'fisher_short_timing': fisher_short_timing,
            'tactical_long': tactical_long,
            'tactical_short': tactical_short
        }, index=df.index)
    
    def _phase5_risk(self, df: pd.DataFrame, kalman_prices: pd.Series) -> pd.DataFrame:
        """
        Phase 5: Calculate ATR-based risk parameters.
        Uses Kalman smoothed price for stop-loss anchor.
        """
        atr = ATR.calculate(df, period=self.atr_period)
        
        # Stop Loss using Kalman price (more stable than raw close)
        stop_loss_long = kalman_prices - (self.atr_stop_multiplier * atr)
        stop_loss_short = kalman_prices + (self.atr_stop_multiplier * atr)
        
        return pd.DataFrame({
            'atr': atr,
            'stop_loss_long': stop_loss_long,
            'stop_loss_short': stop_loss_short
        }, index=df.index)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using the 5-phase validation system.
        
        Returns:
            pd.Series with values:
                1: Long signal
               -1: Short signal
                0: No signal
               -99: Emergency close (crash regime detected)
        """
        if df.empty or len(df) < 50:
            return pd.Series(0, index=df.index if not df.empty else [])
        
        # Phase 1: Regime Classification
        regime = self._phase1_regime(df)
        
        # Initialize signals
        signals = pd.Series(0, index=df.index)
        
        # Emergency close on crash detection
        if regime.is_crash:
            signals.iloc[-1] = -99
            return signals
        
        # No trading if regime not stable
        if not regime.is_tradeable:
            return signals
        
        # Phase 2a: Kalman Filter
        kalman_df = self._phase2_kalman(df)
        
        # Phase 2b: KAMA
        kama_df = self._phase2_kama(df)
        
        # Phase 3: SMC Analysis
        smc_df = self._phase3_smc(df)
        
        # Phase 4: Tactical Indicators
        tactical_df = self._phase4_tactical(df)
        
        # Phase 5: Risk Parameters (calculated but not used for signal generation)
        risk_df = self._phase5_risk(df, kalman_df['kalman_price'])
        
        # Combine all phases for signal generation
        # Long Signal requirements:
        # - Kalman velocity > 0 (primary trend direction)
        # - KAMA confirms uptrend OR is trending
        # - Z-score shows pullback toward KAMA in uptrend (if enabled)
        # - Tactical confirms (ADX + Aroon + Fisher timing)
        # - SMC confluence (optional boost, not required unless configured)
        
        kalman_long = kalman_df['kalman_velocity'] > 0
        kama_long = (kama_df['kama_trend'] >= 0) | kama_df['kama_is_trending']
        
        # Use the new tactical_long which includes Fisher timing
        tactical_long = tactical_df['tactical_long']
        
        # Z-score pullback filter: Enter on statistically significant pullback toward KAMA
        if self.use_zscore_timing:
            zscore_long = kama_df['zscore_long_pullback']
        else:
            zscore_long = True
        
        # Base long conditions (Kalman + KAMA + Z-Score Pullback + Tactical)
        long_conditions = kalman_long & kama_long & tactical_long & zscore_long
        
        # Apply SMC filter if required
        if self.smc_require_fvg:
            long_conditions = long_conditions & smc_df['near_bullish_fvg']
        elif self.smc_require_structure:
            long_conditions = long_conditions & (smc_df['has_bullish_bos'] | smc_df['has_bullish_choch'])
        
        # Short Signal requirements (inverse of long)
        kalman_short = kalman_df['kalman_velocity'] < 0
        kama_short = (kama_df['kama_trend'] <= 0) | kama_df['kama_is_trending']
        
        # Use the new tactical_short which includes Fisher timing
        tactical_short = tactical_df['tactical_short']
        
        # Z-score pullback filter for shorts
        if self.use_zscore_timing:
            zscore_short = kama_df['zscore_short_pullback']
        else:
            zscore_short = True
        
        short_conditions = kalman_short & kama_short & tactical_short & zscore_short
        
        # Apply SMC filter if required
        if self.smc_require_fvg:
            short_conditions = short_conditions & smc_df['near_bearish_fvg']
        elif self.smc_require_structure:
            short_conditions = short_conditions & (smc_df['has_bearish_bos'] | smc_df['has_bearish_choch'])
        
        signals[long_conditions] = 1
        signals[short_conditions] = -1
        
        return signals
    
    def generate_signals_with_meta(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate signals with full metadata for analysis.
        Includes all phase data and SMC confluence information.
        
        Returns:
            Tuple of (signals, metadata_df)
        """
        if df.empty or len(df) < 50:
            empty_signals = pd.Series(0, index=df.index if not df.empty else [])
            empty_meta = pd.DataFrame(index=df.index if not df.empty else [])
            return empty_signals, empty_meta
        
        # Phase 1
        regime = self._phase1_regime(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Phase 2a: Kalman
        kalman_df = self._phase2_kalman(df)
        
        # Phase 2b: KAMA
        kama_df = self._phase2_kama(df)
        
        # Phase 3: SMC
        smc_df = self._phase3_smc(df)
        
        # Phase 4: Tactical
        tactical_df = self._phase4_tactical(df)
        
        # Phase 5: Risk
        risk_df = self._phase5_risk(df, kalman_df['kalman_price'])
        
        # Build comprehensive metadata
        meta_df = pd.concat([kalman_df, kama_df, smc_df, tactical_df, risk_df], axis=1)
        
        # Add regime info
        meta_df['regime_id'] = regime.regime_id
        meta_df['stable_trend_prob'] = regime.stable_trend_prob
        meta_df['crash_prob'] = regime.crash_prob
        meta_df['is_tradeable'] = regime.is_tradeable
        
        # Calculate confluence score (0-5)
        # Each layer that agrees on direction adds 1 point
        confluence_long = pd.Series(0, index=df.index, dtype=int)
        confluence_short = pd.Series(0, index=df.index, dtype=int)
        
        # Kalman direction
        confluence_long += (kalman_df['kalman_velocity'] > 0).astype(int)
        confluence_short += (kalman_df['kalman_velocity'] < 0).astype(int)
        
        # KAMA direction
        confluence_long += (kama_df['kama_trend'] > 0).astype(int)
        confluence_short += (kama_df['kama_trend'] < 0).astype(int)
        
        # SMC confluence
        if self.use_smc:
            confluence_long += smc_df['smc_long_confluence'].astype(int)
            confluence_short += smc_df['smc_short_confluence'].astype(int)
        
        # Tactical (ADX regime + Aroon) - using entry_allowed which includes ADX regime rules
        confluence_long += (tactical_df['entry_allowed'] & tactical_df['has_up_momentum']).astype(int)
        confluence_short += (tactical_df['entry_allowed'] & tactical_df['has_down_momentum']).astype(int)
        
        # Fisher timing bonus (+1 if Fisher confirms)
        if self.use_fisher_timing:
            confluence_long += tactical_df['fisher_long_timing'].astype(int)
            confluence_short += tactical_df['fisher_short_timing'].astype(int)
        
        # Z-Score pullback bonus (+1 if entering on pullback toward KAMA)
        if self.use_zscore_timing:
            confluence_long += kama_df['zscore_long_pullback'].astype(int)
            confluence_short += kama_df['zscore_short_pullback'].astype(int)
        
        meta_df['confluence_long'] = confluence_long
        meta_df['confluence_short'] = confluence_short
        
        # Generate signals with all filters
        if regime.is_crash:
            signals.iloc[-1] = -99
        elif regime.is_tradeable:
            kalman_long = kalman_df['kalman_velocity'] > 0
            kama_long = (kama_df['kama_trend'] >= 0) | kama_df['kama_is_trending']
            tactical_long = tactical_df['tactical_long']
            
            # Add z-score filter
            if self.use_zscore_timing:
                zscore_long = kama_df['zscore_long_pullback']
            else:
                zscore_long = True
            
            long_cond = kalman_long & kama_long & tactical_long & zscore_long
            
            if self.smc_require_fvg:
                long_cond = long_cond & smc_df['near_bullish_fvg']
            elif self.smc_require_structure:
                long_cond = long_cond & (smc_df['has_bullish_bos'] | smc_df['has_bullish_choch'])
            
            kalman_short = kalman_df['kalman_velocity'] < 0
            kama_short = (kama_df['kama_trend'] <= 0) | kama_df['kama_is_trending']
            tactical_short = tactical_df['tactical_short']
            
            # Add z-score filter
            if self.use_zscore_timing:
                zscore_short = kama_df['zscore_short_pullback']
            else:
                zscore_short = True
            
            short_cond = kalman_short & kama_short & tactical_short & zscore_short
            
            if self.smc_require_fvg:
                short_cond = short_cond & smc_df['near_bearish_fvg']
            elif self.smc_require_structure:
                short_cond = short_cond & (smc_df['has_bearish_bos'] | smc_df['has_bearish_choch'])
            
            signals[long_cond] = 1
            signals[short_cond] = -1
        
        meta_df['signal'] = signals
        
        return signals, meta_df
    
    def calculate_position_size(self, signal: int, price: float, 
                                portfolio_value: float, risk_params: Dict) -> float:
        """
        Calculate position size based on ATR volatility sizing.
        
        Formula: Units = (Account * Risk%) / ATR
        
        Args:
            signal: Trade direction (1 or -1)
            price: Current price
            portfolio_value: Total account value
            risk_params: Dict with 'atr' value
            
        Returns:
            Position size in units
        """
        if signal == 0:
            return 0.0
        
        atr = risk_params.get('atr', price * 0.02)  # Default to 2% if ATR not provided
        risk_amount = portfolio_value * self.account_risk_pct
        
        # Position size in currency
        position_value = risk_amount / (atr / price) if atr > 0 else 0
        
        # Convert to units
        units = position_value / price if price > 0 else 0
        
        return abs(units)
    
    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """
        Calculate stop loss using Kalman-smoothed ATR method.
        
        Args:
            entry_price: Entry price (preferably Kalman estimate)
            side: 'long' or 'short'
            atr: Current ATR value
            
        Returns:
            Stop loss price
        """
        if side == 'long':
            return entry_price - (atr * self.atr_stop_multiplier)
        else:
            return entry_price + (atr * self.atr_stop_multiplier)

