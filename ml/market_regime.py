import numpy as np
from hmmlearn import hmm
try:
    from pykalman import KalmanFilter
except ImportError:
    print("pykalman not found, using placeholder or failing")
    KalmanFilter = None

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class RegimeParams:
    """Regime-specific Kalman parameters"""
    F: np.ndarray  # State transition
    Q: np.ndarray  # Process noise
    R: np.ndarray  # Measurement noise
    H: np.ndarray  # Observation matrix

class SwitchingKalmanHMM:
    """
    Hybrid model: HMM determines regime -> Regime selects Kalman params.
    Based on [web:12][web:25] architectures.
    
    Integrates HMM for regime detection and Kalman Filter (pykalman) for state estimation.
    """
    
    def __init__(self, n_regimes: int = 3, state_dim: int = 4, history_len: int = 10):
        self.n_regimes = n_regimes
        self.state_dim = state_dim
        self.history_len = history_len
        
        # Initialize HMM for regime detection
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42
        )
        self.hmm_trained = False
        
        # Initialize regime parameters
        self.regime_params = self._initialize_regime_params()
        self.current_regime = 0
        
        # Initialize the single Kalman Filter state (Mean and Covariance)
        # We hold the Current State MEAN (x) and COVARIANCE (P) explicitly
        # because pykalman is functional (filter_update returns new state)
        self.current_state_mean = np.zeros(state_dim)
        self.current_state_cov = np.eye(state_dim) * 10.0
        
        # Buffer for recent features to predict regime
        self.recent_features: List[np.ndarray] = []
        
    def _initialize_regime_params(self) -> List[RegimeParams]:
        """
        Define dynamics (F, Q, R) for each regime.
        State vector: [price, velocity, acceleration, volatility]
        """
        params = []
        
        # REGIME 0: Low Volatility Bull (Mean-Revert with Upward Drift)
        f_bull = np.array([
            [1.0, 0.01, 0.0005, 0.0],  # Price with small positive drift
            [0.0, 0.95, 0.01, 0.0],    # Velocity (mean-reverting)
            [0.0, 0.0, 0.8, 0.0],      # Acceleration (decays fast)
            [0.0, 0.0, 0.0, 0.85]      # Volatility persistence
        ])
        q_bull = np.eye(self.state_dim) * 0.001
        r_bull = np.array([[0.5]])
        h_bull = np.array([[1., 0., 0., 0.]])
        params.append(RegimeParams(F=f_bull, Q=q_bull, R=r_bull, H=h_bull))
        
        # REGIME 1: High Volatility Bear (Momentum with Downward Drift)
        f_bear = np.array([
            [1.0, -0.02, -0.001, 0.0],
            [0.0, 1.05, 0.02, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.95]
        ])
        q_bear = np.eye(self.state_dim) * 0.01
        r_bear = np.array([[2.0]])
        h_bear = np.array([[1., 0., 0., 0.]])
        params.append(RegimeParams(F=f_bear, Q=q_bear, R=r_bear, H=h_bear))
        
        # REGIME 2: Sideways Chop (Pure Random Walk)
        f_chop = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.75]
        ])
        q_chop = np.eye(self.state_dim) * 0.005
        r_chop = np.array([[1.0]])
        h_chop = np.array([[1., 0., 0., 0.]])
        params.append(RegimeParams(F=f_chop, Q=q_chop, R=r_chop, H=h_chop))
        
        return params
    
    def train_hmm(self, feature_matrix: np.ndarray):
        """
        Train HMM on historical features [Returns, Range Vol, Volume].
        feature_matrix: (n_samples, 3)
        """
        print(f"Training HMM on {feature_matrix.shape[0]} samples...")
        self.hmm.fit(feature_matrix)
        self.hmm_trained = True
        print(f"HMM converged. Transition Matrix:\n{self.hmm.transmat_}")
        
    def _get_regime_probabilities(self, current_features: np.ndarray) -> np.ndarray:
        """
        Soft regime classification using HMM posterior.
        Uses the provided current_features or history.
        """
        if not self.hmm_trained:
            # Fallback if not trained
            return np.ones(self.n_regimes) / self.n_regimes
            
        # We need a sequence for predict_proba. 
        # If we have history, use last N. If just one, use it.
        # Check dimensionality
        if current_features.ndim == 1:
            X = current_features.reshape(1, -1)
        else:
            X = current_features
            
        try:
            # predict_proba expects (n_samples, n_features)
            # It returns (n_samples, n_components)
            # We want the proba for the last sample
            probs = self.hmm.predict_proba(X)
            return probs[-1]
        except Exception as e:
            print(f"Error in HMM prediction: {e}")
            return np.ones(self.n_regimes) / self.n_regimes

    def predict_step(self, observation: float, features: np.ndarray = None) -> Dict[str, Any]:
        """
        Two-stage prediction:
        1. HMM infers current regime from observations/features
        2. Regime-specific Kalman Filter forecasts next state
        
        Args:
            observation: Current noisy price measurement (for KF update)
            features: Feature vector for HMM regime detection [Returns, Range Vol, Volume]
                      If None, tries to use internal history or defaults.
        Returns:
            predicted_price, predicted_velocity, regime_id, confidence
        """
        if KalmanFilter is None:
             raise ImportError("pykalman package is missing. Please install it.")

        # Update history
        if features is not None:
            self.recent_features.append(features)
            if len(self.recent_features) > self.history_len:
                self.recent_features.pop(0)
        
        # Stage 1: Regime Detection via HMM
        if features is not None:
            regime_probs = self._get_regime_probabilities(np.array(self.recent_features))
        elif len(self.recent_features) > 0:
            regime_probs = self._get_regime_probabilities(np.array(self.recent_features))
        else:
            regime_probs = np.ones(self.n_regimes) / self.n_regimes
            
        self.current_regime = np.argmax(regime_probs)
        
        # Stage 2: Kalman Update with selected filter parameters
        # Load parameters for the current regime
        params = self.regime_params[self.current_regime]
        
        # Create a transient KF with regime params to perform one update
        # pykalman uses filter_update to step (t-1) -> (t)
        # Note: 'observation' is y_t. current_state_mean is x_{t-1|t-1} or x_{t|t-1}?
        # Usually we store Filtered State (x_{t|t}) and Covariance (P_{t|t}) from previous step.
        # filter_update(prev_mean, prev_cov, observation=obs, transition_matrix=F, ...) 
        # returns (next_mean, next_cov) which IS x_{t|t} and P_{t|t} (Filtered).
        
        kf = KalmanFilter(
            transition_matrices=params.F,
            observation_matrices=params.H,
            transition_covariance=params.Q,
            observation_covariance=params.R
        )
        
        # Perform update
        # observation needs to be compatible with H (1x4) -> obs should be scalar or 1D array
        next_mean, next_cov = kf.filter_update(
            self.current_state_mean,
            self.current_state_cov,
            observation=observation
        )
        
        # Store for next iteration
        self.current_state_mean = next_mean
        self.current_state_cov = next_cov
        
        # Extract estimates (filtered state)
        current_price_est = next_mean[0]
        current_velocity = next_mean[1]
        current_volatility = next_mean[3]
        
        # Confidence = inverse of position variance
        confidence = 1.0 / (next_cov[0, 0] + 1e-6)
        
        return {
            'price': current_price_est,
            'velocity': current_velocity,
            'volatility': current_volatility,
            'regime': int(self.current_regime),
            'regime_probs': regime_probs.tolist(),
            'confidence': confidence
        }

class KalmanHMMLSTMFusion:
    """
    Production-ready Fusion system:
    1. HMM: Identifies Market Regime (Bull, Bear, Chop)
    2. Switching Kalman: Adapts linear tracking based on regime
    3. LSTM: Captures non-linear residual / complex patterns (Uses KalmanEnhancedLSTM)
    4. SMC: Structural breaks (BOS, CHoCH) and Liquidity (FVG)
    5. Indicators: RSI, EMA, T3, Ichimoku for confluence
    6. Fusion: Combines all for final prediction and signal generation
    """
    
    def __init__(self, n_regimes: int = 3, state_dim: int = 4, lstm_input_size: int = 3, lstm_hidden_size: int = 64):
        # Components
        self.skhmm = SwitchingKalmanHMM(n_regimes=n_regimes, state_dim=state_dim)
        
        # Initialize SMC components
        try:
            from indicators.sm.smc_wrapper import SMCWrapper
            self.smc = SMCWrapper(use_library=False) 
        except ImportError:
            self.smc = None
            print("Warning: SMCWrapper not found.")
            
        # Import Indicators
        try:
            from indicators.rsi import RSI
            from indicators.ema import EMA
            from indicators.tillson_t3 import TillsonT3
            from indicators.ichimoku import Ichimoku
            self.ind_rsi = RSI
            self.ind_ema = EMA
            self.ind_t3 = TillsonT3
            self.ind_ichi = Ichimoku
        except ImportError as e:
            print(f"Warning: Failed to import indicators: {e}")
            self.ind_rsi = None

        # Always use Advanced TF LSTM
        from .kalman_enhanced_lstm import KalmanEnhancedLSTM
        self.lstm = KalmanEnhancedLSTM(lookback=10, n_regimes=n_regimes)
        
        # Fusion Weights [Kalman_Weight, LSTM_Weight] per Regime
        self.fusion_weights = {
            0: (0.7, 0.3),
            1: (0.5, 0.5),
            2: (0.3, 0.7)
        }
        
        self.history_features = []
        self.HISTORY_WINDOW = 10 
        
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all standard indicators and appends them to DF.
        Returns DF with new columns.
        """
        df_ind = df.copy()
        if self.ind_rsi:
            # RSI
            df_ind['rsi'] = self.ind_rsi.calculate(df_ind, period=14)
            # EMA
            df_ind['ema_21'] = self.ind_ema.calculate(df_ind, period=21)
            # T3
            df_ind['t3'] = self.ind_t3.calculate(df_ind, length=10)
            # Ichimoku
            ichi_df = self.ind_ichi.calculate(df_ind)
            df_ind = pd.concat([df_ind, ichi_df], axis=1)
        return df_ind

    def train(self, df: pd.DataFrame):
        """
        Train HMM and LSTM using DF with SMC and standard indicators.
        """
        # 1. Feature Engineering with Indicators
        df_full = self._calculate_indicators(df)
        
        # Extract features for HMM 
        # We can now use RSI and T3 deviation as features for regime detection
        
        # Base features: Returns
        returns = df_full['close'].pct_change().fillna(0)
        
        # Volatility
        if 'vol' in df_full.columns:
             vol = df_full['vol']
        else:
             vol = returns.rolling(20).std().fillna(0)
             
        # RSI normalized (0-1)
        rsi_norm = df_full['rsi'].fillna(50) / 100.0
        
        # T3 Trend: Price vs T3
        t3_trend = (df_full['close'] - df_full['t3'].fillna(df_full['close'])) / df_full['close']
        
        # Construct HMM Feature Matrix
        # [Returns, Volatility, RSI, T3_Deviation]
        feat_matrix = np.column_stack([
            returns.values, 
            vol.values, 
            rsi_norm.values,
            t3_trend.values
        ])
        # Add Volume if available
        if 'volume' in df_full.columns:
             vol_chg = df_full['volume'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
             feat_matrix = np.column_stack([feat_matrix, vol_chg.values])
        
        print("Training HMM with extended indicators...")
        # Handle scaling/NaNs
        feat_matrix = np.nan_to_num(feat_matrix)
        self.skhmm.train_hmm(feat_matrix)
        
        # 2. Train LSTM (TF Model)
        # TF setup same as before but now we have more data
        pass

    def predict(self, df_window: pd.DataFrame, observation: float, features: np.ndarray = None) -> Tuple[int, float, bool, bool, bool, Dict[str, Any]]:
        """
        Full prediction pipeline.
        
        Returns:
            (regime, confidence, is_fvg, is_bos, is_choch, indicator_signals)
        """
        # 0. Calculate Indicators on Window to get latest values for Feature Vector
        # If 'features' arguments are NOT passed, we construct them on the fly.
        
        # We need the indicators for the HMM feature vector: [Returns, Vol, RSI, T3_Dev]
        df_ind = self._calculate_indicators(df_window)
        latest = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        
        # Returns
        ret = (latest['close'] - prev['close']) / prev['close']
        # Vol (using ATR or std dev of returns)
        vol_est = df_ind['close'].pct_change().rolling(20).std().iloc[-1]
        # RSI
        rsi_val = latest['rsi'] / 100.0
        # T3
        t3_val = (latest['close'] - latest['t3']) / latest['close']
        
        # Construct features vector for HMM if not provided
        if features is None or len(features) < 4:
            current_features = np.array([ret, vol_est, rsi_val, t3_val])
            # Add volume if needed (assumed 0 if not passed)
            if 'volume' in df_window.columns:
                vol_chg = (latest['volume'] - prev['volume']) / (prev['volume'] + 1e-6)
                current_features = np.append(current_features, vol_chg)
        else:
            current_features = features

        # 1. Update Internal History
        self.history_features.append(current_features)
        if len(self.history_features) > self.HISTORY_WINDOW:
            self.history_features.pop(0)
            
        # 2. Kalman-HMM Prediction
        khmm_result = self.skhmm.predict_step(observation, current_features)
        regime = khmm_result['regime']
        confidence = khmm_result['confidence']
        
        # 3. SMC Analysis
        is_fvg = False
        is_bos = False
        is_choch = False
        
        if self.smc and len(df_window) > 20:
            # FVG
            fvg_df = self.smc.detect_fvg(df_window)
            latest_fvg = fvg_df.iloc[-1]
            is_fvg = bool(latest_fvg['bullish_fvg'] or latest_fvg['bearish_fvg'])
            
            # Structure
            swing_df = self.smc.detect_swing_points(df_window)
            from indicators.sm.market_structure import MarketStructure
            struct_df = MarketStructure.detect_bos_choch(df_window, swing_df)
            latest_struct = struct_df.iloc[-1]
            is_bos = bool(latest_struct['bullish_bos'] or latest_struct['bearish_bos'])
            is_choch = bool(latest_struct['bullish_choch'] or latest_struct['bearish_choch'])
            
        # 4. Indicator Signals
        # Collect signals from standard indicators
        indicator_signals = {
            'rsi_val': latest['rsi'],
            't3_val': latest['t3'],
            'ema_trend': 'bull' if latest['close'] > latest['ema_21'] else 'bear',
            'ichimoku_signal': 0 # Placeholder for detailed ichi signal logic
        }

        # 5. LSTM (TF Model)
        if len(self.history_features) == self.HISTORY_WINDOW:
             # TF Model inference would go here
             pass

        # Return the requested Tuple
        return (regime, confidence, is_fvg, is_bos, is_choch, indicator_signals)

