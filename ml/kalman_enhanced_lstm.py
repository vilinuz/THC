import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
import numpy as np
import pandas as pd

class KalmanEnhancedLSTM:
    """
    LSTM that ingests:
    1. Kalman-filtered price (smoothed signal) [web:5]
    2. Kalman velocity/acceleration (momentum features)
    3. HMM regime probabilities
    4. SMC structural indicators (OB, FVG distances)
    5. Sentiment score
    """
    
    def __init__(self, lookback=100, n_regimes=3):
        self.lookback = lookback
        self.n_regimes = n_regimes
        self.model = self._build_architecture()
        
    def _build_architecture(self):
        """
        Multi-input LSTM with separate heads for different feature types.
        """
        # Input 1: Kalman-Filtered Time Series (Price, Vel, Accel, Vol)
        kalman_input = Input(shape=(self.lookback, 4), name='kalman_series')
        kalman_lstm = LSTM(128, return_sequences=True)(kalman_input)
        kalman_lstm = Dropout(0.2)(kalman_lstm)
        kalman_lstm = LSTM(64)(kalman_lstm)
        
        # Input 2: SMC Structural Features (Distance to OB, FVG flags)
        smc_input = Input(shape=(self.lookback, 3), name='smc_features')
        smc_lstm = LSTM(32)(smc_input)
        
        # Input 3: HMM Regime Probabilities (Current snapshot)
        regime_input = Input(shape=(self.n_regimes,), name='regime_probs')
        regime_dense = Dense(16, activation='relu')(regime_input)
        
        # Input 4: Sentiment Score (Scalar)
        sentiment_input = Input(shape=(1,), name='sentiment')
        
        # Fusion Layer
        concatenated = Concatenate()([
            kalman_lstm,
            smc_lstm,
            regime_dense,
            sentiment_input
        ])
        
        # Deep Fusion
        fusion = Dense(128, activation='relu')(concatenated)
        fusion = Dropout(0.3)(fusion)
        fusion = Dense(64, activation='relu')(fusion)
        
        # Output: Next candle close price
        output = Dense(1, activation='linear', name='price_prediction')(fusion)
        
        model = Model(
            inputs=[kalman_input, smc_input, regime_input, sentiment_input],
            outputs=output
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='huber',  # Robust to outliers [web:8]
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_training_data(self, df: pd.DataFrame, kalman_predictions: dict, smc_df: pd.DataFrame, sentiment_df: pd.DataFrame):
        """
        Constructs the multi-input training dataset.
        
        Args:
            df: Raw OHLCV DataFrame
            kalman_predictions: Output from SwitchingKalmanHMM history or similar dict
                                Expects keys: 'price', 'velocity', 'volatility', 'regime_probs'
                                (as lists or arrays aligned with df)
            smc_df: SMC Engine output (FVG/OB distances)
                    Expects cols: 'dist_ob_bull', 'dist_ob_bear', 'in_fvg'
            sentiment_df: Sentiment scores timeline
                          Expects col: 'sentiment_score'
        """
        # Note: We need to ensure alignment. This assumes all DFs have same index or length.
        
        # Create Acceleration from Velocity if not present
        if 'velocity' not in kalman_predictions:
             raise ValueError("kalman_predictions must contain 'velocity'")
        
        velocity = np.array(kalman_predictions['velocity'])
        acceleration = np.gradient(velocity)
        
        # Kalman features: [Filtered Price, Velocity, Accel, Vol]
        price_arr = np.array(kalman_predictions['price'])
        velocity_arr = velocity
        accel_arr = acceleration
        vol_arr = np.array(kalman_predictions['volatility'])
        
        kalman_features = np.column_stack([
            price_arr,
            velocity_arr,
            accel_arr,
            vol_arr
        ])
        
        # SMC features: [Dist_to_Bull_OB, Dist_to_Bear_OB, In_FVG_Flag]
        # In case smc_df has different length, we take matching tail or head.
        # Assuming aligned for now.
        smc_features = smc_df[['dist_ob_bull', 'dist_ob_bear', 'in_fvg']].values
        
        # Create rolling windows
        X_kalman, X_smc, X_regime, X_sentiment, y = [], [], [], [], []
        
        regime_probs = np.array(kalman_predictions['regime_probs']) # shape (N, n_regimes)
        sentiment_scores = sentiment_df['sentiment_score'].values
        target_close = df['close'].values
        
        min_len = min(len(df), len(kalman_features), len(smc_features), len(sentiment_scores))
        
        # We start from 'lookback' index up to min_len
        for i in range(self.lookback, min_len):
            X_kalman.append(kalman_features[i-self.lookback:i])
            X_smc.append(smc_features[i-self.lookback:i])
            X_regime.append(regime_probs[i]) # Current regime state at prediction time
            X_sentiment.append(sentiment_scores[i]) # Current sentiment
            y.append(target_close[i])  # Target: actual next close (wait, shift?)
            # Usually LSTM predicts t+1 using 0..t. 
            # If target_close[i] is price at time t, we are predicting current price?
            # Or is 'df' shifted? Standard: Predict next step.
            # If i is current time, inputs represent t-lookback..t-1. 
            # If we want to predict t, we use inputs ending at t-1.
            # The slicing [i-self.lookback : i] gives indices i-lookback ... i-1. (Length lookback).
            # So X ends at t-1.
            # y uses index i, which is time t.
            # So this predicts Close(t) using history up to t-1. Correct.
        
        return {
            'kalman_series': np.array(X_kalman),
            'smc_features': np.array(X_smc),
            'regime_probs': np.array(X_regime),
            'sentiment': np.array(X_sentiment).reshape(-1, 1)
        }, np.array(y)
