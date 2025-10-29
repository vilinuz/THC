import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit

class XGBoostModel:
    """XGBoost model for trading signals"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'objective': 'binary:logistic',
            'max_depth': 7,
            'learning_rate': 0.01,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc'
        }
        self.model = None
        
    def prepare_labels(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
        """
        Create labels for classification
        
        Args:
            horizon: Periods ahead to predict
            threshold: Minimum return to be considered as positive class
        """
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        labels = (future_returns > threshold).astype(int)
        return labels
        
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """Train XGBoost model"""
        # Time-based split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'model': self.model,
            'feature_importance': feature_importance,
            'train_score': self.model.score(X_train, y_train),
            'val_score': self.model.score(X_val, y_val)
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def generate_signals(
        self,
        X: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.Series:
        """
        Generate trading signals based on model predictions

        Returns:
            Series with 1 (buy), -1 (sell), 0 (hold)
        """
        probas = self.predict_proba(X)[:, 1]  # Probability of positive class

        signals = pd.Series(0, index=X.index)
        signals[probas > threshold] = 1
        signals[probas < (1 - threshold)] = -1

        return signals
