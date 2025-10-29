 import pandas as pd
 import numpy as np
 from typing import List, Dict
 
 class FeatureEngineer:
     """Create features for machine learning models"""
     
     @staticmethod
     def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
         """Create price-based features"""
         features = pd.DataFrame(index=df.index)
         
         # Returns
         features['returns'] = df['close'].pct_change()
         features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
         
         # Price changes
         features['high_low_ratio'] = df['high'] / df['low']
         features['close_open_ratio'] = df['close'] / df['open']
         
         # Rolling statistics
         for window in [5, 10, 20, 50]:
             features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
             features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
             features[f'price_momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
             
         return features
         
     @staticmethod
     def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
         """Create volume-based features"""
         features = pd.DataFrame(index=df.index)
         
         # Volume changes
         features['volume_change'] = df['volume'].pct_change()
         features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
         
         # Price-volume features
         features['pv_trend'] = df['close'].pct_change() * df['volume']
         
         # Rolling volume statistics
         for window in [5, 10, 20]:
             features[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
              features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
 
         return features
 
     @staticmethod
     def create_technical_features(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
         """Create features from technical indicators"""
         features = pd.DataFrame(index=df.index)
 
         # Add all indicator values as features
         for name, values in indicators.items():
             if isinstance(values, pd.Series):
                 features[f'ind_{name}'] = values
             elif isinstance(values, pd.DataFrame):
                 for col in values.columns:
                     features[f'ind_{name}_{col}'] = values[col]
 
         return features
 
     @staticmethod
     def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
         """Create time-based features"""
         features = pd.DataFrame(index=df.index)
 
         # Extract time components
         features['hour'] = df.index.hour
         features['day_of_week'] = df.index.dayofweek
         features['day_of_month'] = df.index.day
         features['month'] = df.index.month
 
         # Cyclical encoding
         features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
         features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
         features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
         features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
 
         return features
 
     @staticmethod
     def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
         """Create lagged features"""
         features = pd.DataFrame(index=df.index)
 
          for col in columns:
             for lag in lags:
                 features[f'{col}_lag_{lag}'] = df[col].shift(lag)
 
         return features
 
     @staticmethod
     def combine_all_features(
         df: pd.DataFrame,
         indicators: Dict,
         include_lags: bool = True
     ) -> pd.DataFrame:
         """Combine all feature types"""
         all_features = pd.DataFrame(index=df.index)
 
         # Add base OHLCV
         all_features = pd.concat([all_features, df[['open', 'high', 'low', 'close', 'volume']]], axis=1)
 
         # Add all feature types
         all_features = pd.concat([
             all_features,
             FeatureEngineer.create_price_features(df),
             FeatureEngineer.create_volume_features(df),
             FeatureEngineer.create_technical_features(df, indicators),
             FeatureEngineer.create_time_features(df)
         ], axis=1)
 
         if include_lags:
             lag_features = FeatureEngineer.create_lag_features(
                 all_features,
                 ['returns', 'volume_change'],
                 lags=[1, 2, 3, 5, 10]
             )
             all_features = pd.concat([all_features, lag_features], axis=1)
 
         return all_features.dropna()
