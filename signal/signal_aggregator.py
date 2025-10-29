"""
Aggregate signals from multiple sources
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class SignalAggregator:
    """Aggregate signals from multiple strategies and indicators"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {}
        
    def aggregate(
        self,
        signals: Dict[str, pd.Series],
        method: str = 'weighted_average'
    ) -> pd.Series:
        """
        Aggregate multiple signal sources
        
        Args:
            signals: Dict of signal name -> signal series
            method: 'weighted_average', 'majority_vote', 'unanimous'
        """
        if method == 'weighted_average':
            return self._weighted_average(signals)
        elif method == 'majority_vote':
            return self._majority_vote(signals)
        elif method == 'unanimous':
            return self._unanimous(signals)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    def _weighted_average(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Weighted average of signals"""
        # Align all signals
        df = pd.DataFrame(signals)
        
        # Apply weights
        if self.weights:
            for name in df.columns:
                df[name] = df[name] * self.weights.get(name, 1.0)
                
        # Calculate weighted average
        total_weight = sum(self.weights.get(name, 1.0) for name in df.columns)
        aggregated = df.sum(axis=1) / total_weight
        
        # Convert to discrete signals
        result = pd.Series(0, index=aggregated.index)
        result[aggregated > 0.5] = 1
        result[aggregated < -0.5] = -1
        
        return result
        
    def _majority_vote(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Majority voting"""
        df = pd.DataFrame(signals)
        
        # Count votes
        buy_votes = (df == 1).sum(axis=1)
        sell_votes = (df == -1).sum(axis=1)
        total_signals = len(df.columns)
        
        # Majority decision
        result = pd.Series(0, index=df.index)
        result[buy_votes > total_signals / 2] = 1
        result[sell_votes > total_signals / 2] = -1
        
        return result
        
    def _unanimous(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Require unanimous agreement"""
        df = pd.DataFrame(signals)
        
        result = pd.Series(0, index=df.index)
        result[(df == 1).all(axis=1)] = 1
        result[(df == -1).all(axis=1)] = -1
        
        return result
