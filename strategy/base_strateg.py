from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Returns:
            Series with 1 (buy), -1 (sell), 0 (hold)
        """
        pass
        
    @abstractmethod
    def calculate_position_size(
        self,
        signal: int,
        price: float,
        portfolio_value: float,
        risk_params: Dict
    ) -> float:
        """Calculate position size based on signal and risk parameters"""
        pass
        
    def validate_signal(self, signal: int, df: pd.DataFrame, idx: int) -> bool:
        """Validate signal against basic conditions"""
        # Implement basic validation logic
        return signal != 0
        
    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calculate stop loss price"""
        multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)
        
        if side == 'long':
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)
            
    def get_take_profit(self, entry_price: float, stop_loss: float, side: str) -> float:
        """Calculate take profit price"""
        multiplier = self.config.get('take_profit_multiplier', 3.0)
        risk = abs(entry_price - stop_loss)
        
        if side == 'long':
            return entry_price + (risk * multiplier)
        else:
            return entry_price - (risk * multiplier)
