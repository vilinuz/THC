from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    @abstractmethod
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data"""
        pass
        
    @abstractmethod
    async def fetch_realtime(self, symbol: str) -> Dict:
        """Fetch real-time price data"""
        pass
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate fetched data"""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_cols)
