import ccxt
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from .base_fetcher import BaseDataFetcher
import asyncio

class BinanceFetcher(BaseDataFetcher):
    """Fetch data from Binance exchange"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange = ccxt.binance({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} if config.get('futures') else {}
        })
        
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1h',
        start_date: datetime = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            since = int(start_date.timestamp() * 1000) if start_date else None
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since,
                limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            raise Exception(f"Error fetching data from Binance: {e}")
            
    async def fetch_realtime(self, symbol: str) -> Dict:
        """Fetch real-time ticker data"""
        try:
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['quoteVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
        except Exception as e:
            raise Exception(f"Error fetching real-time data: {e}"):
