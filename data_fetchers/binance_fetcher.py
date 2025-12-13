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
        """Fetch OHLCV data from Binance with pagination"""
        try:
            all_ohlcv = []
            since = int(start_date.timestamp() * 1000) if start_date else None
            end_ts = int(end_date.timestamp() * 1000) if end_date else int(datetime.now().timestamp() * 1000)
            
            while True:
                # Calculate limit for this batch if near end_date to avoid fetching excessive future data? 
                # CCXT usually handles it, but let's stick to standard limit
                
                # Check if we reached the end
                if since and since >= end_ts:
                    break
                    
                batch = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv,
                    symbol,
                    timeframe,
                    since,
                    limit
                )
                
                if not batch:
                    break
                    
                all_ohlcv.extend(batch)
                
                # Update since to the timestamp of the last candle + 1ms
                last_ts = batch[-1][0]
                since = last_ts + 1
                
                # If we got fewer candles than limit, we likely reached the end
                if len(batch) < limit:
                    break
                    
                # Rate limit is handled by ccxt enableRateLimit=True
            
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates(subset=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Filter by end_date just in case
            if end_date:
                df = df[df.index <= end_date]
            
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
            raise Exception(f"Error fetching real-time data: {e}")
