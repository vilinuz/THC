import asyncio
from typing import List, Dict, Optional
from cryptofeed import FeedHandler
from cryptofeed.defines import (
    L2_BOOK, L3_BOOK, TRADES, TICKER, CANDLES, 
    FUNDING, OPEN_INTEREST, LIQUIDATIONS, INDEX,
    BINANCE, BINANCE_FUTURES, COINBASE
)
from cryptofeed.backends.redis import BookRedis, TradeRedis, TickerRedis, CandleRedis, FundingRedis, OpenInterestRedis, LiquidationsRedis
# Postgres backends are usually custom or generic in newer cryptofeed versions, 
# checking typical usage. Usually 'Postgres' backend class exists or Generic.
# For safety, we will assume generic integration or use available ones.
# In recent cryptofeed, backend logic is standardized.

try:
    from cryptofeed.backends.postgres import BookPostgres, TradePostgres, TickerPostgres, CandlePostgres, FundingPostgres, OpenInterestPostgres, LiquidationsPostgres
except ImportError:
    # Fallback or updated path
    print("Warning: Specific Postgres backends not found. Using generic callback structure if needed.")
    BookPostgres = None

class CryptoFeedManager:
    """
    Manages real-time data ingestion from Exchanges into Redis and Postgres.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.fh = FeedHandler()
        self.redis_host = config.get('redis_host', 'localhost')
        self.redis_port = config.get('redis_port', 6379)
        self.postgres_dsn = config.get('postgres_dsn', None)
        
        # Pairs to subscribe to
        self.pairs = config.get('pairs', ['BTC-USDT', 'ETH-USDT'])
        self.futures_pairs = config.get('futures_pairs', ['BTC-USDT-PERP', 'ETH-USDT-PERP'])
        
    def _get_redis_callbacks(self):
        """Configure Redis Backend Callbacks"""
        # We write latest state to Redis for Real-time access
        return {
            L2_BOOK: BookRedis(host=self.redis_host, port=self.redis_port, snapshots_only=False),
            TRADES: TradeRedis(host=self.redis_host, port=self.redis_port),
            TICKER: TickerRedis(host=self.redis_host, port=self.redis_port),
            CANDLES: CandleRedis(host=self.redis_host, port=self.redis_port),
            FUNDING: FundingRedis(host=self.redis_host, port=self.redis_port),
            OPEN_INTEREST: OpenInterestRedis(host=self.redis_host, port=self.redis_port),
            LIQUIDATIONS: LiquidationsRedis(host=self.redis_host, port=self.redis_port)
        }

    def _get_postgres_callbacks(self):
        """Configure Postgres Backend Callbacks (Archival)"""
        if not self.postgres_dsn:
            return {}
            
        # We log specific events to DB
        callbacks = {}
        if BookPostgres:
             # Typically we don't log full L2/L3 books to SQL rows unless using JSONB
             # Skipping L2/L3 for Postgres to save space, usually snapshot or diffs
             callbacks[TRADES] = TradePostgres(self.postgres_dsn)
             callbacks[CANDLES] = CandlePostgres(self.postgres_dsn)
             # Derivatives
             callbacks[FUNDING] = FundingPostgres(self.postgres_dsn)
             callbacks[LIQUIDATIONS] = LiquidationsPostgres(self.postgres_dsn)
        return callbacks

    def start(self):
        """Start the Feed Handler"""
        callbacks = self._get_redis_callbacks()
        pg_callbacks = self._get_postgres_callbacks()
        
        # Merge callbacks (Cryptofeed supports list of callbacks)
        # We need to structure it: {L2_BOOK: [RedisCallback, PB_Callback], ...}
        
        combined_callbacks = {}
        
        # Helper to add
        def add_cb(channel, cb):
            if channel not in combined_callbacks: combined_callbacks[channel] = []
            combined_callbacks[channel].append(cb)

        for chan, cb in callbacks.items(): add_cb(chan, cb)
        for chan, cb in pg_callbacks.items(): add_cb(chan, cb)
        
        print("Starting CryptoFeed Ingestion...")
        print(f"Subscribing to {len(self.pairs)} Spot pairs and {len(self.futures_pairs)} Futures pairs.")
        
        # Prepare Exchange Config
        # Cryptofeed expects a specific structure for auth
        exchange_config = {}
        if self.config.get('binance_api_key') and self.config.get('binance_api_secret'):
            exchange_config = {
                'key_id': self.config['binance_api_key'],
                'key_secret': self.config['binance_api_secret']
            }
            print("Binance Credentials loaded.")
        
        # 1. Spot Data (Binance)
        # L2, Trades, Ticker, Candles
        try:
             self.fh.add_feed(BINANCE, channels=[L2_BOOK, TRADES, TICKER, CANDLES], 
                              symbols=self.pairs, callbacks=combined_callbacks,
                              config=exchange_config)
        except Exception as e:
             print(f"Error adding Spot feed: {e}")

        # 2. Futures/Perp Data (Binance Futures)
        # Funding, OI, Liquidations, Index, plus standard
        futures_chans = [TRADES, TICKER, FUNDING, OPEN_INTEREST, LIQUIDATIONS]
        # Binance Futures supports CANDLES too
        # INDEX might be specific symbol format
        
        try:
             self.fh.add_feed(BINANCE_FUTURES, channels=futures_chans, 
                              symbols=self.futures_pairs, callbacks=combined_callbacks,
                              config=exchange_config)
        except Exception as e:
             print(f"Error adding Futures feed: {e}")

        # L3 Book generally Coinbase
        # if 'COINBASE' in config...
        
        self.fh.run()

if __name__ == "__main__":
    import yaml
    import os
    
    # Load config.yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    conf = {
        'redis_host': 'localhost',
        'pairs': ['BTC-USDT', 'ETH-USDT'],
        'futures_pairs': ['BTC-USDT-PERP'],
    }
    
    try:
        with open(config_path, 'r') as f:
            full_conf = yaml.safe_load(f)
            if 'data_ingestion' in full_conf:
                conf.update(full_conf['data_ingestion'])
            # Also check redis settings in top level cache if missing
            if 'redis_host' not in full_conf.get('data_ingestion', {}):
                 conf['redis_host'] = full_conf.get('cache', {}).get('host', 'localhost')
            
    except Exception as e:
        print(f"Warning: Could not load config.yaml ({e}), using defaults.")

    manager = CryptoFeedManager(conf)
    manager.start()
