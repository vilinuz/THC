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
        
        self.metrics = {
            'obi': {}, # Order Book Imbalance
            'velocity': {}, # Trade Velocity
            'liquidations': {} # Liquidation Volume
        }

    async def _custom_l2_callback(self, book, receipt_timestamp):
        """Calculate Order Book Imbalance (OBI)"""
        try:
            # Calculate Imbalance on top 10 levels
            # book.bids and book.asks are dicts {price: size}
            # Need to handle different book variants, but standard L2 is {price: size}
            
            # Helper to sum top N
            def sum_top_n(d, n=10):
                return sum(list(d.values())[:n])

            bid_vol = sum_top_n(book.bids)
            ask_vol = sum_top_n(book.asks)
            
            if bid_vol + ask_vol > 0:
                obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                self.metrics['obi'][book.symbol] = obi
                
        except Exception as e:
            # print(f"OBI calc error: {e}")
            pass
            
        # Forward to Redis if configured
        if self.redis_backends.get(L2_BOOK):
            await self.redis_backends[L2_BOOK](book, receipt_timestamp)

    async def _custom_trade_callback(self, trade, receipt_timestamp):
        """Calculate Tick Velocity / VWAP"""
        # trade is a Trade object or dict
        # Logic: Accumulate volume/price for velocity
        
        # Forward to Redis
        if self.redis_backends.get(TRADES):
            await self.redis_backends[TRADES](trade, receipt_timestamp)

    async def _custom_liquidation_callback(self, liquidation, receipt_timestamp):
        """Detect Liquidations"""
        # Forward to Redis
        if self.redis_backends.get(LIQUIDATIONS):
            await self.redis_backends[LIQUIDATIONS](liquidation, receipt_timestamp)

    def start(self):
        """Start the Feed Handler"""
        # Initialize Redis Backends directly to use in custom callbacks
        self.redis_backends = self._get_redis_callbacks()
        pg_callbacks = self._get_postgres_callbacks()
        
        # Prepare Combined Callbacks
        # We use our Custom Callbacks as the PRIMARY entry point for L2, Trades, Liq
        # They will forward to Redis internally.
        # For others (Ticker, etc), we use Redis directly.
        
        callbacks = {
            L2_BOOK: self._custom_l2_callback,
            TRADES: self._custom_trade_callback,
            LIQUIDATIONS: self._custom_liquidation_callback,
            TICKER: self.redis_backends[TICKER],
            CANDLES: self.redis_backends[CANDLES],
            FUNDING: self.redis_backends[FUNDING],
            OPEN_INTEREST: self.redis_backends[OPEN_INTEREST]
        }
        
        # Add Postgres to the mix (Cryptofeed allows list of callbacks)
        # But since we use methods for the main ones, we need to handle list logic manually or 
        # use Cryptofeed's native support for lists.
        # Simplest: Wrapper calls Redis, then we let FeedHandler call Postgres if we pass a list?
        # Limit: add_feed callbacks arg takes {CHANNEL: [cb1, cb2]}
        
        final_callbacks = {}
        for chan, main_cb in callbacks.items():
            final_callbacks[chan] = [main_cb]
            # Add PG if exists
            if chan in pg_callbacks:
                final_callbacks[chan].append(pg_callbacks[chan])
        
        print("Starting CryptoFeed Ingestion...")
        print(f"Subscribing to {len(self.pairs)} Spot pairs and {len(self.futures_pairs)} Futures pairs.")
        
        # Prepare Exchange Config
        exchange_config = {}
        if self.config.get('binance_api_key') and self.config.get('binance_api_secret'):
            exchange_config = {
                'key_id': self.config['binance_api_key'],
                'key_secret': self.config['binance_api_secret']
            }
            print("Binance Credentials loaded.")
        
        # 1. Spot Data (Binance)
        try:
             self.fh.add_feed(BINANCE, channels=[L2_BOOK, TRADES, TICKER, CANDLES], 
                              symbols=self.pairs, callbacks=final_callbacks,
                              config=exchange_config)
        except Exception as e:
             print(f"Error adding Spot feed: {e}")

        # 2. Futures/Perp Data (Binance Futures)
        futures_chans = [TRADES, TICKER, FUNDING, OPEN_INTEREST, LIQUIDATIONS]
        
        try:
             self.fh.add_feed(BINANCE_FUTURES, channels=futures_chans, 
                              symbols=self.futures_pairs, callbacks=final_callbacks,
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
