
import unittest
import sys
import os
sys.path.append('/home/vilivom/src/THC')
from unittest.mock import MagicMock, AsyncMock

# Mock cryptofeed modules before importing feed_manager
sys.modules['cryptofeed'] = MagicMock()
sys.modules['cryptofeed.defines'] = MagicMock()
sys.modules['cryptofeed.backends.redis'] = MagicMock()
sys.modules['cryptofeed.backends.postgres'] = MagicMock()

# Define defines
L2_BOOK = 'L2_BOOK'
TRADES = 'TRADES'
LIQUIDATIONS = 'LIQUIDATIONS'
TICKER = 'TICKER'
CANDLES = 'CANDLES'
FUNDING = 'FUNDING'
OPEN_INTEREST = 'OPEN_INTEREST'
BINANCE = 'BINANCE'
BINANCE_FUTURES = 'BINANCE_FUTURES'

sys.modules['cryptofeed.defines'].L2_BOOK = L2_BOOK
sys.modules['cryptofeed.defines'].TRADES = TRADES
sys.modules['cryptofeed.defines'].LIQUIDATIONS = LIQUIDATIONS
sys.modules['cryptofeed.defines'].TICKER = TICKER
sys.modules['cryptofeed.defines'].CANDLES = CANDLES
sys.modules['cryptofeed.defines'].FUNDING = FUNDING
sys.modules['cryptofeed.defines'].OPEN_INTEREST = OPEN_INTEREST
sys.modules['cryptofeed.defines'].BINANCE = BINANCE
sys.modules['cryptofeed.defines'].BINANCE_FUTURES = BINANCE_FUTURES

# Now import manager
from data_ingestion.feed_manager import CryptoFeedManager

class TestFeedManager(unittest.IsolatedAsyncioTestCase):
    async def test_obi_calculation(self):
        config = {
            'redis_host': 'localhost',
            'pairs': ['BTC-USDT']
        }
        manager = CryptoFeedManager(config)
        # Manually init metrics since start() isn't called
        manager.metrics = {'obi': {}, 'velocity': {}, 'liquidations': {}}
        manager.redis_backends = {} # No redis
        
        # Mock Book object
        book = MagicMock()
        book.symbol = 'BTC-USDT'
        # bids: {price: size}
        book.bids = {100: 1.0, 99: 2.0} # Total 3.0
        book.asks = {101: 1.0, 102: 0.5} # Total 1.5
        
        # Expected OBI: (3.0 - 1.5) / (3.0 + 1.5) = 1.5 / 4.5 = 0.333...
        
        await manager._custom_l2_callback(book, 1234567890)
        
        self.assertIn('BTC-USDT', manager.metrics['obi'])
        self.assertAlmostEqual(manager.metrics['obi']['BTC-USDT'], 1.5/4.5)
        print(f"Calculated OBI: {manager.metrics['obi']['BTC-USDT']}")

if __name__ == '__main__':
    unittest.main()
