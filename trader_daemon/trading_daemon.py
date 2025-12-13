"""
Trading daemon for continuous operation
"""
import asyncio
import signal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingDaemon:
    """Trading bot daemon"""
    
    def __init__(self, config: dict, platform):
        self.config = config
        self.platform = platform
        self.running = False
        self.check_interval = config['daemon']['check_interval']
        
    async def start(self):
        """Start daemon"""
        self.running = True
        logger.info("Trading daemon started")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                await self._trading_loop()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)
        finally:
            await self.stop()
            
    async def _trading_loop(self):
        """Main trading loop"""
        logger.info(f"Running trading loop at {datetime.now()}")
        
        # Fetch latest data
        for asset in self.config['assets']:
            if asset['enabled']:
                await self.platform.fetch_and_store_data(
                    symbol=asset['symbol'],
                    timeframe='1h',
                    days=1  # Only fetch latest data
                )
                
        # Generate signals
        # Execute trades (if live trading enabled)
        # Update reports
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    async def stop(self):
        """Stop daemon gracefully"""
        logger.info("Trading daemon stopped")
        self.running = False
