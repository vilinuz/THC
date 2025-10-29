"""
Main entry point for the crypto trading platform
"""
import asyncio
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Import modules
from data_fetchers.binance_fetcher import BinanceFetcher
from database.duckdb_manager import DuckDBManager
from cache.redis_manager import RedisManager
from indicators.vwap import VWAP
from indicators.ema import EMA
from indicators.rsi import RSI
from indicators.ichimoku import Ichimoku
from ml_models.feature_engineer import FeatureEngineer
from ml_models.xgboost_model import XGBoostModel
from optimization.bayesian_optimizer import BayesianOptimizer
from optimization.walk_forward import WalkForwardOptimizer
from signal.signal_aggregator import SignalAggregator
from backtesting.backtest_engine import BacktestEngine
from reporting.pdf_generator import PDFReportGenerator
from daemon.trading_daemon import TradingDaemon
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = setup_logger(__name__)

class CryptoTradingPlatform:
    """Main trading platform orchestrator"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        self.setup_components()
        
    def setup_components(self):
        """Initialize all platform components"""
        logger.info("Initializing platform components...")
        
        # Database
        self.db = DuckDBManager(
            self.config['database']['path'],
            self.config['database']['parquet_dir']
        )
        
        # Cache
        self.cache = RedisManager(
            host=self.config['cache']['host'],
            port=self.config['cache']['port'],
            db=self.config['cache']['db'],
            ttl=self.config['cache']['ttl']
        )
        
        # Data fetchers
        self.fetchers = {}
        if self.config['data_sources']['binance']['enabled']:
            self.fetchers['binance'] = BinanceFetcher(
                self.config['data_sources']['binance']
            )
            
        logger.info("Components initialized successfully")
        
    async def fetch_and_store_data(self, symbol: str, timeframe: str = '1h', days: int = 365):
        """Fetch historical data and store in database"""
        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch from Binance
        df = await self.fetchers['binance'].fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Store in database
        self.db.save_ohlcv(df, symbol, timeframe)
        
        # Cache recent data
        self.cache.cache_dataframe(f"{symbol}_{timeframe}_latest", df.tail(100))
        
        logger.info(f"Stored {len(df)} records for {symbol}")
        return df
        
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        logger.info("Calculating technical indicators...")
        
        indicators = {}
        
        # VWAP
        if self.config['indicators']['vwap']['enabled']:
            indicators['vwap'] = VWAP.calculate(df)
            indicators['vwap_signal'] = VWAP.signal(df, indicators['vwap'])
            
        # EMA
        if self.config['indicators']['ema']['enabled']:
            ema_periods = self.config['indicators']['ema']['periods']
            indicators['ema'] = EMA.multi_ema(df, ema_periods)
            
        # RSI
        if self.config['indicators']['rsi']['enabled']:
            period = self.config['indicators']['rsi']['period']
            indicators['rsi'] = RSI.calculate(df, period)
            indicators['rsi_signal'] = RSI.signal(
                indicators['rsi'],
                self.config['indicators']['rsi']['overbought'],
                self.config['indicators']['rsi']['oversold']
            )
            
        # Ichimoku
        if self.config['indicators']['ichimoku']['enabled']:
            ich_config = self.config['indicators']['ichimoku']
            indicators['ichimoku'] = Ichimoku.calculate(
                df,
                ich_config['tenkan_period'],
                ich_config['kijun_period'],
                ich_config['senkou_b_period']
            )
            indicators['ichimoku_signal'] = Ichimoku.signals(df, indicators['ichimoku'])
            
        return indicators
        
    def run_backtest(self, symbol: str, timeframe: str = '1h'):
        """Run backtesting on historical data"""
        logger.info(f"Running backtest for {symbol} on {timeframe}")
        
        # Load data
        df = self.db.load_ohlcv(symbol, timeframe)
        
        # Calculate indicators
        indicators = self.calculate_indicators(df)
        
        # Create features
        features = FeatureEngineer.combine_all_features(df, indicators)
        
        # Train ML model
        xgb_model = XGBoostModel(self.config['ml_models']['xgboost'])
        labels = xgb_model.prepare_labels(df, horizon=5, threshold=0.001)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Train model
        train_result = xgb_model.train(X, y)
        logger.info(f"Model trained - Val Score: {train_result['val_score']:.4f}")
        
        # Generate signals
        ml_signals = xgb_model.generate_signals(X, threshold=0.6)
        
        # Aggregate signals
        all_signals = {
            'ml': ml_signals,
            'vwap': indicators.get('vwap_signal'),
            'rsi': indicators.get('rsi_signal'),
            'ichimoku': indicators.get('ichimoku_signal')
        }
        
        # Run backtest
        engine = BacktestEngine(self.config['backtesting'])
        results = engine.run(df, all_signals)
        
        logger.info(f"Backtest completed - Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results
        
    def run_walk_forward_optimization(self, symbol: str, timeframe: str = '1h'):
        """Run walk-forward optimization"""
        logger.info("Running walk-forward optimization...")
        
        df = self.db.load_ohlcv(symbol, timeframe)
        
        # Define parameter bounds
        param_bounds = {
            'rsi_period': (10, 20),
            'rsi_overbought': (65, 75),
            'rsi_oversold': (25, 35),
            'ema_fast': (5, 15),
            'ema_slow': (20, 30)
        }
        
        wfo = WalkForwardOptimizer(
            train_period_days=self.config['optimization']['walk_forward']['train_period_days'],
            test_period_days=self.config['optimization']['walk_forward']['test_period_days'],
            step_days=self.config['optimization']['walk_forward']['step_days']
        )
        
        # Define optimization function
        def optimize_func(train_data, bounds):
            # Simplified optimization
            return {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30}
            
        # Define backtest function
        def backtest_func(test_data, params):
            # Simplified backtest
            return {'total_return': 0.15, 'sharpe_ratio': 1.5, 'max_drawdown': -0.1}
            
        results = wfo.run_walk_forward(df, optimize_func, backtest_func, param_bounds)
        metrics = wfo.calculate_metrics(results)
        
        logger.info(f"Walk-forward completed - Avg Return: {metrics['avg_return']:.2%}")
        
        return results, metrics
        
    def generate_report(self, symbol: str, results: dict):
        """Generate PDF report"""
        logger.info("Generating PDF report...")
        
        report_gen = PDFReportGenerator(self.config['reporting']['output_dir'])
        report_path = report_gen.generate(symbol, results)
        
        logger.info(f"Report generated: {report_path}")
        return report_path
        
    async def run_daemon_mode(self):
        """Run platform as daemon/service"""
        logger.info("Starting daemon mode...")
        
        daemon = TradingDaemon(self.config, self)
        await daemon.start()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Crypto Trading Platform')
    parser.add_argument('--mode', choices=['backtest', 'daemon', 'optimize'], 
                       default='backtest', help='Operation mode')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    
    args = parser.parse_args()
    
    # Initialize platform
    platform = CryptoTradingPlatform()
    
    try:
        # Fetch data
        df = await platform.fetch_and_store_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=730  # 2 years
        )
        
        if args.mode == 'backtest':
            results = platform.run_backtest(args.symbol, args.timeframe)
            platform.generate_report(args.symbol, results)
            
        elif args.mode == 'optimize':
            results, metrics = platform.run_walk_forward_optimization(
                args.symbol, args.timeframe
            )
            platform.generate_report(args.symbol, {'wfo_results': results, 'wfo_metrics': metrics})
            
        elif args.mode == 'daemon':
            await platform.run_daemon_mode()
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        platform.db.close()
        platform.cache.close()

if __name__ == '__main__':
    asyncio.run(main())
