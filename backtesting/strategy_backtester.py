"""
Multi-Layer Strategy Backtester

A comprehensive backtesting engine specifically designed for the MultiLayerStrategy.
Features:
- Real-world data fetching from Binance
- Proper position management with stop-losses
- Detailed trade logging and analytics
- Performance metrics calculation
- Equity curve visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

# Import strategy
from strategy.multi_layer_strategy import MultiLayerStrategy

# Import existing metrics
from backtesting.performance_metrics import PerformanceMetrics


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    side: str  # 'long' or 'short'
    size: float
    stop_loss: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'signal', 'stop_loss', 'emergency', 'end_of_data'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    # Metadata
    confluence_score: int = 0
    kalman_velocity: float = 0.0
    kama_trend: int = 0
    smc_confluence: bool = False


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Summary metrics
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade_duration: timedelta
    
    # Data
    trades: List[Trade]
    equity_curve: pd.DataFrame
    signals_df: pd.DataFrame
    
    # Configuration
    initial_capital: float
    final_capital: float
    config: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'summary': {
                'total_return': self.total_return,
                'total_return_pct': self.total_return_pct,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown_pct,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'calmar_ratio': self.calmar_ratio
            },
            'trade_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'avg_trade_duration_hours': self.avg_trade_duration.total_seconds() / 3600
            },
            'capital': {
                'initial': self.initial_capital,
                'final': self.final_capital
            }
        }
    
    def print_summary(self):
        """Print a formatted summary of results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"   Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"   Final Capital:    ${self.final_capital:,.2f}")
        print(f"   Total Return:     ${self.total_return:,.2f} ({self.total_return_pct:.2%})")
        print(f"   Max Drawdown:     {self.max_drawdown_pct:.2%}")
        
        print(f"\n📈 RISK METRICS")
        print(f"   Sharpe Ratio:     {self.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio:    {self.sortino_ratio:.2f}")
        print(f"   Calmar Ratio:     {self.calmar_ratio:.2f}")
        print(f"   Profit Factor:    {self.profit_factor:.2f}")
        
        print(f"\n🎯 TRADE STATISTICS")
        print(f"   Total Trades:     {self.total_trades}")
        print(f"   Win Rate:         {self.win_rate:.2%}")
        print(f"   Winning Trades:   {self.winning_trades}")
        print(f"   Losing Trades:    {self.losing_trades}")
        print(f"   Avg Win:          ${self.avg_win:,.2f}")
        print(f"   Avg Loss:         ${self.avg_loss:,.2f}")
        print(f"   Avg Duration:     {self.avg_trade_duration}")
        print("="*60 + "\n")


class MultiLayerBacktester:
    """
    Backtester specifically designed for MultiLayerStrategy.
    
    Features:
    - Supports both long and short positions
    - Respects stop-loss levels from strategy
    - Handles emergency close signals (-99)
    - Tracks confluence scores and SMC signals
    - Calculates comprehensive metrics
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Capital settings
        self.initial_capital = self.config.get('initial_capital', 10000)
        self.commission = self.config.get('commission', 0.001)  # 0.1%
        self.slippage = self.config.get('slippage', 0.0005)  # 0.05%
        
        # Position sizing
        self.position_size_pct = self.config.get('position_size_pct', 0.95)  # Use 95% of capital
        self.use_strategy_sizing = self.config.get('use_strategy_sizing', False)
        
        # Risk management
        self.use_stop_loss = self.config.get('use_stop_loss', True)
        self.max_positions = self.config.get('max_positions', 1)  # 1 = no pyramiding
        
        # Strategy configuration
        self.strategy_config = self.config.get('strategy_config', {})
        
        # Initialize strategy
        self.strategy = MultiLayerStrategy(self.strategy_config)
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        if side == 'long':
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def _calculate_commission(self, value: float) -> float:
        """Calculate commission for a trade."""
        return value * self.commission
    
    def run(self, df: pd.DataFrame, train_ratio: float = 0.3) -> BacktestResult:
        """
        Run backtest on provided data.
        
        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            train_ratio: Ratio of data to use for HMM training (0.0 to skip training)
            
        Returns:
            BacktestResult with all metrics and trade data
        """
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient data for backtesting. Need at least 100 bars.")
        
        # Split data for training if requested
        if train_ratio > 0:
            train_size = int(len(df) * train_ratio)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            print(f"Training HMM on {len(train_df)} bars...")
            self.strategy.train(train_df)
            print("Training complete.")
        else:
            test_df = df.copy()
        
        # Generate signals with metadata
        print(f"Generating signals for {len(test_df)} bars...")
        signals, meta_df = self.strategy.generate_signals_with_meta(test_df)
        
        # Initialize state
        cash = self.initial_capital
        position: Optional[Trade] = None
        trades: List[Trade] = []
        equity_curve: List[Dict] = []
        
        print("Running backtest simulation...")
        
        for i in range(len(test_df)):
            timestamp = test_df.index[i]
            row = test_df.iloc[i]
            signal = signals.iloc[i]
            meta = meta_df.iloc[i]
            
            current_price = row['close']
            high = row['high']
            low = row['low']
            
            # Check stop-loss if in position
            if position is not None and self.use_stop_loss:
                stop_triggered = False
                exit_price = None
                
                if position.side == 'long' and low <= position.stop_loss:
                    stop_triggered = True
                    exit_price = self._apply_slippage(position.stop_loss, 'short')
                elif position.side == 'short' and high >= position.stop_loss:
                    stop_triggered = True
                    exit_price = self._apply_slippage(position.stop_loss, 'long')
                
                if stop_triggered:
                    # Close position at stop
                    position.exit_time = timestamp
                    position.exit_price = exit_price
                    position.exit_reason = 'stop_loss'
                    
                    if position.side == 'long':
                        pnl = (exit_price - position.entry_price) * position.size
                    else:
                        pnl = (position.entry_price - exit_price) * position.size
                    
                    pnl -= self._calculate_commission(position.size * exit_price)
                    position.pnl = pnl
                    position.pnl_pct = pnl / (position.entry_price * position.size)
                    
                    cash += position.size * exit_price + pnl
                    trades.append(position)
                    position = None
            
            # Handle emergency close
            if signal == -99 and position is not None:
                exit_price = self._apply_slippage(current_price, 
                                                   'short' if position.side == 'long' else 'long')
                position.exit_time = timestamp
                position.exit_price = exit_price
                position.exit_reason = 'emergency'
                
                if position.side == 'long':
                    pnl = (exit_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - exit_price) * position.size
                
                pnl -= self._calculate_commission(position.size * exit_price)
                position.pnl = pnl
                position.pnl_pct = pnl / (position.entry_price * position.size)
                
                cash += position.size * exit_price + pnl
                trades.append(position)
                position = None
            
            # Process signals
            elif signal == 1 and position is None:
                # Open long
                entry_price = self._apply_slippage(current_price, 'long')
                position_value = cash * self.position_size_pct
                size = position_value / entry_price
                cost = size * entry_price + self._calculate_commission(size * entry_price)
                
                if cost <= cash:
                    cash -= cost
                    
                    # Get stop loss from strategy
                    stop_loss = meta.get('stop_loss_long', entry_price * 0.95)
                    
                    position = Trade(
                        entry_time=timestamp,
                        entry_price=entry_price,
                        side='long',
                        size=size,
                        stop_loss=stop_loss,
                        confluence_score=int(meta.get('confluence_long', 0)),
                        kalman_velocity=float(meta.get('kalman_velocity', 0)),
                        kama_trend=int(meta.get('kama_trend', 0)),
                        smc_confluence=bool(meta.get('smc_long_confluence', False))
                    )
            
            elif signal == -1:
                # Close long if exists
                if position is not None and position.side == 'long':
                    exit_price = self._apply_slippage(current_price, 'short')
                    position.exit_time = timestamp
                    position.exit_price = exit_price
                    position.exit_reason = 'signal'
                    
                    pnl = (exit_price - position.entry_price) * position.size
                    pnl -= self._calculate_commission(position.size * exit_price)
                    position.pnl = pnl
                    position.pnl_pct = pnl / (position.entry_price * position.size)
                    
                    cash += position.size * exit_price + pnl
                    trades.append(position)
                    position = None
                
                # Open short (if no position)
                if position is None:
                    entry_price = self._apply_slippage(current_price, 'short')
                    position_value = cash * self.position_size_pct
                    size = position_value / entry_price
                    cost = self._calculate_commission(size * entry_price)
                    
                    if cost <= cash:
                        cash -= cost
                        
                        stop_loss = meta.get('stop_loss_short', entry_price * 1.05)
                        
                        position = Trade(
                            entry_time=timestamp,
                            entry_price=entry_price,
                            side='short',
                            size=size,
                            stop_loss=stop_loss,
                            confluence_score=int(meta.get('confluence_short', 0)),
                            kalman_velocity=float(meta.get('kalman_velocity', 0)),
                            kama_trend=int(meta.get('kama_trend', 0)),
                            smc_confluence=bool(meta.get('smc_short_confluence', False))
                        )
            
            # Calculate portfolio value
            if position is not None:
                if position.side == 'long':
                    position_value = position.size * current_price
                else:
                    # For short: initial value + unrealized P&L
                    unrealized_pnl = (position.entry_price - current_price) * position.size
                    position_value = position.size * position.entry_price + unrealized_pnl
                portfolio_value = cash + position_value
            else:
                portfolio_value = cash
            
            equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'position_value': position_value if position else 0,
                'position_side': position.side if position else None,
                'signal': signal
            })
        
        # Close any remaining position at end
        if position is not None:
            final_price = test_df['close'].iloc[-1]
            exit_price = self._apply_slippage(final_price, 
                                              'short' if position.side == 'long' else 'long')
            position.exit_time = test_df.index[-1]
            position.exit_price = exit_price
            position.exit_reason = 'end_of_data'
            
            if position.side == 'long':
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
            
            pnl -= self._calculate_commission(position.size * exit_price)
            position.pnl = pnl
            position.pnl_pct = pnl / (position.entry_price * position.size)
            
            trades.append(position)
        
        # Build equity DataFrame
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        
        # Calculate metrics
        result = self._calculate_results(trades, equity_df, meta_df)
        
        print("Backtest complete!")
        
        return result
    
    def _calculate_results(self, trades: List[Trade], equity_df: pd.DataFrame, 
                          signals_df: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        
        final_capital = equity_df['portfolio_value'].iloc[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Use existing metrics calculator
        returns = equity_df['portfolio_value'].pct_change().dropna()
        
        sharpe = PerformanceMetrics.sharpe_ratio(returns)
        sortino = PerformanceMetrics.sortino_ratio(returns)
        max_dd = PerformanceMetrics.max_drawdown(equity_df['portfolio_value'])
        profit_factor = PerformanceMetrics.profit_factor(returns)
        calmar = PerformanceMetrics.calmar_ratio(equity_df['portfolio_value'])
        
        # Trade statistics
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
        
        total_trades = len(trades)
        n_winning = len(winning_trades)
        n_losing = len(losing_trades)
        win_rate = n_winning / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate average trade duration
        durations = []
        for t in trades:
            if t.exit_time and t.entry_time:
                durations.append(t.exit_time - t.entry_time)
        avg_duration = np.mean(durations) if durations else timedelta(0)
        
        return BacktestResult(
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd * self.initial_capital,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=n_winning,
            losing_trades=n_losing,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_duration,
            trades=trades,
            equity_curve=equity_df,
            signals_df=signals_df,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            config=self.config
        )
    
    async def run_with_live_data(self, symbol: str = 'BTC/USDT', 
                                  timeframe: str = '1h',
                                  days: int = 90) -> BacktestResult:
        """
        Fetch real data from Binance and run backtest.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            days: Number of days of historical data
            
        Returns:
            BacktestResult
        """
        from data_fetchers.binance_fetcher import BinanceFetcher
        
        print(f"Fetching {days} days of {symbol} {timeframe} data from Binance...")
        
        fetcher = BinanceFetcher({})
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = await fetcher.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"Fetched {len(df)} candles.")
        
        return self.run(df)


def run_quick_backtest(df: pd.DataFrame = None, 
                       symbol: str = 'BTC/USDT',
                       initial_capital: float = 10000,
                       **strategy_kwargs) -> BacktestResult:
    """
    Convenience function for quick backtesting.
    
    Args:
        df: Optional pre-loaded DataFrame. If None, generates sample data.
        symbol: Symbol name for logging
        initial_capital: Starting capital
        **strategy_kwargs: Additional strategy configuration
        
    Returns:
        BacktestResult
    """
    # Generate sample data if not provided
    if df is None:
        print("No data provided. Generating sample data...")
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        # Generate realistic price movement
        returns = np.random.randn(n) * 0.02
        close = 40000 * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.randn(n) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n) * 0.01))
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.005),
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(100, 10000, n) * 1000
        }, index=dates)
    
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {symbol}")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Bars: {len(df)}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"{'='*60}\n")
    
    config = {
        'initial_capital': initial_capital,
        'strategy_config': strategy_kwargs
    }
    
    backtester = MultiLayerBacktester(config)
    result = backtester.run(df, train_ratio=0.3)
    
    result.print_summary()
    
    return result


# CLI entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest MultiLayerStrategy')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    
    args = parser.parse_args()
    
    result = run_quick_backtest(
        initial_capital=args.capital,
        symbol=args.symbol
    )
