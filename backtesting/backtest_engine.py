"""
Backtesting engine for trading strategies
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .performance_metrics import PerformanceMetrics


class BacktestEngine:
    """Backtest trading strategies"""

    def __init__(self, config: Dict):
        self.initial_capital = config.get("initial_capital", 10000)
        self.commission = config.get("commission", 0.001)
        self.slippage = config.get("slippage", 0.0005)

    def run(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> Dict:
        """
        Run backtest

        Args:
            df: OHLCV data
            signals: Dict of signal sources
        """
        # Combine signals (simple average for now)
        combined_signal = pd.DataFrame(signals).mean(axis=1)
        combined_signal = combined_signal.apply(
            lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)
        )

        # Initialize tracking
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0
        trades = []
        equity_curve = []

        for i in range(len(df)):
            current_price = df["close"].iloc[i]
            signal = combined_signal.iloc[i]

            # Execute trades
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short
                    entry_trade = trades[-1]
                    exit_price = current_price
                    pnl = abs(position) * (
                        entry_trade["entry_price"] - exit_price
                    )  # Short PnL: (Entry - Exit) * Size

                    # Update closed trade
                    entry_trade["exit_price"] = exit_price
                    entry_trade["exit_time"] = df.index[i]
                    entry_trade["pnl"] = pnl

                    # Let's stick to minimal changes, just update dictionary.
                    cash += abs(position) * current_price + pnl
                    position = 0

                # Open long
                position_value = cash * 0.95  # Use 95% of cash
                position = position_value / current_price
                cash -= position * current_price * (1 + self.commission)

                trades.append(
                    {
                        "entry_idx": i,
                        "entry_time": df.index[i],
                        "entry_price": current_price,
                        "side": "long",
                        "size": position,
                    }
                )

            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long
                    entry_trade = trades[-1]
                    exit_price = current_price
                    pnl = position * (exit_price - entry_trade["entry_price"])

                    # Update closed trade
                    entry_trade["exit_price"] = exit_price
                    entry_trade["exit_time"] = df.index[i]
                    entry_trade["pnl"] = pnl

                    cash += position * current_price + pnl
                    position = 0

                # Open short
                position_value = cash * 0.95
                position = -(position_value / current_price)
                # For shorting in this simple engine, we assume simplistic handling:
                # Cash stays same (collateral), we verify equity later.
                # Actually, standard logic for simplistic backtester:
                # Cash -= (Initial Margin of Short).
                # Let's assume 1x leverage short -> "sell" borrows asset.
                # Just mimic the logic:
                # Cost is commission.
                cash -= abs(position) * current_price * self.commission
                
                trades.append({
                    "entry_idx": i,
                    "entry_time": df.index[i],
                    "entry_price": current_price,
                    "side": "short",
                    "size": position,
                })

            # Update portfolio value
            portfolio_value = cash + abs(position) * current_price
            equity_curve.append(
                {
                    "timestamp": df.index[i],
                    "portfolio_value": portfolio_value,
                    "position": position,
                }
            )

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
        metrics = PerformanceMetrics.calculate_all(equity_df, self.initial_capital)

        # Calculate Win Rate
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        return {
            "equity_curve": equity_df,
            "trades": trades,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "win_rate": win_rate,
            **metrics,
        }
