import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestEngine
from db.duckdb_manager import DuckDBManager
from optimization.simulated_annealing import SimulatedAnnealingOptimizer
from strategy.multi_layer_strategy import MultiLayerStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WFO")


class WalkForwardOptimizer:
    def __init__(
        self,
        config: Dict,
        db_manager: DuckDBManager,
        symbols: List[str],
        train_window_months: int = 3,
        test_window_months: int = 1,
    ):
        self.config = config
        self.db = db_manager
        self.symbols = symbols
        self.train_window = train_window_months
        self.test_window = test_window_months
        self.results = []

        # Define parameter space
        self.param_bounds = {
            "adx_period": (10, 30, 1),
            "adx_consolidation": (15, 25, 1),
            "kama_period": (5, 20, 1),
            "zscore_entry_threshold": (-2.0, -0.5, 0.1),
            "atr_stop_multiplier": (1.5, 4.0, 0.1),
        }

        self.initial_params = {
            "adx_period": 14,
            "adx_consolidation": 20,
            "kama_period": 10,
            "zscore_entry_threshold": -1.0,
            "atr_stop_multiplier": 2.0,
        }

    def _get_date_ranges(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.
        """
        ranges = []
        current_date = start_date

        while (
            current_date + timedelta(days=30 * (self.train_window + self.test_window))
            <= end_date
        ):
            train_start = current_date
            train_end = train_start + timedelta(days=30 * self.train_window)
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.test_window)

            if test_end > end_date:
                break

            ranges.append((train_start, train_end, test_start, test_end))
            current_date = current_date + timedelta(
                days=30 * self.test_window
            )  # Slide by test window

        return ranges

    async def run(self, start_date_str: str, end_date_str: str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        windows = self._get_date_ranges(start_date, end_date)
        logger.info(f"Generated {len(windows)} Walk-Forward Windows.")

        cumulative_pnl = 0.0

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}")
            logger.info(f"  Train: {train_start.date()} -> {train_end.date()}")
            logger.info(f"  Test:  {test_start.date()} -> {test_end.date()}")

            # 1. Prepare Train Data
            train_data_map = {}
            for symbol in self.symbols:
                df = self.db.get_aggregated_ohlcv(symbol, train_start, train_end, "1h")
                if not df.empty and len(df) > 100:
                    train_data_map[symbol] = df

            if not train_data_map:
                logger.warning("No training data used/found. Skipping window.")
                continue

            # 2. Optimize on Train Data
            optimizer = SimulatedAnnealingOptimizer(
                strategy_class=MultiLayerStrategy,
                data_map=train_data_map,
                param_bounds=self.param_bounds,
                initial_params=self.initial_params,
                iterations=15,  # Fast optimization
                initial_temp=10.0,
                cooling_rate=0.85,
            )

            best_params, best_score, _ = optimizer.run()
            logger.info(f"  Best Train Sharpe: {best_score:.4f}")
            logger.info(f"  Best Params: {best_params}")

            # 3. Test on Out-of-Sample Test Data
            window_pnl = 0.0

            for symbol in self.symbols:
                test_df = self.db.get_aggregated_ohlcv(
                    symbol, test_start, test_end, "1h"
                )
                if test_df.empty:
                    continue

                strategy = MultiLayerStrategy(config=best_params)
                signals = strategy.generate_signals(test_df)

                engine = BacktestEngine({"initial_capital": 10000, "commission": 0.001})
                res = engine.run(test_df, {"strategy": signals})

                pnl = res.get("total_return", 0.0) * 10000  # Absolute PnL assumption
                window_pnl += pnl

                logger.info(
                    f"    {symbol} Test Return: {res.get('total_return', 0.0):.2%}"
                )

            cumulative_pnl += window_pnl
            logger.info(f"  Window PnL: {window_pnl:.2f}")

            self.results.append(
                {
                    "window": i,
                    "train_start": train_start,
                    "test_start": test_start,
                    "params": best_params,
                    "train_sharpe": best_score,
                    "test_pnl": window_pnl,
                }
            )

            # Update initial params for next window (adaptivity)
            self.initial_params = best_params

        logger.info("=" * 50)
        logger.info(f"Walk-Forward Analysis Complete.")
        logger.info(f"Total Cumulative PnL: ${cumulative_pnl:.2f}")

        # Save results
        pd.DataFrame(self.results).to_csv("wfo_results.csv", index=False)


async def main():
    DB_PATH = "data/market.duckdb"
    PARQUET_DIR = "data/parquet"

    db = DuckDBManager(DB_PATH, PARQUET_DIR)

    # We will use ETHUSDT for now as it's the most reliable data source we have confirmed
    symbols = ["ETHUSDT"]

    # Check if other keys are available
    avail = db.conn.execute("SELECT DISTINCT symbol FROM market_trades").df()
    all_symbols = set(avail["symbol"].tolist())

    target_symbols = ["ETHUSDT", "ADAUSDT", "APTUSDT", "KSMUSDT"]
    valid_symbols = [s for s in target_symbols if s in all_symbols]

    logger.info(f"Running WFO on: {valid_symbols}")

    optimizer = WalkForwardOptimizer(
        config={},
        db_manager=db,
        symbols=valid_symbols,
        train_window_months=3,
        test_window_months=1,
    )

    # Run WFO over 2024
    await optimizer.run("2024-01-01", "2024-12-31")


if __name__ == "__main__":
    asyncio.run(main())
