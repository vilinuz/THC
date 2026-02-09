import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backtesting.strategy_backtester import MultiLayerBacktester
from db.duckdb_manager import DuckDBManager


class TestRealDataBacktest(unittest.TestCase):
    def setUp(self):
        # Use simple config for backtest
        self.config = {
            "database": {"path": "data/market.duckdb"},
            "strategy": {"initial_capital": 10000.0, "position_size": 0.1},
        }

        self.db_path = str(project_root / self.config["database"]["path"])
        self.parquet_dir = str(project_root / "data/parquet")
        self.db_manager = DuckDBManager(self.db_path, self.parquet_dir)

    def test_backtest_btc_2025_september(self):
        """
        Backtest strategy on real BTCUSDT data from September 2025.
        Requires data to be already ingested.
        """
        symbol = "BTCUSDT"
        start_date = "2025-09-01 00:00:00"
        end_date = "2025-09-30 23:59:59"

        # 1. Fetch Aggregated OHLCV Data
        print(f"\nFetching 1h candles for {symbol} from DB...")
        df = self.db_manager.get_aggregated_ohlcv(
            symbol, start_date, end_date, timeframe="1h"
        )

        if df.empty:
            self.skipTest(f"No data found for {symbol} in Sep 2025. Is data ingested?")

        print(f"Loaded {len(df)} candles.")

        # 2. Initialize Strategy and Backtester
        # 2. Initialize Backtester
        # MultiLayerBacktester initializes its own strategy instance internally based on config
        backtester = MultiLayerBacktester(
            config={
                "initial_capital": self.config["strategy"]["initial_capital"],
            }
        )

        # 3. Run Backtest
        print("Running backtest...")
        # Assuming run() takes a DataFrame. Need to check signature if unsure.
        # Based on previous view, run(data: pd.DataFrame) -> dict
        # Skip training for test (small data causes singular matrix in HMM)
        results = backtester.run(df, train_ratio=0.0)

        # 4. Assertions and Reporting
        print("\nBacktest Results:")
        results_dict = results.to_dict()
        for k, v in results_dict.items():
            print(f"{k}: {v}")

        # Basic sanity check
        self.assertIsInstance(results.total_return, float)
        self.assertIsInstance(results.sharpe_ratio, float)
        self.assertIsInstance(results.max_drawdown, float)

    def tearDown(self):
        self.db_manager.close()


if __name__ == "__main__":
    unittest.main()
