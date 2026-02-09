import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from db.duckdb_manager import DuckDBManager  # noqa: E402

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BinanceFetcher")


class BinanceHistoryFetcher:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)

        # Initialize DuckDB Manager
        db_config = self.config.get("database", {})
        self.db_path = str(project_root / db_config.get("path", "data/market.duckdb"))
        self.parquet_dir = str(
            project_root / db_config.get("parquet_dir", "data/parquet")
        )

        self.db_manager = DuckDBManager(self.db_path, self.parquet_dir)

    def _load_config(self, config_path):
        """Load configuration from yaml"""
        path = project_root / config_path
        if not path.exists():
            logger.warning(f"Config file not found at {path}, using defaults")
            return {"database": {"path": "data/market.duckdb"}}

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def download_and_ingest(self, symbol="BTCUSDT", year="2023", month="01"):
        """
        Downloads monthly tick data from Binance Vision and ingests into DuckDB.
        Skips if already ingested.
        """
        # 1. Check if already exists in DB
        try:
            year_int = int(year)
            month_int = int(month)
        except ValueError:
            logger.error("Year and Month must be integers/digits")
            return

        if self.db_manager.check_data_exists(symbol, year_int, month_int):
            logger.info(
                f"Data for {symbol} {year}-{month} already exists in DB. Skipping."
            )
            return

        # 2. Construct URL
        # Normalize year/month to 2 digits for URL
        month_str = f"{month_int:02d}"
        year_str = str(year_int)

        base_url = "https://data.binance.vision/data/spot/monthly/trades"
        filename = f"{symbol}-trades-{year_str}-{month_str}.zip"
        url = f"{base_url}/{symbol}/{filename}"

        logger.info(f"Downloading {url}...")

        # 3. Download
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download. Status Code: {response.status_code}")
                return

            # 4. Extract and Process
            logger.info("Download complete. Extracting and parsing...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as csv_file:
                    # Binance CSVs: id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
                    df = pd.read_csv(
                        csv_file,
                        names=[
                            "trade_id",
                            "price",
                            "qty",
                            "quote_qty",
                            "time",
                            "is_buyer_maker",
                            "is_best_match",
                        ],
                    )

            # 5. Transform
            # 5. Transform
            # Auto-detect timestamp unit
            # 2024 in ms ~ 1.7e12
            # 2024 in us ~ 1.7e15
            first_ts = df["time"].iloc[0]
            if first_ts > 1e14:
                unit = "us"
            else:
                unit = "ms"

            df["time"] = pd.to_datetime(df["time"], unit=unit)

            # 6. Save to DB
            logger.info(f"Ingesting {len(df)} trades into DuckDB...")
            self.db_manager.save_market_trades(df, symbol, year_int, month_int)
            self.db_manager.log_ingestion_status(
                symbol, year_int, month_int, "COMPLETED"
            )

            logger.info(f"Successfully ingested {symbol} {year}-{month}")

        except Exception as e:
            logger.error(f"Error processing {symbol} {year}-{month}: {e}")
            self.db_manager.log_ingestion_status(
                symbol, year_int, month_int, f"FAILED: {str(e)}"
            )
        finally:
            # Check for close? DuckDBManager has close(), but usually fine to keep open if class persists.
            # But here we might want to close if it's a script run.
            pass

    def close(self):
        self.db_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download and Ingest Binance Tick Data"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading Pair Symbol"
    )
    parser.add_argument("--year", type=str, required=True, help="Year (YYYY)")
    parser.add_argument("--month", type=str, required=True, help="Month (MM)")

    args = parser.parse_args()

    fetcher = BinanceHistoryFetcher()
    try:
        fetcher.download_and_ingest(args.symbol, args.year, args.month)
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
