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

        # 3. Download to temp file
        import tempfile
        import os
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                temp_zip_path = temp_zip.name
                
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code != 200:
                    logger.error(f"Failed to download. Status Code: {response.status_code}")
                    os.unlink(temp_zip_path)
                    return
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_zip.write(chunk)
            
            # 4. Extract and Process
            logger.info("Download complete. Extracting and parsing from disk...")
            with zipfile.ZipFile(temp_zip_path) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as csv_file:
                    # Binance CSVs: id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
                    chunk_iter = pd.read_csv(
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
                        chunksize=1_000_000,
                    )

                    is_first_chunk = True
                    total_rows = 0

                    for chunk in chunk_iter:
                        if chunk.empty:
                            continue

                        # 5. Transform
                        first_ts = chunk["time"].iloc[0]
                        if first_ts > 1e14:
                            unit = "us"
                        else:
                            unit = "ms"

                        chunk["time"] = pd.to_datetime(chunk["time"], unit=unit)

                        # 6. Save to DB
                        logger.info(f"Ingesting {len(chunk)} trades chunk into DuckDB...")
                        self.db_manager.save_market_trades(
                            chunk, symbol, year_int, month_int, append_only=not is_first_chunk
                        )
                        
                        total_rows += len(chunk)
                        is_first_chunk = False

            if total_rows == 0:
                logger.warning(f"Empty Data for {symbol} {year}-{month}")
                os.unlink(temp_zip_path)
                return

            self.db_manager.log_ingestion_status(
                symbol, year_int, month_int, "COMPLETED"
            )

            logger.info(f"Successfully ingested {total_rows} total trades for {symbol} {year}-{month}")

        except Exception as e:
            logger.error(f"Error processing {symbol} {year}-{month}: {e}")
            self.db_manager.log_ingestion_status(
                symbol, year_int, month_int, f"FAILED: {str(e)}"
            )
        finally:
            if 'temp_zip_path' in locals() and os.path.exists(temp_zip_path):
                os.unlink(temp_zip_path)

    def download_range(self, symbol: str, start_date: str, end_date: str):
        """
        Download data for a range of dates.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        current = start.replace(day=1)
        while current <= end:
            year = str(current.year)
            month = f"{current.month:02d}"
            
            logger.info(f"Processing {symbol} for {year}-{month}")
            self.download_and_ingest(symbol, year, month)
            
            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
    def close(self):
        self.db_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download and Ingest Binance Tick Data"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading Pair Symbol"
    )
    parser.add_argument("--year", type=str, help="Year (YYYY)")
    parser.add_argument("--month", type=str, help="Month (MM)")
    parser.add_argument("--start_date", type=str, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End Date (YYYY-MM-DD)")

    args = parser.parse_args()

    fetcher = BinanceHistoryFetcher()
    try:
        if args.start_date and args.end_date:
            fetcher.download_range(args.symbol, args.start_date, args.end_date)
        elif args.year and args.month:
            fetcher.download_and_ingest(args.symbol, args.year, args.month)
        else:
            print("Please provide --year and --month OR --start_date and --end_date")
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
