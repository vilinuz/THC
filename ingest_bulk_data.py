import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from data_fetchers.full_data_fetcher import BinanceHistoryFetcher

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BulkIngester")


def ingest_month(fetcher, symbol, year, month):
    str_month = f"{month:02d}"
    logger.info(f"Processing {symbol} {year}-{str_month}...")
    try:
        fetcher.download_and_ingest(symbol, str(year), str_month)
    except Exception as e:
        logger.error(f"Failed {symbol} {year}-{str_month}: {e}")


def main():
    fetcher = BinanceHistoryFetcher()

    symbols = ["ETHUSDT", "ADAUSDT", "APTUSDT", "KSMUSDT"]
    years = [2025]
    months = range(1, 13)  # 1 to 12

    # Also Jan 2026
    extra_months = [(2026, 1)]

    tasks = []

    # We can probably parallelize a bit, but let's be gentle with Binance Vision IP limits.
    # Sequential per symbol might be safer, or parallel symbols sequential months.

    for symbol in symbols:
        logger.info(f"--- Starting {symbol} ---")

        # Full Year
        for year in years:
            for month in months:
                ingest_month(fetcher, symbol, year, month)

        # Extra months
        for y, m in extra_months:
            ingest_month(fetcher, symbol, y, m)

    fetcher.close()
    logger.info("Bulk Ingestion Complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
