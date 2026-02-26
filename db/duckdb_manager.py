from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class DuckDBManager:
    """Manage market data with DuckDB and Parquet"""

    def __init__(self, db_path: str, parquet_dir: str):
        self.db_path = db_path
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database and create tables"""
        self.conn = duckdb.connect(self.db_path)

        # Create schema
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timestamp, timeframe)
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                signal_type VARCHAR,
                strength DOUBLE,
                source VARCHAR,
                metadata JSON
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                side VARCHAR,
                price DOUBLE,
                quantity DOUBLE,
                pnl DOUBLE,
                strategy VARCHAR
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_trades (
                trade_id BIGINT,
                price DOUBLE,
                qty DOUBLE,
                quote_qty DOUBLE,
                time TIMESTAMP,
                is_buyer_maker BOOLEAN,
                is_best_match BOOLEAN,
                symbol VARCHAR,
                year INTEGER,
                month INTEGER,
                PRIMARY KEY (trade_id, symbol)
            );
        """)

        # Auto-migration for existing tables with INTEGER trade_id
        try:
            self.conn.execute(
                "ALTER TABLE market_trades ALTER COLUMN trade_id TYPE BIGINT"
            )
        except Exception:
            # Ignore error if table doesn't exist or column is already BIGINT (DuckDB might throw if no change needed or other issues)
            pass

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_ingestion_log (
                symbol VARCHAR,
                year INTEGER,
                month INTEGER,
                status VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, year, month)
            );
        """)

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save OHLCV data to both parquet and database"""
        # Save to parquet
        safe_symbol = symbol.replace("/", "-")
        parquet_path = self.parquet_dir / f"{safe_symbol}_{timeframe}.parquet"
        df.to_parquet(parquet_path, index=True)

        # Also update database
        df_copy = df.reset_index()
        df_copy["symbol"] = symbol
        df_copy["timeframe"] = timeframe

        # Ensure column order matches table schema
        # Schema: symbol, timestamp, timeframe, open, high, low, close, volume
        df_copy = df_copy[
            [
                "symbol",
                "timestamp",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

        self.conn.execute("""
            INSERT OR REPLACE INTO ohlcv
            SELECT * FROM df_copy
        """)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from parquet (faster) or database"""
        safe_symbol = symbol.replace("/", "-")
        parquet_path = self.parquet_dir / f"{safe_symbol}_{timeframe}.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            return df
        else:
            # Fallback to database
            query = """
                SELECT * FROM ohlcv
                WHERE symbol = ?
                AND timeframe = ?
            """
            params = [symbol, timeframe]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            return self.conn.execute(query, params).df()

    def save_signal(self, signal_data: dict):
        """Save trading signal to database"""
        self.conn.execute(
            """
            INSERT INTO signals (timestamp, symbol, signal_type, strength, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                signal_data["timestamp"],
                signal_data["symbol"],
                signal_data["signal_type"],
                signal_data["strength"],
                signal_data["source"],
                str(signal_data.get("metadata", {})),
            ],
        )

    def get_signals(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve signals for analysis"""
        return self.conn.execute(
            """
            SELECT * FROM signals
            WHERE symbol = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """,
            [symbol, start_date, end_date],
        ).df()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def check_data_exists(self, symbol: str, year: int, month: int) -> bool:
        """Check if data for specific month has been successfully ingested"""
        result = self.conn.execute(
            """
            SELECT count(*) FROM data_ingestion_log 
            WHERE symbol = ? AND year = ? AND month = ? AND status = 'COMPLETED'
        """,
            [symbol, year, month],
        ).fetchone()

        return result[0] > 0

    def log_ingestion_status(self, symbol: str, year: int, month: int, status: str):
        """Log the status of data ingestion"""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO data_ingestion_log (symbol, year, month, status, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            [symbol, year, month, status],
        )

    def save_market_trades(self, df: pd.DataFrame, symbol: str, year: int, month: int, append_only: bool = False):
        """Save bulk market trades to DuckDB"""
        # Ensure DataFrame has correct columns and types
        df_copy = df.copy()
        df_copy["symbol"] = symbol
        df_copy["year"] = year
        df_copy["month"] = month

        # Determine if we should append or overwrite.
        # Typically for monthly batches, if we re-download, we might want to replace.
        # But for huge datasets, append is faster.
        # Given we check ingestion log, a re-run usually implies a retry or force update.
        # Let's delete existing for this month/symbol first to avoid dups if re-running.

        if not append_only:
            self.conn.execute(
                "DELETE FROM market_trades WHERE symbol = ? AND year = ? AND month = ?",
                [symbol, year, month],
            )

        # Use DuckDB's efficient appending from Pandas
        self.conn.register("temp_df", df_copy)
        self.conn.execute("""
            INSERT INTO market_trades 
            SELECT trade_id, price, qty, quote_qty, time, is_buyer_maker, is_best_match, symbol, year, month 
            FROM temp_df
        """)
        self.conn.unregister("temp_df")

    def get_aggregated_ohlcv(
        self, symbol: str, start_date: str, end_date: str, timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Aggregate raw trades into OHLCV candles using DuckDB.
        Supported timeframes: 1m, 1h, 1d (bucket logic can be expanded)
        """
        # Map timeframe to DuckDB interval
        tf_map = {
            "1m": "1 minute",
            "5m": "5 minutes",
            "15m": "15 minutes",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
        }
        interval = tf_map.get(timeframe, "1 hour")

        query = f"""
            SELECT
                time_bucket(INTERVAL '{interval}', time) AS timestamp,
                first(price) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price) AS close,
                sum(qty) AS volume
            FROM market_trades
            WHERE symbol = ?
            AND time BETWEEN ? AND ?
            GROUP BY 1
            ORDER BY 1
        """

        df = self.conn.execute(query, [symbol, start_date, end_date]).df()
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
