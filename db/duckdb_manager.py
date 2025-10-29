import duckdb
import pandas as pd
import os
from pathlib import Path
from typing import Optional, List
import pyarrow.parquet as pq

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

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save OHLCV data to both parquet and database"""
        # Save to parquet
        parquet_path = self.parquet_dir / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(parquet_path, index=True)

        # Also update database
        df_copy = df.reset_index()
        df_copy['symbol'] = symbol
        df_copy['timeframe'] = timeframe

        self.conn.execute("""
            INSERT OR REPLACE INTO ohlcv
            SELECT * FROM df_copy
        """)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load OHLCV data from parquet (faster) or database"""
        parquet_path = self.parquet_dir / f"{symbol}_{timeframe}.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            return df
        else:
            # Fallback to database
            query = f"""
                SELECT * FROM ohlcv
                WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
            """
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"

            return self.conn.execute(query).df()

    def save_signal(self, signal_data: dict):
        """Save trading signal to database"""
        self.conn.execute("""
            INSERT INTO signals (timestamp, symbol, signal_type, strength, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            signal_data['timestamp'],
            signal_data['symbol'],
            signal_data['signal_type'],
            signal_data['strength'],
            signal_data['source'],
            str(signal_data.get('metadata', {}))
        ])

    def get_signals(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve signals for analysis"""
        return self.conn.execute(f"""
            SELECT * FROM signals
            WHERE symbol = '{symbol}'
            AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY timestamp DESC
        """).df()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
