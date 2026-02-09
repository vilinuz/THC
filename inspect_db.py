import duckdb
import pandas as pd

DB_PATH = "data/market.duckdb"

conn = duckdb.connect(DB_PATH)

print("Tables:")
print(conn.execute("SHOW TABLES").df())

print("\nOHLCV Symbols:")
try:
    print(conn.execute("SELECT DISTINCT symbol FROM ohlcv").df())
except Exception as e:
    print(f"Error querying ohlcv: {e}")

print("\nMarket Trades Symbols:")
try:
    print(conn.execute("SELECT DISTINCT symbol FROM market_trades").df())
except Exception as e:
    print(f"Error querying market_trades: {e}")
    
conn.close()
