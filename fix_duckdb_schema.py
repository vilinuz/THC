import os

import duckdb

DB_PATH = "data/market.duckdb"


def fix_schema():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)

    try:
        # Check current schema
        print("Checking current schema...")
        schema_df = conn.execute("DESCRIBE market_trades").df()
        trade_id_type = schema_df[schema_df["column_name"] == "trade_id"][
            "column_type"
        ].values[0]

        print(f"Current trade_id type: {trade_id_type}")

        if trade_id_type == "INTEGER":
            print("Migrating trade_id to BIGINT by recreating table...")

            # 1. Create new table with BIGINT
            conn.execute("""
                CREATE TABLE market_trades_new (
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

            # 2. Copy data
            print("Copying data...")
            conn.execute("INSERT INTO market_trades_new SELECT * FROM market_trades")

            # 3. Drop old table
            print("Dropping old table...")
            conn.execute("DROP TABLE market_trades")

            # 4. Rename new table
            print("Renaming new table...")
            conn.execute("ALTER TABLE market_trades_new RENAME TO market_trades")

            print("Migration complete.")

            # Verify
            schema_df = conn.execute("DESCRIBE market_trades").df()
            new_type = schema_df[schema_df["column_name"] == "trade_id"][
                "column_type"
            ].values[0]
            print(f"New trade_id type: {new_type}")

            if new_type == "BIGINT":
                print("SUCCESS: Schema updated successfully.")
            else:
                print("FAILURE: Schema update failed.")
        else:
            print("Schema already correct (BIGINT). No action needed.")

    except Exception as e:
        print(f"Error during migration: {e}")
        # Attempt rollback if possible (DuckDB transactions work but simple script might not need complex rollback for this task)
    finally:
        conn.close()


if __name__ == "__main__":
    fix_schema()
