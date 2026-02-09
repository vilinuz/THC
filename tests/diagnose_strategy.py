
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backtesting.strategy_backtester import MultiLayerBacktester
from db.duckdb_manager import DuckDBManager


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def run_diagnostic(name, config_overrides, db_manager):
    print(f"\n--- Running Diagnostic: {name} ---")
    set_seeds()
    
    symbol = "BTCUSDT"
    start_date = "2025-09-01 00:00:00"
    end_date = "2025-09-30 23:59:59"
    
    df = db_manager.get_aggregated_ohlcv(symbol, start_date, end_date, timeframe="1h")
    if df.empty:
        print("No data found.")
        return None

    # Base config
    base_config = {
        "initial_capital": 10000.0,
        "strategy_config": {
            "use_smc": True,
            "stable_trend_threshold": 0.6, # Default
        }
    }
    
    # Apply overrides
    for k, v in config_overrides.items():
        base_config["strategy_config"][k] = v

    backtester = MultiLayerBacktester(config=base_config)
    
    # Run twice to check determinism within this config
    results = []
    for i in range(2):
        res = backtester.run(df, train_ratio=0.0)
        results.append(res)
        
    r1 = results[0]
    r2 = results[1]
    
    is_deterministic = abs(r1.total_return - r2.total_return) < 1e-6
    det_status = "✅ Stable" if is_deterministic else "❌ Unstable"
    
    print(f"Result: Return=${r1.total_return:.2f} ({r1.total_return_pct:.2%}) | Trades={r1.total_trades} | {det_status}")
    if not is_deterministic:
        print(f"   Run 2: Return=${r2.total_return:.2f}")
        
    return {
        "name": name,
        "return": r1.total_return,
        "trades": r1.total_trades,
        "deterministic": is_deterministic
    }

def main():
    config_db = {"path": "data/market.duckdb"}
    db_path = str(project_root / config_db["path"])
    parquet_dir = str(project_root / "data/parquet")
    db_manager = DuckDBManager(db_path, parquet_dir)
    
    diagnostics = [
        ("Baseline (All On)", {}),
        ("Disable SMC", {"use_smc": False}),
        ("Disable HMM (Accept All)", {"stable_trend_threshold": 0.0}),
        ("Disable SMC + HMM", {"use_smc": False, "stable_trend_threshold": 0.0}),
    ]
    
    summary = []
    try:
        for name, overrides in diagnostics:
            res = run_diagnostic(name, overrides, db_manager)
            summary.append(res)
    finally:
        db_manager.close()
        
    print("\n\n=== Diagnostic Summary ===")
    print(f"{'Configuration':<25} | {'Return':<10} | {'Trades':<6} | {'Stability':<10}")
    print("-" * 60)
    for s in summary:
        stab = "✅" if s['deterministic'] else "❌"
        print(f"{s['name']:<25} | ${s['return']:<9.2f} | {s['trades']:<6} | {stab}")

if __name__ == "__main__":
    main()
