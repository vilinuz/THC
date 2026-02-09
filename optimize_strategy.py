import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd

from db.duckdb_manager import DuckDBManager
from optimization.simulated_annealing import SimulatedAnnealingOptimizer
from strategy.multi_layer_strategy import MultiLayerStrategy

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def map_tickers(user_inputs: List[str]) -> List[str]:
    """Map user friendly names to likely pairs (Binance Format: BTCUSDT)"""
    mapping = {"solana": "SOL", "render": "RNDR", "fartcoin": "FARTCOIN"}

    pairs = []
    for item in user_inputs:
        ticker = mapping.get(item.lower(), item.upper())
        # Format for market_trades seems to be BTCUSDT (no slash) based on inspection
        pairs.append(f"{ticker}USDT")
    return pairs


async def main():
    # 1. Configuration
    DB_PATH = "data/market.duckdb"
    PARQUET_DIR = "data/parquet"
    TIMEFRAME = "1h"
    START_DATE = "2025-01-01"  # Corrected for 1-year analysis
    END_DATE = "2026-03-01"

    # User's requested list
    raw_tickers = [
        "btc",
        "eth",
        "solana",
        "link",
        "qnt",
        "bnb",
        "near",
        "ada",
        "apt",
        "ksm",
    ]

    symbols = map_tickers(raw_tickers)
    logger.info(f"Target Symbols: {symbols}")

    # 2. Load Data
    logger.info("Connecting to DuckDB...")
    db = DuckDBManager(DB_PATH, PARQUET_DIR)

    data_map = {}

    try:
        # Check available symbols in market_trades first to save time
        avail_df = db.conn.execute("SELECT DISTINCT symbol FROM market_trades").df()
        available_symbols = (
            set(avail_df["symbol"].tolist()) if not avail_df.empty else set()
        )
        logger.info(f"Available symbols in DB: {available_symbols}")

        for symbol in symbols:
            if symbol not in available_symbols:
                logger.warning(f"Symbol {symbol} not found in market_trades. Skipping.")
                continue

            logger.info(f"Aggregating data for {symbol}...")
            # Use aggregation from trades since ohlcv might be empty
            df = db.get_aggregated_ohlcv(symbol, START_DATE, END_DATE, TIMEFRAME)

            if df is None or df.empty:
                logger.warning(f"No aggregated data for {symbol}. Skipping.")
                continue

            logger.info(f"Loaded {len(df)} candles for {symbol}")
            data_map[symbol] = df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    finally:
        db.close()

    if not data_map:
        logger.error("No data loaded for any symbol. Aborting.")
        return

    # 3. Define Parameter Space
    param_bounds = {
        "adx_period": (10, 30, 1),
        "adx_consolidation": (15, 25, 1),
        "kama_period": (5, 20, 1),
        "zscore_entry_threshold": (-2.0, -0.5, 0.1),
        "atr_stop_multiplier": (1.5, 4.0, 0.1),
    }

    # Initial Guess
    initial_params = {
        "adx_period": 14,
        "adx_consolidation": 20,
        "kama_period": 10,
        "zscore_entry_threshold": -1.0,
        "atr_stop_multiplier": 2.0,
    }

    # 4. Run Optimizer
    # MANUAL PARAMS FROM OPTIMIZATION (Sharpe 0.1086 on ETH)
    best_params = {
        "adx_period": 13,
        "adx_consolidation": 16,
        "kama_period": 9,
        "zscore_entry_threshold": -1.1,
        "atr_stop_multiplier": 1.9,
    }

    print("Using Optimized Parameters:", best_params)

    # SKIP OPTIMIZATION to generate plots quickly
    # optimizer = SimulatedAnnealingOptimizer(
    #     strategy_class=MultiLayerStrategy,
    #     data_map=data_map,
    #     param_bounds=param_bounds,
    #     initial_params=initial_params,
    #     iterations=20,  # Short run for verification
    #     initial_temp=10.0,
    #     cooling_rate=0.85,
    # )

    # best_params, best_score, history = optimizer.run()

    # print("\n" + "=" * 50)
    # print("OPTIMIZATION COMPLETE")
    # print("=" * 50)
    # print(f"Best Sharpe Ratio: {best_score:.4f}")
    # print("Best Parameters:")
    # for k, v in best_params.items():
    #     print(f"  {k}: {v}")

    # # Save history
    # hist_df = pd.DataFrame(history)
    # hist_df.to_csv("optimization_history.csv", index=False)
    # print("\nHistory saved to optimization_history.csv")

    print("\n" + "=" * 50)
    print("DETAILED PERFORMANCE (Best Params)")
    print("=" * 50)

    from backtesting.backtest_engine import BacktestEngine
    from backtesting.plotting import plot_backtest_results

    # Run backtest for each asset with best params
    for symbol, df in data_map.items():
        try:
            strategy = MultiLayerStrategy(config=best_params)
            signals = strategy.generate_signals(df)

            # Note: generate_signals modifies df in-place or usage of phases does.
            # We explicitly check if 'adx' etc are present.
            # MultiLayerStrategy stores intermediate dfs in self.last_phase_dfs or similar if we modified it?
            # Actually, the strategy methods return separate DFs (kama_df, tactical_df).
            # We might need to call them again to get indicators for plotting,
            # OR refactor generate_signals to attach them.
            # For now, let's just re-calculate the necessary phases for plotting to ensure we hava columns.
            # But wait, generate_signals calls internal methods. We can manually call them to enrich DF.

            # Enrich DF for plotting
            kama_df = strategy._phase2_kama(df)
            tactical_df = strategy._phase4_tactical(df)

            df_plot = df.copy()
            df_plot["kama_price"] = kama_df["kama"]
            df_plot["adx"] = tactical_df["adx"]

            engine = BacktestEngine({"initial_capital": 10000, "commission": 0.001})
            results = engine.run(df_plot, {"strategy": signals})

            win_rate = results.get("win_rate", 0.0)
            trades = results.get("total_trades", 0)
            sharpe = results.get("sharpe_ratio", 0.0)
            ret = results.get("total_return", 0.0)

            print(f"SYMBOL: {symbol}")
            print(f"  Win Rate:      {win_rate:.2%}")
            print(f"  Total Trades:  {trades}")
            print(f"  Winning Trades: {results.get('winning_trades', 0)}")
            print(f"  Sharpe Ratio:  {sharpe:.2f}")
            print(f"  Total Return:  {ret:.2%}")

            # Generate Plot
            plot_file = plot_backtest_results(
                symbol,
                df_plot,
                signals,
                results.get("trades", []),
                results.get("equity_curve", pd.DataFrame()),
                output_dir="backtest_plots",
            )
            print(f"  Chart: {plot_file}")
            print("-" * 30)

        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
