
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from db.duckdb_manager import DuckDBManager
from backtesting.strategy_backtester import MultiLayerBacktester

def run_visualization():
    # Load Data (Use a known sample)
    config_db = {"path": "data/market.duckdb"}
    db_path = str(project_root / config_db["path"])
    parquet_dir = str(project_root / "data/parquet")
    db_manager = DuckDBManager(db_path, parquet_dir)
    
    symbol = "BTCUSDT"
    start_date = "2025-09-01 00:00:00"
    end_date = "2025-09-15 00:00:00"
    
    print("Loading data for visualization...")
    df = db_manager.get_aggregated_ohlcv(symbol, start_date, end_date, timeframe="1h")
    db_manager.close()
    
    if len(df) == 0:
        print("No data found for visualization.")
        return

    # Run Backtest to get signals
    print("Running backtest for signal generation...")
    backtester = MultiLayerBacktester(
        config={
            "initial_capital": 10000.0,
            "strategy_config": {
                "use_smc": True,
                "hmm_lookback": 100,
                "use_bcpd": True
            },
        }
    )
    
    # Train HMM first to ensure valid signals if possible
    try:
        backtester.strategy.train(df.iloc[:400])
    except:
        pass # Best effort
        
    result = backtester.run(df, train_ratio=0.0)
    
    # Extract DataFrames
    prices = df
    signals = result.signals_df
    # We might need to access internal strategy states if we want to plot indicators like KAMA, HMM, BCPD.
    # The current strategy backtester doesn't readily expose the full indicator history unless we modify it 
    # or re-calculate it. 
    # For now, we visualize Price + Buy/Sell Markers + Equity Curve (if trades exist).
    
    # Create Subplots: Price, Equity
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3],
                        subplot_titles=("Price & Signals", "Equity Curve"))

    # 1. Candlestsick
    fig.add_trace(go.Candlestick(
        x=prices.index,
        open=prices['open'], high=prices['high'],
        low=prices['low'], close=prices['close'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Buy/Sell Markers
    if result.trades:
        buy_times = [t.entry_time for t in result.trades if t.side == 'long']
        buy_prices = [t.entry_price for t in result.trades if t.side == 'long']
        sell_times = [t.entry_time for t in result.trades if t.side == 'short']
        sell_prices = [t.entry_price for t in result.trades if t.side == 'short']
        
        exit_times = [t.exit_time for t in result.trades]
        exit_prices = [t.exit_price for t in result.trades]

        fig.add_trace(go.Scatter(
            x=buy_times, y=buy_prices, mode='markers', 
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Long Entry'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sell_times, y=sell_prices, mode='markers', 
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Short Entry'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices, mode='markers', 
            marker=dict(symbol='x', size=8, color='black'),
            name='Exit'
        ), row=1, col=1)

    # 3. Equity Curve
    if not result.equity_curve.empty:
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index, 
            y=result.equity_curve['portfolio_value'],
            mode='lines', name='Equity', line=dict(color='blue')
        ), row=2, col=1)

    # Layout Updates
    fig.update_layout(
        title=f"Backtest Analysis: {symbol}",
        yaxis_title="Price",
        yaxis2_title="Capital",
        xaxis_rangeslider_visible=False,
        height=800
    )

    output_path = project_root / "backtest_chart.html"
    fig.write_html(str(output_path))
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    run_visualization()
