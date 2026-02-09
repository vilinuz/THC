import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_backtest_results(symbol: str, df: pd.DataFrame, signals: pd.Series, trades: list, equity_curve: pd.DataFrame, output_dir: str = "plots"):
    """
    Generates an interactive HTML plot for the backtest results.
    
    Args:
        symbol: The asset symbol (e.g., 'ETHUSDT').
        df: OHLCV DataFrame with indicators columns (e.g., 'kama_price', 'adx').
        signals: Series with 1 (Buy), -1 (Sell), 0 (Hold).
        trades: List of trade dictionaries (entry_price, exit_price, entry_time, exit_time, side).
        equity_curve: DataFrame with 'portfolio_value'.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots: Price (Main), Equity, ADX
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{symbol} Price & Signals", "Equity Curve", "ADX Strength")
    )

    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Price'
    ), row=1, col=1)

    # 2. Indicators (KAMA)
    if 'kama_price' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['kama_price'],
            line=dict(color='orange', width=2),
            name='KAMA'
        ), row=1, col=1)

    # 3. Buy/Sell Markers
    # We need exact timestamps and prices for markers.
    # We can extract them from the trades list or the signals series.
    # Using trades list provides executed prices.
    
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    for trade in trades:
        if trade.get('side') == 'long':
            buy_times.append(trade['entry_time'])
            buy_prices.append(trade['entry_price'])
            # If closed, mark exit
            if 'exit_time' in trade:
                sell_times.append(trade['exit_time'])
                sell_prices.append(trade['exit_price'])
        # Short trades logic could be added here (inverse)

    if buy_times:
        fig.add_trace(go.Scatter(
            x=buy_times, y=buy_prices,
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Buy'
        ), row=1, col=1)

    if sell_times:
        fig.add_trace(go.Scatter(
            x=sell_times, y=sell_prices,
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Sell'
        ), row=1, col=1)

    # 4. Equity Curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve['portfolio_value'],
        line=dict(color='blue', width=2),
        fill='tozeroy',
        name='Equity'
    ), row=2, col=1)

    # 5. ADX (Indicator)
    if 'adx' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['adx'],
            line=dict(color='purple', width=1.5),
            name='ADX'
        ), row=3, col=1)
        # Add ADX Threshold line
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]], y=[25, 25],
            mode='lines', line=dict(color='gray', dash='dash'),
            name='Trend Threshold'
        ), row=3, col=1)

    # Layout
    fig.update_layout(
        title=f"{symbol} Backtest Analysis",
        xaxis_rangeslider_visible=False,
        height=900,
        template="plotly_dark"
    )

    filename = f"{output_dir}/{symbol}_backtest.html"
    fig.write_html(filename)
    print(f"Plot saved to {filename}")
    return filename
