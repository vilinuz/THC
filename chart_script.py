import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create a system architecture diagram using plotly
# Define node positions for each layer
fig = go.Figure()

# Define colors for different layers (using brand colors)
colors = {
    'datasource': '#1FB8CD',  # Strong cyan
    'analysis': '#2E8B57',    # Sea green  
    'ml': '#D2BA4C',          # Moderate yellow
    'strategy': '#5D878F',    # Cyan
    'execution': '#DB4545',   # Bright red
    'infrastructure': '#B4413C' # Moderate red
}

# Define component positions and labels
components = [
    # Data Sources Layer (y=5)
    {'name': 'Binance API', 'x': 1, 'y': 5, 'color': 'datasource'},
    {'name': 'yfinance', 'x': 3, 'y': 5, 'color': 'datasource'},
    
    # Data Processing Layer (y=4)
    {'name': 'Data Fetchers', 'x': 2, 'y': 4, 'color': 'datasource'},
    {'name': 'DuckDB Storage', 'x': 0.5, 'y': 3.5, 'color': 'datasource'},
    {'name': 'Parquet Files', 'x': 2, 'y': 3.5, 'color': 'datasource'},
    {'name': 'Redis Cache', 'x': 3.5, 'y': 3.5, 'color': 'datasource'},
    
    # Analysis Layer (y=3)
    {'name': 'Tech Indicators', 'x': 0.5, 'y': 2.5, 'color': 'analysis'},
    {'name': 'Smart Money', 'x': 2, 'y': 2.5, 'color': 'analysis'},
    {'name': 'Feature Eng', 'x': 3.5, 'y': 2.5, 'color': 'analysis'},
    
    # ML Layer (y=2)
    {'name': 'XGBoost', 'x': 0.5, 'y': 1.5, 'color': 'ml'},
    {'name': 'LSTM', 'x': 2, 'y': 1.5, 'color': 'ml'},
    {'name': 'LLM Analyzer', 'x': 3.5, 'y': 1.5, 'color': 'ml'},
    {'name': 'Signal Aggreg', 'x': 2, 'y': 1, 'color': 'ml'},
    
    # Strategy Layer (y=1)
    {'name': 'Trading Strats', 'x': 0.5, 'y': 0.5, 'color': 'strategy'},
    {'name': 'Bayesian Opt', 'x': 2, 'y': 0.5, 'color': 'strategy'},
    {'name': 'Walk-Forward', 'x': 3.5, 'y': 0.5, 'color': 'strategy'},
    
    # Execution Layer (y=0)
    {'name': 'Backtest Eng', 'x': 1, 'y': -0.5, 'color': 'execution'},
    {'name': 'Position Mgr', 'x': 3, 'y': -0.5, 'color': 'execution'},
    {'name': 'Perf Metrics', 'x': 2, 'y': -1, 'color': 'execution'},
    
    # Output Layer (y=-1)
    {'name': 'PDF Reports', 'x': 0.5, 'y': -1.5, 'color': 'execution'},
    {'name': 'Dashboard', 'x': 2, 'y': -1.5, 'color': 'execution'},
    {'name': 'Trading Daemon', 'x': 3.5, 'y': -1.5, 'color': 'execution'},
    
    # Infrastructure (side)
    {'name': 'Config Mgr', 'x': 5, 'y': 2, 'color': 'infrastructure'},
    {'name': 'Logger', 'x': 5, 'y': 1, 'color': 'infrastructure'},
    {'name': 'Risk Manager', 'x': 5, 'y': 0, 'color': 'infrastructure'},
]

# Create component lookup for connections
comp_lookup = {comp['name']: i for i, comp in enumerate(components)}

# Define connections (source, target)
connections = [
    ('Binance API', 'Data Fetchers'),
    ('yfinance', 'Data Fetchers'),
    ('Data Fetchers', 'DuckDB Storage'),
    ('Data Fetchers', 'Parquet Files'),
    ('Data Fetchers', 'Redis Cache'),
    ('DuckDB Storage', 'Tech Indicators'),
    ('DuckDB Storage', 'Smart Money'),
    ('DuckDB Storage', 'Feature Eng'),
    ('Redis Cache', 'Tech Indicators'),
    ('Redis Cache', 'Smart Money'),
    ('Redis Cache', 'Feature Eng'),
    ('Tech Indicators', 'XGBoost'),
    ('Tech Indicators', 'LSTM'),
    ('Tech Indicators', 'LLM Analyzer'),
    ('Smart Money', 'XGBoost'),
    ('Smart Money', 'LSTM'),
    ('Smart Money', 'LLM Analyzer'),
    ('Feature Eng', 'XGBoost'),
    ('Feature Eng', 'LSTM'),
    ('Feature Eng', 'LLM Analyzer'),
    ('XGBoost', 'Signal Aggreg'),
    ('LSTM', 'Signal Aggreg'),
    ('LLM Analyzer', 'Signal Aggreg'),
    ('Signal Aggreg', 'Trading Strats'),
    ('Signal Aggreg', 'Bayesian Opt'),
    ('Signal Aggreg', 'Walk-Forward'),
    ('Trading Strats', 'Backtest Eng'),
    ('Bayesian Opt', 'Backtest Eng'),
    ('Walk-Forward', 'Backtest Eng'),
    ('Backtest Eng', 'Position Mgr'),
    ('Backtest Eng', 'Perf Metrics'),
    ('Perf Metrics', 'PDF Reports'),
    ('Perf Metrics', 'Dashboard'),
    ('Position Mgr', 'Trading Daemon'),
]

# Add connection lines
for source, target in connections:
    if source in comp_lookup and target in comp_lookup:
        src_idx = comp_lookup[source]
        tgt_idx = comp_lookup[target]
        src_comp = components[src_idx]
        tgt_comp = components[tgt_idx]
        
        fig.add_trace(go.Scatter(
            x=[src_comp['x'], tgt_comp['x']],
            y=[src_comp['y'], tgt_comp['y']],
            mode='lines',
            line=dict(color='#333333', width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))

# Add nodes
for comp in components:
    fig.add_trace(go.Scatter(
        x=[comp['x']],
        y=[comp['y']],
        mode='markers+text',
        marker=dict(
            size=80,
            color=colors[comp['color']],
            line=dict(width=2, color='white')
        ),
        text=comp['name'],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name=comp['color'].title(),
        legendgroup=comp['color'],
        showlegend=comp['color'] not in [trace.legendgroup for trace in fig.data if hasattr(trace, 'legendgroup')],
        hoverinfo='text',
        hovertext=comp['name']
    ))

# Add layer labels on the left side
layer_labels = [
    {'label': 'Data Sources', 'y': 5},
    {'label': 'Data Processing', 'y': 3.75},
    {'label': 'Analysis & Compute', 'y': 2.5},
    {'label': 'ML & Signals', 'y': 1.25},
    {'label': 'Strategy & Optimize', 'y': 0.5},
    {'label': 'Execution & Output', 'y': -1.25},
    {'label': 'Infrastructure', 'y': 1}
]

for layer in layer_labels:
    fig.add_annotation(
        x=-0.8,
        y=layer['y'],
        text=layer['label'],
        showarrow=False,
        font=dict(size=11, color='#333333'),
        textangle=-90,
        xanchor='center',
        yanchor='middle'
    )

fig.update_layout(
    title='Crypto Trading Platform Architecture',
    showlegend=True,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5
    ),
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-1.5, 6]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-2, 5.5]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Save the chart
fig.write_image('crypto_trading_architecture.png')
fig.write_image('crypto_trading_architecture.svg', format='svg')

print("System architecture chart created successfully!")
print("Components organized by functional layers with clear data flow")