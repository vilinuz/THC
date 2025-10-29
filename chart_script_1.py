import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create a flowchart using Plotly with shapes and annotations
fig = go.Figure()

# Define positions for each step (x, y coordinates)
# Step 1 - Data Collection (Blue)
step1_positions = [(0.5, 0.95), (0.5, 0.90), (0.5, 0.85)]
step1_labels = ["Raw OHLCV Data", "Data Validation", "Data Storage"]

# Step 2 - Feature Engineering (Green) 
step2_positions = [(0.5, 0.78), (0.2, 0.70), (0.4, 0.70), (0.6, 0.70), (0.8, 0.70), (0.5, 0.62)]
step2_labels = ["OHLCV Processing", "Price Features", "Volume Features", "Technical Features", "Time Features", "Feature Matrix"]

# Step 3 - Model Training (Orange)
step3_positions = [(0.5, 0.55), (0.2, 0.48), (0.5, 0.48), (0.8, 0.48)]
step3_labels = ["Train/Test Split", "XGBoost", "LSTM", "LLM"]

# Step 4 - Optimization (Purple)
step4_positions = [(0.35, 0.40), (0.65, 0.40), (0.5, 0.33)]
step4_labels = ["Bayesian Optimizer", "Walk-Forward Valid", "Optimized Models"]

# Step 5 - Signal Generation (Red)
step5_positions = [(0.2, 0.26), (0.5, 0.26), (0.8, 0.26), (0.5, 0.19)]
step5_labels = ["Buy/Sell Signals", "Confidence Scores", "Risk Metrics", "Signal Aggregator"]

# Step 6 - Validation (Yellow)
step6_positions = [(0.5, 0.12), (0.5, 0.05)]
step6_labels = ["Signal Validator", "Valid Signals"]

# Step 7 - Output
step7_positions = [(0.2, -0.02), (0.5, -0.02), (0.8, -0.02)]
step7_labels = ["Strategy Executor", "Backtest Engine", "Database"]

# Color scheme
colors = {
    'data': '#B3E5EC',      # Light cyan
    'process': '#A5D6A7',   # Light green  
    'model': '#FFCDD2',     # Light red
    'optimize': '#D8BFD8',  # Light purple
    'signal': '#FFCDD2',    # Light red
    'validate': '#FFEB8A'   # Light yellow
}

# Function to add boxes and text
def add_box(fig, x, y, text, color, width=0.12, height=0.04):
    fig.add_shape(
        type="rect",
        x0=x-width/2, y0=y-height/2,
        x1=x+width/2, y1=y+height/2,
        fillcolor=color,
        line=dict(color="#333333", width=1)
    )
    fig.add_annotation(
        x=x, y=y,
        text=text,
        showarrow=False,
        font=dict(size=10, color="#133343"),
        xanchor="center",
        yanchor="middle"
    )

# Function to add arrows
def add_arrow(fig, x1, y1, x2, y2):
    fig.add_annotation(
        x=x2, y=y2,
        ax=x1, ay=y1,
        xref="x", yref="y",
        axref="x", ayref="y",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="#333333",
        showarrow=True,
        text=""
    )

# Add Step 1 - Data Collection (Blue)
for i, (pos, label) in enumerate(zip(step1_positions, step1_labels)):
    add_box(fig, pos[0], pos[1], label, colors['data'])
    if i < len(step1_positions) - 1:
        add_arrow(fig, pos[0], pos[1]-0.02, step1_positions[i+1][0], step1_positions[i+1][1]+0.02)

# Add Step 2 - Feature Engineering (Green)
add_box(fig, step2_positions[0][0], step2_positions[0][1], step2_labels[0], colors['process'])
add_arrow(fig, 0.5, 0.83, 0.5, 0.80)

# Feature branches
for i in range(1, 5):
    add_box(fig, step2_positions[i][0], step2_positions[i][1], step2_labels[i], colors['process'], width=0.10)
    add_arrow(fig, 0.5, 0.76, step2_positions[i][0], step2_positions[i][1]+0.02)

# Feature Matrix
add_box(fig, step2_positions[5][0], step2_positions[5][1], step2_labels[5], colors['process'])
for i in range(1, 5):
    add_arrow(fig, step2_positions[i][0], step2_positions[i][1]-0.02, 0.5, 0.64)

# Add Step 3 - Model Training (Orange)
add_box(fig, step3_positions[0][0], step3_positions[0][1], step3_labels[0], colors['model'])
add_arrow(fig, 0.5, 0.60, 0.5, 0.57)

for i in range(1, 4):
    add_box(fig, step3_positions[i][0], step3_positions[i][1], step3_labels[i], colors['model'], width=0.10)
    add_arrow(fig, 0.5, 0.53, step3_positions[i][0], step3_positions[i][1]+0.02)

# Add Step 4 - Optimization (Purple)
for i, (pos, label) in enumerate(zip(step4_positions[:2], step4_labels[:2])):
    add_box(fig, pos[0], pos[1], label, colors['optimize'], width=0.13)

add_box(fig, step4_positions[2][0], step4_positions[2][1], step4_labels[2], colors['optimize'])

# Arrows from models to optimizers
for model_pos in step3_positions[1:]:
    add_arrow(fig, model_pos[0], model_pos[1]-0.02, 0.35, 0.42)  # To Bayesian
    add_arrow(fig, model_pos[0], model_pos[1]-0.02, 0.65, 0.42)  # To Walk-Forward

# Arrows from optimizers to optimized models
add_arrow(fig, 0.35, 0.38, 0.5, 0.35)
add_arrow(fig, 0.65, 0.38, 0.5, 0.35)

# Add Step 5 - Signal Generation (Red)
for i, (pos, label) in enumerate(zip(step5_positions[:3], step5_labels[:3])):
    add_box(fig, pos[0], pos[1], label, colors['signal'], width=0.11)
    add_arrow(fig, 0.5, 0.31, pos[0], pos[1]+0.02)

add_box(fig, step5_positions[3][0], step5_positions[3][1], step5_labels[3], colors['signal'])
for i in range(3):
    add_arrow(fig, step5_positions[i][0], step5_positions[i][1]-0.02, 0.5, 0.21)

# Add Step 6 - Validation (Yellow)
for i, (pos, label) in enumerate(zip(step6_positions, step6_labels)):
    add_box(fig, pos[0], pos[1], label, colors['validate'])
    if i == 0:
        add_arrow(fig, 0.5, 0.17, 0.5, 0.14)
    else:
        add_arrow(fig, 0.5, 0.10, 0.5, 0.07)

# Add Step 7 - Output (Blue)
for i, (pos, label) in enumerate(zip(step7_positions, step7_labels)):
    add_box(fig, pos[0], pos[1], label, colors['data'], width=0.11)
    add_arrow(fig, 0.5, 0.03, pos[0], pos[1]+0.02)

# Update layout
fig.update_layout(
    title="ML & Signal Generation Pipeline",
    xaxis=dict(range=[-0.05, 1.05], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[-0.08, 1.0], showgrid=False, showticklabels=False, zeroline=False),
    showlegend=False,
    plot_bgcolor='white',
    font=dict(family="Arial", size=12)
)

# Save the chart
fig.write_image("ml_pipeline_flowchart.png")
fig.write_image("ml_pipeline_flowchart.svg", format="svg")

print("ML Pipeline flowchart created successfully!")