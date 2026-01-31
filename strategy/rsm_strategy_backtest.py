import pandas as pd
import numpy as np

# Fetch Data
data = yf.download("ETH-USD", start="2023-01-01", end="2024-01-01", interval="1d", progress=False)

system = QuantSystem()
logs = []

print("Running Streaming Backtest... (This simulates live trading)")

for date, row in data.iterrows():
    # Simulate the live feed
    price = row['Close']
    vol = row['Volume']
    
    # Run the System
    signal, shock, bull_prob, smooth = system.ingest(price, vol)
    
    # Execute Trades (Simplified)
    if signal.startswith("BUY") and system.position == 0:
        system.position = 1 # Enter
        entry_price = price
    elif signal.startswith("SELL") and system.position == 1:
        system.position = 0 # Exit
        pnl = (price - entry_price) / entry_price
        system.equity.append(system.equity[-1] * (1 + pnl))
    
    logs.append({
        'Date': date,
        'Price': price,
        'SmoothPrice': smooth,
        'Signal': signal,
        'ShockProb': shock,
        'BullProb': bull_prob,
        'Equity': system.equity[-1]
    })

# ==========================================
# 4. ANALYSIS
# ==========================================
df_res = pd.DataFrame(logs).set_index('Date')

# Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# Price vs Smooth (Kalman)
axes[0].plot(df_res['Price'], color='gray', alpha=0.5, label='Raw Price')
axes[0].plot(df_res['SmoothPrice'], color='blue', label='Kalman Filter')
axes[0].set_title('Step 1: Kalman Noise Cleaning')
axes[0].legend()

# Shock Probability (BCPD)
axes[1].plot(df_res['ShockProb'], color='red')
axes[1].axhline(0.5, linestyle='--', color='black')
axes[1].set_title('Step 2: Bayesian Shock Detection (Emergency Brake)')

# Bull Probability (HMM)
axes[2].plot(df_res['BullProb'], color='green')
axes[2].axhline(0.7, linestyle='--', color='green')
axes[2].axhline(0.3, linestyle='--', color='red')
axes[2].set_title('Step 3: Regime Probability (The Strategist)')

# Equity Curve
axes[3].plot(df_res['Equity'], color='purple', linewidth=2)
axes[3].set_title('Step 4: Portfolio Equity (Walk-Forward Result)')

plt.tight_layout()
plt.show()