
import sys
import os
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.multi_layer_strategy import MultiLayerStrategy, CausalContext
from backtesting.backtest_engine import BacktestEngine
from scripts.test_causal_discovery import BinanceVisionDownloader

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'causal_best_params.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return {
            "kama_period": 25,
            "kama_fast": 2,
            "kama_slow": 30,
            "adx_threshold": 20,
            "aroon_threshold": 60,
            "atr_stop_multiplier": 2.0,
            "rsi_period": 11,
            "use_fisher_timing": True,
            "use_smc": False
        }

def run_universe_backtest():
    downloader = BinanceVisionDownloader()
    
    # Configuration
    start_date = datetime.now() - timedelta(days=180) # Last 6 months
    end_date = datetime.now()
    interval = '1h'
    
    leader_symbol = 'BTCUSDT'
    
    # Asset Universe
    follower_symbols = [
        'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'TRXUSDT',
        'DOTUSDT', 'LINKUSDT', 'LTCUSDT',
        'UNIUSDT', 'ATOMUSDT', 'ETCUSDT', 'FILUSDT', 'ICPUSDT', 
        'RENDERUSDT', 'KASUSDT', 'HBARUSDT', 'ONDOUSDT', 
        'NEARUSDT', 'JASMYUSDT', 'JUPUSDT', 'ENAUSDT',
        'APTUSDT', 'SUIUSDT'
    ]
    
    # 1. Fetch Leader Data
    print(f"Fetching LEADER {leader_symbol}...")
    leader_df = downloader.get_data(leader_symbol, interval, start_date, end_date)
    if leader_df.empty:
        print("Failed to fetch Leader. Exiting.")
        return

    # 2. Setup Params
    optimized_params = load_config()
    print(f"Loaded Optimized Params: {optimized_params}")
    
    full_params = {
        'initial_capital': 10000,
        'commission': 0.001,
        **optimized_params
    }

    results = []
    
    # 3. Iterate Universe
    print(f"\n--- Running Backtest on {len(follower_symbols)} Assets ---")
    
    for symbol in follower_symbols:
        print(f"Processing {symbol}...", end='\r')
        
        # Fetch Data
        df = downloader.get_data(symbol, interval, start_date, end_date)
        if df.empty or len(df) < 500:
            print(f"Skipping {symbol}: Insufficient Data")
            continue
            
        # Align with Leader
        aligned_leader, aligned_follower = leader_df.align(df, join='inner', axis=0)
        
        if len(aligned_follower) < 500:
            continue
            
        # Setup Strategy
        strategy = MultiLayerStrategy(full_params)
        
        # Inject Causal Context
        ctx = CausalContext(
            leader_df=aligned_leader,
            optimal_lag=1, # Default lag
            ete_rising=True,
            leader_name="BTC",
            is_valid=True
        )
        strategy.set_leader_context(ctx)
        
        # Run Backtest
        try:
            signals = strategy.generate_signals(aligned_follower)
            engine = BacktestEngine(full_params)
            backtest_res = engine.run(aligned_follower, {'causal': signals})
            
            # Store Metrics
            results.append({
                'Symbol': symbol,
                'Total Return': backtest_res['total_return'],
                'Max Drawdown': backtest_res['max_drawdown'],
                'Sharpe Ratio': backtest_res['sharpe_ratio'],
                'Win Rate': backtest_res['win_rate'],
                'Trades': len(backtest_res['trades']) # Fix: Use len(trades) list
            })
            
        except Exception as e:
            print(f"\nError processing {symbol}: {e}")
            continue
            
    # 4. Aggregate & Report
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Total Return', ascending=False)
        
        print("\n=== Universe Backtest Results (Ranked by Return) ===")
        print(results_df.to_string(index=False))
        
        # Save
        os.makedirs('reports', exist_ok=True)
        results_df.to_csv('reports/universe_backtest_results.csv', index=False)
        print(f"\nSaved to reports/universe_backtest_results.csv")
        
        # Portfolio Stats
        avg_ret = results_df['Total Return'].mean()
        avg_dd = results_df['Max Drawdown'].mean()
        positive_assets = len(results_df[results_df['Total Return'] > 0])
        
        print(f"\n--- Portfolio Summary ---")
        print(f"Average Return: {avg_ret:.2%}")
        print(f"Average Max Drawdown: {avg_dd:.2%}")
        print(f"Profitable Assets: {positive_assets}/{len(results_df)}")
        
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_universe_backtest()
