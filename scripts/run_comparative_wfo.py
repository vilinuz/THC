import sys
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_causal_discovery import BinanceVisionDownloader
from backtesting.wfo_engine import WalkForwardOptimizationEngine
from strategy.strategy_beta import StrategyBeta
from strategy.strategy_gamma import StrategyGamma
from strategy.regimeline_strategy import RegimelineStrategy

def mock_hmm_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Mock HMM states for Strategy Beta (used in earlier tests)."""
    df_out = df.copy()
    vol = df_out['close'].pct_change().rolling(24).std()
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    vol_norm = vol_norm.fillna(0.5)
    
    state_1, state_2, state_3 = [], [], []
    for v in vol_norm:
        if v > 0.7:
            s1, s2, s3 = 0.1, 0.1, 0.8
        elif v < 0.3:
            s1, s2, s3 = 0.2, 0.7, 0.1
        else:
            s1, s2, s3 = 0.7, 0.2, 0.1
        s1 += random.uniform(-0.1, 0.1)
        s2 += random.uniform(-0.1, 0.1)
        s3 += random.uniform(-0.1, 0.1)
        s1, s2, s3 = max(0, s1), max(0, s2), max(0, s3)
        total = s1 + s2 + s3
        state_1.append(s1/total)
        state_2.append(s2/total)
        state_3.append(s3/total)
        
    df_out['hmm_state_1_prob'] = state_1
    df_out['hmm_state_2_prob'] = state_2
    df_out['hmm_state_3_prob'] = state_3
    return df_out

def run_strategy_wfo(data, strategy_class, param_grid, symbol, strategy_name, sa_iterations=30):
    print(f"\n--- Running WFO for {strategy_name} on {symbol} ---")
    
    # Enrich data if Beta
    if strategy_name == "StrategyBeta":
        df_for_strat = mock_hmm_regimes(data)
    else:
        df_for_strat = data
        
    wfo_engine = WalkForwardOptimizationEngine(
        data=df_for_strat,
        strategy_class=strategy_class,
        param_grid=param_grid,
        l_train_days=180, # 6 months
        l_test_days=30,   # 1 month step
        optimization_method="simulated_annealing",
        sa_initial_temp=15.0,
        sa_cooling_rate=0.92,
        sa_iterations=sa_iterations
    )
    
    wfe, oos_returns, _ = wfo_engine.run()
    
    # Base stats
    cum_ret = (1 + oos_returns).prod() - 1 if not oos_returns.empty else 0
    ann_ret = oos_returns.mean() * 24 * 365 if not oos_returns.empty else 0
    shp = (oos_returns.mean() / (oos_returns.std() + 1e-9) * np.sqrt(24 * 365)) if not oos_returns.empty else 0
    
    result = {
        'Strategy': strategy_name,
        'Symbol': symbol,
        'Walk-Forward Efficency (%)': round(wfe, 2),
        'Cumulative OOS Return (%)': round(cum_ret * 100, 2),
        'Annualized OOS Return (%)': round(ann_ret * 100, 2),
        'Sharpe Ratio': round(shp, 2)
    }
    return result

def main():
    downloader = BinanceVisionDownloader()
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    interval = '1h'
    months = 9
    start_date = datetime.now() - timedelta(days=30*months)
    end_date = datetime.now()
    
    # Pre-define param grids
    param_grids = {
        'StrategyBeta': {
            'ema_fast': [5, 9, 15],
            'ema_slow': [20, 26, 30],
            'bb_lookback': [15, 20, 30],
            'bb_std': [1.5, 2.0, 2.5],
            'rsi_period': [10, 14, 21]
        },
        'StrategyGamma': {
            'adx_period': [10, 14, 20],
            'adx_trend_threshold': [20, 25],
            'chop_period': [10, 14, 20],
            'chop_trend_threshold': [50.0, 61.8],
            't3_fast_len': [5, 8],
            't3_slow_len': [15, 21],
            't3_volume_factor': [0.6, 0.7],
            'fisher_period': [9, 14, 20],
            'fisher_overbought': [1.5, 2.0],
            'fisher_oversold': [-2.0, -1.5]
        },
        'RegimelineStrategy': {
            'ema_fast_len': [10, 20],
            'ema_slow_len': [40, 60],
            'atr_len': [14, 20],
            'adx_bull_min': [20, 25],
            'bull_pb_atr': [0.3, 0.5],
            'bear_buf_atr': [0.3, 0.5]
        }
    }
    
    strategies_to_test = [
        ("StrategyBeta", StrategyBeta),
        ("StrategyGamma", StrategyGamma),
        ("RegimelineStrategy", RegimelineStrategy)
    ]
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n=============================================")
        print(f"Fetching {months} months of {symbol} {interval} Data")
        print(f"=============================================")
        df = downloader.get_data(symbol, interval, start_date, end_date)
        
        if df.empty or len(df) < 500:
            print(f"Could not load enough data for {symbol}.")
            continue
            
        print(f"{symbol} Data loaded: {len(df)} bars")
        
        for name, strat_class in strategies_to_test:
            res = run_strategy_wfo(
                data=df,
                strategy_class=strat_class,
                param_grid=param_grids[name],
                symbol=symbol,
                strategy_name=name,
                sa_iterations=25 # Slightly lower for multiple runs
            )
            all_results.append(res)
            
    # Format and save
    results_df = pd.DataFrame(all_results)
    print("\n\n=== FINAL COMPARATIVE WFO RESULTS ===")
    print(results_df.to_string(index=False))
    
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/comparative_wfo_results.csv'
    results_df.to_csv(report_path, index=False)
    print(f"\nComparative results saved to {report_path}")

if __name__ == "__main__":
    main()
