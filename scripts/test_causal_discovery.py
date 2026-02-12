
import io
import sys
import os
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.causal_inference import CausalVolatilityTrading

class BinanceVisionDownloader:
    """
    Downloads historical data directly from data.binance.vision
    """
    BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
    
    COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", 
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]
    
    def __init__(self, cache_dir="data/binance_vision"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_month(self, symbol, interval, year, month):
        """Download and parse a specific month's ZIP file"""
        month_str = f"{month:02d}"
        filename = f"{symbol}-{interval}-{year}-{month_str}.zip"
        url = f"{self.BASE_URL}/{symbol}/{interval}/{filename}"
        
        cache_path = os.path.join(self.cache_dir, filename.replace('.zip', '.csv'))
        
        if os.path.exists(cache_path):
            # print(f"Loading cached {filename}...")
            return pd.read_csv(cache_path)
            
        print(f"Downloading {url}...")
        try:
            response = requests.get(url)
            if response.status_code == 404:
                print(f"Data not found for {year}-{month} (404)")
                return pd.DataFrame()
            
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # The zip usually contains one csv file named identically to the zip but .csv
                csv_filename = filename.replace('.zip', '.csv')
                with z.open(csv_filename) as f:
                    # Binance monthly data has NO header
                    df = pd.read_csv(f, names=self.COLUMNS, header=None)
                    # Cache it
                    df.to_csv(cache_path, index=False)
                    return df
                    
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return pd.DataFrame()

    def get_data(self, symbol, interval, start_date, end_date):
        """Fetch data for a range of months"""
        dfs = []
        current = start_date.replace(day=1)
        now = datetime.now()
        
        while current <= end_date:
            # Skip future or current month (Binance Vision only has past months)
            if current.year == now.year and current.month == now.month:
                current = current.replace(month=current.month + 1 if current.month < 12 else 1, 
                                          year=current.year + 1 if current.month == 12 else current.year)
                continue

            df = self.download_month(symbol, interval, current.year, current.month)
            if not df.empty:
                dfs.append(df)
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        if not dfs:
            return pd.DataFrame()
            
        full_df = pd.concat(dfs).sort_values('open_time')
        
        # DEBUG
        print(f"\n[DEBUG] {symbol} raw open_time head:")
        print(full_df['open_time'].head())
        print(f"Max: {full_df['open_time'].max()}")
        
        # Ensure open_time is numeric
        full_df['open_time'] = pd.to_numeric(full_df['open_time'], errors='coerce')
        full_df = full_df.dropna(subset=['open_time'])
        
        # Check scale
        mean_ts = full_df['open_time'].mean()
        if mean_ts > 1e14: # Microseconds?
             print(f"[WARN] Timestamp detected as micros/nanos ({mean_ts}). Scaling down.")
             full_df['open_time'] = full_df['open_time'] / 1000
             if mean_ts > 1e17: # Nanos
                 full_df['open_time'] = full_df['open_time'] / 1000

        # Convert to datetime (ms)
        try:
            full_df['timestamp'] = pd.to_datetime(full_df['open_time'], unit='ms')
        except Exception as e:
            print(f"[ERROR] Date conversion failed: {e}")
            print(full_df['open_time'].head())
            return pd.DataFrame()
            
        full_df.set_index('timestamp', inplace=True)
        
        # Filter range
        full_df = full_df[start_date:end_date]
        
        # Keep numeric cols
        num_cols = ['open', 'high', 'low', 'close', 'volume']
        full_df = full_df[num_cols].astype(float)
        
        return full_df

def run_causal_test():
    downloader = BinanceVisionDownloader()
    
    # Configuration
    start_date = datetime.now() - timedelta(days=180) # Last 6 months
    end_date = datetime.now()
    interval = '1h'
    
    # Leader
    leader_symbol = 'BTCUSDT'
    
    # Universe of Potential Followers (High Volume pairs)
    follower_symbols = [
        'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
        'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'TRXUSDT',
        'DOTUSDT', 'LINKUSDT', 'LTCUSDT',
        'UNIUSDT', 'ATOMUSDT', 'ETCUSDT', 'FILUSDT', 'ICPUSDT', 
        'RENDERUSDT', 'KASUSDT', 'HBARUSDT', 'ONDOUSDT', 
        'NEARUSDT', 'JASMYUSDT', 'JUPUSDT', 'ENAUSDT',
        'APTUSDT', 'SUIUSDT'
    ]
    
    print(f"--- Fetching Data (6 Months, {interval}) ---")
    
    # Fetch Leader
    print(f"\nFetching LEADER: {leader_symbol}")
    leader_df = downloader.get_data(leader_symbol, interval, start_date, end_date)
    
    if leader_df.empty:
        print("Critical Error: Could not fetch Leader data. Exiting.")
        return

    # Fetch Followers
    universe_data = {}
    
    # Add leader to universe for clustering context
    universe_data[leader_symbol] = leader_df
    
    for sym in follower_symbols:
        print(f"Fetching {sym}...", end="\r")
        df = downloader.get_data(sym, interval, start_date, end_date)
        if not df.empty and len(df) > 500:
            universe_data[sym] = df
    print("\nData fetched.")

    # Initialize Causal Engine
    causal_engine = CausalVolatilityTrading(lookback_period=24*30) # 1 month lookback for causality check windows? 
    # Actually checking entire 6 months for this test report
    
    # 1. Run Clustering (Just to see where they land)
    print("\n--- Phase 1: GMM Clustering (Volatility Regimes) ---")
    clusters = causal_engine.cluster_universe(universe_data)
    for cid, tickers in clusters.items():
        regime = "Mid-Vol (Tradeable)" if cid == causal_engine.mid_vol_cluster_id else "Other"
        print(f"Cluster {cid} [{regime}]: {tickers}")

    # 2. Run Causality Tests (BTC vs Universe)
    print(f"\n--- Phase 2: Causal Discovery (Leader: {leader_symbol}) ---")
    
    results = []
    
    for ticker, df in universe_data.items():
        if ticker == leader_symbol:
            continue
            
        # Align data
        aligned_leader, aligned_follower = leader_df.align(df, join='inner', axis=0)
        
        if len(aligned_leader) < 500:
            print(f"Skipping {ticker}: Insufficient overlapping data ({len(aligned_leader)})")
            continue
            
        # A. Granger Causality
        # We need to reshape for the function: check_granger_causality(leader_series, follower_series)
        # The current implementation checks if Leader -> Follower
        is_causal, f_score = causal_engine.check_granger_causality(
            aligned_leader['close'], 
            aligned_follower['close'], 
            max_lag=24 # Check up to 24h lag
        )
        
        # B. Optimal Lag (DTW/Correlation)
        optimal_lag, correlation = causal_engine.estimate_lead_lag(
            aligned_leader['close'], 
            aligned_follower['close'], 
            max_lag=48
        )
        
        # C. Transfer Entropy (ETE/STE)
        # Using a subset for speed if needed, but 4000 points is fast enough for STE
        try:
             ete = causal_engine.calculate_transfer_entropy(
                aligned_leader['close'], 
                aligned_follower['close'], 
                lag=max(1, optimal_lag)
            )
        except Exception as e:
            ete = 0.0
            
        results.append({
            'Ticker': ticker,
            'Granger Causal': is_causal,
            'F-Score': round(f_score, 2),
            'Optimal Lag (hrs)': optimal_lag,
            'Correlation': round(correlation, 4),
            'Transfer Entropy': round(ete, 4),
            'Data Points': len(aligned_leader)
        })
        
    # Display Results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Sort by Transfer Entropy (Information Flow)
        results_df = results_df.sort_values(by='Transfer Entropy', ascending=False)
        
        print("\n=== Causal Inference Results ===")
        print(results_df.to_string(index=False))
        
        # Save
        results_df.to_csv("reports/causal_discovery_results.csv", index=False)
        print("\nReport saved to reports/causal_discovery_results.csv")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    run_causal_test()
