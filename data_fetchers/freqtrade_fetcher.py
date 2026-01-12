
import subprocess
import pandas as pd
import os
import json
import logging
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)

class FreqtradeFetcher:
    """
    Wrapper around Freqtrade CLI to fetch and load data.
    """
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = config.get('data_dir', 'user_data/data/binance')
        self.exchange = config.get('exchange', 'binance')
        self.format = config.get('format', 'json')
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except OSError as e:
                logger.warning(f"Could not create data directory {self.data_dir}: {e}")

    def fetch_ohlcv(self, symbol: str, timeframe: str, days: int = 100) -> pd.DataFrame:
        """
        Download data via Freqtrade CLI and load into DataFrame.
        """
        # Convert symbol to Freqtrade format (e.g., BTC/USDT)
        # Freqtrade expects standard slash notation
        ft_symbol = symbol.replace('-', '/')
        
        # 1. Download Data
        cmd = [
            "freqtrade", "download-data",
            "--pairs", ft_symbol,
            "--timeframes", timeframe,
            "--days", str(days),
            "--exchange", self.exchange,
            "--format", self.format,
            "--data-dir", os.path.dirname(self.data_dir) # Freqtrade appends /data/{exchange}
        ]
        
        # If user data dir is custom, we might need to adjust. 
        # Freqtrade usually outputs to {user_data_dir}/data/{exchange}
        # The --data-dir arg specifies the user_data dir in older versions 
        # or the root data dir in newer ones.
        # Let's try to assume 'user_data' is the root context
        
        # Simplified: We just run it and expect it to land in default or configured location
        try:
            logger.info(f"Executing Freqtrade command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Freqtrade download failed: {e.stderr}")
            return pd.DataFrame()
        except FileNotFoundError:
            logger.error("Freqtrade executable not found. Is it installed and in PATH?")
            return pd.DataFrame()

        # 2. Load Data
        return self._load_data(ft_symbol, timeframe)

    def _load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load downloaded data from disk.
        """
        # Freqtrade filename format: Pair_Timeframe.json
        # e.g. BTC_USDT_1h.json
        filename_symbol = symbol.replace('/', '_')
        filename = f"{filename_symbol}-{timeframe}.{self.format}"
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found at: {file_path}")
            return pd.DataFrame()
            
        try:
            if self.format == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'], unit='ms') # Freqtrade uses timestamp ms
            elif self.format == 'feather':
                df = pd.read_feather(file_path)
            # Add other formats as needed
            
            # Standardize
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Rename columns to lowercase if needed (Freqtrade usually is lowercase)
            df.columns = [c.lower() for c in df.columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data file {file_path}: {e}")
            return pd.DataFrame()
