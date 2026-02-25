import pandas as pd
import numpy as np

class ADX:
    """Average Directional Index"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate ADX
        """
        df = df.copy()
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        
        # Calculate TR (True Range)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate DM (Directional Movement)
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Smooth TR and DM
        # Standard ADX uses Wilder's Smoothing which is roughly EMA(2*n-1) or similar.
        # But often simpler is rolling sum for first, then smooth.
        # Let's use Wilder's Smoothing: next = prev - (prev/n) + current
        # Which is equivalent to EMA with alpha=1/n
        
        # Using ewm(alpha=1/period, adjust=False) matches Wilder's
        
        tr_smooth = df['tr'].ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = df['plus_dm'].ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = df['minus_dm'].ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (Smooth DX)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx

    @staticmethod
    def calculate_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate ADX and DMI (+DI, -DI)
        """
        df = df.copy()
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        
        # Calculate TR (True Range)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate DM (Directional Movement)
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        tr_smooth = df['tr'].ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = df['plus_dm'].ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = df['minus_dm'].ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX (Smooth DX)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
