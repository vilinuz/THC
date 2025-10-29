import pandas as pd
import numpy as np

class Ichimoku:
    """Ichimoku Cloud trading indicator"""
    
    @staticmethod
    def calculate(
        df: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ) -> pd.DataFrame:
        """
        Calculate all Ichimoku components
        
        Returns DataFrame with all Ichimoku lines
        """
        result = pd.DataFrame(index=df.index)
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        result['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        result['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A) - shifted forward
        result['senkou_span_a'] = (
            (result['tenkan_sen'] + result['kijun_sen']) / 2
        ).shift(displacement)
        
        # Senkou Span B (Leading Span B) - shifted forward
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        result['senkou_span_b'] = (
            (senkou_b_high + senkou_b_low) / 2
        ).shift(displacement)

        # Chikou Span (Lagging Span) - shifted backward
        result['chikou_span'] = df['close'].shift(-displacement)

        return result

    @staticmethod
    def signals(df: pd.DataFrame, ichimoku_df: pd.DataFrame) -> pd.Series:
        """
        Generate Ichimoku trading signals

        Returns:
            Series with 1 (bullish), -1 (bearish), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)

        # Cloud color (bullish or bearish)
        cloud_bullish = ichimoku_df['senkou_span_a'] > ichimoku_df['senkou_span_b']

        # Price above/below cloud
        cloud_top = ichimoku_df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        cloud_bottom = ichimoku_df[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        price_above_cloud = df['close'] > cloud_top
        price_below_cloud = df['close'] < cloud_bottom

        # TK Cross
        tk_bullish_cross = (
            (ichimoku_df['tenkan_sen'] > ichimoku_df['kijun_sen']) &
            (ichimoku_df['tenkan_sen'].shift(1) <= ichimoku_df['kijun_sen'].shift(1))
        )

        tk_bearish_cross = (
            (ichimoku_df['tenkan_sen'] < ichimoku_df['kijun_sen']) &
            (ichimoku_df['tenkan_sen'].shift(1) >= ichimoku_df['kijun_sen'].shift(1))
        )

        # Strong bullish signal: TK cross above cloud in bullish cloud
        signals[tk_bullish_cross & price_above_cloud & cloud_bullish] = 1

        # Strong bearish signal: TK cross below cloud in bearish cloud
        signals[tk_bearish_cross & price_below_cloud & ~cloud_bullish] = -1
         170 │         return signals

    @staticmethod
    def cloud_strength(ichimoku_df: pd.DataFrame) -> pd.Series:
        """
        Calculate cloud thickness as strength indicator

        Returns:
            Series with cloud thickness values
        """
        return abs(
            ichimoku_df['senkou_span_a'] - ichimoku_df['senkou_span_b']
        )
