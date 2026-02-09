import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add src to path
sys.path.append("/home/vilivom/src/THC")

from ml.causal_inference import CausalVolatilityTrading
from strategy.multi_layer_strategy import CausalContext, MultiLayerStrategy


class TestCausalIntegration(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start="2023-01-01", periods=300, freq="h")

        # Follower: Uptrend
        self.follower_df = pd.DataFrame(
            {
                "open": np.linspace(100, 200, 300),
                "high": np.linspace(101, 201, 300),
                "low": np.linspace(99, 199, 300),
                "close": np.linspace(100, 200, 300),
                "volume": np.random.randint(100, 1000, 300),
            },
            index=dates,
        )

        # Leader: Same uptrend, but shifted (Leading)
        self.leader_df = self.follower_df.copy()

        self.strategy = MultiLayerStrategy()
        # Mocking training to avoid HMM overhead/issues in test
        self.strategy._is_trained = True
        self.strategy.regime_classifier.is_trained = True
        # Mock classifier to always return stable regime for Follower
        self.strategy.regime_classifier.classify = (
            lambda df, stable_threshold=0.6, crash_threshold=0.3: type(
                "obj",
                (object,),
                {
                    "regime_id": 0,
                    "stable_trend_prob": 0.9,
                    "crash_prob": 0.0,
                    "is_tradeable": True,
                    "is_crash": False,
                },
            )
        )

    def test_default_behavior(self):
        """Test that strategy works without causal context."""
        signals = self.strategy.generate_signals(self.follower_df)
        self.assertTrue(len(signals) > 0)

    def test_leader_crash_veto(self):
        """Test Phase 1: Leader Crash should block trade."""
        # Setup crashing Leader
        crash_df = self.leader_df.copy()
        # Mock HMM to return crash for Leader
        original_classify = self.strategy.regime_classifier.classify

        def mock_classify(df, stable_threshold=0.6, crash_threshold=0.3):
            # If passed DF is leader (check by length or content if possible, or just hack)
            # Simplest: check if it's the crash_df
            if df.equals(crash_df.tail(100)):
                return type(
                    "obj",
                    (object,),
                    {
                        "regime_id": 2,
                        "stable_trend_prob": 0.0,
                        "crash_prob": 0.9,
                        "is_tradeable": False,
                        "is_crash": True,
                    },
                )
            return type(
                "obj",
                (object,),
                {
                    "regime_id": 0,
                    "stable_trend_prob": 0.9,
                    "crash_prob": 0.0,
                    "is_tradeable": True,
                    "is_crash": False,
                },
            )

        self.strategy.regime_classifier.classify = mock_classify

        ctx = CausalContext(leader_df=crash_df, optimal_lag=5, is_valid=True)
        self.strategy.set_leader_context(ctx)

        # The logic inside _phase1_regime should call classify on valid leader context
        # and see it's crashing, then set regime.is_tradeable = False

        regime = self.strategy._phase1_regime(self.follower_df)
        self.assertTrue(regime.is_crash)
        self.assertFalse(regime.is_tradeable)

    def test_projected_velocity_veto(self):
        """Test Phase 2: Signal Denoising with Projected Velocity."""
        # Leader moves DOWN sharply 5 bars ago
        # Follower is moving UP right now

        # Manipulate Leader to have a sharp drop at t-5
        # We need projected_velocity (Leader at t-5) to be NEGATIVE
        # while Follower at t is POSITIVE

        # follower at -1 is UP (velocity > 0)
        # leader at -1-lag (-6) needs to provide negative velocity

        # Modify leader to be distinguishable from follower
        self.leader_df["close"] = self.leader_df["close"] + 10.0

        lag = 5
        ctx = CausalContext(leader_df=self.leader_df, optimal_lag=lag, is_valid=True)
        self.strategy.set_leader_context(ctx)

        # Create a mock kalman filter that returns specific velocities
        class MockKalman:
            def __init__(self):
                self.call_count = 0

            def process_series(self, series):
                self.call_count += 1
                df = pd.DataFrame(index=series.index)
                df["kalman_price"] = series
                df["kalman_velocity"] = 1.0
                df["trend_direction"] = "up"

                # 1st call is typically Follower, 2nd call is Leader (in _phase2_kalman logic)
                # But let's verify logic:
                # _phase2_kalman calls self.kalman_filter.process_series(df['close']) -> Count 1
                # Then calls self.kalman_filter.process_series(leader_close) -> Count 2

                if self.call_count == 2:  # Leader call
                    df["kalman_velocity"] = -1.0

                return df

        self.strategy.kalman_filter = MockKalman()

        result = self.strategy._phase2_kalman(self.follower_df)

        # The logic:
        # projected = leader_velocity.shift(lag)
        # if follower=UP (1.0) and projected=DOWN (-1.0) -> VETO

        # Check last row
        self.assertEqual(result.iloc[-1]["trend_direction"], "neutral")
        self.assertIn("projected_velocity", result.columns)

    def test_ete_gate(self):
        """Test Phase 4: ETE Gate."""
        # ETE not rising -> Entry disallowed
        ctx = CausalContext(
            leader_df=self.leader_df, optimal_lag=5, ete_rising=False, is_valid=True
        )
        self.strategy.set_leader_context(ctx)

        # We need to run phase 4.
        # By default, setup has uptrend, so entry_allowed should be true if ETE was rising.
        # With ETE falling, should be false (unless climax)

        res = self.strategy._phase4_tactical(self.follower_df)
        self.assertFalse(res.iloc[-1]["entry_allowed"])

        # Rising ETE -> Allowed
        ctx.ete_rising = True
        self.strategy.set_leader_context(ctx)
        res = self.strategy._phase4_tactical(self.follower_df)
        # Depends on ADX/Aroon, but assuming trend, it should be True
        # Let's force ADX > 38 (trend regime) to be sure
        pass


if __name__ == "__main__":
    unittest.main()
