"""
Smart Money Concepts Module

Implements institutional trading concepts:
- Order Blocks (OB)
- Fair Value Gaps (FVG)
- Liquidity Zones
- Market Structure (BOS/CHoCH)
"""

from .order_blocks import OrderBlocks
from .fvg import FairValueGap
from .liquidity import Liquidity
from .market_structure import MarketStructure
from .smc_wrapper import SMCWrapper
from .smc_strategy import SMCStrategy

__all__ = [
    'OrderBlocks',
    'FairValueGap',
    'Liquidity',
    'MarketStructure',
    'SMCWrapper',
    'SMCStrategy'
]

__version__ = '1.0.0'
