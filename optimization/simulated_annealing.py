import copy
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestEngine

# Imports from project
from strategy.multi_layer_strategy import MultiLayerStrategy


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing Optimizer for Trading Strategy Parameters.
    """

    def __init__(
        self,
        strategy_class,
        data_map: Dict[str, pd.DataFrame],
        param_bounds: Dict[str, Tuple[float, float, float]],  # (min, max, step)
        initial_params: Dict[str, float],
        backtest_config: Dict = None,
        iterations: int = 100,
        initial_temp: float = 10.0,
        cooling_rate: float = 0.95,
    ):
        """
        Args:
            strategy_class: The strategy class to optimize (e.g., MultiLayerStrategy).
            data_map: Dictionary of {symbol: DataFrame} to backtest against.
            param_bounds: Dictionary defining the search space for each parameter.
                          Format: {'param_name': (min_val, max_val, step_size)}
            initial_params: Starting point for optimization.
            backtest_config: Configuration dict for the BacktestEngine.
            iterations: Number of optimization steps.
            initial_temp: Starting temperature.
            cooling_rate: Rate at which temperature decays (geometric schedule).
        """
        self.strategy_class = strategy_class
        self.data_map = data_map
        self.param_bounds = param_bounds
        self.curr_params = initial_params.copy()
        self.best_params = initial_params.copy()
        
        self.backtest_config = backtest_config or {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        }
        
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        
        # State tracking
        self.best_score = -float('inf')
        self.curr_score = -float('inf')
        self.history = []

    def objective_function(self, params: Dict) -> float:
        """
        Evaluate parameters by running backtests on all assets and averaging the Sharpe Ratio.
        """
        scores = []
        
        # We can parallelize this inner loop if needed, but for now serial is safer for memory.
        for symbol, df in self.data_map.items():
            try:
                # 1. Instantiate Strategy with params
                # Combine base config with optimization params
                strategy_config = params.copy()
                strategy = self.strategy_class(config=strategy_config)
                
                # 2. Generate Signals
                # The strategy might need training (HMM)
                # Ideally, we train once per optimization step if params affect training,
                # but HMM params are usually hyperparameters (n_regimes) not tuned here.
                # If 'hmm_lookback' is optimized, we must re-train.
                # Assuming simple backtest for now without retraining heavy models every step if possible.
                # But MultiLayerStrategy.generate_signals calls internal phases.
                # NOTE: For speed, we might want to pre-calculate indicators that don't depend on params,
                # but strategy is monolithic. We run full generation.
                
                # Verify enough data
                if len(df) < 200:
                    continue
                    
                # Train if necessary (usually robust strategies need training)
                if hasattr(strategy, 'train'):
                    # strategy.train(df) # Skipping full HMM training per step for speed unless strictly needed
                    pass 

                signals = strategy.generate_signals(df)
                
                # 3. Run Backtest
                engine = BacktestEngine(self.backtest_config)
                # BacktestEngine expects a dict of signal series
                results = engine.run(df, {'strategy': signals})
                
                # 4. Extract Metric (Sharpe)
                sharpe = results.get('sharpe_ratio', 0.0)
                
                # Penalize low trade count to avoid overfitting to 1 lucky trade
                num_trades = len(results.get('trades', []))
                if num_trades < 5:
                    sharpe = -1.0 # Penalty
                
                if math.isnan(sharpe):
                    sharpe = -1.0
                    
                scores.append(sharpe)
                
            except Exception as e:
                print(f"Error evaluating {symbol}: {e}")
                scores.append(-2.0) # Penalty for failure
                
        if not scores:
            return -float('inf')
            
        # Optimize for Mean Sharpe across all assets
        avg_sharpe = sum(scores) / len(scores)
        return avg_sharpe

    def _get_neighbor(self, params: Dict) -> Dict:
        """
        Generate a neighbor solution by perturbing one parameter.
        """
        new_params = params.copy()
        
        # Pick one random parameter to change
        param_name = random.choice(list(self.param_bounds.keys()))
        min_val, max_val, step = self.param_bounds[param_name]
        
        current_val = new_params[param_name]
        
        # Random step: +step or -step
        direction = random.choice([-1, 1])
        new_val = current_val + (direction * step)
        
        # Clamp to bounds
        new_val = max(min_val, min(max_val, new_val))
        
        # Round to precision based on step (rough check)
        # e.g. step 0.1 -> round to 1 decimal
        decimals = 0
        if isinstance(step, float) and step < 1:
            decimals = str(step)[::-1].find('.')
            new_val = round(new_val, decimals)
        elif isinstance(step, int):
            new_val = int(round(new_val))
            
        new_params[param_name] = new_val
        return new_params

    def run(self):
        """
        Execute the Simulated Annealing optimization loop.
        """
        print(f"Starting SA Optimization with {self.iterations} iterations...")
        print(f"Initial params: {self.curr_params}")
        
        # Initial evaluation
        self.curr_score = self.objective_function(self.curr_params)
        self.best_score = self.curr_score
        
        print(f"Initial Score (Sharpe): {self.curr_score:.4f}")
        
        temp = self.initial_temp
        
        for i in range(self.iterations):
            # 1. Generate Neighbor
            neighbor_params = self._get_neighbor(self.curr_params)
            
            # 2. Evaluate Neighbor
            neighbor_score = self.objective_function(neighbor_params)
            
            # 3. Acceptance Probability
            if neighbor_score > self.curr_score:
                accept = True
            else:
                # Metropolis criterion
                # maximize objective -> delta = new - old
                delta = neighbor_score - self.curr_score
                # Probability = exp(delta / T)
                # Since delta is negative, this probability is < 1
                try:
                    prob = math.exp(delta / temp)
                except OverflowError:
                    prob = 0.0
                
                accept = random.random() < prob
                
            # 4. Update interactions
            if accept:
                self.curr_params = neighbor_params
                self.curr_score = neighbor_score
                
                # Track global best
                if self.curr_score > self.best_score:
                    self.best_score = self.curr_score
                    self.best_params = self.curr_params.copy()
                    print(f"[{i+1}/{self.iterations}] NEW BEST! Score: {self.best_score:.4f} | Params: {self.best_params}")
            
            self.history.append({
                'iteration': i,
                'temp': temp,
                'score': self.curr_score,
                'best_score': self.best_score,
                'params': self.curr_params.copy()
            })
            
            # Cooling
            temp *= self.cooling_rate
            
            if (i+1) % 10 == 0:
                print(f"Iter {i+1}: Temp={temp:.4f}, Curr={self.curr_score:.4f}, Best={self.best_score:.4f}")
                
        return self.best_params, self.best_score, self.history
