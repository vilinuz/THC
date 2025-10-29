from bayes_opt import BayesianOptimization
import numpy as np
from typing import Dict, Callable, Tuple
import pandas as pd

class BayesianOptimizer:
    """Bayesian optimizer for trading strategies"""
    
    def __init__(self, param_bounds: Dict, random_state: int = 42):
        self.param_bounds = param_bounds
        self.random_state = random_state
        self.optimizer = None
        self.best_params = None
        self.best_score = None
        
    def optimize(
        self,
        objective_function: Callable,
        n_iterations: int = 50,
        n_initial_points: int = 10
    ) -> Dict:
        """
        Run Bayesian optimization
        
        Args:
            objective_function: Function to maximize (e.g., Sharpe ratio)
            n_iterations: Number of optimization iterations
            n_initial_points: Random exploration points
            
        Returns:
            Dict with best parameters and score
        """
        self.optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=self.param_bounds,
            random_state=self.random_state,
            verbose=2
        )
        
        # Run optimization
        self.optimizer.maximize(
            init_points=n_initial_points,
            n_iter=n_iterations
        )

        self.best_params = self.optimizer.max['params']
        self.best_score = self.optimizer.max['target']

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_iterations': pd.DataFrame(self.optimizer.res)
        }

    def get_exploration_history(self) -> pd.DataFrame:
        """Get history of explored parameters and scores"""
        if self.optimizer is None:
            return pd.DataFrame()

        history = []
        for res in self.optimizer.res:
            row = res['params'].copy()
            row['target'] = res['target']
            history.append(row)

        return pd.DataFrame(history)
