import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable
import logging
from itertools import product

logger = logging.getLogger(__name__)

class WalkForwardOptimizationEngine:
    """
    Simulates the real-world operational cycle of a quantitative trading desk 
    by systematically moving a training (In-Sample) window and a testing 
    (Out-of-Sample) window forward through time.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        strategy_class: Any, 
        param_grid: Dict[str, list],
        l_train_days: int = 730, # 24 months approx
        l_test_days: int = 90, # 3 months approx
        objective_function: Callable[[pd.Series], float] = None,
        optimization_method: str = "grid_search",
        sa_initial_temp: float = 10.0,
        sa_cooling_rate: float = 0.95,
        sa_iterations: int = 100
    ):
        """
        :param data: The comprehensive historical dataset. Must have DatetimeIndex.
        :param strategy_class: Class of the strategy to test (e.g. StrategyAlpha).
        :param param_grid: Dictionary of parameter names to lists of possible values to search.
        :param l_train_days: Length of In-Sample window in days.
        :param l_test_days: Length of Out-of-Sample window in days (step size).
        :param objective_function: Function to maximize (default is Sharpe Ratio approximation).
        :param optimization_method: 'grid_search' or 'simulated_annealing'.
        :param sa_initial_temp: Initial temperature for SA.
        :param sa_cooling_rate: Cooling rate for SA (alpha).
        :param sa_iterations: Number of iterations for SA.
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.l_train_days = pd.Timedelta(days=l_train_days)
        self.l_test_days = pd.Timedelta(days=l_test_days)
        self.objective_function = objective_function or self._default_sharpe
        self.optimization_method = optimization_method
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations = sa_iterations
        
    def _default_sharpe(self, returns: pd.Series) -> float:
        """Default objective function: Annualized Sharpe Ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        # Assume daily returns for annualization factor, adjust if timeframe differs
        return np.sqrt(365) * (returns.mean() / returns.std())

    def _generate_param_combinations(self) -> list:
        """Create a list of all parameter combinations from param_grid."""
        keys, values = zip(*self.param_grid.items())
        experiment_params = [dict(zip(keys, v)) for v in product(*values)]
        return experiment_params

    def _evaluate_strategy(self, data: pd.DataFrame, params: Dict) -> float:
        """
        Evaluates the strategy on the provided data split using given parameters.
        Returns the objective function score.
        """
        try:
            strategy = self.strategy_class(params)
            # generate_signals should return a Series mapped to index
            signals = strategy.generate_signals(data)
            
            # Simple return calculation: signal * forward return
            # Assumes 'close' column exists
            forward_returns = data['close'].pct_change().shift(-1)
            strategy_returns = signals * forward_returns
            
            # Drop NaNs
            strategy_returns = strategy_returns.dropna()
            
            return self.objective_function(strategy_returns)
        except Exception as e:
            logger.error(f"Error evaluating strategy with params {params}: {e}")
            return 0.0

    def optimize_in_sample(self, train_data: pd.DataFrame) -> Dict:
        """
        Explores the parameter space within the L_train window to find optimal theta_star
        that maximizes the predefined objective function.
        """
        if self.optimization_method == "simulated_annealing":
            return self._optimize_simulated_annealing(train_data)
        else:
            return self._optimize_grid_search(train_data)

    def _optimize_grid_search(self, train_data: pd.DataFrame) -> Dict:
        best_score = -float('inf')
        best_params = {}
        
        combinations = self._generate_param_combinations()
        
        # In a real scenario, we might use Bayesian Optimization (like optuna) here instead of Grid Search.
        for params in combinations:
            score = self._evaluate_strategy(train_data, params)
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params

    def _optimize_simulated_annealing(self, train_data: pd.DataFrame) -> Dict:
        """
        Simulated Annealing metaheuristic for hyperparameter optimization to avoid local optima 
        and handle non-convex objective spaces efficiently.
        """
        import random
        # Initialization
        current_params = {k: random.choice(v) for k, v in self.param_grid.items()}
        # Energy is inverse of objective function (we want to minimize energy)
        current_energy = -1.0 * self._evaluate_strategy(train_data, current_params)
        
        best_params = current_params.copy()
        best_energy = current_energy
        
        temp = self.sa_initial_temp
        
        for i in range(self.sa_iterations):
            # Perturbation: Randomly mutate one hyperparameter
            new_params = current_params.copy()
            param_to_mutate = random.choice(list(self.param_grid.keys()))
            new_params[param_to_mutate] = random.choice(self.param_grid[param_to_mutate])
            
            # Evaluation
            new_energy = -1.0 * self._evaluate_strategy(train_data, new_params)
            delta_energy = new_energy - current_energy
            
            # Acceptance Criterion (Metropolis-Hastings)
            if delta_energy <= 0:
                current_params = new_params
                current_energy = new_energy
            else:
                prob = np.exp(-delta_energy / temp) if temp > 1e-10 else 0.0
                if random.random() < prob:
                    current_params = new_params
                    current_energy = new_energy
                    
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = current_params.copy()
                
            # Cooling Schedule
            temp *= self.sa_cooling_rate
            
        return best_params

    def validate_out_of_sample(self, test_data: pd.DataFrame, optimal_params: Dict) -> pd.Series:
        """
        Evaluates the frozen optimal_params on strictly unseen data in the adjacent L_test window.
        """
        strategy = self.strategy_class(optimal_params)
        signals = strategy.generate_signals(test_data)
        
        forward_returns = test_data['close'].pct_change().shift(-1)
        oos_returns = signals * forward_returns
        return oos_returns.dropna()

    def run(self) -> Tuple[float, pd.Series, pd.DataFrame]:
        """
        Executes the Walk-Forward Optimization.
        Rolls the window through the entire dataset.
        Returns the Walk-Forward Efficiency (WFE) ratio, the composite OOS equity curve returns,
        and a summary dataframe of the windows.
        """
        
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        
        current_train_start = start_date
        
        all_oos_returns = []
        all_is_returns = []
        window_summaries = []
        
        while True:
            train_end = current_train_start + self.l_train_days
            test_end = train_end + self.l_test_days
            
            if test_end > end_date:
                # Stop if we don't have enough data for a full test window
                break
                
            # Data Partitioning
            train_data = self.data.loc[current_train_start:train_end]
            test_data = self.data.loc[train_end:test_end]
            
            # In-Sample Optimization
            optimal_params = self.optimize_in_sample(train_data)
            
            # Get IS returns using optimal params for WFE calculation
            is_strategy = self.strategy_class(optimal_params)
            is_signals = is_strategy.generate_signals(train_data)
            is_returns = (is_signals * train_data['close'].pct_change().shift(-1)).dropna()
            
            # Out-of-Sample Validation
            oos_returns = self.validate_out_of_sample(test_data, optimal_params)
            
            all_is_returns.append(is_returns)
            all_oos_returns.append(oos_returns)
            
            window_summaries.append({
                'train_start': current_train_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'optimal_params': optimal_params,
                'is_score': self.objective_function(is_returns),
                'oos_score': self.objective_function(oos_returns)
            })
            
            # Rolling the Window
            current_train_start += self.l_test_days
            
        # Compile final results
        composite_oos_returns = pd.concat(all_oos_returns) if all_oos_returns else pd.Series()
        composite_is_returns = pd.concat(all_is_returns) if all_is_returns else pd.Series()
        
        summary_df = pd.DataFrame(window_summaries)
        
        # Calculate Walk-Forward Efficiency (Annualized OOS Return / Annualized IS Return)
        # Assuming daily returns
        ann_oos_return = composite_oos_returns.mean() * 365
        ann_is_return = composite_is_returns.mean() * 365
        
        if ann_is_return <= 0:
            wfe = 0.0 # Avoid division by zero or negative IS returns logic flaws
        else:
            wfe = (ann_oos_return / ann_is_return) * 100
            
        logger.info(f"WFO Complete. WFE: {wfe:.2f}%")
        if wfe < 50:
            logger.warning("Strategy is deemed overfitted (WFE < 50%).")
        elif wfe > 60:
            logger.info("Highly robust strategy demonstrated (WFE > 60%).")
            
        return wfe, composite_oos_returns, summary_df
