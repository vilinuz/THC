import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist, euclidean
from scipy import stats

class CausalVolatilityTrading:
    """
    Implements the Causal Volatility Trading framework by Ivan Lettery.
    
    Components:
    1. GMM Clustering: Identifies 'Mid-Volatility' regimes/assets.
    2. Granger Causality Test (GCT): Identifies directional predictive links.
    3. Effective Transfer Entropy (ETE): Measures information flow strength.
    4. DTW-KNN: Estimates optimal time lag for execution.
    """
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.gmm_model = None
        self.mid_vol_cluster_id = None
    
    # ==============================================================================
    # 1. GMM Clustering
    # ==============================================================================
    
    def extract_volatility_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for clustering:
        - Annualized Volatility
        - Volatility of Volatility (VoV)
        - Average True Range (Normalized)
        """
        if len(df) < 50:
            return np.array([0, 0, 0])
            
        returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # 1. Annualized Volatility
        vol = returns.rolling(window=20).std()
        annual_vol = vol.mean() * np.sqrt(252)
        
        # 2. Volatility of Volatility
        vov = vol.std()
        
        # 3. Normalized ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_norm = (tr.rolling(14).mean() / close).mean()
        
        return np.array([annual_vol, vov, atr_norm])
        
    def cluster_universe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[int, List[str]]:
        """
        Cluster assets into 3 groups based on volatility profiles.
        Returns dictionary mapping Cluster ID -> List of Tickers.
        Identifies the 'Mid-Volatility' cluster automatically.
        """
        features_list = []
        valid_tickers = []
        
        for ticker, df in data_dict.items():
            feat = self.extract_volatility_features(df)
            if not np.isnan(feat).any():
                features_list.append(feat)
                valid_tickers.append(ticker)
        
        if not features_list:
            return {}
            
        X = np.array(features_list)
        
        # Fit GMM
        self.gmm_model = GaussianMixture(n_components=3, random_state=42)
        labels = self.gmm_model.fit_predict(X)
        
        clusters = {0: [], 1: [], 2: []}
        cluster_vols = {0: [], 1: [], 2: []}
        
        for i, ticker in enumerate(valid_tickers):
            clusters[labels[i]].append(ticker)
            cluster_vols[labels[i]].append(X[i][0]) # Use Annualized Vol as metric
            
        # Identify "Mid-Volatility" cluster
        # Calculate mean vol for each cluster
        avg_vols = {k: np.mean(v) if v else 0 for k, v in cluster_vols.items()}
        sorted_clusters = sorted(avg_vols.items(), key=lambda item: item[1])
        
        # sorted_clusters is [(id, low_vol), (id, mid_vol), (id, high_vol)]
        if len(sorted_clusters) >= 2:
            self.mid_vol_cluster_id = sorted_clusters[1][0]
        else:
            self.mid_vol_cluster_id = sorted_clusters[0][0]
            
        return clusters

    def is_mid_volatility(self, ticker: str, clusters: Dict[int, List[str]]) -> bool:
        """Check if a ticker belongs to the Mid-Volatility cluster."""
        if self.mid_vol_cluster_id is None or ticker not in sum(clusters.values(), []):
            return True # Default to allow if clustering failed
        return ticker in clusters.get(self.mid_vol_cluster_id, [])

    # ==============================================================================
    # 2. Causal Discovery (Granger Causality - Simplified)
    # ==============================================================================
    
    def check_granger_causality(self, leader_series: pd.Series, follower_series: pd.Series, max_lag: int = 5) -> Tuple[bool, float]:
        """
        Check if Leader G-causes Follower.
        Uses a Restricted vs Unrestricted Regression approach (F-test proxy).
        
        Returns: (is_causal, p_value_proxy)
        """
        # Ensure aligned
        df = pd.DataFrame({'Y': follower_series, 'X': leader_series}).dropna()
        if len(df) < 30:
            return False, 1.0
            
        # Unrestricted Model: Y_t = sum(a*Y_t-i) + sum(b*X_t-i)
        # Restricted Model:   Y_t = sum(a*Y_t-i)
        
        rss_restricted = 0
        rss_unrestricted = 0
        
        y = df['Y'].values
        x = df['X'].values
        n = len(y)
        
        # We test for lag=1 to max_lag
        best_p_val = 1.0
        is_causal = False
        
        # Simplified: Just test optimal lag or a fixed lag window? 
        # We'll use a single pass with all lags included
        
        # Prepare feature matrices
        features_res = []
        features_unres = []
        targets = []
        
        start_idx = max_lag
        
        for t in range(start_idx, n):
            # Autoregressive terms
            y_lags = y[t-max_lag:t][::-1]
            
            # Exogenous terms
            x_lags = x[t-max_lag:t][::-1]
            
            features_res.append(y_lags)
            features_unres.append(np.concatenate([y_lags, x_lags]))
            targets.append(y[t])
            
        X_res = np.array(features_res)
        X_unres = np.array(features_unres)
        y_target = np.array(targets)
        
        # Fit Restricted
        model_res = LinearRegression().fit(X_res, y_target)
        preds_res = model_res.predict(X_res)
        rss_res = np.sum((y_target - preds_res) ** 2)
        
        # Fit Unrestricted
        model_unres = LinearRegression().fit(X_unres, y_target)
        preds_unres = model_unres.predict(X_unres)
        rss_unres = np.sum((y_target - preds_unres) ** 2)
        
        # F-Test
        # F = ((RSS_r - RSS_ur) / p) / (RSS_ur / (n - k))
        # p = number of restrictions (lags of X, i.e., max_lag)
        # k = number of parameters in unrestricted (2 * max_lag + 1)
        # n = number of observations
        
        T = len(y_target)
        p = max_lag
        k = 2 * max_lag + 1
        
        if rss_unrestricted == 0:
            return True, 0.0
            
        F_stat = ((rss_res - rss_unrestricted) / p) / (rss_unrestricted / (T - k))
        
        # Approximate p-value using chi2 as we don't have scipy.stats.f easily available everywhere 
        # (Though we imported scipy.stats, so we can use f.sf)
        try:
            p_value = stats.f.sf(F_stat, p, T - k)
        except:
            # Fallback simple threshold if scipy fails
            p_value = 0.01 if F_stat > 3.0 else 0.5
            
        return p_value < 0.05, p_value

    # ==============================================================================
    # 3. Lead-Lag Estimation (DTW + Cross-Corr)
    # ==============================================================================

    def estimate_lead_lag(self, leader_series: pd.Series, follower_series: pd.Series, max_lag: int = 10) -> int:
        """
        Estimate optimal time lag where Leader predicts Follower.
        Uses combined approach: DTW distance check and Cross-Correlation.
        """
        # Align
        s1 = leader_series.values
        s2 = follower_series.values
        
        # 1. Simple Cross Correlation
        corrs = []
        lags = range(1, max_lag + 1)
        for lag in lags:
            # Shift leader forward by lag (Leader[t-lag] vs Follower[t])
            # If Leader leads, then Leader[t] should look like Follower[t+lag]
            # So Corr(Leader[t], Follower[t+lag]) -> which is Corr(Leader[:-lag], Follower[lag:])
            
            c = np.corrcoef(s1[:-lag], s2[lag:])[0, 1]
            corrs.append(c)
            
        optimal_lag_cc = lags[np.argmax(corrs)]
        
        # 2. DTW refinement (Optional, but per paper)
        # We can simulate DTW lag finding by sliding windows and calculating DTW distance
        # Ideally, we want the lag that MINIMIZES DTW distance.
        
        dtw_dists = []
        for lag in lags:
            # Shifted leader
            l_shifted = s1[:-lag]
            f_target = s2[lag:]
            
            # Normalize for DTW
            l_norm = (l_shifted - np.mean(l_shifted)) / (np.std(l_shifted) + 1e-6)
            f_norm = (f_target - np.mean(f_target)) / (np.std(f_target) + 1e-6)
            
            # Fast simplified DTW: just Euclidean on aligned path (since we forced lag)
            # Full DTW is O(N^2), overkill here. The paper uses DTW to *align* then KNN to classify.
            # Here we just want the lag. The lag that minimizes Euclidean distance after shift 
            # is effectively the optimal alignment for a "rigid" lead-lag.
            d = euclidean(l_norm, f_norm)
            dtw_dists.append(d)
            
        optimal_lag_dtw = lags[np.argmin(dtw_dists)]
        
        # Weighted decision (give preference to DTW/Euclidean minimum as it handles noise better)
        return optimal_lag_dtw

    # ==============================================================================
    # 4. Effective Transfer Entropy (ETE) (Symbolic Proxy)
    # ==============================================================================

    def calculate_transfer_entropy(self, leader: pd.Series, follower: pd.Series, lag: int = 1) -> float:
        """
        Calculate Symbolic Transfer Entropy (STE) as a robust proxy for ETE.
        TE(X->Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-lag)
        
        Returns:
            Positive float representing information flow strength (bits).
        """
        if len(leader) < 50:
            return 0.0
            
        # 1. Symbolize Data (3 states: Down, Flat, Up)
        def symbolize(series):
            diff = series.diff().fillna(0)
            threshold = diff.std() * 0.5
            symbols = np.zeros(len(series), dtype=int)
            symbols[diff > threshold] = 2    # Up
            symbols[diff < -threshold] = 0   # Down
            symbols[abs(diff) <= threshold] = 1 # Flat
            return symbols
            
        X = symbolize(leader) # Source
        Y = symbolize(follower) # Target
        
        # Shift X by lag to align X_{t-lag} with Y_t
        X_lag = np.roll(X, lag)
        Y_prev = np.roll(Y, 1)
        
        # Remove first 'lag' samples to clean up rolling artifacts
        valid_idx = slice(max(lag, 1), None)
        Y_t = Y[valid_idx]
        Y_t_1 = Y_prev[valid_idx]
        X_t_l = X_lag[valid_idx]
        
        # Calculate Conditional Entropy H(Y_t | Y_t-1)
        # H(A|B) = H(A,B) - H(B)
        def entropy(labels):
            value, counts = np.unique(labels, return_counts=True, axis=0)
            probs = counts / len(labels)
            return -np.sum(probs * np.log2(probs + 1e-10))
            
        # Joint (Y_t, Y_t-1)
        joint_y_y1 = np.vstack([Y_t, Y_t_1]).T
        h_y_y1 = entropy(joint_y_y1)
        h_y1 = entropy(Y_t_1)
        h_cond_y_y1 = h_y_y1 - h_y1
        
        # Calculate Conditional Entropy H(Y_t | Y_t-1, X_t-l)
        # H(A|B,C) = H(A,B,C) - H(B,C)
        joint_all = np.vstack([Y_t, Y_t_1, X_t_l]).T
        joint_cond = np.vstack([Y_t_1, X_t_l]).T
        
        h_all = entropy(joint_all)
        h_cond = entropy(joint_cond)
        h_cond_y_y1_x = h_all - h_cond
        
        # TE = H(Y|Y_prev) - H(Y|Y_prev, X_lag)
        te = h_cond_y_y1 - h_cond_y_y1_x
        
        return max(0.0, te)