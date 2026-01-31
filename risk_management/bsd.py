import pandas as pd
import numpy as np

class BayesianShockDetector:
    """ BCPD Implementation (Optimized for Speed) """
    def __init__(self, hazard=0.01, window=30):
        self.hazard = hazard
        self.window = window
        self.R = np.array([1.0])
        self.muT = np.array([0.0]) # Mean of returns
        self.kT = np.array([1.0])
        self.aT = np.array([1.0])
        self.bT = np.array([1.0]) # Variance of returns

    def update(self, log_ret):
        # 1. Predictive Prob (Student T)
        df = 2 * self.aT
        scale = np.sqrt(self.bT * (self.kT + 1) / (self.aT * self.kT))
        pred_probs = stats.t.pdf(log_ret, df, loc=self.muT, scale=scale)
        
        # 2. Growth & Changepoint Probabilities
        growth_probs = pred_probs * self.R * (1 - self.hazard)
        cp_prob = np.sum(pred_probs * self.R * self.hazard)
        
        evidence = np.sum(growth_probs) + cp_prob
        self.R = np.append(cp_prob, growth_probs) / evidence
        
        # 3. Update Suff Stats
        new_muT = (self.kT * self.muT + log_ret) / (self.kT + 1)
        new_kT = self.kT + 1
        new_aT = self.aT + 0.5
        new_bT = self.bT + (self.kT * (log_ret - self.muT)**2) / (2 * (self.kT + 1))
        
        self.muT = np.append(0, new_muT) # 0 is prior mean
        self.kT = np.append(1, new_kT)
        self.aT = np.append(1, new_aT)
        self.bT = np.append(1, new_bT)
        
        # Pruning
        if len(self.R) > self.window:
            self.R = self.R[:self.window]
            self.R /= np.sum(self.R)
            self.muT = self.muT[:self.window]
            self.kT = self.kT[:self.window]
            self.aT = self.aT[:self.window]
            self.bT = self.bT[:self.window]
            
        return self.R[0] # Probability of run length 0 (Shock)
