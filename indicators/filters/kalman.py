import pandas as pd
import numpy as np

class KalmanFilter:
    """ Wraps a Kalman Filter for online smoothing """
    def __init__(self, transition_covariance=0.01, observation_covariance=1.0):
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance
        )
        self.state_mean = None
        self.state_cov = None

    def update(self, price):
        if self.state_mean is None:
            self.state_mean = price
            self.state_cov = 1.0
            return price
        
        # Online update of the Kalman Filter
        self.state_mean, self.state_cov = self.kf.filter_update(
            self.state_mean, self.state_cov, observation=price
        )
        return float(self.state_mean)
