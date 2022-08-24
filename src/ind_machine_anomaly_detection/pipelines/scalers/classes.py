# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SpecFreqScaler(BaseEstimator, TransformerMixin):
    """ Transform the features by applying logarithm and scaling the
    results by a range determined by the minimum and maximum values
    of the all matrix of features. The idea here is scaling without
    losing the information contained in the shape of the data (important
    in a Spectrum in Frequency).
    The transformation is applied only in columns related to the Spectrum in
    Frequency, which has the following pattern as name: {channel_name}_X_{number}.

    ...

    Parameters:
        channel (str): Name of the channel which spectrum data will be scaled
    """

    def __init__(self, channel='ch1'):
        self.channel = channel
        self.columns = None
        self.x_log_min = None
        self.x_log_max = None

    def fit(self, X, y=None):
        self.columns = self._get_channel_columns(X)
        # Checking if channel columns exist in X
        if len(self.columns) == 0:
            raise OSError(f"Channel {self.channel} not found in input data")

        X_ = X[self.columns].values
        X_log = np.log(X_)
        self.x_log_min = X_log.min()
        self.x_log_max = X_log.max()

    def fit_transform(self, X, y=None):
        self.fit(X,y=y)
        return self.transform(X,y=y)

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_ = X[self.columns].values
        X_log = np.log(X_)
        X_copy[self.columns] = (X_log - self.x_log_min) / (self.x_log_max - self.x_log_min)
        return X_copy

    def _get_channel_columns(self, X):
        return [col for col in X.columns if col.split('_', 2)[:2] == [self.channel, 'X']]

class FreqPosMinMaxScaler(BaseEstimator, TransformerMixin):
    """ Scale the data by the range determined by the minimum
    and maximum values of all the matrix of features.
    The transformation is applied only in columns related to the Frequency
    Position, which has the following pattern as name: {channel_name}_freqmax_{number}.

    ...
    Parameters:
        channel (str): Name of channel which data will be transformed
    """

    def __init__(self, channel='ch1'):
        self.channel = channel
        self.columns = None
        self.x_min = None
        self.x_max = None

    def fit(self, X, y=None):
        self.columns = self._get_channel_columns(X)
        # Checking if channel columns exist in X
        if len(self.columns) == 0:
            raise OSError(f"Channel {self.channel} not found in input data")

        X_ = X[self.columns].values
        self.x_min = X_.min()
        self.x_max = X_.max()

    def fit_transform(self, X, y=None):
        self.fit(X,y=y)
        return self.transform(X,y=y)

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_ = X[self.columns].values
        X_copy[self.columns] = (X_ - self.x_min) / (self.x_max - self.x_min)
        return X_copy

    def _get_channel_columns(self, X):
        return [col for col in X.columns if col.split('_', 2)[:2] == [self.channel, 'freqmax']]

class MinMaxScalerCustomized(BaseEstimator, TransformerMixin):
    """ Transform the features of the specified columns by their minimum
    and maximum range.

    ...
    Parameters:
        channel (str): Name of the channel which data will be scaled
        variables (list): Name of the features, related to the given channel, that will be scaled
    """

    def __init__(self, channel='ch1', variables=['ffund', 'rms', 'mean', 'median', 'skew', 'kurtosis']):
        self.channel = channel
        self.variables = variables
        self.columns = None
        self.x_min = None
        self.x_max = None

    def fit(self, X, y=None):
        self.columns = self._get_channel_columns(X)
        # Check if channel columns exist in X
        self._check_if_channel_in_input()
        # Check if all variables related to channel exist in X
        self._check_if_variables_in_input()

        X_ = X[self.columns].values
        self.x_min = X_.min(axis=0)
        self.x_max = X_.max(axis=0)

    def fit_transform(self, X, y=None):
        self.fit(X,y=y)
        return self.transform(X,y=y)

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_ = X[self.columns].values
        X_copy[self.columns] = (X_ - self.x_min) / (self.x_max - self.x_min)
        return X_copy

    def _get_channel_columns(self, X):
        channel_cols = [col for col in X.drop(columns=set(['label']) & set(X.columns)).columns if col.split('_',1)[0] == self.channel]
        selected_cols = [col for col in channel_cols if col.split('_',2)[1] in self.variables]
        return selected_cols

    def _check_if_channel_in_input(self):
        # Checking if channel columns exist in X
        if len(self.columns) == 0:
            raise OSError(f"Channel {self.channel} not found in input data")

    def _check_if_variables_in_input(self):
        for var in self.variables:
            cols_var = [col for col in self.columns if col.split('_')[1] == var]
            if len(cols_var) == 0:
                raise OSError(f"Variable '{var}' related to channel {self.channel} not found in input data")
