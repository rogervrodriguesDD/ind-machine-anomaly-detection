# -*- utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from ind_machine_anomaly_detection.pipelines.scalers.classes import (
    SpecFreqScaler,
    FreqPosMinMaxScaler,
    MinMaxScalerCustomized,
)

def test_class_spec_freq_scaler():

    # Creating example data: a set of sine signals (scaled to range 0.0  and 1.0),
    # exponentiated (given that the scaler calculate the logarithm)
    N_samples = 16
    N_points = 200

    t = np.linspace(0, N_points, N_points)
    X = np.zeros(shape=(N_samples, N_points))

    for i in range(N_samples):
        f = np.random.randint(low=5, high=20, size=1)
        theta = np.random.rand()
        X[i,:] = 0.5 * ( 1 + np.sin(2 * np.pi * f * t + theta))

    X_exp = np.exp(X)
    df = pd.DataFrame(X_exp, columns=[f"ch1_X_{i}" for i in range(1, N_points + 1)])

    # Scaling the data
    df_scaled = SpecFreqScaler(channel='ch1').fit_transform(df)

    assert pytest.approx(X, df_scaled.values)

    # Testing for absent channel
    with pytest.raises(OSError) as excinfo:
        SpecFreqScaler(channel='foo').fit_transform(df)

    assert "Channel foo not found in input data"

def test_class_freq_pos_min_max_scaler():

    # Creating example data
    N_samples = 16
    N_points = 200

    X = np.random.normal(loc=2.5, scale=5.0, size=(N_samples, N_points))
    df = pd.DataFrame(X, columns=[f"ch1_freqmax_{i}" for i in range(1, N_points + 1)])

    # Counting number of min and max values (must be equal after transformation)
    max_X = X.max()
    min_X = X.min()

    count_max_X = np.isclose(X, max_X).sum()
    count_min_X = np.isclose(X, min_X).sum()

    # Scaling the data
    df_scaled = FreqPosMinMaxScaler(channel='ch1').fit_transform(df)

    # Counting the number of min and max in the scaled data
    min_X_scaled = df_scaled.values.min()
    max_X_scaled = df_scaled.values.max()

    count_min_X_scaled = np.isclose(df_scaled.values, min_X_scaled).sum()
    count_max_X_scaled = np.isclose(df_scaled.values, max_X_scaled).sum()

    assert count_min_X == count_min_X_scaled
    assert count_max_X == count_max_X_scaled
    assert pytest.approx(min_X_scaled, 0.0)
    assert pytest.approx(max_X_scaled, 1.0)

    # Testing for absent channel
    with pytest.raises(OSError) as excinfo:
        FreqPosMinMaxScaler(channel='foo').fit_transform(df)

    assert "Channel foo not found in input data"


def test_class_min_max_scaler_customized():

    # Creating example data
    N_samples = 16
    variables_names = ['foo', 'bar', 'baz', 'etc']
    variables_with_mult_cols = ['foo']
    N_cols_mult = 3

    N_total_cols = len(variables_names) - len(variables_with_mult_cols) + \
                N_cols_mult * len(variables_with_mult_cols)

    X = np.random.normal(loc=2.5, scale=5.0, size=(N_samples, N_total_cols))

    cols_names = [f"ch1_{col}_{i}" for col in variables_with_mult_cols for i in range(1, N_cols_mult + 1)]
    cols_names.extend([f"ch1_{col}" for col in variables_names if col not in variables_with_mult_cols])

    df = pd.DataFrame(X, columns=cols_names)

    # Scaling the data
    df_scaled = MinMaxScalerCustomized(channel='ch1', variables=variables_names).fit_transform(df)

    assert np.allclose(df_scaled.min(axis=0), 0.0)
    assert np.allclose(df_scaled.max(axis=0), 1.0)

    # Testing for absent channel
    with pytest.raises(OSError) as excinfo:
        MinMaxScalerCustomized(channel='foo', variables=variables_names).fit_transform(df)

    assert "Channel foo not found in input data" in str(excinfo)

    # Testing for absent variable
    with pytest.raises(OSError) as excinfo:
        MinMaxScalerCustomized(channel='ch1', variables=variables_names + ['mean']).fit_transform(df)

    assert "Variable 'mean' related to channel ch1 not found in input data" in str(excinfo)
