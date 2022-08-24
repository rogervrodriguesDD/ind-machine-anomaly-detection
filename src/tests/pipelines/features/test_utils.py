# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from ind_machine_anomaly_detection.pipelines.features.utils import (
    calc_fft,
    create_df_esp_freq,
    calculate_spec_frequency_all_instances,
    get_freq_peaks_positions_values,
)

from tests.test_fixtures import (
    test_config,
    test_machine_data,
)

def test_calc_fft():

    # Creating sample signal with two defined frequencies
    fs = 1000
    N = 70000
    f1, A1 = 100, 1
    f2, A2 = 300, 0.5

    t = np.linspace(0, N / fs, N)
    x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

    df = pd.DataFrame(np.hstack([t.reshape(-1, 1), x.reshape(-1, 1)]), columns=['time', 'ch1'])
    df['sample'] = 1
    df = df.set_index(['sample', 'time'])

    f, X = calc_fft(df, col='ch1', samples=[1], N=N, fs=fs)

    # Getting the calculated frequencies
    f1_spectrum = f[np.argmax(X)]
    f1_amplitude = float(X[np.argmax(X)])
    X[np.argmax(X)] = 0.0
    f2_spectrum = f[np.argmax(X)]
    f2_amplitude = float(X[np.argmax(X)])

    assert f1 == pytest.approx(f1_spectrum, abs=1e-2)
    assert A1 == pytest.approx(f1_amplitude, abs=1e-1)
    assert f2 == pytest.approx(f2_spectrum, abs=1e-2)
    assert A2 == pytest.approx(f2_amplitude, abs=1e-1)

def test_create_df_esp_freq(test_machine_data):

    # Setting the necessary parameters
    idx_instances = test_machine_data.dataframe_instances.index.get_level_values(level=0).unique()
    samples = np.random.choice(idx_instances, size=5, replace=False)
    N_max = 200
    cols_list = ['ch1', 'ch2']

    df_esp_freq = None
    for col in cols_list:

        # Calculating the spectrum frequency for a set of samples
        _, X_mat = calc_fft(df=test_machine_data.dataframe_instances,
                            col=col,
                            samples=samples,
                            N=test_machine_data.instances_data_catalog.machine_data_instances_number_meas_points,
                            fs=test_machine_data.instances_data_catalog.machine_data_instances_sampling_frequency
                            )

        # Testing the `create_df_esp_freq` function
        df_esp_freq = create_df_esp_freq(X_mat,
                                        samples=samples,
                                        N_max=N_max,
                                        col=col,
                                        axis_concat=0,
                                        data_freq=df_esp_freq)

    assert len(df_esp_freq.columns) == N_max * len(cols_list)
    assert len(set(samples).difference(set(df_esp_freq.index))) == 0
    assert df_esp_freq.columns.get_level_values(level=0).unique().tolist() == cols_list

def test_calculate_spec_frequency_all_instances(test_machine_data):

    idx_instances = test_machine_data.dataframe_instances.index.get_level_values(level=0).unique()
    N_max = 200

    df_esp_freq = calculate_spec_frequency_all_instances(data=test_machine_data,
                                                        N_max=N_max)

    assert len(df_esp_freq.columns) == N_max * 3
    assert len(set(idx_instances).difference(set(df_esp_freq.index))) == 0
    assert df_esp_freq.columns.get_level_values(level=0).unique().tolist() == ['ch1', 'ch2', 'ch3']

def test_get_peaks_positions_values():

    # Creating sample signal of a damped system with defined fundamental frequency
    fs = 15000
    N = 70000
    t = np.linspace(0, N / fs, N)

    # Systems parameters (One Degree of Freedom System)
    fund_f, zeta, C, V = 5, 0.01, 125, 625

    x = np.zeros(shape=N)
    list_freq_peaks_signal = []
    for i in range(6):
        f_ = (10 * i  + 1)* fund_f
        zeta_ = zeta / np.sqrt(2 * i + 1)
        C_ = C * (2 * i + 1)
        V_ = V

        x += C_ * np.exp(-zeta_ * 2 * np.pi * f_ * t) * np.cos(2 * np.pi * f_ * np.sqrt(1 - zeta_**2) * t) + \
             V_ * np.exp(-zeta_ * 2 * np.pi * f_ * t) * np.sin(2 * np.pi * f_ * np.sqrt(1 - zeta_**2) * t)

        list_freq_peaks_signal.append(f_)

    # Creating a DataFrame in the proper format
    df = pd.DataFrame(np.hstack([t.reshape(-1, 1), x.reshape(-1, 1)]), columns=['time', 'ch1'])
    df['sample'] = 1
    df = df.set_index(['sample', 'time'])

    # Calculating the spectrum in frequency
    f, X = calc_fft(df, col='ch1', samples=[1], N=N, fs=fs)
    df_esp_freq = create_df_esp_freq(X, [1], 5000, 'ch1')

    # Finding the peaks positions
    df_peaks = get_freq_peaks_positions_values(df_esp_freq, 'ch1')
    list_idx_peaks_founded = [int( col.split('_')[-1] ) for col in df_peaks.columns]

    # Getting the frequencies values (round at 0 decimals is needed to get the desired results)
    list_freq_founded = [np.round(f[idx]) for idx in list_idx_peaks_founded]

    assert len(set(list_freq_peaks_signal).difference(set(list_freq_founded))) == 0
    assert pytest.approx(df_esp_freq.values.max(), df_peaks.values.max())

def test_get_freq_peaks_positions_values_with_multiple_channels(test_machine_data):

    idx_instances = test_machine_data.dataframe_instances.index.get_level_values(level=0).unique()
    N_max = 200

    df_esp_freq = calculate_spec_frequency_all_instances(data=test_machine_data,
                                                        N_max=N_max)

    df_peaks_ch1 = get_freq_peaks_positions_values(df_esp_freq, channel='ch1')
    df_peaks_ch2 = get_freq_peaks_positions_values(df_esp_freq, channel='ch2')
    df_peaks_ch3 = get_freq_peaks_positions_values(df_esp_freq, channel='ch3')

    # Renaming original df columns
    df_esp_freq.columns = df_esp_freq.columns.map('_'.join)

    assert np.allclose(df_peaks_ch1, df_esp_freq[df_peaks_ch1.columns])
    assert np.allclose(df_peaks_ch2, df_esp_freq[df_peaks_ch2.columns])
    assert np.allclose(df_peaks_ch3, df_esp_freq[df_peaks_ch3.columns])
    assert len(df_esp_freq) == len(df_peaks_ch1)
    assert len(df_esp_freq) == len(df_peaks_ch2)
    assert len(df_esp_freq) == len(df_peaks_ch3)
