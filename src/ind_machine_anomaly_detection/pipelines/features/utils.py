# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.fft import fft, rfft
from scipy import signal
import typing as t

from ind_machine_anomaly_detection.config.core import PipelineConfig
from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV  # noqa


def calc_fft(
        df: pd.DataFrame,
        col: str,
        samples: t.List[int],
        N: int,
        fs: float
        ) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the FFT for the given DataFrame and desired column and
    samples identification numbers.

    Args:
        df (pd.DataFrame): Multi-level index DataFrame with signal data.
                           Expected indices order: Sample identification,
                           measurement point
        col (str): Name of the column to be used when calculating the FFT
        samples (list): List of samples indices to be used for filtering the
                        DataFrame
        N (int): Number of measurement points of the original signal
        fs (float): Sampling frequency

    Returns:
        f (np.array): Vector with the N/2 frequency values of the spectrum
        mat_X (np.array): Array with shape (N/2, number of samples) containing
                          the absolute value of the Discrete Fourier
                          Tranformation of the Signal. The limit frequency is
                          considered to be half of the sampling frequency
                          (therefore N/2).
    """

    df_ = df.copy()
    f = np.linspace(0, fs / 2, int(N/2))
    mat_X = np.zeros(shape=(int(N/2), len(samples)))

    for i, sample in enumerate(samples):

        x = df_.loc[sample][col].values
        mat_X[:, i] = np.abs(fft(x) * 2 / N)[:int(N/2)]

    return f, mat_X


def create_df_esp_freq(
        X_mat: np.ndarray,
        samples: t.List[int],
        N_max: int,
        col: str,
        axis_concat: int = 0,
        data_freq: bool = None
        ) -> pd.DataFrame:
    """
    Create a DataFrame for a given Spectral output of a FFT operation and its
    informations.
    If a previous created DF is passed as `data_freq` argument, the new DF
    will be concated to it using the `axis_concat` argument as reference.

    Args:
        X_mat (np.array): Matrix containing the output of the FFT operation
                          (Spectrum in frequency).
        samples (list): List of samples indices to be used when creating the
                        Multi-level index.
        N_max (int): Maximum number of frequency points to be registered.
        col (str): Name of the column used when calculating the FFT.
        axis_concat (int): Reference for the concatenation method

    Returns:
        data_freq (pd.DataFrame): When None, a new DataFrame is returned.
                                  Else, the new DF is concatenated to the given
                                  one, considering the `axis_concat` reference.
    """

    indexes = ['X_{}'.format(f) for f in range(N_max)]
    multidx_cols = [*zip([col]*len(indexes), indexes)]
    indexes_ = pd.MultiIndex.from_tuples(multidx_cols)

    data_freq_ = pd.DataFrame(
        data=X_mat[:N_max],
        columns=samples,
        index=indexes_
    )

    data_freq_ = data_freq_.T

    if data_freq is None:
        data_freq = data_freq_

    else:
        data_freq = pd.concat(
            [data_freq, data_freq_],
            axis=axis_concat,
            ignore_index=False
        )

    return data_freq


def calculate_spec_frequency_all_instances(
        *,
        data: MachineSensorDataCSV,
        N_max: int
        ) -> pd.DataFrame:
    """
    Apply the FFT on all samples measurements, for the three channels, and
    returns a concatenated DataFrame with the results.

    Args:
        data (MachineSensorDataCSV): Object that contains all datasets of the
                                     machine sensors.
        N_max (int): Maximum number of frequency points to be returned as
                                    columns (for each channel).

    Returns:
        data_spec_freq (pd.DataFrame): DataFrame with the spectral output,
            where the rows are related to the sample identification and
            columns are the amplitude of the i-th measured frequency.
    """

    df_samples = data.dataframe_instances
    idx_samples = df_samples.index.get_level_values(level=0).unique().tolist()

    number_points = data.instances_data_catalog \
        .machine_data_instances_number_meas_points

    sampling_frequency = data.instances_data_catalog \
        .machine_data_instances_sampling_frequency

    data_spec_freq = None

    for col in ['ch1', 'ch2', 'ch3']:

        _, fft = calc_fft(
            df_samples,
            col,
            idx_samples,
            N=number_points,
            fs=sampling_frequency
        )

        # Creating a dataframe with the amplitudes in the frequency domain
        axis_concat = 1
        data_spec_freq = create_df_esp_freq(
            fft,
            idx_samples,
            N_max,
            col,
            axis_concat,
            data_spec_freq
        )

    return data_spec_freq


def get_freq_peaks_positions_values(
        data_f: pd.DataFrame,
        channel: str,
        min_perc_samples_count: float = 0.25
        ) -> pd.DataFrame:
    """
    Get in the Spectral output the positions and values of the peaks for
    at least a percentual of the samples spectrums and the given channel name.

    Args:
        data_f (pd.DataFrame): DataFrame with Spectral output of FFT on the
                               data
        channel (str): Name of the channel (first level column)
        min_perc_samples_count (int): Percentual of samples considered to
                               return the peak

    Returns:
        data_freq_peaks (pd.DataFrame): DataFrame with peaks positions
        (columns names) and values for each sample.
    """

    list_peaks_idx = []
    list_peaks_values = []

    # Getting peaks indices
    for i, x in data_f[channel].iterrows():

        peaks_idx, _ = signal.find_peaks(x, threshold=np.quantile(x, 0.15))
        list_peaks_idx.append(peaks_idx)
        list_peaks_values.append(x[peaks_idx].values)

    # Counting the number of times each peak was found
    count_peaks = pd.DataFrame(
        [0] * len(data_f[channel].columns),
        index=[*range(len(data_f[channel].columns))],
        columns=['count']
        )

    for peaks_idx in list_peaks_idx:
        count_peaks.loc[peaks_idx] += 1

    # Getting the peaks that where considered for at least half of the samples
    get_peaks_idx = count_peaks.loc[
        count_peaks['count'] > min_perc_samples_count * len(data_f[channel])
        ]\
        .index

    # Selecting only the columns related to the peaks
    data_freq_peaks = data_f[channel].iloc[:, list(get_peaks_idx)]

    # Renaming the columns to adding the prefix "ch{}_"
    rename_cols = {
        col: '{}_{}'.format(channel, col) for col in data_freq_peaks.columns
    }

    data_freq_peaks = data_freq_peaks.rename(columns=rename_cols)

    return data_freq_peaks


def get_freq_peaks_positions_values_all_channels(
        *,
        data_freq: pd.DataFrame,
        min_perc_samples_count: float = 0.25
        ) -> pd.DataFrame:
    """
    Get in the Spectral output the positions and values of the peaks for all
    all spectrums and channels. The name of the channel is added as a prefix
    to the original column.

    Args:
        data_freq (pd.DataFrame): DataFrame with Spectral output of FFT on the
                                  data

    Returns:
        data_freq_peaks (pd.DataFrame): DataFrame with peaks positions
                                 (columns names) and values for each sample.
    """

    list_data_freq_peaks = []
    data_freq_peaks = get_freq_peaks_positions_values(data_freq, channel='ch1')
    list_channels = ['ch1', 'ch2', 'ch3']
    for i, channel in enumerate(list_channels):
        data_freq_peaks_ = get_freq_peaks_positions_values(
            data_freq,
            channel=channel,
            min_perc_samples_count=min_perc_samples_count
        )

        list_data_freq_peaks.append(data_freq_peaks_)

    data_freq_peaks = pd.concat(
        list_data_freq_peaks,
        axis=1,
        ignore_index=False
    )

    return data_freq_peaks


def select_cols_names_channel(
        df: pd.DataFrame,
        channel: str = 'ch1'
        ) -> t.List[str]:
    """
    Returns the columns of the DataFrame related to given channel.
    The search consists in comparing the last component of splitted
    column name to the channel.

    Args:
        df (pd.DataFrame): DataFrame which columns will be considered
        channel (str): Name of the channel

    Returns:
        (list): List of columns related to channel
    """
    return [col for col in df.columns if col.split('_', 1)[0] == channel]


def select_cols_channel(
        df: pd.DataFrame,
        channel: str = 'ch1'
        ) -> pd.DataFrame:
    """
    Returns the DataFrame `df` with only the columns related to given
    channel.

    Args:
        df (pd.DataFrame): DataFrame which columns will be considered
        channel (str): Name of the channel

    Returns:
        df_filtered (pd.DataFrame): Filtered DataFrame
    """
    cols_names = select_cols_names_channel(df, channel=channel)
    return df[cols_names]


def get_ordered_peaks_positions(
        df: pd.DataFrame,
        channel: str,
        n_freq_points: int,
        n_max_positions: int = 3
        ) -> pd.DataFrame:
    """
    Get the normalized peaks positions ordered by the values peaks.
    The idea here is to have the information, which of the given peak
    has the highest value (that probably makes it the fundamental frequency),
    followed by the other peaks.
    The normalization of the positions is done by the number of samples, which
    is the number of peaks considered.

    Args:
        df (pd.DataFrame): DataFrame with peaks positions and values
        channel (str): Name of the channel
        n_freq_points (int): Total number of frequency points of the original
                             spectrum of frequency
        n_max_positions (int): Number of ordered peaks desired

    Returns:
        data_ord_peak_pos (pd.DataFrame): DataFrame with ordered peaks
                             positions.
    """

    df_ = select_cols_channel(df, channel=channel)

    array_peaks_pos = np.zeros(shape=(len(df_), n_max_positions))

    i = 0
    for idx_df, row in df_.iterrows():
        row_ = row.copy()
        for pos in range(1, n_max_positions+1):

            idx_max = row_.argmax()
            position = int(row_.index[idx_max].split('_')[-1])
            row_ = row_.drop(index=row_.index[idx_max])
            array_peaks_pos[i, pos-1] = position / n_freq_points

        i += 1

    cols_df_ord = [
        '{}_freqmax_{}'.format(channel, i) for i in range(1, n_max_positions+1)
    ]

    data_ord_peak_pos = pd.DataFrame(
        data=array_peaks_pos,
        columns=cols_df_ord,
        index=df_.index
    )

    return data_ord_peak_pos


def get_ordered_peaks_positions_all_channels(
        *,
        data_freq_peaks: pd.DataFrame,
        n_freq_points: int,
        n_max_positions: int = 5
        ) -> pd.DataFrame:
    """
    Get the normalized peaks positions ordered by the values peaks for all
    channels, using the function `get_ordered_peaks_positions`.

    Args:
        df (pd.DataFrame): DataFrame with peaks positions and values
        n_max_positions (int): Number of ordered peaks desired

    Returns:
        data_ord_peak_pos (pd.DataFrame): DataFrame with ordered peaks
                                          positions.
    """

    list_data_peaks_pos = []
    list_channels = ['ch1', 'ch2', 'ch3']
    for i, channel in enumerate(list_channels):

        data_ord_peaks_pos_ = get_ordered_peaks_positions(
            data_freq_peaks,
            channel=channel,
            n_freq_points=n_freq_points,
            n_max_positions=n_max_positions
        )

        list_data_peaks_pos.append(data_ord_peaks_pos_)

    data_ord_peaks_pos = pd.concat(
        list_data_peaks_pos,
        axis=1,
        ignore_index=False
    )

    return data_ord_peaks_pos


def parabolic(f: np.ndarray, x: int) -> t.Tuple[float, float]:
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known. Returns
    the coordinates of the vertex of a parabola that goes through point
    x and its two neighbors

    Args:
        f (np.array):  Vector which the curve to be interpolated
        x (int): Index of `f` where the vertex is expected

    Returns:
        vx (float): horizontal coordinate of the vertex of the parabola
        yx (float): vertical coordinate of the vertex of the parabola
    """

    if int(x) != x:
        raise ValueError('x must be an integer sample index')
    else:
        x = int(x)
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def calculate_thdn(
        df: pd.DataFrame,
        col: str,
        samples: t.List[int],
        N: int = 70000,
        beta_filter: float = 38,
        n_harm: int = 5,
        wind_fund_ratio: float = 0.1,
        f_max: int = 250
        ) -> t.Tuple[t.List[int], t.List[float], t.List[float], t.List[float]]:
    """
    Calculate the Total Harmonic Distortion plus Noise for a the given set of
    signals in time domain.

    Args:
        df (pd.DataFrame): DataFrame with set of signals in time domain
        col (str): Name of the channel
        samples (list): List of measurement samples to be evaluated
        N (int): Number of measurement points for each sample
        beta_filter (float): Shape parameter of the Kaiser window function
                             (Default = 38)
        n_harm (int): Number of harmonic regions, in addition to fundamental,
                             considered in the spectrum frequency
                             (Default = 5)
        wind_fund_ratio (float): Size of the window considered in the harmonic
        regions (Default = 0.1)
        f_max (int): Maximum frequency considered (Default = 250)

    Returns:
        samples_sorted (list of ints): List of sorted samples id
        f_fund (list of floats): List with fundamental frequencies founded per
                              sample
        rms_total (list of floats): List with RMS Total calculated per sample
        rms_noise (list of floats): List with RMS Noise calculated per sample
    """

    # When unstacked, the indices (samples id) are sorted
    df_ = df.loc[samples][col].unstack().copy()
    samples_sorted = df_.index

    # Matrix format: row: frequency, column: sample
    mat_X = np.zeros(shape=(int(N/2 + 1), len(samples)))
    mat_X_noise = mat_X.copy()

    f_fund = []
    rms_total = []
    rms_noise = []

    # Defining and window
    window = signal.windows.kaiser(M=N, beta=beta_filter)
    windowed = (df_.values * window).T  # Row: Time, column: Sample

    for i, sample in enumerate(samples_sorted):

        x = windowed[:, i]
        X = rfft(x)
        mat_X[:, i] = np.abs(X)

        fund_f_idx = int(np.argmax(np.abs(X[:f_max])))
        true_f_idx = parabolic(np.log(np.abs(X)), fund_f_idx)[0]
        f_fund.append(true_f_idx)

        rms_total.append(20*np.log10(np.sqrt(np.abs(X.T @ X))))

        # Setting fundamental and harmonic regions to zero
        for nn in range(1, n_harm + 1):

            lowermin = int(true_f_idx * nn * (1 - wind_fund_ratio))
            uppermin = int(true_f_idx * nn * (1 + wind_fund_ratio))

            X[lowermin: uppermin] = 0.0

        mat_X_noise[:, i] = np.abs(X)

        rms_noise.append(20*np.log10(np.sqrt(np.abs(X.T @ X))))

    return samples_sorted, f_fund, rms_total, rms_noise


def get_df_thdn_by_channel(
        data: MachineSensorDataCSV,
        pipeline_config: PipelineConfig,
        channel: str = 'ch1',
        add_channel_prefix: bool = True
        ) -> pd.DataFrame:
    """
    Get the Total Harmonic Distortion plus Noise, calculate the RMS signal and
    noise ratio, and returns the estimated values in DataFrame.

    Args:
        data (MachineSensorDataCSV): Object with set of signals in time domain.
        pipeline_config (PipelineConfig): Configuration object with needed
                            parameters to call function `calculate_thdn`.
        channel (str): Name of the channel.
        add_channel_prefix (bool): If True, add the channel name to columns as
                            prefix.

    Returns:
        df_thdn (pd.DataFrame): DataFrame with results of the TDHn estimation
    """

    samples = data.dataframe_instances.index \
                  .get_level_values(level=0).unique()

    df = data.dataframe_instances.copy()

    number_points = data.instances_data_catalog. \
        machine_data_instances_number_meas_points

    samples_sorted, f_fund, rms_total, rms_noise = calculate_thdn(
        df,
        channel,
        samples,
        N=number_points,
        beta_filter=pipeline_config.thdn_beta_filter,
        n_harm=pipeline_config.thdn_n_harm,
        wind_fund_ratio=pipeline_config.thdn_wind_func_ratio,
        f_max=pipeline_config.thdn_f_max
    )

    # Creating a DF with the results
    df_thdn = pd.DataFrame(
        index=samples_sorted,
        data={
            'ffund': f_fund,
            'rms_total': rms_total,
            'rms_noise': rms_noise
            }
    )

    df_thdn['rms_ratio'] = df_thdn['rms_noise'] / df_thdn['rms_total']
    df_thdn = df_thdn.drop(columns='rms_noise')

    if add_channel_prefix:
        rename_cols = {
            col: f"{channel}_{col}" for col in df_thdn.columns
        }

        df_thdn = df_thdn.rename(columns=rename_cols)

    # Reordering to the original index order
    df_thdn = df_thdn.loc[samples]

    return df_thdn


def calculate_thdn_all_channels(
        *,
        data: MachineSensorDataCSV,
        pipeline_config: PipelineConfig
        ) -> pd.DataFrame:
    """
    Calculate Total Harmonic Distortion plus Noise measurement for all
    channels.

    Args:
        data (MachineSensorDataCSV): DataFrame with set of signals in time
                             domain
        pipeline_config (PipelineConfig): Configuration object with parameters
                             needed to estimate tdhn (see `calculate_thdn`
                             documentation).

    Returns:
        df_thdn (pd.DataFrame): DataFrame with results of the TDHn
                            estimation
    """

    list_df_thdn = []
    list_channels = ['ch1', 'ch2', 'ch3']

    for channel in list_channels:

        df_thdn_ = get_df_thdn_by_channel(data, pipeline_config,
                                          channel=channel)

        list_df_thdn.append(df_thdn_)

    df_thdn = pd.concat(list_df_thdn, axis=1, ignore_index=False)

    return df_thdn


def calculating_stats_summary_all_instances(
        data: MachineSensorDataCSV
        ) -> pd.DataFrame:
    """
    Calculate a statistical summary table on all instances data (samples and
    channels) in time domain. The parameters calculated are: mean, median,
    kurtosis, and skewness.

    Args:
        data (MachineSensorDataCSV): DataFrame with signal data in time domain

    Returns:
        df_stats_summary (pd.DataFrame): DataFrame with statistical summary
                             table
    """

    list_grouping_func = ['mean', 'median', 'kurtosis', 'skew']

    list_dfs = []
    df_ = data.dataframe_instances
    for col in ['ch1', 'ch2', 'ch3']:
        df_stats_summary_ = df_[col].unstack().agg(list_grouping_func, axis=1)

        rename = {
            name: col + '_' + name for name in df_stats_summary_.columns
        }

        list_dfs.append(df_stats_summary_.rename(columns=rename))

    df_stats_summary = pd.concat(list_dfs, axis=1, ignore_index=False)

    return df_stats_summary
