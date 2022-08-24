import pandas as pd

from ind_machine_anomaly_detection.pipelines.features.utils import (
    calculate_spec_frequency_all_instances,
    get_freq_peaks_positions_values_all_channels,
    get_ordered_peaks_positions_all_channels,
    calculate_thdn_all_channels,
    calculating_stats_summary_all_instances,
)


def build_features_candidates(data, pipeline_config, training=True):
    """
    Build all potential features candidates on the machine data sensors
    in time domain.

    The following preprocessing steps are used to build the features:
    1. Calculate the spectrum in frequency of the sensors data (all channels);
    2. Get the peaks positions of the spectrum in frequency obtained by the previous step;
    3. Get the ordered values of the peaks, given their amplitude;
    4. Estimate the TDH+N of the sensors data (all channels);
    5. Calculate the statistical summary table (mean, median, std, skewness, kurtosis)
        of the sensors data in time domain.

    Since those steps have deterministic behavior, and there is no need to save metadata,
    it was decided not to wrap this function into a sklearn Pipeline component.
    ...

    Args:
        data (MachineSensorDataCSV): Object with all data instances and information
        pipeline_config (PipelineConfig): Configuration object with all needed parameters
                                        to build the features

    Returns:
        data_feat_candidates (pd.DataFrame): DataFrame with all features candidates and the labels
    """

    data_freq = calculate_spec_frequency_all_instances(
                    data,
                    N_max=pipeline_config.number_max_freq_spectrum
                )

    # The function `get_freq_peaks_positions_values_all_channels` must be used ONLY during training
    #   During prediction, those values are selected on the features filtering step
    if training:
        data_peaks_pos_values = get_freq_peaks_positions_values_all_channels(
                                    data_freq,
                                    min_perc_samples_count=pipeline_config.min_perc_samples_count_peaks_freq_spectrum,
                                )
    else:
        data_peaks_pos_values = data_freq.copy()
        data_peaks_pos_values.columns = data_peaks_pos_values.columns.map('_'.join)

    data_ord_peaks_pos = get_ordered_peaks_positions_all_channels(
                            data_peaks_pos_values,
                            n_max_positions=pipeline_config.number_ordered_peaks_freq_spectrum,
                            n_freq_points=pipeline_config.number_max_freq_spectrum
                        )

    data_thdn = calculate_thdn_all_channels(data, pipeline_config)

    data_stats_summ = calculating_stats_summary_all_instances(data)

    data_feat_candidates = pd.concat(
        [
            data_peaks_pos_values,
            data_ord_peaks_pos,
            data_thdn,
            data_stats_summ,
            data.get_labels()
        ],
        axis=1,
        ignore_index=False
        )

    return data_feat_candidates
