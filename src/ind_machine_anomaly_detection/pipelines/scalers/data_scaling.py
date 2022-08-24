# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline

from ind_machine_anomaly_detection.pipelines.scalers.classes import (
    SpecFreqScaler,
    FreqPosMinMaxScaler,
    MinMaxScalerCustomized
)

def get_pipeline_scaling(pipeline_config):
    """
    Create a sklean Pipeline object with preprocessing scaling steps.
    The following scaling steps are available:
    1. 'scaler_spec_{channel}': Scaler for Spectrum in Frequency (class SpecFreqScaler)
    2. 'scaler_freq_pos_{channel}': Scaler of the Positions of the Peaks (class FreqPosMinMaxScaler)
    3. 'scaler_minmax_{channel}': Customized MinMaxScaler for the listed variables (class MinMaxScalerCustomized)

    Obs,: The parameters of which channel or variables for each step is set in the Pipeline Configuration
    parameters in the `conf.yml` file.

    Args:
        pipeline_config (PipelineConfig object): Configuration object with channels and variables for each available
                                                steps.

    Returns:
        pipeline_scaling (sklearn.Pipeline): Pipeline with scaling steps.
    """

    # Listing the steps to be included
    include_scaling_steps = []

    if pipeline_config.channels_spectrum_in_frequency_scalers is not None:
        spec_freq_scalers_steps = [
            (
            f"scaler_spec_{channel}",
            SpecFreqScaler(channel=channel)
            )
            for channel in pipeline_config.channels_spectrum_in_frequency_scalers
        ]
        include_scaling_steps.extend(spec_freq_scalers_steps)

    if pipeline_config.channels_peaks_positions_scalers is not None:
        freq_pos_scalers_steps = [
            (
            f"scaler_freq_pos_{channel}",
            FreqPosMinMaxScaler(channel=channel)
            )
            for channel in pipeline_config.channels_peaks_positions_scalers
        ]
        include_scaling_steps.extend(freq_pos_scalers_steps)

    minmax_scalers_steps = [
        (
        f"scaler_minmax_{channel}",
        MinMaxScalerCustomized(channel=channel, variables=variables)
        )
        for channel, variables in pipeline_config.channels_and_variables_min_max_scalers.items()
    ]
    include_scaling_steps.extend(minmax_scalers_steps)

    pipeline_scaling = Pipeline(include_scaling_steps)

    return pipeline_scaling
