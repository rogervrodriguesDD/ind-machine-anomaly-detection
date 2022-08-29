# -*- coding: utf-8 -*-
import typing as t
import pandas as pd

from ind_machine_anomaly_detection.config.core import PipelineConfig


def select_features(
        *,
        df_features_candidates: pd.DataFrame,
        pipeline_config: PipelineConfig
        ) -> t.Tuple[pd.DataFrame, pd.Series]:
    """ Return a DataFrame with only the selected features, as set
    in the `config.yml` file.

    Args:
        df_features_candidates (pd.DataFrame): DataFrame with all built
                                               features
        pipeline_config (PipelineConfig): Configuration object with feature
                                          columns parameter.

    Returns:
        df_features (pd.DataFrame): DataFrame with the selected features
        df_labels (pd.Series): Series with the labels
    """

    df_features = df_features_candidates[pipeline_config.features_cols]

    if 'label' in df_features_candidates:
        df_labels = df_features_candidates['label']
    else:
        df_labels = None

    return df_features, df_labels
