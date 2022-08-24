# -*- coding: utf-8 -*-
import pandas as pd

def select_features(df_features_candidates, pipeline_config):
    """ Return a DataFrame with only the selected features, as set
    in the `config.yml` file.

    Args:
        df_features_candidates (pd.DataFrame): DataFrame with all built features
        pipeline_config (PipelineConfig): Configuration object with feature columns parameter.

    Returns:
        df_features (pd.DataFrame): DataFrame with the selected features
        df_labels (pd.Series): Series with the labels
    """

    df_features = df_features_candidates[pipeline_config.features_cols]
    df_labels = df_features_candidates['label']

    return df_features, df_labels
