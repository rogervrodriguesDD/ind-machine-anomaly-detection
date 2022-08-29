# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ind_machine_anomaly_detection.pipelines.features import (
    build_features_candidates
    )

from ind_machine_anomaly_detection.pipelines.train_test_split import split_data


def test_split_data_first_format(test_machine_data):

    original_df_inst = test_machine_data.dataframe_instances.copy()

    data_train, data_test = split_data(test_machine_data)

    def calc_ratio_before_after(data):

        labels = data[:][1]
        ratio_before_after = labels.values.mean()

        return ratio_before_after

    ratio_before_after_train = calc_ratio_before_after(data_train)
    ratio_before_after_test = calc_ratio_before_after(data_test)

    samples_train = data_train.dataframe_instances.index \
        .get_level_values(level=0).unique()

    samples_test = data_test.dataframe_instances.index \
        .get_level_values(level=0).unique()

    assert pytest.approx(ratio_before_after_train) == 0.5
    assert pytest.approx(ratio_before_after_test) == 0.5

    assert np.allclose(original_df_inst.loc[samples_train].values,
                       data_train.dataframe_instances.values)
    assert np.allclose(original_df_inst.loc[samples_test].values,
                       data_test.dataframe_instances.values)


def test_split_data_second_format(test_config, test_machine_data):

    _, original_df_labels = test_machine_data[:]

    test_features = build_features_candidates(
        data=test_machine_data,
        pipeline_config=test_config.pipeline_config
    )

    original_df_feat = test_features.drop(columns='label').copy()

    X_train, X_test, y_train, y_test = split_data(
        original_df_feat,
        original_df_labels
        )

    ratio_before_after_train = y_train.values.mean()
    ratio_before_after_test = y_test.values.mean()

    samples_train = X_train.index
    samples_test = X_test.index

    assert pytest.approx(ratio_before_after_train) == 0.5
    assert pytest.approx(ratio_before_after_test) == 0.5

    assert np.allclose(original_df_feat.loc[samples_train].values,
                       X_train.values)
    assert np.allclose(original_df_feat.loc[samples_test].values,
                       X_test.values)
