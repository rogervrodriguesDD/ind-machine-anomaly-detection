# -*- coding: utf-8 -*-
import pytest

from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV  # noqa


def test_machine_data_object(test_config, test_machine_data):

    data_instances_idx = test_machine_data.dataframe_instances.index \
                         .get_level_values(level=0).unique().tolist()

    num_columns_df_loaded = len(test_machine_data.dataframe_instances.columns)
    num_columns_df_config = len(
        test_config.machine_data_catalog.machine_data_file_columns
        )

    filename_range = test_config.machine_data_catalog \
        .machine_data_instances_filename_range
    first_filename_config = filename_range[0]
    last_filename_config = filename_range[1]

    assert test_machine_data
    assert num_columns_df_loaded == num_columns_df_config
    assert data_instances_idx[0] == first_filename_config
    assert data_instances_idx[-1] == last_filename_config


def test_machine_data_object_with_missing_labels_encoding_config(
        test_config,
        test_labels_data_catalog_with_missing_labels_encoding):

    labels_data_catalog = test_labels_data_catalog_with_missing_labels_encoding

    test_data = MachineSensorDataCSV(
        instances_data_catalog=test_config.machine_data_catalog,
        labels_data_catalog=labels_data_catalog,
        load_instances=True,
        load_labels=False
        )

    # Trying to load labels (an error is expected)
    with pytest.raises(OSError) as excinfo:
        test_data.load_labels()

    expected_error = "Error when replacing labels values for encoded ones"

    assert expected_error in str(excinfo)


def test_machine_data_object_missing_values(
        test_machine_data_with_missing_values):

    idx_missing_instances, idx_missing_labels = \
        test_machine_data_with_missing_values.get_samples_with_missing_values()

    assert len(idx_missing_instances) == 10
    assert len(idx_missing_labels) == 10
