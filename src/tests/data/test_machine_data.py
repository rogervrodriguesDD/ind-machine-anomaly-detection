# -*- coding: utf-8 -*-
import pytest

from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV
from tests.test_fixtures import (
    test_config,
    test_labels_data_catalog_with_missing_labels_encoding,
    test_machine_data,
    test_machine_data_with_missing_values,
)

def test_machine_data_object(test_config, test_machine_data):

    data_instances_idx = test_machine_data.dataframe_instances.index.get_level_values(level=0).unique().tolist()

    assert test_machine_data
    assert len(test_machine_data.dataframe_instances.columns) == len(test_config.machine_data_catalog.machine_data_file_columns)
    assert data_instances_idx[0] == test_config.machine_data_catalog.machine_data_instances_filename_range[0]
    assert data_instances_idx[-1] == test_config.machine_data_catalog.machine_data_instances_filename_range[-1]

def test_machine_data_object_with_missing_labels_encoding_config(test_config,
                                                            test_labels_data_catalog_with_missing_labels_encoding):

    test_data = MachineSensorDataCSV(instances_data_catalog=test_config.machine_data_catalog,
                                    labels_data_catalog=test_labels_data_catalog_with_missing_labels_encoding,
                                    load_instances=True,
                                    load_labels=False)

    # Trying to load labels (an error is expected)
    with pytest.raises(OSError) as excinfo:
        test_data.load_labels()

    assert "Error when replacing labels values for encoded ones" in str(excinfo)

def test_machine_data_object_missing_values(test_machine_data_with_missing_values):

    idx_missing_instances, idx_missing_labels = test_machine_data_with_missing_values.get_samples_with_missing_values()

    assert len(idx_missing_instances) == 10
    assert len(idx_missing_labels) == 10
