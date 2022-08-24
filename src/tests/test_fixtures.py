# -*- encoding utf-8 -*-
import numpy as np
import pytest

from ind_machine_anomaly_detection.config.core import config, LabelDataCatalog
from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV

@pytest.fixture
def test_config():
    yield config

@pytest.fixture
def test_labels_data_catalog_with_missing_labels_encoding(test_config):

    labels_encoding_dict = test_config.labels_data_catalog.machine_data_labels_encoding
    labels_encoding_dict_with_missing_labels = {
        k: int(labels_encoding_dict[k]) for k in list(labels_encoding_dict.keys())[:-1]
    }

    labels_data_catalog_with_missing_labels_encoding = LabelDataCatalog(
        machine_data_labels_file=test_config.labels_data_catalog.machine_data_labels_file,
        machine_data_labels_encoding=labels_encoding_dict_with_missing_labels
    )

    yield labels_data_catalog_with_missing_labels_encoding

@pytest.fixture
def test_machine_data(test_config):
    test_machine_data = MachineSensorDataCSV(instances_data_catalog=test_config.machine_data_catalog,
                                            labels_data_catalog=test_config.labels_data_catalog,
                                            load_instances=True,
                                            load_labels=True)

    yield test_machine_data

@pytest.fixture
def test_machine_data_with_missing_values(test_machine_data):

    original_df_instances = test_machine_data.dataframe_instances.copy()
    original_df_labels = test_machine_data.dataframe_labels.copy()

    # Choosing random measurements points
    idx_original_df = original_df_instances.index
    samples_list = original_df_instances.index.get_level_values(level=0).unique().tolist()

    # Instance data
    samples_insert_nan = set(np.random.choice(samples_list, size=10, replace=False))
    idx_insert_nan = [
        (sample, i) \
            for sample in samples_insert_nan \
            for i in np.random.randint(
                            low=0,
                            high=test_machine_data.instances_data_catalog.machine_data_instances_number_meas_points,
                            size=1000
                    )
    ]

    col_insert_nan = np.random.choice(original_df_instances.columns, size=1)

    # Labels data
    samples_insert_nan_labels = np.random.choice(samples_list, size=10, replace=False)

    # Setting choosen samples and measurement points as NaN
    modified_df_instances = original_df_instances.copy()
    modified_df_labels = original_df_labels.copy()

    modified_df_instances.at[idx_insert_nan, col_insert_nan] = np.nan
    modified_df_labels.at[samples_insert_nan_labels] = np.nan

    test_machine_data.dataframe_instances = modified_df_instances.copy()
    test_machine_data.dataframe_labels = modified_df_labels.copy()

    yield test_machine_data
