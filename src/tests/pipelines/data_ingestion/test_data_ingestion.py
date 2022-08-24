# -*- coding: utf-8 -*-
from ind_machine_anomaly_detection.pipelines.data_ingestion import clean_data

from tests.test_fixtures import test_config, test_machine_data, test_machine_data_with_missing_values

def test_clean_data(test_machine_data_with_missing_values):

    idx_missing_inst_before, _ = test_machine_data_with_missing_values.get_samples_with_missing_values()

    cleaned_data = clean_data(test_machine_data_with_missing_values)
    idx_missing_inst_after, idx_missing_lab_after = cleaned_data.get_samples_with_missing_values()

    assert len(idx_missing_inst_before) > 0
    assert len(idx_missing_inst_after) == 0
    assert len(idx_missing_lab_after) == 0
    assert len(test_machine_data_with_missing_values.dataframe_instances) > len(cleaned_data.dataframe_instances)
