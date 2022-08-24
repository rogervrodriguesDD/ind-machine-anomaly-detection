# -*- coding: utf-8 -*-
import copy

def clean_data(data):
    """
    Copy the `data` object and drop the indices for which there are missing values
    in the instances OR labels data.

    Args:
        data (MachineSensorDataCSV): Object with loaded instances and labels datasets.

    Returns:
        data_cp (MachineSensorDataCSV): Object with cleaned instances and labels datasets.
    """

    # The MachineSensorDataCSV object is mutable, thefore we use `copy` module
    data_cp = copy.copy(data)

    idx_missing_instances, idx_missing_labels = data_cp.get_samples_with_missing_values()
    idx_drop = set(idx_missing_instances).union(set(idx_missing_labels))

    data_cp.dataframe_instances = data_cp.dataframe_instances.drop(index=idx_drop)
    data_cp.dataframe_labels = data_cp.dataframe_labels.drop(idx_drop)

    return data_cp
