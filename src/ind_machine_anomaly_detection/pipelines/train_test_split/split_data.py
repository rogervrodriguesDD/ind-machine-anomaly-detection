# -*- coding: utf-8 -*-
import copy
import pandas as pd
from sklearn.model_selection import train_test_split

from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV

def _split_df_data(X, y):
        return train_test_split(X,
                                y,
                                test_size=0.25,
                                shuffle=True,
                                random_state=42,
                                stratify=y)

def _reshape_to_original_df_format(X):
    X = pd.DataFrame(X.reshape(-1,5), columns=['ch1', 'ch2', 'ch3', 'sample', 'sample_index'])
    for col in ['sample', 'sample_index']:
        X[col] = X[col].astype('int')
    X = X.set_index(['sample', 'sample_index'])
    return X

def split_data(data, y=None):
    """ Split the data object into training and testing datasets.
    The parameters for the spliting process are set in the '_split_df_data'
    function.
    The returned object has the same type as the one passed as input.

    Args:
        data (MachineSensorDataCSV or pd.DataFrame): Object (or DataFrame) with instances data.
        y (np.array): Array with labels (Default=None)

    Returns:
        If data is 'MachineSensorDataCSV':
        data_train, data_test: Objects with instances data of training and testing sets

        If data is 'pd.DataFrame':
        X_train, X_test, y_train, y_test: DataFrames and Series with instances or features
                                        values of training and testing sets.
    """

    if isinstance(data, MachineSensorDataCSV):
        data_train = copy.copy(data)
        data_test = copy.copy(data)

        X, y = data.dataframe_instances.copy(), data.dataframe_labels.copy()

        # Converting X to a proper shape (including the two-level indices)
        X['sample'] = X.index.get_level_values(level=0)
        X['sample_index'] = X.index.get_level_values(level=1)
        X_reshaped = X.values.reshape(-1, data.instances_data_catalog.machine_data_instances_number_meas_points, 5)

        X_train, X_test, y_train, y_test = _split_df_data(X_reshaped, y)

        # Returning X_train, X_test to the proper shape
        X_train = _reshape_to_original_df_format(X_train)
        X_test = _reshape_to_original_df_format(X_test)

        # Substituting dataframes in the datas objects
        data_train.dataframe_instances = X_train
        data_train.dataframe_labels = y_train

        data_test.dataframe_instances = X_test
        data_test.dataframe_labels = y_test

        return data_train, data_test

    elif isinstance(data, pd.DataFrame):

        X_train, X_test, y_train, y_test = _split_df_data(data, y)

        return X_train, X_test, y_train, y_test

    raise OSError("'data' object does not has the proper type (MachineSensorDataCSV or pd.DataFrame)")
