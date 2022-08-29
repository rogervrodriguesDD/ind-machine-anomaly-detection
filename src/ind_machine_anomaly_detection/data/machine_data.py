# -*- coding: utf-8 -*-
from pathlib import Path
from io import TextIOWrapper
import pandas as pd
import typing as t

from ind_machine_anomaly_detection.config.core import (
    PROJECT_ROOT,
    MachineDataCatalog,
    LabelDataCatalog
)


class MachineSensorDataCSV():
    """
    Create a new object that contains the datasets of the machine sensors.
    It is considered that the instances data to be loaded is a set of files in
    csv format, which name is the number of identification of the sample.
    The labels dataset file, on the other hand, is one unique file with name
    configurable in the `config.yml` file.
    ...

    Attributes:
        instances_data_catalog (MachineDataCatalog): Catalog with configuration
                                    variables related to machine sensor data
        labels_data_catalog (LabelDataCatalog): Catalog with configuration
                                    variables related to labels data
        load_instances (bool): Load machine sensor data when initializing
        load_labels (bool): Load labels data when initializing
        dataframe_instances (pd.DataFrame): Multilevel index DataFrame with
                                    instances data. Indices order: Sample
                                    identification, measurement point
        dataframe_labels (pd.Series): Series with labels data. Index is set
                                    as the sample identification, starting
                                    from the number 1.
    """

    def __init__(self,
                 instances_data_catalog: MachineDataCatalog,
                 labels_data_catalog: LabelDataCatalog,
                 load_instances: bool = False,
                 load_labels: bool = False,
                 instances_file: t.Union[TextIOWrapper, None] = None
                 ) -> None:

        self.instances_data_catalog = instances_data_catalog
        self.labels_data_catalog = labels_data_catalog

        if load_instances:
            if instances_file is not None:
                self.load_instances_one_sample(instances_file)

            else:
                self.load_instances_all_samples()

        if load_labels:
            self.load_labels()

    def __getitem__(self, i) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return all samples instances and encoded labels.
        """

        return (self.dataframe_instances.loc[i],
                self.dataframe_labels.loc[i]
                    .replace(
                        self.labels_data_catalog.machine_data_labels_encoding
                    ).astype(int))

    def _load_instance_file(
            self,
            filepath: t.Union[Path, TextIOWrapper],
            columns_names: t.Sequence[str]
            ) -> pd.DataFrame:

        df = pd.read_csv(filepath, names=columns_names)
        return df

    def _test_labels_encoding(self) -> None:
        """
        Test if there is any missing key-value pair needed to replace the
        labels values.
        """

        labels_loaded = set(self.dataframe_labels.label.unique())

        labels_encoded = set(
            self.labels_data_catalog.machine_data_labels_encoding.keys()
        )

        missing_encoded_labels = labels_loaded.difference(labels_encoded)

        if len(missing_encoded_labels) > 0:
            raise OSError("Error when replacing labels values for encoded " +
                          "ones. Check the `config.yml` file for the " +
                          "following keys:" +
                          f" {', '.join(missing_encoded_labels)}.")

        return None

    def load_instances_one_sample(
            self,
            instances_file: TextIOWrapper
            ) -> pd.DataFrame:
        """
        Load one unique csv file with machine sensors data.

        Attribute redefined:
            dataframe_all_samples (pd.DataFrame): Dataframe with instances data
                    The DF has Multilevel index, where the first level is the
                    samples identification, and second level is the measurement
                    point identification.
        """

        mdata_instance_columns = self.instances_data_catalog \
            .machine_data_file_columns

        df = self._load_instance_file(instances_file, mdata_instance_columns)

        df['sample'] = 1
        df['sample_index'] = df.index.tolist()
        df = df.set_index(['sample', 'sample_index'])

        self.dataframe_instances = df

    def load_instances_all_samples(self) -> None:
        """
        Load set of csv files with machine sensors data. The path of the folder
        and the name of the files to be loaded is determined the parameters set
        in the `config.yml` file. The name is the range between the initial and
        final values.

        Attribute redefined:
            dataframe_all_samples (pd.DataFrame): Dataframe with instances data
                    The DF has Multilevel index, where the first level is the
                    samples identification, and second level is the measurement
                    point identification.
        """

        mdata_instance_folder = PROJECT_ROOT.joinpath(
            self.instances_data_catalog.machine_data_instances_folder
            ).resolve()
        mdata_instance_first_filename = self.instances_data_catalog \
            .machine_data_instances_filename_range[0]
        mdata_instance_last_filename = self.instances_data_catalog \
            .machine_data_instances_filename_range[1]
        mdata_instance_columns = self.instances_data_catalog \
            .machine_data_file_columns

        dfs = []
        for i in range(mdata_instance_first_filename,
                       mdata_instance_last_filename+1):

            mdata_instance_filename = f"{i}.csv"
            mdata_filepath = mdata_instance_folder.joinpath(
                mdata_instance_filename
                ).resolve()

            df_ = self._load_instance_file(
                    mdata_filepath,
                    mdata_instance_columns
                 )

            df_['sample'] = i
            df_['sample_index'] = df_.index.tolist()
            df_ = df_.set_index(['sample', 'sample_index'])
            dfs.append(df_)

        self.dataframe_instances = pd.concat(dfs, axis=0, ignore_index=False)

    def load_labels(self) -> None:
        """
        Load labels csv file, considering the parameters set in the
        `config.yml` file.

        Attribute redefined:
            dataframe_labels (pd.Series): Series with sample identification as
            index.
        """
        labels_filepath = PROJECT_ROOT.joinpath(
            self.labels_data_catalog.machine_data_labels_file
            ).resolve()

        self.dataframe_labels = pd.read_csv(labels_filepath)

        self.dataframe_labels['sample'] = \
            [i+1 for i in self.dataframe_labels.index]

        self.dataframe_labels = self.dataframe_labels.set_index('sample')

        # Testing if all the labels values will be correctly replaced
        self._test_labels_encoding()

    def get_samples_by_label(
            self,
            label: str
            ) -> t.Tuple[t.List[int], pd.DataFrame]:
        """
        Returns the set of data samples which has the desired label (in the
        case of this project, 'Before' or 'After').

        Returns:
            idx (list): list of samples identifications which labels match
                        the desired one.
            dataframe_filtered_samples (pd.DataFrame): filtered DataFrame
                        according to idx
        """

        idx = self.dataframe_labels.loc[self.dataframe_labels.label == label] \
            .index.tolist()
        return idx, self.dataframe_instances.loc[idx]

    def get_labels(self, encoded: bool = False) -> pd.DataFrame:
        """
        Returns the data labels. If encoded is True, the original values will
        be substituted according to the parameters given in `conf.yml` file.
        """
        if encoded:
            return self.dataframe_labels.replace(
                    self.labels_data_catalog.machine_data_labels_encoding
                    ).astype(int)
        return self.dataframe_labels

    def get_samples_with_missing_values(self) -> t.Tuple[pd.Index, pd.Index]:
        """
        Return the indices for which there are any missing values.

        Returns:
            idx_missing_instances (pd.Index): Indices for which are missing
                values in the instances dataframe.
            idx_missing_labels (pd.Index): Indices for which are missing labels
        """

        idx_missing_instances = \
            self.dataframe_instances[
                self.dataframe_instances.isna().sum(axis=1) > 0
            ] \
            .index \
            .get_level_values(level=0).unique()

        idx_missing_labels = \
            self.dataframe_labels.loc[
                self.dataframe_labels.isna().values
            ] \
            .index

        return idx_missing_instances, idx_missing_labels
