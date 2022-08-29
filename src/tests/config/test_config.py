# -*- coding: utf-8 -*-
from pathlib import Path
import pytest
from pydantic import ValidationError

from ind_machine_anomaly_detection.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml
    )

TEST_CONFIG_TEXT = """
# Project overview
project_name: ind-machine-anomaly-detection

# Package config
package_name: ind_machine_anomaly_detection
pipeline_name: svc_classification
pipeline_save_file: svc_classification_output
training_data_file: features_train.csv
testing_data_file: features_test.csv

# Data catalog
machine_data_instances_folder: data/raw
machine_data_instances_filename_range:
  - 1
  - 40
machine_data_file_columns:
  - ch1
  - ch2
  - ch3
machine_data_instances_number_meas_points: 70000
machine_data_instances_sampling_frequency: 1

machine_data_labels_file: data/raw/labels.csv
machine_data_labels_encoding:
  Before: 1
  After: 0

# Pipeline config
## Scaling pipeline (Document about this parameters)
channels_spectrum_in_frequency_scalers:
   - ch1
   - ch2
   - ch3

channels_peaks_positions_scalers:
  - ch1
  - ch2
  - ch3

channels_and_variables_min_max_scalers:
  ch1:
    - ffund
    - rms
    - mean
    - median
    - skew
    - kurtosis

  ch2:
    - ffund
    - rms
    - mean
    - median
    - skew
    - kurtosis

  ch3:
    - ffund
    - rms
    - mean
    - median
    - skew
    - kurtosis

## Features building pipeline
### Related to spectrum in frequency
number_max_freq_spectrum: 250
min_perc_samples_count_peaks_freq_spectrum: 0.50
number_ordered_peaks_freq_spectrum: 5
### Related to THD+N estimation
thdn_beta_filter: 38
thdn_n_harm: 5
thdn_wind_func_ratio: 0.1
thdn_f_max: 250

## Features selection
features_cols:
  - ch3_skew
  - ch2_mean
  - ch1_freqmax_3
  - ch2_ffund
  - ch1_kurtosis
  - ch1_skew
  - ch2_rms_ratio
  - ch2_X_154
  - ch2_X_166
  - ch1_X_166
  - ch3_rms_ratio
  - ch2_X_167
  - ch3_X_91
  - ch3_X_77
  - ch3_kurtosis

# Model configuration (SVC)
C: 1.0
kernel: linear
allowed_kernel_types:
  - linear
"""

INCOMPLETE_CONFIG_TEXT = """
# Project overview
project_name: ind-machine-anomaly-detection

# Package config
package_name: ind_machine_anomaly_detection
pipeline_name: svc_classification
pipeline_save_file: svc_classification_output
training_data_file: features_train.csv
testing_data_file: features_test.csv

# Data catalog
machine_data_instances_folder: data/raw
machine_data_file_columns:
  - ch1
  - ch2
  - ch3
machine_data_instances_number_meas_points: 70000
machine_data_instances_sampling_frequency: 1

machine_data_labels_file: data/raw/labels.csv
machine_data_labels_encoding:
  Before: 1
  After: 0

# Pipeline config
## Scaling pipeline (Document about this parameters)
channels_spectrum_in_frequency_scalers:
   - ch1
   - ch2
   - ch3

channels_peaks_positions_scalers:
  - ch1
  - ch2
  - ch3

channels_and_variables_min_max_scalers:
  ch1:
    - ffund
    - rms
    - mean
    - median
    - skew
    - kurtosis

  ch2:
    - ffund
    - rms
    - mean
    - median
    - skew
    - kurtosis

  ch3:
    - ffund
    - rms
    - mean
    - median
    - skew
    - kurtosis

## Features building pipeline
### Related to spectrum in frequency
number_max_freq_spectrum: 250
min_perc_samples_count_peaks_freq_spectrum: 0.50
number_ordered_peaks_freq_spectrum: 5
### Related to THD+N estimation
thdn_beta_filter: 38
thdn_n_harm: 5
thdn_wind_func_ratio: 0.1
thdn_f_max: 250

## Features selection
features_cols:
  - ch3_skew
  - ch2_mean
  - ch1_freqmax_3
  - ch2_ffund
  - ch1_kurtosis
  - ch1_skew
  - ch2_rms_ratio
  - ch2_X_154
  - ch2_X_166
  - ch1_X_166
  - ch3_rms_ratio
  - ch2_X_167
  - ch3_X_91
  - ch3_X_77
  - ch3_kurtosis

# Model configuration (SVC)
C: 1.0
kernel: linear
allowed_kernel_types:
  - linear
"""


class TestConfiguration(object):

    def test_fetch_config_structure(self, tmpdir):

        # Writing a tmp file
        configs_dir = Path(tmpdir)
        config_1 = configs_dir / "sample_config.yml"
        config_1.write_text(TEST_CONFIG_TEXT)
        parsed_config = fetch_config_from_yaml(cfg_path=config_1)

        # Creating the config object
        config = create_and_validate_config(parsed_config=parsed_config)

        assert config.app_config
        assert config.machine_data_catalog
        assert config.labels_data_catalog

    def test_missing_config_field_raises_error(self, tmpdir):

        # Writing a tmp file
        configs_dir = Path(tmpdir)
        config_1 = configs_dir / "sample_config.yml"
        config_1.write_text(INCOMPLETE_CONFIG_TEXT)
        parsed_config = fetch_config_from_yaml(cfg_path=config_1)

        # Creating the config object (Validation error expected)
        with pytest.raises(ValidationError) as excinfo:
            create_and_validate_config(parsed_config=parsed_config)

        assert "field required" in str(excinfo.value)
        assert "machine_data_instances_filename_range" in str(excinfo.value)
