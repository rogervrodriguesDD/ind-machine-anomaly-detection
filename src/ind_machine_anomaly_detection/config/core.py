# -*- coding: utf-8 -*-
from pathlib import Path
import typing as t

from pydantic import BaseModel, validator
from strictyaml import load, YAML

# Project Directories
CONFIG_FOLDER_PATH = Path(__file__).resolve().parent
CONFIG_FILE_PATH = CONFIG_FOLDER_PATH.joinpath("config.yml")
PACKAGE_ROOT = CONFIG_FOLDER_PATH.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
TRAINED_MODEL_DIR = PROJECT_ROOT.joinpath("models").resolve()

class AppConfig(BaseModel):
    """
    Create Application-level configuration object.
    """

    project_name: str
    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    training_data_file: str
    testing_data_file: str

class MachineDataCatalog(BaseModel):

    machine_data_instances_folder: str
    machine_data_instances_filename_range: t.Sequence[int]
    machine_data_file_columns: t.Sequence[str]
    machine_data_instances_number_meas_points: int
    machine_data_instances_sampling_frequency: float

class LabelDataCatalog(BaseModel):

    machine_data_labels_file: str
    machine_data_labels_encoding: dict

class PipelineConfig(BaseModel):

    # Scalers
    channels_spectrum_in_frequency_scalers: t.Optional[t.Sequence[str]]
    channels_peaks_positions_scalers: t.Optional[t.Sequence[str]]
    channels_and_variables_min_max_scalers: dict

    # Features building
    number_max_freq_spectrum: int
    min_perc_samples_count_peaks_freq_spectrum: float
    number_ordered_peaks_freq_spectrum: int
    thdn_beta_filter: float
    thdn_n_harm: int
    thdn_wind_func_ratio: float
    thdn_f_max: int

    # Feature selection
    features_cols: t.Sequence[str]

    # Model config
    allowed_kernel_types: t.Sequence[str]
    C: float
    kernel: str

    @validator('kernel')
    def check_kernel_type(cls, value, values):
        """
        Following the research phase, kernels types are restricted to
        'linear'
        """
        allowed_kernel_types = values.get('allowed_kernel_types')
        if value not in allowed_kernel_types:
            raise ValueError(
                f"the kernel type specified: {value}, "
                f"is not in the allowed set: {allowed_kernel_types}"
            )
        return value

class Config(BaseModel):

    app_config: AppConfig
    machine_data_catalog: MachineDataCatalog
    labels_data_catalog: LabelDataCatalog
    pipeline_config: PipelineConfig

def find_config_file() -> Path:
    """
    Locate the configuration file.

    Args: None
    Returns:
        CONFIG_FILE_PATH (Path): Path to the configuration YAML file.
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH

    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """
    Parse YAML containing the package configuration.

    Args:
        cfg_path (Path, optional): Path to the configuration YAML to be loaded.

    Returns:
        parsed_config (YAML): Parsed YAML object with configurations parameters.
    """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """
    Run validation on config values.

    Args:
        parsed_config (YAML): Parsed YAML object which parameters will be unpacked
                            and loaded as attributes of the 'config' object.
    Returns:
        _config (Config): Master configuration object with validated configuration values.
    """
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        machine_data_catalog=MachineDataCatalog(**parsed_config.data),
        labels_data_catalog=LabelDataCatalog(**parsed_config.data),
        pipeline_config=PipelineConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
