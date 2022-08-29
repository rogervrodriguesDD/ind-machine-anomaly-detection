# -*- coding: utf-8 -*-
from io import TextIOWrapper
import logging
import typing as t

from ind_machine_anomaly_detection.config.core import (
    config,
)

from ind_machine_anomaly_detection.pipelines.features import (
    build_features_candidates,
    select_features
)

from ind_machine_anomaly_detection.data.data_management import load_pipeline
from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV  # noqa : E501


_logger = logging.getLogger(__name__)


def predict(file: t.Optional[TextIOWrapper]) -> dict:

    # Loading the data
    data = MachineSensorDataCSV(
            config.machine_data_catalog,
            config.labels_data_catalog,
            load_instances=True,
            load_labels=False,
            instances_file=file
            )

    # Building the features and selected only the necessary ones
    data_feat_candidates = build_features_candidates(
        data=data,
        pipeline_config=config.pipeline_config,
        training=False
        )

    X, _ = select_features(
        df_features_candidates=data_feat_candidates,
        pipeline_config=config.pipeline_config
        )

    # Loading the persisted model
    trained_model_file = f"{config.app_config.pipeline_save_file}.pkl"
    pipeline = load_pipeline(file_name=trained_model_file)

    # Predicting the labels
    y_pred = pipeline.predict(X)
    _logger.warning(
        f"Making predictions with model persisted as '{trained_model_file}'. "
        f"Prediction: {y_pred[0]}"
    )

    result = {
        'prediction': f"{y_pred[0]}",
        'model': 'svc_model'
        }

    return result
