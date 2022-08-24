# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline

from ind_machine_anomaly_detection.config.core import config
from ind_machine_anomaly_detection.data.data_management import save_pipeline
from ind_machine_anomaly_detection.data.machine_data import MachineSensorDataCSV
from ind_machine_anomaly_detection.models import svc_model
from ind_machine_anomaly_detection.pipelines.data_ingestion import clean_data
from ind_machine_anomaly_detection.pipelines.features import build_features_candidates, select_features
from ind_machine_anomaly_detection.pipelines.train_test_split import split_data
from ind_machine_anomaly_detection.pipelines.scalers import get_pipeline_scaling

def train_model():
    """Train the model"""

    # Loading the data
    data = MachineSensorDataCSV(config.machine_data_catalog,
                                config.labels_data_catalog,
                                load_instances=True,
                                load_labels=True)

    # Cleaning and spliting the data
    data_cleaned = clean_data(data)
    data_train, _ = split_data(data_cleaned)

    # Building the features and selected only the necessary ones
    data_feat_candidates = build_features_candidates(data_train, config.pipeline_config)
    X, y = select_features(data_feat_candidates, config.pipeline_config)

    # Getting the scaling steps for the complete pipeline
    pipeline_scalers = get_pipeline_scaling(config.pipeline_config)

    # Setting the complete pipeline
    pipeline_steps = pipeline_scalers.steps
    model = svc_model(config.pipeline_config)
    pipeline_steps.append(("svc_model", model))
    pipeline = Pipeline(pipeline_steps)

    # Training the model
    pipeline.fit(X, y)

    # Persisting trained model
    save_pipeline(pipeline_to_persist=pipeline)
