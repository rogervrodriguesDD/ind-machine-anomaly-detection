# -*- coding: utf-8 -*-
from ind_machine_anomaly_detection.config.core import PROJECT_ROOT
from ind_machine_anomaly_detection.data.data_management import (
    remove_old_pipelines,
)
from ind_machine_anomaly_detection.data.data_management import check_for_trained_model  # noqa : E501
from ind_machine_anomaly_detection.pipelines import train_model, predict


def test_train_model():

    # Deleting persisted models
    remove_old_pipelines(files_to_keep=[])

    # Training the model (a new persisted model file is created)
    train_model()

    assert check_for_trained_model()


def test_predict_file():

    filepath = PROJECT_ROOT.joinpath("data/raw/4.csv").resolve()

    with open(filepath, "r", encoding="utf-8") as f:
        result = predict(f)

    assert result
