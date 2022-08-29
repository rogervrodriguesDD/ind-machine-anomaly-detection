# -*- coding: utf-8 -*-
from sklearn.svm import SVC

from ind_machine_anomaly_detection.config.core import PipelineConfig


def svc_model(pipeline_config: PipelineConfig) -> SVC:
    return SVC(C=pipeline_config.C, kernel=pipeline_config.kernel)
