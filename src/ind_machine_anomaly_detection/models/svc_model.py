# -*- encoding: utf-8 -*-
from sklearn.svm import SVC

def svc_model(pipeline_config):
    return SVC(C=pipeline_config.C, kernel=pipeline_config.kernel)
