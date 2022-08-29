import logging

from ind_machine_anomaly_detection.config.core import config, PACKAGE_ROOT

logging.getLogger(config.app_config.package_name).addHandler(logging.StreamHandler())  # noqa


with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
