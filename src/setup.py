# -*- coding: utf-8 -*-
from pathlib import Path
from setuptools import find_packages, setup

NAME = "ind_machine_anomaly_detection"
DESCRIPTION = 'Anomaly detection of an industrial machine based on sensors data.'
AUTHOR = 'Roger Rodrigues'

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
with open(PACKAGE_DIR / "VERSION") as f:
    __version__ = f.read().strip()

def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()

setup(
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    author=AUTHOR,
    packages=find_packages(exclude=("tests",)),
    package_data={NAME: ["VERSION"]},
    install_requires=list_reqs(),
    include_package_data=True,
    license='',
)
