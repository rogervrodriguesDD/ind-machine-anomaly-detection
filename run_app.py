# -*- coding: utf-8 -*-
import argparse
import pathlib
import sys

from ind_machine_anomaly_detection.config.core import config, TRAINED_MODEL_DIR
from ind_machine_anomaly_detection.data.data_management import check_for_trained_model
from ind_machine_anomaly_detection.pipelines import train_model, predict


def _setup_parser_run_app_config():
    """
    Setup Python's ArgumentParser with Application-level configuration.
    The default values of the arguments are given by the 'predict' for
    '--command' and 'None' for '--file'
    """

    parser = argparse.ArgumentParser(add_help=False)

    def allowed_commands(input_c):
        if input_c.lower() not in ['predict', 'train']:
            raise ValueError
        return input_c.lower()

    parser.add_argument('--help', '-h', action='help')

    parser.add_argument('--command', '-c', type=allowed_commands,
                       default='predict', help='Options: train, predict')

    parser.add_argument('--file', '-f', nargs='?',
                       type=argparse.FileType('r', encoding='UTF-8'),
                       default=None, required=False)

    return parser


def main():

    parser = _setup_parser_run_app_config()
    run_app_config = parser.parse_args()

    trained_model_exist = check_for_trained_model()

    if run_app_config.command == 'train':
        train_model()

    if run_app_config.command == 'predict':

        if run_app_config.file is not None:
            file_or_filename = run_app_config.file
        else:
            raise OSError(
                "For prediction, specify the file using the -f argument"
                )

        if not trained_model_exist:
            train_model()


        result = predict(file_or_filename)
        print('Result: ', result)



if __name__ == '__main__':
    """
    Run the model.

    Example of command:
    ```
    python run_app.py --command train
    ```

    or,
    ```
    python run_app.py --command predict --file data/raw/4.csv
    ```
    """
    main()
