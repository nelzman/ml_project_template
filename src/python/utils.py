import datetime as dt
import json
import logging
import os
import time

import joblib
import yaml


def load_yaml_config(config_file: str) -> dict:
    """
    :param config_file: path to the yaml file
    :return: dictionary of configuration

    - load configuration from yaml file
    """
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_pickle(
    full_path: str = None,
    filename: str = None,
) -> object:
    """
    :param full_path: full path to save data to
    :param filename: name of the file to load data from
    :return: object loaded from pickle file

    - load data from pickle file
    """
    data = joblib.load(f"{full_path}/{filename}")
    return data


def setup_logging(filename: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    :param filename: file name where log file is saved
    :param log_level: log level to be used by the logger [logging.DEBUG, .INFO, .WARN, .ERROR or .CRITICAL]
    :return: logging.Logger

    - set up for the logger of data ingressor, preprocessor, training pipeline and scoring
    """
    timestamp = int(dt.datetime.now().timestamp())
    logs_path_all = "artifacts/logs"
    os.makedirs(logs_path_all, exist_ok=True)
    logging.basicConfig(
        filename=f"{logs_path_all}/{filename}_{timestamp}.txt",
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger(f"{__name__}_{os.getpid()}")

    #  adjust the color of log to info in grey, warning in yellow, error in red
    class ColorHandler(logging.StreamHandler):
        GRAY8 = "38;5;8"
        GRAY7 = "38;5;7"
        YELLOW = "33"
        RED = "31"
        WHITE = "0"

        def emit(self, record: logging) -> None:
            """
            :param: specified logging record

            - log the specified logging record
            """
            # Don't use white for any logging, to help distinguish from user print statements
            level_color_map = {
                logging.DEBUG: self.GRAY8,
                logging.INFO: self.GRAY7,
                logging.WARNING: self.YELLOW,
                logging.ERROR: self.RED,
            }

            csi = f"{chr(27)}["  # control sequence introducer
            color = level_color_map.get(record.levelno, self.WHITE)

            print(f"{csi}{color}m{record.msg}{csi}m")

    if not logger.handlers:
        logger.addHandler(ColorHandler())
    logger.setLevel(log_level)
    logger.timestamp = timestamp
    # Delete log files created 7 days ago automatically
    now = time.time()
    time_calculated_to_del_log = now - 7 * 86400
    for file in os.listdir(logs_path_all):
        if os.path.getmtime(os.path.join(logs_path_all, file)) < time_calculated_to_del_log:
            os.remove(os.path.join(logs_path_all, file))
    return logger


def load_json(full_path: str, filename: str, mode: str = "r", file_encoding: str = "utf-8") -> dict:
    """
    :param full_path: The full path to the json file to load
    :param filename: name of the file to load data from
    :param mode: The mode in which the file is to be opened (r, w, a...)
    :param file_encoding: The encoding used when the file will be opened
    :return: A python dictionary of data representing the file`s contents
    :rtype: dict

    - loads a json file from a given specified path.
    """
    with open(f"{full_path}/{filename}", mode, encoding=file_encoding) as json_file:
        return json.load(json_file)
