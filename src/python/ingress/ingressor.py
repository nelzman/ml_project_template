import logging

import pandas as pd


class DataIngressor:
    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def ingress(self, location: str = "local"):

        if location == "local":
            data = pd.read_csv("artifacts/data/Housing.csv")
        else:
            raise NotImplementedError("Only location == 'local' is supported until now!")
        return data
