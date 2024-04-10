from scipy import stats
import logging
import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(
        self, config: dict, logger: logging.Logger, save_data_for_tests: bool, use_azure_ml: bool, workspace: object
    ) -> None:
        self._config = config
        self._logger = logger
        self._save_data_for_tests = save_data_for_tests
        self._use_azure_ml = use_azure_ml
        self._workspace = workspace

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: dataset to preprocess
        :return: preprocessed dataset

        - preprocess the data
        """
        data = self._fill_nan_values(data)
        data = self._remove_outliers(data)
        data = self._encode_categorical_features(data)
        return data

    def _fill_nan_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: dataset to preprocess
        :return: preprocessed dataset

        - fill nan values with the mean of the numeric columns
        """
        numeric_columns = data.select_dtypes(include=np.number).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: dataset to preprocess
        :return: preprocessed dataset

        - remove outliers from the dataset
        """
        numeric_columns = data.select_dtypes(include=np.number).columns
        data_numeric = data[numeric_columns]
        z_scores = stats.zscore(data_numeric)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        return data[filtered_entries]

    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: dataset to preprocess
        :return: preprocessed dataset

        - encode categorical features
        """
        data = pd.get_dummies(data)
        # data2 = data.copy()
        # for col in data.columns:
        #    if col in data.select_dtypes(include=object).columns:
        #        data2[col] = [True if value == "yes" else False for value in data[col]]
        return data
