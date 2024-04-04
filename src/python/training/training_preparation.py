import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TrainingPreparation:
    def __init__(self, config: dict, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def construct_training_data(self, data: pd.DataFrame, columns: list = []) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
        """
        :param datasets: datasets to construct train-test data
        :param segment: segment to get training and testing data
        :return: a tuple with (training_x, training_y, test_x, test_y, preselected columns, cross-validation-settings)

        - split the data into training and test-data
        - get columns to use for training
        """
        x_train, y_train, x_test, y_test = self._split_train_test_data(data)
        x_train, x_test, pre_selected_columns = self._get_pre_selected_columns(x_train, x_test, columns)
        x_train, y_train, x_test, y_test = self._drop_columns_where_all_nan(x_train, y_train, x_test, y_test)
        return x_train, y_train, x_test, y_test, pre_selected_columns

    def _split_train_test_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param data: dataset to construct train-test data
        :return: training and testing dataframe

        - construct training and testing data with the following config-behaviour for the parameters train and test:
            - both empty: train-test-split on all data that was preprocessed
            - train given, test not given: filter preprocessed data for the train-campaigns and make a train-test-split
            - train not given, test given: train on all data and evaluate on test-campaigns
            - both given: filter preprocessed data for train-campaigns and train on that,
              then evaluate on data filtered for test-campaign
            - when test is given and attribute the "test_on_last_heat_only: true" on config,
                we test on the last heat only! UsEcAsE: Ideal for accuracy check on real data only. RH's only have a Post Mortem
        - then split features and target from training and testing data
        """

        training, testing = train_test_split(data, test_size=self._config["training"]["test_size"], shuffle=True)

        y_train = training[self._config["training"]["target"]]
        x_train = training.drop(labels=self._config["training"]["target"], axis=1)
        y_test = testing[self._config["training"]["target"]]
        x_test = testing.drop(labels=self._config["training"]["target"], axis=1)

        return x_train, y_train, x_test, y_test

    def _get_pre_selected_columns(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, columns: list = []
    ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        :param x_train: training-dataframe with features
        :param x_test: test-dataframe with features
        :param segment: segment for the pre_selected_columns
        :return: training and testing dataframe and the preselected columns

        - get dataframes with preselected columns
        """
        if not columns:
            columns = x_train.columns
        x_train = x_train[columns]
        x_test = x_test[columns]
        return x_train, x_test, columns

    def _drop_columns_where_all_nan(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        :param x_train: training-dataframe with features
        :param y_train: target of the train-set
        :param x_test: test-dataframe with features
        :param y_test: target of the test-set
        :return: training and testing dataframe with features

        - deletes columns that are all nan
        """
        train_nan_cols = x_train.columns[x_train.isna().all()]
        if train_nan_cols.to_list():
            test_index = x_test[~x_test[train_nan_cols].isna().all(axis=1)].index
            test_to_train_index = test_index[: int(np.ceil(len(test_index) / 2))]
            x_train = pd.concat([x_train, x_test.loc[test_to_train_index]])
            y_train = pd.concat([y_train, y_test[test_to_train_index]])
            x_test = x_test.drop(test_to_train_index)
            y_test = y_test.drop(test_to_train_index)
        initial_col_number = x_train.shape[1]
        x_train.dropna(axis=1, how="all", inplace=True)
        x_test = x_test[x_train.columns]
        if initial_col_number > len(x_train.columns):
            self._logger.warning("Columns were dropped before Training due to being full of nans")
        return x_train, y_train, x_test, y_test

