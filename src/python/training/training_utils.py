import importlib
from typing import Callable

import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.interpolate import splev, splrep


class TrainingUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def z_score_filter_on_target(train_x: pd.DataFrame, train_y: pd.DataFrame, value: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param train_x: features to train on
        :param train_y: target to train on
        :param value: threshold for deleting data
        :return: Dataframes for features and target

        - apply z-score on data
        - filter data that is too far away from the center of the z-score
        """
        z_scores = stats.zscore(train_y)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = abs_z_scores < value
        train_y = train_y[filtered_entries]
        train_x = train_x[filtered_entries]
        return train_x, train_y

    @staticmethod
    def z_score_filter_on_wear_rate(dataset: pd.DataFrame, value: int = 2, target: str = "WEAR_RATE") -> pd.DataFrame:
        """
        :param dataset: dataset with WEAR_RATE
        :param value: threshold for deleting data
        :param target: target variable
        :return: Dataframe

        - apply z-score on data
        - filter data that is too far away from the center of the z-score
        """
        z_scores = stats.zscore(dataset[target])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = abs_z_scores < value
        dataset = dataset[filtered_entries]
        return dataset

    @staticmethod
    def smooth_target_values(measurement_data: pd.DataFrame) -> pd.DataFrame:
        """
        :param measurement_data: dataset with measurements
        :return: Dataframe

        - smoothen out the target values
        """
        measurement_data = measurement_data.sort_values(by=["HEAT"])
        # cubic spline interpolation requires minimum 3 measurements
        if len(measurement_data["HEAT"]) >= 3:
            b_spline_1d_representation = splrep(
                measurement_data["HEAT"],
                measurement_data["RBL"],
                s=int(measurement_data["HEAT"].iloc[-1]) * 42,
            )
            evaluated_b_spline_values = splev(measurement_data["HEAT"], b_spline_1d_representation)
            measurement_data["SMOOTH_RBL"] = evaluated_b_spline_values
            measurement_data["SMOOTH_DELTA_RBL"] = measurement_data["SMOOTH_RBL"].diff()
        else:
            measurement_data["SMOOTH_RBL"] = measurement_data["RBL"]
            measurement_data["SMOOTH_DELTA_RBL"] = measurement_data["SMOOTH_RBL"].diff()
        return measurement_data

    @staticmethod
    def extrapolate_rbls(rbls: list) -> list:
        """
        :param rbls: list with raw brick lengths (rbls)
        :return: list

        - extrapolates rbls
        """
        if len(rbls) == 1:
            return rbls
        heats = np.arange(1, len(rbls) + 1)
        f = interpolate.interp1d(heats, rbls, kind="linear", fill_value="extrapolate")
        next_heat = heats[-1]
        while f(next_heat) >= 0 and next_heat >= 2 * heats[-1]:
            next_heat = next_heat + 1
            print(next_heat, f(next_heat))
            rbls.append(f(next_heat))
        return rbls

    @staticmethod
    def extract_numeric_data(raw_process_data: pd.DataFrame) -> pd.DataFrame:
        """
        :param raw_process_data: dataframe of the raw process data
        :return: dataframe with only numeric columns

        - returns a new dataframe with only the numeric columns of the input dataframe
        """
        numeric_columns = raw_process_data.select_dtypes(["int", "float"]).columns.tolist()
        numeric_data = raw_process_data[numeric_columns]

        # zero_columns = numeric_data.columns[(numeric_data == 0).all()].tolist()
        first_columns = numeric_data.iloc[:, :2].columns.tolist()
        # remove columns with all 0's as well as the first 4 columns in the raw data
        numeric_data = numeric_data.drop(first_columns, axis=1)

        return numeric_data

    @staticmethod
    def get_estimator(algorithm: str) -> Callable:
        """
        :param algorithm: algorithm to use
        :return: estimator object

        - load and return estimator constructor
        """
        if algorithm in ["LinearRegression", "LassoLars", "Ridge", "BayesianRidge"]:
            algorithm_module_str = "sklearn.linear_model"
        elif algorithm in ["LGBMRegressor"]:
            algorithm_module_str = "lightgbm"
        elif algorithm in ["XGBRegressor"]:
            algorithm_module_str = "xgboost"
        elif algorithm in ["MLPRegressor"]:
            algorithm_module_str = "sklearn.neural_network"
        elif algorithm in ["SVR"]:
            algorithm_module_str = "sklearn.svm"
        elif algorithm in ["ExplainableBoostingRegressor"]:
            algorithm_module_str = "interpret.glassbox"
        elif algorithm in ["RandomForestRegressor"]:
            algorithm_module_str = "sklearn.ensemble"
        else:
            raise ValueError(
                f"Algorithm {algorithm} not found in available algorithms. Choose another one or change the config/method."
            )

        module = importlib.import_module(algorithm_module_str)
        return getattr(module, algorithm)
