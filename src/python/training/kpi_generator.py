from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, mean_squared_error, r2_score


class KPIGenerator:
    """
    :param y_true: the ground-truth to compare the prediction to
    :param y_pred: prediction to calculate the KPIs for
    :return: none

    - class that generates different KPIs for model evaluation
    """

    def __init__(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        count_param_coeffs: int = None,
        train_obs: int = None,
    ) -> None:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param count_param_coeffs: parameters coefficients used for scoring
        :param train_obs: observations used for training
        :return: none

        - initializes all the necessary attributes for the KPIGenerator object.
        """

        if y_true is not None and y_pred is not None and len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred have different lengths: {str(len(y_true))} vs. {str(len(y_pred))}")
        self.y_true = y_true.tolist() if isinstance(y_true, pd.Series) else y_true
        self.y_pred = y_pred.tolist() if isinstance(y_pred, pd.Series) else y_pred
        self.count_param_coeffs = count_param_coeffs
        self.train_obs = train_obs

        self.mean_absolute_percentage_error = None
        self.delta_standard_deviation = None
        self.delta_relative_standard_deviation = None
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.norm_root_mean_squared_error = None
        self.explained_variance_score = None
        self.regional_accuracy = None
        self.r2_score = None
        self.adjusted_r2_score = None

        self.kpi_dict = {}
        self.delta_kpi_dict = {}

    def _get_start_time_point(self, time_horizon: int = None) -> int:
        """
        :param time_horizon: time horizon for the score in days
        :return: time point to start the time-horizon

        - get the np.arrays starting point for the time-horizon until the last value
        """

        if time_horizon is None:
            return 0
        elif time_horizon < 0:
            raise ValueError("time_horizon can not be negative")
        elif time_horizon > len(self.y_pred):
            raise ValueError("time_horizon can not be larger than number of predictions")
        else:
            return len(self.y_true) - time_horizon

    def calculate_r2_score(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: r2-score for the time-horizon

        - calculates the r2-score for a time-horizo
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        self.r2_score = r2_score(y_true=y_true_th, y_pred=y_pred_th)
        return self.r2_score

    def calculate_adjusted_r2_score(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: adjusted r2-score for the time-horizon

        - calculates the adjusted r2-score for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        self.adjusted_r2_score = 1 - (1 - r2_score(y_true=y_true_th, y_pred=y_pred_th)) * (self.train_obs - 1) / (
            self.train_obs - (self.count_param_coeffs + 1)
        )
        return self.adjusted_r2_score

    def calculate_mean_squared_error(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: mean_squared_error for the time-horizon

        - calculates the mean_squared_error for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        self.mean_squared_error = mean_squared_error(y_true=y_true_th, y_pred=y_pred_th)

        return self.mean_squared_error

    def calculate_root_mean_squared_error(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: root_mean_squared_error for the time-horizon

        - calculates the root_mean_squared_error for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        self.root_mean_squared_error = np.sqrt(mean_squared_error(y_true=y_true_th, y_pred=y_pred_th))
        return self.root_mean_squared_error

    def calculate_norm_root_mean_squared_error(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
        normalize_method: str = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :param  normalize_method: normalization method. Can be one of ['mean', 'range', 'std', 'iqr'].
                None refers to root_mean_squared_error
        :return: normalized root_mean_squared_error for the time-horizon

        - calculates the normalized root_mean_squared_error for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )

        if normalize_method is None:
            norm_factor = 1
        elif normalize_method == "mean":
            norm_factor = np.mean(y_true_th)
        elif normalize_method == "range":
            norm_factor = np.max(y_true_th) - np.min(y_true_th)
        elif normalize_method == "std":
            norm_factor = np.std(y_true_th)
        elif normalize_method == "iqr":
            norm_factor = np.quantile(y_true_th, 0.75) - np.quantile(y_true_th, 0.25)
        else:
            raise ValueError(
                'normalize_method set to an unknown method. Please choose a value from "mean", "range", "sd", "iqr" and None'
            )

        self.norm_root_mean_squared_error = np.sqrt(mean_squared_error(y_true=y_true_th, y_pred=y_pred_th)) / norm_factor
        return self.norm_root_mean_squared_error

    def calculate_explained_variance_score(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: explained_variance_score for the time-horizon

        - calculates the explained_variance_score for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        self.explained_variance_score = explained_variance_score(y_true=y_true_th, y_pred=y_pred_th)
        return self.explained_variance_score

    def calculate_delta_standard_deviation(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: the standard deviation of the deltas for a time-horizon

        - calculates the standard deviation of the deltas for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        deltas = y_pred_th - y_true_th
        self.delta_standard_deviation = np.std(deltas).astype(np.float64)
        return self.delta_standard_deviation

    def calculate_delta_relative_standard_deviation(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: relative standard deviation of the deltas for the time-horizon

        - calculates the relative standard deviation of the deltas for the time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        deltas = y_pred_th - y_true_th
        self.delta_relative_standard_deviation = np.std(deltas) / np.mean(y_pred_th)
        self.delta_relative_standard_deviation = self.delta_relative_standard_deviation.astype(np.float64)
        return self.delta_relative_standard_deviation

    def calculate_mean_absolute_percentage_error(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: mean_absolute_percentage_error for the time-horizon

        - calculates the mean_absolute_percentage_error for the time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )
        self.mean_absolute_percentage_error = mean_absolute_percentage_error(y_true=y_true_th, y_pred=y_pred_th)
        return self.mean_absolute_percentage_error

    def calculate_regional_accuracy(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
        region: float = 0.2,
        calc_type: str = "start",
    ) -> float:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :param  region: percentage to under/overpredict
        :param  calc_type: type of region-definition. 'start' draws the region based on the starting brick length.
                'truth' calculates the region based on every y_true value.
        :return: region-KPI for the time-horizon

        - calculates the percentage of predictions within the range of their true- or start-value.
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        start_time_point = self._get_start_time_point(time_horizon)
        y_true_th, y_pred_th = (
            self.y_true[start_time_point:],
            self.y_pred[start_time_point:],
        )

        if calc_type == "start":
            region_width = y_true_th[0] * region
            kpi = [
                1 if (y_pred_th[i] >= y_true_th[i] - region_width) and (y_pred_th[i] <= y_true_th[i] + region_width) else 0
                for i in range(len(y_true_th))
            ]
        elif calc_type == "truth":
            kpi = [
                y_pred_th[i] / y_true_th[i] - 1 if y_true_th[i] != 0 else y_pred_th[i] / (y_true_th[i] + 0.01) - 1
                for i in range(len(y_true_th))
            ]
            kpi = [1 if (value >= -region) and (value <= region) else 0 for value in kpi]
        self.regional_accuracy = np.sum(kpi) / len(self.y_true)

        return self.regional_accuracy

    def calculate_all_kpis(
        self,
        y_true: Union[np.array, list] = None,
        y_pred: Union[np.array, list] = None,
        time_horizon: int = None,
    ) -> dict:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :param time_horizon: time horizon for the score in days
        :return: dict with KPIs

        - calculates all KPIs for a time-horizon
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        self.kpi_dict["mean_absolute_percentage_error"] = self.calculate_mean_absolute_percentage_error(time_horizon=time_horizon)
        self.kpi_dict["delta_standard_deviation"] = self.calculate_delta_standard_deviation(time_horizon=time_horizon)
        self.kpi_dict["delta_relative_standard_deviation"] = self.calculate_delta_relative_standard_deviation(
            time_horizon=time_horizon
        )
        self.kpi_dict["explained_variance_score"] = self.calculate_explained_variance_score(time_horizon=time_horizon)
        self.kpi_dict["mean_squared_error"] = self.calculate_mean_squared_error(time_horizon=time_horizon)
        self.kpi_dict["root_mean_squared_error"] = self.calculate_root_mean_squared_error(time_horizon=time_horizon)
        self.kpi_dict["regional_accuracy_start"] = self.calculate_regional_accuracy(time_horizon=time_horizon, calc_type="start")
        self.kpi_dict["regional_accuracy_truth"] = self.calculate_regional_accuracy(time_horizon=time_horizon, calc_type="truth")
        self.kpi_dict["r2_score"] = self.calculate_r2_score(time_horizon=time_horizon)
        if self.count_param_coeffs is not None:
            self.kpi_dict["adjusted_r2_score"] = self.calculate_adjusted_r2_score(time_horizon=time_horizon)

        return self.kpi_dict

    def calculate_delta_kpis(self, y_true: Union[np.array, list] = None, y_pred: Union[np.array, list] = None) -> dict:
        """
        :param y_true: the ground-truth to compare the prediction to
        :param y_pred: prediction to calculate the KPIs for
        :return: dict with delta kpis

        - calculate kpis for the deltas
        """

        self.y_true = y_true if y_true is not None else self.y_true
        self.y_pred = y_pred if y_pred is not None else self.y_pred

        pos_neg_deltas = self.y_pred - self.y_true
        delta_rbls = abs(pos_neg_deltas)

        over_under_prediction = [1 if i > 0 else 0 if i == 0 else -1 for i in pos_neg_deltas]

        if len(delta_rbls) > 0:
            quantiles = np.quantile(delta_rbls, q=[0.1, 0.25, 0.5, 0.75, 0.9])
            self.delta_kpi_dict = {
                "number_of_measurements": len(delta_rbls),
                "mean": np.mean(delta_rbls),
                "min": np.min(delta_rbls),
                "max": np.max(delta_rbls),
                "p_10": quantiles[0],
                "p_25": quantiles[1],
                "median": quantiles[2],
                "p_75": quantiles[3],
                "p_90": quantiles[4],
                "last_heat_delta": delta_rbls[-1],
                "over_under_prediction_kpi": np.mean(over_under_prediction),
                "std": np.std(delta_rbls),
                "var": np.var(delta_rbls),
            }

            return self.delta_kpi_dict
        else:
            return {}
