import datetime as dt
import logging
import os
from typing import Callable, Union

import joblib
import pandas as pd
from sklearn.ensemble import VotingRegressor
# now you can import normally from sklearn.impute
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.python.training.metric_generator import MetricGenerator
from src.python.training.training_evaluator import TrainingEvaluator
from src.python.training.training_preparation import TrainingPreparation
from src.python.training.training_utils import TrainingUtils
from src.python.training.training_visualizer import TrainingVisualizer


class TrainingWorkflow:
    def __init__(
        self,
        config: dict,
        data: pd.DataFrame,
        logger: logging.Logger,
        save_files_for_tests: bool,
    ) -> None:
        self._config = config
        self._data = data
        self._logger = logger
        self._train_preparation = TrainingPreparation(config=self._config, logger=self._logger)
        self._save_files_for_tests = save_files_for_tests
        self._TEST_PATH = f"src/python/test/unit/test_training/test_files/"
        self._evaluate_metric = self._config["training"]["evaluate_metric"]
        self.model_tree = {}
        self._training_results_dict = {}
        self._metric_generator = MetricGenerator(y_true=None, y_pred=None)

    def train(
        self,
        algorithms: Union[str, list] = ["XGBRegressor"],
        ensemble_weights: list = None,
        plot_results: bool = True,
        save_trees: bool = False,
    ) -> tuple[dict, dict, dict]:
        """
        :param algorithms: list of strings to train different algorithms or a string to train one algorithm
        :param ensemble_weights: weights for the ensemble-estimator
        :param plot_results: plot the training results
        :param save_trees: save training results
        :return: model_tree

        - trains different algorithms
        - trains an ensemble model
        - saves results to pkl and json files
        """

        # setup parameters:
        algorithms = [algorithms] if isinstance(algorithms, str) else algorithms

        training_start = dt.datetime.now()

        # logging:
        self._logger.info("----------------------------------------------------------")
        self._logger.info(f"Training Start: {training_start}")
        self._logger.info(f"Training Start Timestamp: {self._logger.timestamp}")

        # training of different models:

        # dataset preparation:
        (
            x_train,
            y_train,
            x_test,
            y_test,
            _pre_selected_columns,
        ) = self._train_preparation.construct_training_data(self._data, [])

        self._logger.info("----------------------------------------------------------")
        self._logger.info(f"training:")

        # train for different algorithms
        self.model_tree = self._train_models(
            algorithms,
            x_train,
            y_train,
            x_test,
            y_test,
            ensemble_weights,
            used_features=_pre_selected_columns,
        )

        # calculate training results
        training_evaluator = TrainingEvaluator(self._config, self.model_tree)
        self._training_results_dict = training_evaluator.calculate_training_results(x_train, y_train, x_test, y_test)

        # plot training results:
        if plot_results:
            training_visualizer = TrainingVisualizer(self._config, self._training_results_dict)
            training_visualizer.plot_training_results(self._logger.timestamp, True)

        # save training results for testing of the TrainingVisualizer:
        if self._save_files_for_tests:
            with open(f"{self._TEST_PATH}/test_training_results_dict.pkl", "wb") as f:
                joblib.dump(self._training_results_dict, f)

        # save complete model files:
        if save_trees:
            os.makedirs("./artifacts/models/", exist_ok=True)
            with open("./artifacts/models/housing_model_tree.pkl", "wb") as f:
                joblib.dump(self.model_tree, f)

        training_end = dt.datetime.now()
        self._logger.info("----------------------------------------------------------")
        self._logger.info(f"Training End: {training_end}")
        self._logger.info(f"Training Duration: {training_end - training_start}")

        return self.model_tree

    def _train_models(
        self,
        algorithms: Union[str, list],
        x_train: pd.DataFrame,
        y_train: list,
        x_test: pd.DataFrame,
        y_test: list,
        ensemble_weights: list,
        used_features: list,
    ) -> tuple[dict, dict]:
        """
        :param algorithms: list of strings to train different algorithms or a string to train one algorithm
        :param x_train: features of the training-set
        :param y_train: target of the training-set
        :param x_test: features of the test-set
        :param y_test: target of the test-set
        :param ensemble_weights: weights for the ensemble-estimator
        :return: bool whether training was successfull. Model-Results get saved to self.model_tree

        - training of in config specified models
        - training of an ensemble model
        - saving results to files
        """

        # list to be filled with ensemble estimators:
        ensemble_estimators = []
        multi_metric = {
            "nrmse": "neg_root_mean_squared_error",
            "mean_absolute_percentage_error": make_scorer(
                self._metric_generator.calculate_mean_absolute_percentage_error,
                greater_is_better=False,
            ),
            "r2": make_scorer(r2_score),
            "explained_variance": make_scorer(explained_variance_score),
        }
        cv_train = KFold(n_splits=self._config["training"]["cv_folds"], shuffle=True, random_state=42)
        self._logger.info("train algorithms:")
        for algorithm in algorithms:
            # train model:
            _estimator, _param_grid = self._get_estimator_and_parameters(algorithm)

            pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", _estimator),
                ]
            )

            grid = GridSearchCV(
                pipeline,
                refit=self._evaluate_metric,
                verbose=0,
                cv=cv_train,
                scoring=multi_metric,
                n_jobs=-1,
                param_grid=_param_grid,
            )

            grid = grid.fit(x_train, y_train)

            y_pred = grid.predict(x_test)
            metric_generator = MetricGenerator(y_true=y_test, y_pred=y_pred)
            model_metrics = metric_generator.calculate_all_metrics()
            grid.best_test_score_ = -1 * model_metrics[self._evaluate_metric]
            grid.scores_ = model_metrics
            grid.used_features = used_features

            # print and log training results:
            self._logger.info(f"\tAlgorithm: {algorithm}")
            self._logger.info(f"\t\tBest parameters: {grid.best_params_}")
            self._logger.info(f"\t\tBest performance on cv-testset: {grid.best_score_}")
            self._logger.info(f"\t\tBest performance on testset: {grid.best_test_score_}")

            self._save_model_test_file(grid)

            self.model_tree[algorithm] = grid
            # add best estimator to ensemble model:
            ensemble_estimators.append((algorithm, self.model_tree[algorithm].best_estimator_))

        # train Ensemble-Algorithm
        algorithm = "Ensemble"
        model = VotingRegressor(ensemble_estimators, weights=ensemble_weights)
        model.fit(x_train, y_train)
        model.cv_results_ = cross_validate(model, x_train, y_train, cv=cv_train, scoring=multi_metric, n_jobs=-1)
        model.best_score_ = model.cv_results_[f"test_{self._evaluate_metric}"].mean()
        model.used_features = used_features

        y_pred = model.predict(x_test)
        metric_generator = MetricGenerator(y_true=y_test, y_pred=y_pred)
        model_metrics = metric_generator.calculate_all_metrics()
        model.best_test_score_ = -1 * model_metrics[self._evaluate_metric]
        model.scores_ = model_metrics

        # print and log training results:
        self._logger.info(f"\tAlgorithm: {algorithm}")
        self._logger.info(f"\t\tBest performance on cv-testset: {model.best_score_}")
        self._logger.info(f"\t\tBest performance on testset: {model.best_test_score_}")

        self._save_model_test_file(model, is_ensemble=True)

        self.model_tree[algorithm] = model

        return self.model_tree

    def _save_model_test_file(self, grid: object, is_ensemble: bool = False) -> None:
        """
        :param grid: model object
        :param is_ensemble: if the model is an ensemble model this parameter should be set to True for correct filename def
        :return:
        """
        if self._save_files_for_tests:
            filename = f'test_grid_{"ens_" if is_ensemble else ""}.pkl'

            with open(f"{self._TEST_PATH}/{filename}", "wb") as f:
                joblib.dump(grid, f)
            self._logger.info(f"saving model test files for {filename}")

    def _get_estimator_and_parameters(self, algorithm: str, prefix: str = "model__") -> tuple[Callable, dict]:
        """
        :param algorithm: algorithm to use
        :param prefix: prefix for the model name
        :return: the algorithm-function and the parameter-grid

        - get the function for training and it's parameter grids
        - throw exception if the model is not found
        """
        if algorithm in self._config["training"]["model_parameters"].keys():
            params = {f"{prefix}{k}": v for k, v in self._config["training"]["model_parameters"][algorithm].items()}
        else:
            # uses standarparams for training:
            params = {}

        estimator = TrainingUtils.get_estimator(algorithm)

        return estimator(), params
