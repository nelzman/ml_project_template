import numpy as np
import pandas as pd
import statsmodels.api as sm


class TrainingEvaluator:
    def __init__(self, config: dict, model_tree: dict) -> None:
        self._config = config
        self.model_tree = model_tree
        self._training_results_dict = {}

    def calculate_training_results(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        :param x_train: training dataframe with feature-variables
        :param y_train: training list with the target-variable
        :param x_test: testing dataframe with feature-variables
        :param y_test: testing list with the target-variable
        :return: dictionary with training results

        - calculates results for the models
        """        

        # fill training_results:
        self._training_results_dict = {"data": {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}}
        self._training_results_dict["algorithm_names"] = list(self.model_tree.keys())
        (
            self._training_results_dict["cooks_distance_data"],
            self._training_results_dict["cooks_distance"],
        ) = self._calculate_cooks_distance(x_train, y_train)

        # correlation:
        data = x_train.copy()
        data["target"] = y_train
        data_corr = data.corr()[:-1]
        data_corr.fillna(0, inplace=True)
        data_corr.index.set_names(["column_name"], inplace=True)
        data_corr.reset_index(inplace=True)
        self._training_results_dict["rbl_corr"] = data_corr

        self._training_results_dict["cv_results"] = []
        self._training_results_dict["test_results"] = []
        for model_name, model_attr in self.model_tree.items():
            self._training_results_dict[model_name] = {}

            # calculate predicted y:
            self._training_results_dict[model_name]["y_pred"] = model_attr.predict(x_train)
            self._training_results_dict[model_name]["y_test_pred"] = model_attr.predict(x_test)

            # calculate residuals:
            self._training_results_dict[model_name]["y_train_residuals"] = (
                self._training_results_dict["data"]["y_train"] - self._training_results_dict[model_name]["y_pred"]
            )
            self._training_results_dict[model_name]["y_test_residuals"] = (
                self._training_results_dict["data"]["y_test"] - self._training_results_dict[model_name]["y_test_pred"]
            )

            # calculate training results:
            if model_name == "Ensemble":
                model_cv_results = model_attr.cv_results_[f'test_{self._config["training"]["evaluate_metric"]}']
            else:
                model_cv_results = [
                    model_attr.cv_results_[f'split{i}_test_{self._config["training"]["evaluate_metric"]}'][model_attr.best_index_]
                    for i in range(self._config["training"]["cv_folds"])
                ]
            self._training_results_dict["test_results"].append([model_attr.best_test_score_])
            self._training_results_dict["cv_results"].append(model_cv_results)

        return self._training_results_dict

    def _calculate_cooks_distance(self, x: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        """
        :param x: dataframe with feature-variables
        :param y: list with the target-variable

        - calculates cooks-distance based on a ordinary least squares estimate
        """
        x = x.astype(dtype="float")
        cd_data = x.copy()
        cd_data["y"] = y
        cd_data = cd_data[cd_data["y"].notna()]
        cd_data = cd_data.dropna(axis=1)
        cd_model = sm.OLS(cd_data["y"], cd_data.drop(labels=["y"], axis=1)).fit()
        return cd_data, cd_model.get_influence().cooks_distance

    @staticmethod
    def get_best_model(config: dict, model_tree: dict) -> dict:
        """
        :param config: customer config
        :param model_tree: tree of trained models
        :return: dictionary of best model with best score

        - gets the best model
        """
        best_models = {}

        best_models = {}
        best_val = -np.inf
        for model_name, model_attr in model_tree.items():
            if config["scoring"]["use_algorithm"] == "":
                if model_attr.best_test_score_ > best_val:
                    best_val = model_attr.best_test_score_
                    best_models["best_model"] = model_name
                    best_models["best_score"] = model_attr.best_score_
                    best_models["best_test_score"] = model_attr.best_test_score_
            elif config["scoring"]["use_algorithm"] == model_name:
                best_models["best_model"] = model_name
                best_models["best_score"] = model_attr.best_score_
                best_models["best_test_score"] = model_attr.best_test_score_
            else:
                continue

        if not best_models:
            ValueError(f"Choosen scoring model with name {config['scoring']['use_algorithm']} not found!")

        return best_models

