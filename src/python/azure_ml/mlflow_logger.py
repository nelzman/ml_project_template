import datetime as dt
import logging
import os.path
import re
import shutil

import mlflow
import numpy as np
import pandas as pd
from azureml.core import Dataset, Model, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.run import Run
from azureml.data.datapath import DataPath
from azureml.exceptions import UserErrorException
from mlflow.entities import Experiment

from src.python.azure_ml.aml_utils import AMLUtils


class MLFlowLogger:
    """
    Class to log run-specific metrics, tags, outputs and artifacts with mlflow usable locally and in aml
    """

    def __init__(
        self, config: dict, logger: logging.Logger, experiment_name: str, use_azure_ml: bool
    ) -> None:
        self._config = config
        self._logger = logger
        self._experiment_name = experiment_name
        self._use_azure_ml = use_azure_ml
        self.workspace = self._setup_mlflow_logging()

    def log_training_runner(self, model_tree: dict, is_pipeline_run: bool = False, update_model_registry: bool = False) -> None:
        """
        :param model_tree: dictionary with the models
        :param is_pipeline_run: dictionary with the models
        :param update_model_registry: whether to update the model registry
        :return: None

        - logs tags, params, files, datasets with mlflow
        - uploads models to the registry
        """
        mlflow.set_tags(self._config["training"])
        mlflow.log_artifact(
            local_path=f"artifacts/logs/"
            f"training_{self._logger.timestamp}.txt",
            artifact_path="training_logs",
        )
        best_scores = []
        metrics_dict = {}

        mlflow.log_artifact(
            local_path=f"artifacts/plots/training/"
            f"training_plots_{self._logger.timestamp}.png",
            artifact_path="evaluation_plots",
        )
        acc_dict = {model: abs(model_value.best_score_) for model, model_value in model_tree.items()}
        best_algo = min(acc_dict, key=acc_dict.get)
        mlflow.set_tags({f"best_algorithm": best_algo})
        best_scores.append(acc_dict[best_algo])
        metrics_dict[f"best_algorithm_score"] = acc_dict[best_algo]
        for model, model_value in model_tree.items():
            metrics_dict[f"score_{model}"] = model_value.best_score_
            metrics_dict[f"mape_{model}"] = model_value.scores_["mean_absolute_percentage_error"]
            metrics_dict[f"accuracy_{model}"] = (
                1 - model_value.scores_["mean_absolute_percentage_error"]
            )
            metrics_dict[f"rmse_{model}"] = model_value.scores_["root_mean_squared_error"]
        mlflow.log_metrics(metrics_dict)

        overall_score = np.mean(best_scores)
        mlflow.log_metric("overall_score", overall_score)

        if self._use_azure_ml:
            run = mlflow.active_run()
            model_name = (
                f"housing_model_tree"
            )
            model_artifact_path = (
                f"./src/artifacts/models/"
                f"housing_model_tree.pkl"
            )
            artifact_path = f"models/{model_name}"
            model_properties = {"run_id": run.info.run_id, "experiment": self._experiment_name}
            try:
                experiment_url = str(Run.get(self.workspace, run_id=model_properties["run_id"]).get_portal_url())
            except Exception:
                experiment_url = ""
            model_tags = {
                "stage": "Development",
                "model_metrics": metrics_dict,
                "overall_score": overall_score,
                "experiment_url": experiment_url,
            }

            # Todo: add description to readme for stages and switch to stages with mflow 2.*
            mlflow.log_artifact(local_path=model_artifact_path, artifact_path=artifact_path)
            self._logger.info("Registering model with aml-sdk")
            model = Model.register(
                workspace=self.workspace,
                model_path=model_artifact_path,
                model_name=model_name,
                tags=model_tags,
                properties=model_properties,
            )

    def _setup_mlflow_logging(self) -> Workspace:
        """
        :return Workspace object or None

        - setup the logging with mlflow in azure-ml or locally
        """
        if self._use_azure_ml:
            try:  # works when running over the aml runner
                run = Run.get_context()
                workspace = run.experiment.workspace
            except AttributeError:  # works for normal data-ingress-runner when download via azureml datastores
                try:
                    credentials = InteractiveLoginAuthentication(tenant_id=self._config["azure_ml"]["tenant_id"])
                except Exception as ex:
                    self._logger.info(ex)
                    credentials = None
                workspace = Workspace.get(
                    name=self._config["azure_ml"]["workspace_name"],
                    subscription_id=self._config["azure_ml"]["subscription_id"],
                    resource_group=self._config["azure_ml"]["resource_group"],
                    auth=credentials,
                )
                self._set_local_experiment()
        else:
            # setup for mlflow local logging
            workspace = None
            self._set_local_experiment()

        return workspace

    def _set_local_experiment(self, days: int = 20) -> list:
        """
        :param days: number of days to keep local mlflow logs
        :return: list with deleted mlflow-runs

        - sets tracking uri for the experiment
        - sets the experiment-name
        - deletes local runs older than 20 days to save disk-space
        """
        tracking_uri = f"artifacts/.mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment(self._experiment_name)
        # delete runs that are older than 7 days
        return self._del_mlflow_runs(experiment=experiment, days=days)

    def _del_mlflow_runs(self, experiment: Experiment, days: int = 20) -> list:
        """
        :param experiment: aml-experiment-object
        :param days: number of days to keep local mlflow logs
        :return: list with deleted mlflow-runs

        - delete old mlflow-runs
        - return these deleted runs as a list
        """

        ts = dt.datetime.timestamp(dt.datetime.now() - dt.timedelta(days=days))
        df = mlflow.search_runs([experiment.experiment_id])
        runs_to_delete = list(df[df["start_time"].values.astype(np.int64) // 10**9 < round(ts)].run_id)
        for run_id in runs_to_delete:
            self._logger.info(f"deleted run from {experiment.name}: {run_id}")
            mlflow.delete_run(run_id)
            folder_to_delete = f"{experiment.artifact_location}/{run_id}"
            if os.path.exists(folder_to_delete):
                shutil.rmtree(folder_to_delete)
        return runs_to_delete

    def _save_dataset_for_step(self, step: str, tag_dict: dict) -> None:
        """
        :param step: step for dataset. one of [data_ingress, preprocess]
        :param tag_dict: dictionary of tags of the step
        :return: None, saves dataset in aml

        - saves datasets for the steps data_ingress and preprocess
        """
        datastore = self.workspace.get_default_datastore()
        self._logger.info("save dataset!")
        dataset_name = f"{step}_all"
        try:
            latest_dataset = Dataset.get_by_name(workspace=self.workspace, name=dataset_name, version="latest")
            latest_dataset_tags = latest_dataset.tags
        except UserErrorException:  # dataset is not found and needs to be constructed
            latest_dataset_tags = {}
        if latest_dataset_tags != tag_dict:
            dataset = Dataset.File.upload_directory(
                src_dir=f"artifacts/data/{step}",
                target=DataPath(datastore, f"{step}/{dataset_name}"),
                overwrite=True,
                show_progress=True,
            )
            dataset.register(workspace=self.workspace, name=dataset_name, tags=tag_dict, create_new_version=True)
