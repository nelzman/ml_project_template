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

    def log_data_ingress_runner(self, dt_tree: tuple, save_dataset: bool = False) -> None:
        """
        :param dt_tree: tuple consiting of:
            raw_process_data_tree: process data tree
            raw_rbl_data_tree: rbl data tree
            raw_meta_data_tree: tree with meta-data
        :param save_dataset: string for the vessel_number
        :return: None

        - logs tags, params, files, datasets with mlflow
        """

        mlflow.log_artifact(
            local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/data_ingress/"
            f"data_ingress_logs/data_ingress_logs_{self._logger.timestamp}.txt",
            artifact_path="data_ingress_logs",
        )
        if self._vessel_type.upper() == "BOF":
            raw_process_data, raw_rbl_data, summary_data, raw_meta_data = dt_tree
        else:
            raw_process_data, raw_rbl_data, raw_meta_data = dt_tree

        tag_dict = self._get_data_tag_dict(
            step="data_ingress", process_data=raw_process_data, measurement_data=raw_rbl_data, meta_data=raw_meta_data
        )

        mlflow.set_tags(tag_dict)

        if self._use_azure_ml and save_dataset:
            self._save_dataset_for_step(step="data_ingress", tag_dict=tag_dict)

    def log_preprocess_runner(self, preprocess_data_tree: object, save_dataset: bool = False) -> None:
        """
        :param preprocess_data_tree: object with the following properties:
            process_data_tree: process data tree
            selected_parameters: list of selected parameters
            target_rbl_data_segment_tree: rbl data tree
        :param save_dataset: string for the vessel_number
        :return: None

        - logs tags, params, files, datasets with mlflow
        """

        mlflow.log_artifact(
            local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/preprocess/"
            f"preprocess_logs/preprocess_logs_{self._logger.timestamp}.txt",
            artifact_path="preprocess_logs",
        )
        if self._vessel_type.upper() == "BOF":
            process_data_tree, rbl_data_tree, summary_data_tree, meta_data_tree = preprocess_data_tree
        else:
            process_data_tree, rbl_data_tree, meta_data_tree = preprocess_data_tree

        tag_dict = self._get_data_tag_dict(
            step="preprocess", process_data=process_data_tree, measurement_data=rbl_data_tree, meta_data=meta_data_tree
        )

        mlflow.set_tags(tag_dict)
        mlflow.set_tags(self._config["preprocess"])

        if self._use_azure_ml and save_dataset:
            self._save_dataset_for_step(step="preprocess", tag_dict=tag_dict)

    def log_plotting_runner(self, rbl_tree: dict, segment_labels: list, returned_plots: list) -> None:
        """
        :param rbl_tree: nested dict with rbl for every vessel, campaign
        :param segment_labels: list of segment names
        :param returned_plots: list of  plotted plot names
        :return: None

        - logs tags, plots with mlflow
        """
        mlflow.set_tags({"plots_logged": returned_plots})
        func_name = self._config.get("laser_data_preparation", {}).get("aggregation_metric")
        for vessel_number, campaigns in rbl_tree.items():
            for campaign in campaigns:
                for segment in segment_labels:
                    mlflow.log_artifact(
                        local_path=f"artifacts/{self._customer}/visualization/rbl_target/{vessel_number}/{campaign}/"
                        f"{segment}/{func_name}.png",
                        artifact_path=f"plots_logged_{segment}_{campaign}_{vessel_number}",
                    )

    def log_training_runner(self, model_tree: dict, is_pipeline_run: bool = False, update_model_registry: bool = False) -> None:
        """
        :param model_tree: dictionary with the models
        :param is_pipeline_run: dictionary with the models
        :param update_model_registry: whether to update the model registry
        :return: None

        - logs tags, params, files, datasets with mlflow
        - uploads models to the registry
        """
        vessel_number = list(self._config["general"]["vesselsAndCampaigns"].keys())[0]
        mlflow.set_tags(self._config["training"])
        mlflow.log_param("customer", self._config["general"]["customer_name"].lower())
        mlflow.log_artifact(
            local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/models/training_logs/"
            f"training_logs_{self._logger.timestamp}.txt",
            artifact_path="training_logs",
        )
        best_scores = []
        metrics_dict = {}
        if self._vessel_type == "rh" and self._customer == "ternium" and is_pipeline_run:
            mlflow.log_artifacts(
                local_dir=f"{self._config['local_storage']['relative_path']}/{self._customer}/training/evaluation_logs",
                artifact_path="evaluation_logs",
            )
        for segment, segment_value in model_tree.items():
            if segment in ["feature_engineering", "training"]:
                continue
            if (
                self._vessel_type == "rh"
                and self._customer == "ternium"
                and is_pipeline_run
                and segment not in ["outlet_snorkel", "inlet_snorkel"]
            ):
                mlflow.log_artifact(
                    local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/training_visualization/"
                    f"evaluation_plots/evaluation_{segment}.png",
                    artifact_path="evaluation_plots",
                )
            mlflow.log_artifact(
                local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/training_visualization/"
                f"training_plots/{segment}/training_plots_{segment}_{self._logger.timestamp}.png",
                artifact_path=f"training_plot_{segment}",
            )
            acc_dict = {model: abs(model_value.best_score_) for model, model_value in segment_value.items()}
            best_algo = min(acc_dict, key=acc_dict.get)
            mlflow.set_tags({f"best_algorithm_{segment}": best_algo})
            # mlflow.log_metric(f"best_algorithm_score_{segment}", acc_dict[best_algo])
            best_scores.append(acc_dict[best_algo])
            metrics_dict[f"best_algorithm_score_{segment}"] = acc_dict[best_algo]
            for model, model_value in segment_value.items():
                metrics_dict[f"score_{segment}_{model}"] = model_tree[segment][model].best_score_
                metrics_dict[f"mape_{segment}_{model}"] = model_tree[segment][model].scores_["mean_absolute_percentage_error"]
                metrics_dict[f"accuracy_{segment}_{model}"] = (
                    1 - model_tree[segment][model].scores_["mean_absolute_percentage_error"]
                )
                metrics_dict[f"rmse_{segment}_{model}"] = model_tree[segment][model].scores_["root_mean_squared_error"]
            mlflow.log_metrics(metrics_dict)
        overall_score = np.mean(best_scores)
        mlflow.log_metric("overall_score", overall_score)

        if self._use_azure_ml and is_pipeline_run:
            run = mlflow.active_run()
            model_name = (
                f"dssteel_modeltree_{self._vessel_type}_{self._customer}_"
                f"{vessel_number}_{self._config['training']['model_version']}"
            )
            model_artifact_path = (
                f"./src/{self._vessel_type}/models/{self._customer}/"
                f"model_tree_{self._vessel_type.upper()}_"
                f"{vessel_number}_{self._config['training']['model_version']}.pkl"
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

            dataset_name = f"preprocess_{self._vessel_type}_{self._customer}_{vessel_number}_all"
            dataset = Dataset.get_by_name(workspace=self.workspace, name=dataset_name, version="latest")
            if update_model_registry:
                self._logger.info("Registering model with aml-sdk")
                # register repo-artifact models:
                self.update_model_in_registry(
                    model_path=model_artifact_path,
                    model_name=model_name,
                    datasets=[("preprocess", dataset)],
                    model_tags=model_tags,
                    model_properties=model_properties,
                )

    def update_model_in_registry(
        self, model_path: str, model_name: str, datasets: list, model_tags: dict, model_properties: dict
    ) -> None:
        """
        :param model_path: path of the model relative to the running code
        :param model_name: name to save the model in the registry
        :param datasets: list of tuples ("relation_to_model", dataset) to link to the model
        :param model_tags: tags to attach to the model
        :param model_properties: properties to attach to the model
        :return: None

        - uploads a model to the registry
        - sets stages of models to 'staging' if they are better than production model
        - if multiple prod-models are available, the best stays on 'prod', the others go to 'archived'
        - Allowed model stages: None, Staging, Production, Archived, Logged, Development, Production, Archived.
        """

        model = Model.register(
            workspace=self.workspace,
            model_path=model_path,
            model_name=model_name,
            datasets=datasets,
            tags=model_tags,
            properties=model_properties,
        )

        AMLUtils.update_model_stage(model, "Development")
        self._logger.info(f"Update stages for model: {model_name}")

        registry_prod_models = Model.list(self.workspace, name=model_name, tags=[["stage", "Production"]])
        if len(registry_prod_models) == 0:  # no prod model
            prod_model = None
        elif len(registry_prod_models) > 1:  # multiple prod models
            prod_model = AMLUtils.clean_prod_model_stages(registry_prod_models)
        else:  # exactly one prod model
            prod_model = registry_prod_models

        # compare current model to prod model
        if self._config["training"]["force_aml_model_update"]:
            AMLUtils.update_model_stage(model, "Staging")
        else:
            prod_model_score = prod_model.tags["overall_score"] if prod_model is not None else np.inf
            current_model_score = model.tags["overall_score"]
            if current_model_score <= prod_model_score:
                AMLUtils.update_model_stage(model, "Staging")

    def log_scoring_runner(self, predictions: dict) -> None:
        """
        :param predictions: predictions
        :return: None

        - logs tags, params, files, datasets with mlflow
        """
        mlflow.set_tags(self._config["scoring"])
        mlflow.log_param("customer", self._customer)
        mlflow.log_artifact(
            local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/predictions/"
            f"scoring_logs/scoring_logs_{self._logger.timestamp}.txt",
            artifact_path="scoring_logs",
        )
        for segment in predictions["reportData"]["hotSpots"]:
            mlflow.log_artifact(
                local_path=f"{self._config['local_storage']['relative_path']}/{self._customer}/scoring_visualization/kpi_plots/"
                f"rbl_deltas/{predictions['vessel']}/{predictions['campaignNr']}/"
                f"rbl_delta_{predictions['vessel']}_{predictions['campaignNr']}_{segment['name']}.png",
                artifact_path=f"scoring_plot_{segment['name']}",
            )
            mlflow.log_metric(f"remainingHeats_{segment['name']}", segment["remainingHeats"])
            mlflow.log_metric(f"predictedLifetime_{segment['name']}", segment["predictedLifetime"])
            if segment["kpis"]:
                mlflow.log_metric(
                    f"mean_absolute_percentage_error_{segment['name']}",
                    segment["kpis"]["mean_absolute_percentage_error"],
                )
                mlflow.log_metric(
                    f"root_mean_squared_error_{segment['name']}",
                    segment["kpis"]["root_mean_squared_error"],
                )

    def log_endreport_runner(self, endreport: str, folder: str, date: str, forecast_type: str) -> None:
        """
        :param endreport: how to construct the endreport
        :param folder: folder for saving the endreport
        :param date: date for the endreport
        :param forecast_type: type for forecasting
        :return: None

        - logs tags, params, files, datasets with mlflow
        - endreport gets generated as a pdf
        """
        if endreport == "yes":
            mlflow.log_artifact(
                local_path=f"{folder}/{date}/{forecast_type}/{date}_END_Campaign_report_ternium_{forecast_type}.pdf",
                artifact_path=f"END_Campaign_report_ternium_{forecast_type}",
            )
        elif endreport == "no":
            mlflow.log_artifact(
                local_path=f"{folder}/{date}/{forecast_type}/{date}_report_ternium_{forecast_type}.pdf",
                artifact_path=f"report_ternium_{forecast_type}",
            )
        elif endreport == "all":
            mlflow.log_artifact(
                local_path=f"{folder}/{date}/{forecast_type}/{date}_END_Campaign_report_ternium_{forecast_type}.pdf",
                artifact_path=f"END_Campaign_report_ternium_{forecast_type}",
            )
            mlflow.log_artifact(
                local_path=f"{folder}/{date}/{forecast_type}/{date}_report_ternium_{forecast_type}.pdf",
                artifact_path=f"report_ternium_{forecast_type}",
            )

    def log_aggregation_runner(self, aggregated_data_by_segment: dict) -> None:
        """
        :param aggregated_data_by_segment: dict with data for the segments
        :return: None

        - logs tags, params, files, datasets with mlflow
        """
        mlflow.set_tags({"segments": list(aggregated_data_by_segment.keys())})
        mlflow.log_param("customer", self._config["general"]["customer_name"].lower())

    def log_outliers_check_runner(self, outlier_percentage: float) -> None:
        """
        :param outlier_percentage: percentage of outliers in the data
        :return: None

        - logs tags, params, files, datasets with mlflow
        """
        mlflow.set_tags(self._config["outlier_check_runner"])
        # log outlier series of process data parameters
        mlflow.set_tags({"outlier_percentage": outlier_percentage})
        mlflow.log_artifact(
            local_path=f"artifacts/{self._customer}/visualization/outlier_check/" f"process_data_boxplot.png",
            artifact_path="boxplot",
        )

    def log_score_model_runner(self, grouped_df: pd.DataFrame) -> None:
        """
        :param grouped_df: df of grouped scoring-results
        :return: None

        - logs tags, params, files, datasets with mlflow
        """
        if self._use_azure_ml:
            datastore = self.workspace.get_default_datastore()
            Dataset.Tabular.register_pandas_dataframe(
                dataframe=grouped_df,
                target=datastore,
                name="apobofml_gob_kpidata",
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

    def _get_data_tag_dict(self, step: str, process_data: dict, measurement_data: dict, meta_data: dict) -> dict:
        """
        :param step: step used, either data_ingress or preprocess
        :param process_data: process_data used
        :param measurement_data: measurement_data used
        :param meta_data: meta data used
        :return: tag dictionary

        - creates the tag dict for data_ingress and preprocessing
        """
        tag_dict = {"vessel_type": self._vessel_type, "customer": self._customer}
        vessels_and_campaigns = {}
        process_campaigns = {}
        measurement_campaigns = {}
        for vessel_number, vessel_info in self._config["general"]["vesselsAndCampaigns"].items():
            vessels_and_campaigns.update({vessel_number: vessel_info["campaigns"]})
            if step == "data_ingress":
                if self._vessel_type == "rh":
                    first_vessel_part = self._config["general"]["vessel_part"][0]
                    measurement_campaigns_for_vessel = [
                        int(i) for i in measurement_data[first_vessel_part][int(vessel_number)].keys()
                    ]
                    process_campaigns_for_vessel = [int(i) for i in process_data[int(vessel_number)].keys()]
                else:
                    measurement_campaigns_for_vessel = [int(i) for i in measurement_data[int(vessel_number)].keys()]
                    process_campaigns_for_vessel = [
                        int(i) for i in process_data[self._config["manifest_column_name"]["campaign"]["name"]].unique()
                    ]
            elif step == "preprocess":
                process_campaigns_for_vessel = [int(i) for i in process_data[int(vessel_number)].keys()]
                if self._vessel_type == "rh":
                    first_vessel_part = self._config["general"]["vessel_part"][0]
                    first_segment = list(measurement_data[first_vessel_part].keys())[0]  # one segment represents all
                    measurement_campaigns_for_vessel = [
                        int(i) for i in measurement_data[first_vessel_part][first_segment][int(vessel_number)].keys()
                    ]
                else:
                    measurement_campaigns_for_vessel = [int(i) for i in measurement_data[int(vessel_number)].keys()]
            else:
                raise ValueError("step needs to be one of [data_ingress, preprocess]")

            process_campaigns.update({vessel_number: list(np.sort(process_campaigns_for_vessel))})
            measurement_campaigns.update({vessel_number: list(np.sort(measurement_campaigns_for_vessel))})

        for key in tag_dict:
            mlflow.log_param(key, tag_dict[key])

        tag_dict["vesselsAndCampaigns"] = str(vessels_and_campaigns)
        tag_dict["process_campaigns"] = str(process_campaigns)
        tag_dict["measurement_campaigns"] = str(measurement_campaigns)
        tag_dict["count_process_files"] = len(meta_data["downloadedFiles"]["processDataFiles"])
        timestamps = self._get_max_timestamp_from_files(raw_meta_data=meta_data, file_type="process")
        tag_dict["latest_process_timestamp"] = str(max(timestamps))

        tag_dict["count_measurement_files"] = len(meta_data["downloadedFiles"]["laserDataFiles"])
        timestamps = self._get_max_timestamp_from_files(raw_meta_data=meta_data, file_type="measurement")
        if len(timestamps) == 0:
            tag_dict["latest_measurement_timestamp"] = str(0)
        else:
            tag_dict["latest_measurement_timestamp"] = str(max(timestamps))

        return tag_dict

    def _get_max_timestamp_from_files(self, raw_meta_data: dict, file_type: str) -> list:
        """
        :param raw_meta_data: dict with raw meta data for the downloaded files and data
        :param file_type: type of files to calculate the max timestamp for
        :return: timestamps of files

        - calculates the max timestamp for process or measurement-files
        """
        if file_type == "process":
            timestamps = []
            for file_name in raw_meta_data["downloadedFiles"]["processDataFiles"]:
                base_name = os.path.basename(file_name)
                try:  # pattern: 1639058135_gob_BOF_1_82_processdata_split_V3.csv
                    timestamps.append(int(base_name.split("_")[0]))
                except ValueError as ve:  # pattern: processdata1626768006.csv
                    self._logger.info(ve)
                    timestamps.append(int(re.findall(r"processdata(\d*).csv", base_name)[0]))
        elif file_type == "measurement":
            timestamps = [
                int(os.path.dirname(file_name).split("/")[-1]) for file_name in raw_meta_data["downloadedFiles"]["laserDataFiles"]
            ]
        else:
            raise ValueError("file_type not specified correctly. Choose one of ['process', 'measurement']")

        return timestamps

    def _save_dataset_for_step(self, step: str, tag_dict: dict) -> None:
        """
        :param step: step for dataset. one of [data_ingress, preprocess]
        :param tag_dict: dictionary of tags of the step
        :return: None, saves dataset in aml

        - saves datasets for the steps data_ingress and preprocess
        """
        vessel_numbers = "|".join(list(self._config["general"]["vesselsAndCampaigns"].keys()))
        datastore = self.workspace.get_default_datastore()
        self._logger.info("save dataset!")
        dataset_name = f"{step}_{self._vessel_type}_{self._customer}_{vessel_numbers}_all"
        try:
            latest_dataset = Dataset.get_by_name(workspace=self.workspace, name=dataset_name, version="latest")
            latest_dataset_tags = latest_dataset.tags
        except UserErrorException:  # dataset is not found and needs to be constructed
            latest_dataset_tags = {}
        if latest_dataset_tags != tag_dict:
            dataset = Dataset.File.upload_directory(
                src_dir=f"{self._config['local_storage']['relative_path']}/" f"{self._customer}/{step}",
                target=DataPath(datastore, f"{step}/{dataset_name}"),
                overwrite=True,
                show_progress=True,
            )
            dataset.register(workspace=self.workspace, name=dataset_name, tags=tag_dict, create_new_version=True)
