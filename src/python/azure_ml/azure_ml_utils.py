import os
import warnings
from datetime import datetime, timezone

import easygui
import xmltodict
from azureml.core import Dataset, Environment, Workspace
from azureml.core.compute import ComputeInstance
from azureml.core.model import Model
from azureml.core.runconfig import DockerConfiguration

from src.python import utils


class AMLUtils:
    """
    Class for different utility methods to support the usage of aml
    """

    def __init__(self, config: dict, workspace: Workspace = None) -> None:
        self._config = config
        self._workspace = workspace

    def get_aml_compute(self, environment: Environment, use_compute_target: str) -> tuple[ComputeInstance, DockerConfiguration]:
        """
        :param environment: environment for local execution
        :param use_compute_target: compute-target to use
        :return: aml_compute-object and docker_config

        - get the aml_compute and docker_config for aml execution
        """
        if use_compute_target == "True":
            azure_ml_compute = ComputeInstance(self._workspace, self._config["azure_ml"]["compute_target"])
            docker_config = DockerConfiguration(use_docker=True)
        else:
            environment.python.user_managed_dependencies = True
            azure_ml_compute = None
            docker_config = None

        return azure_ml_compute, docker_config

    def get_environment(self, env_version: str) -> Environment:
        """
        :param env_version: version specification. choose from:
            latest: get the latest version
            update: update the environment with your current version
            <version-number>: choose the number by applying a int here
        :return: Environment-object

        - get the environment from aml or update it
        """
        if env_version == "update":
            env = Environment.from_dockerfile(
                name="ml_project_template",
                dockerfile="aml_dockerfile",
                conda_specification="./infrastructure/environment.yml",
            )
            env.register(self._workspace)
        elif env_version == "latest":
            env = Environment.get(workspace=self._workspace, name="ml_project_template")
        else:
            env = Environment.get(workspace=self._workspace, name="ml_project_template", version=env_version)

        return env

    def _download_dataset_data(self, dataset_name: str, update_data: bool) -> None:
        """
        :param dataset_name: dataset to download
        :param update_data: whether to update the local data

        - downloads the data from aml datasets if necessary
        """

        dataset = Dataset.get_by_name(workspace=self._workspace, name=dataset_name, version="latest")
        all_campaigns_folder = "artifacts/aml_data/"
        local_file = f"{dataset_name}.pkl"
        if os.path.exists(local_file):
            local_file_time = datetime.fromtimestamp(os.path.getmtime(local_file)).replace(tzinfo=timezone.utc)
            aml_file_time = dataset.data_changed_time
            if (aml_file_time > local_file_time or os.path.getsize(local_file) == 0) and update_data:
                env_name = "prod" if self._use_prd else "dev"
                print(f"Download of dataset {dataset_name} from {env_name}-workspace started!")
                dataset.download(target_path=all_campaigns_folder, overwrite=True)
                print("Download finished!")
            else:
                print("Use up-to-date local file!")
        else:
            print(f"Download of dataset {dataset_name} started!")
            dataset.download(target_path=all_campaigns_folder, overwrite=True)
            print("Download finished!")

    @staticmethod
    def download_aml_model(workspace: Workspace, model_path: str, model_name: str) -> str:
        """
        :param workspace: workspace to download from
        :param model_path: path where to save the model
        :param model_name: model_name to download
        :return: models and parameters

        - download model_tree from registered models in aml
        """

        registry_prod_models = Model.list(workspace=workspace, name=model_name, tags=[["stage", "Production"]])
        if len(registry_prod_models) == 0:
            raise FileNotFoundError(f"No Production model for {model_name} found on workspace: {workspace.name}")
        if len(registry_prod_models) == 1:
            prod_model = registry_prod_models[0]
        else:
            raise FileNotFoundError(f"Too many Production models for {model_name} found on workspace: {workspace.name}")

        print(f"Download prod-model for {model_name}")
        prod_model.download(target_dir=model_path, exist_ok=True)

        return model_path

    @staticmethod
    def clean_prod_model_stages(registry_prod_models: list) -> Model:
        """
        :param registry_prod_models: list with production models
        :return Model

        - cleans up model registry to only have one production model
        """
        prod_versions = [registry_model.version for registry_model in registry_prod_models]
        warnings.warn(
            f"Multiple models available with prod-stage (versions: {prod_versions}). "
            f"Best Model stays on Production, the other models get set to stage 'Archived'."
        )
        prod_model_score = float("inf")
        prod_model = None
        for registry_model in registry_prod_models:
            registry_model_score = registry_model.tags["overall_score"]
            if registry_model_score <= prod_model_score:
                if prod_model is not None:
                    AMLUtils.update_model_stage(prod_model, "Archived")
                prod_model = registry_model
                prod_model_score = registry_model_score
            else:
                AMLUtils.update_model_stage(registry_model, "Archived")
        return prod_model

    @staticmethod
    def update_model_stage(model: Model, new_stage: str) -> None:
        """
        :param model: model from aml
        :param new_stage: new stage to add to the model

        - updates aml-model stages if valid model-stages given
        - Production can only be set in the devops model-promotion pipeline
        """
        valid_stages = ["Development", "Staging", "Archived"]
        if new_stage not in valid_stages:
            raise ValueError(f"New model stage needs to be one of {valid_stages}")
        model.stage = new_stage
        model.update()
        model.add_tags({"stage": new_stage})

    def get_runner_details(
        self,
        ide: str,
        runner: str,
        local_path: str = None,
    ) -> tuple[str, dict, str]:
        """
        :param ide: used ide
        :param runner: used runner
        :param local_path: local path to the .run.xml/launch.json
        :return: runner-dictionary with run-configuration, runner-name

        - constructs a dict with run-configurations based on the .run.xml files for pycharm
        """

        # setup parameters for the run:
        aml_params = {
            "--use_azure_ml": "True",
        }

        if ide == "pycharm":
            r_dict = self._get_runner_dict_pycharm(aml_params_dict=aml_params, local_path=local_path)
        elif ide == "vscode":
            r_dict = self._get_runner_dict_vscode(aml_params_dict=aml_params, local_path=local_path)
        else:
            raise ValueError(f"specified ide {ide} not implemented for this runner")

        r_name = self._input_runner_name(r_dict) if runner == "None" else runner

        return r_dict, r_name

    def _get_runner_dict_pycharm(self, aml_params_dict: dict, local_path: str = None) -> dict:
        """
        :param aml_params_dict: parameters specifically for aml to be added to the parameters
        :param local_path: local path to the .run.xml
        :return: dict with run-configurations

        - constructs a dict with run-configurations based on the .run.xml files for pycharm
        """

        if local_path is None:
            local_path = "src/python/.run"
        r_dict = {}
        for file in os.listdir(local_path):

            if "AzureML" in file:
                continue

            with open(f"{local_path}/{file}", "r", encoding="utf-8") as file:
                my_xml = file.read()

            my_dict = xmltodict.parse(my_xml)
            r_name = my_dict["component"]["configuration"]["@name"]

            for option in my_dict["component"]["configuration"]["option"]:
                if option["@name"] == "SCRIPT_NAME":
                    runner_file = os.path.basename(option["@value"])
                if option["@name"] == "PARAMETERS":
                    runner_params = option["@value"]

            runner_param_nested_list = [param.replace('"', "").split("=") for param in runner_params.split('" "')]
            runner_param_nested_list = [key for key in runner_param_nested_list if key[0] not in aml_params_dict] + [
                [key, value] for key, value in aml_params_dict.items()
            ]
            runner_param_list = [item for sublist in runner_param_nested_list for item in sublist]
            r_dict[r_name] = {"file": runner_file, "param_list": runner_param_list}

        return r_dict

    def _get_runner_dict_vscode(self, aml_params_dict: dict, local_path: str = None) -> dict:
        """
        :param aml_params_dict: parameters specifically for aml to be added to the parameters
        :param local_path: local path to the launch.json
        :return: dict with run-configurations

        - constructs a dict with run-configurations based on the .run.xml files for pycharm
        """

        if local_path is None:
            local_path = ".vscode"
        my_dict = utils.load_json(full_path=local_path, filename="launch.json")
        r_dict = {}
        for conf in my_dict["configurations"]:
            r_name = conf["name"]
            if "AzureML" in r_name:
                continue
            r_params = conf["args"]
            r_file = conf["module"].split(".")[-1]

            runner_param_nested_list = [param.replace('"', "").split("=") for param in r_params]
            runner_param_nested_list = [key for key in runner_param_nested_list if key[0] not in aml_params_dict.keys()] + [
                [key, value] for key, value in aml_params_dict.items()
            ]
            runner_param_list = [item for sublist in runner_param_nested_list for item in sublist]
            r_dict[r_name] = {"file": f"{r_file}.py", "param_list": runner_param_list}
        return r_dict

    def _input_runner_name(self, r_dict: dict) -> str:
        """
        :param r_dict: dictionary with runner configurations
        :return string of runner name

        - pop-up to choose the runner to use
        """
        if len(r_dict.keys()) > 1:
            r_name = easygui.choicebox(
                "Which runner do you want to use?",
                "Runner Selection",
                list(r_dict.keys()),
            )
        else:
            r_name = list(r_dict.keys())[0]
        return r_name
