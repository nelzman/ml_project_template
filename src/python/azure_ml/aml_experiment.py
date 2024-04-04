import os

from azureml.core import Experiment, ScriptRunConfig, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import RunConfiguration

from src.python.azure_ml.aml_utils import AMLUtils


class AMLExperiment:
    """
    Class to setup aml-experiments and logging via mlflow
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        try:
            credentials = InteractiveLoginAuthentication(tenant_id=self._config["aml"]["tenant_id"])
        except Exception as ex:
            print(ex)
            credentials = None
        self._workspace = Workspace.get(
            name=self._config["aml"]["dev"]["workspace_name"],
            subscription_id=self._config["aml"]["subscription_id"],
            resource_group=self._config["aml"]["dev"]["resource_group"],
            auth=credentials,
        )
        self._aml_utils = AMLUtils(self._config, self._workspace)

    def run(
        self,
        file: str, 
        args: dict,
        env_version: str,
        use_compute_target: str,
    ) -> None:
        """
        :param file: filepath to run
        :param args: parameters to run the script
        :param env_version: version specification. choose from:
            latest: get the latest version
            update: update the environment with your current version
            <version-number>: choose the number by applying a int here
        :param use_compute_target: whether to use the compute target specified in the config or local
        :return: None

        - this runner takes another runner and sends it to aml as an experiment
        - things are logged with mlflow inside the runners
        """
        # setup aml:

        env = self._aml_utils.get_environment(env_version)

        aml_compute, docker_config = self._aml_utils.get_aml_compute(environment=env, use_compute_target=use_compute_target)

        run_config = RunConfiguration()
        run_config.environment = env
        run_config.target = aml_compute
        run_config.environment_variables = {"AML": "Distinction between execution environments"}

        exp = Experiment(workspace=self._workspace, name=f"{file}_experiment")

        # sending runners to aml:
        try:
            run_config = ScriptRunConfig(
                source_directory="",
                script=f"src/{file}",
                docker_runtime_config=docker_config,
                arguments=args,
                run_config=run_config,
            )
            execution = exp.submit(run_config)
            print(f"Experiment-URL: {execution.get_portal_url()}")

        except Exception as e:
            print(f"Exception thrown: {e}")
            if "execution" in locals():
                execution.cancel()

        finally:
            print("Runner finished.")
