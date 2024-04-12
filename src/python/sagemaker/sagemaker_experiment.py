from azureml.core import Experiment, ScriptRunConfig, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import RunConfiguration

from src.python.sagemaker.sagemaker_utils import SagemakerUtils

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.experiments.run import Run
from sagemaker.remote_function import remote
import mlflow

class SagemakerExperiment:
    """
    Class to setup sagemaker-experiments and logging via mlflow
    """

    def __init__(self, config: dict) -> None:
        self._config = config

        try:
            credentials = InteractiveLoginAuthentication(tenant_id=self._config["azure_ml"]["tenant_id"])
        except Exception as ex:
            print(ex)
            credentials = None
        self._workspace = Workspace.get(
            name=self._config["azure_ml"]["workspace_name"],
            subscription_id=self._config["azure_ml"]["subscription_id"],
            resource_group=self._config["azure_ml"]["resource_group"],
            auth=credentials,
        )
        self._sagemaker_utils = SagemakerUtils(self._config, self._workspace)
        # Set SageMaker and S3 client variables
        self.sagemaker_session = sagemaker.Session()
        #boto_sess = boto3.Session()

        #role = get_execution_role()
        #default_bucket = self.sagemaker_session.default_bucket()

        #sm = boto_sess.client("sagemaker")
        #region = boto_sess.region_name
    def execute(self):


        sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name="us-west-2"))
        settings = dict(
            sagemaker_session=sm_session,
            role=<The IAM role name>,
            instance_type="ml.m5.xlarge",
            dependencies='./requirements.txt'
        )

        @remote(**settings)
        def divide(x, y):
            return x / y

        print(divide(2, 3.0))

        with Run(experiment_name="test-exp", sagemaker_session=self.sagemaker_session) as run:
            run.log_parameter(name="param1", value=0.5)
            

    def run(
        self,
        ide: str,
        runner: str,
        env_version: str,
        use_compute_target: str,
    ) -> None:
        """
        :param ide: ide used for development
        :param runner: runner to run
        :param env_version: version specification. choose from:
            latest: get the latest version
            update: update the environment with your current version
            <version-number>: choose the number by applying a int here
        :param use_compute_target: whether to use the compute target specified in the config or local

        - this runner takes another runner and sends it to aml as an experiment
        - things are logged with mlflow inside the runners
        """

        runner_dict, runner_name = self._aml_utils.get_runner_details(ide, runner)

        # setup aml:

        env = self._aml_utils.get_environment(env_version)

        aml_compute, docker_config = self._aml_utils.get_aml_compute(environment=env, use_compute_target=use_compute_target)

        run_config = RunConfiguration()
        run_config.environment = env
        run_config.target = aml_compute
        run_config.environment_variables = {"AzureML": "Distinction between execution environments"}

        exp = Experiment(workspace=self._workspace, name=f"{runner_name.replace(' ', '_')}_experiment")

        # sending runners to aml:
        try:
            run_config = ScriptRunConfig(
                source_directory="",
                script=f"src/python/runners/{runner_dict[runner_name]['file']}",
                docker_runtime_config=docker_config,
                arguments=runner_dict[runner_name]["param_list"],
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
