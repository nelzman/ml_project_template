from src.python.sagemaker.sagemaker_utils import SagemakerUtils
from src.python.runners.runner_utils import RunnerUtils

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.experiments.run import Run
from sagemaker.remote_function import remote
from sagemaker.sklearn.estimator import SKLearn
import mlflow


class SagemakerExperiment:
    """
    Class to setup sagemaker-experiments and logging via mlflow
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._sagemaker_utils = SagemakerUtils(self._config)
        self._runner_utils = RunnerUtils(self._config)
        self._sagemaker_utils = SagemakerUtils(self._config, self._workspace)
        # Set SageMaker and S3 client variables
        self.sagemaker_session = sagemaker.Session()
        self.sm_boto3 = boto3.client("sagemaker")

        self.region = self.sagemaker_session.boto_session.region_name

        self.bucket = self.sagemaker_session.default_bucket()  # this could also be a hard-coded bucket name

        print("Using bucket " + self.bucket)

    def run(
        self,
        ide: str,
        runner: str,
        env_version: str,
        use_compute_target: str) -> None:

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
        FRAMEWORK_VERSION = "0.23-1"


        runner_dict, runner_name = self._runner_utils.get_runner_details(ide, runner)

        # setup aml:

        env = self._aml_utils.get_environment(env_version)

        sm_compute, docker_config = self._sagemaker_utils.get_aml_compute(environment=env, use_compute_target=use_compute_target)

        run_config = RunConfiguration()
        run_config.environment = env
        run_config.target = aml_compute
        run_config.environment_variables = {"AzureML": "Distinction between execution environments"}

        exp = Experiment(workspace=self._workspace, name=f"{runner_name.replace(' ', '_')}_experiment")

        sklearn_estimator = SKLearn(
            entry_point="src/python/runners/training_runner.py",
            role=get_execution_role(),
            instance_count=1,
            instance_type=sm_compute,  # "ml.c5.xlarge",
            framework_version=FRAMEWORK_VERSION,
            base_job_name="housing-scikit",
            hyperparameters=runner_dict[runner_name]["param_list"],
        )

        sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=True)
        
        #sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name="us-west-2"))
        #settings = dict(
        #    sagemaker_session=sm_session,
        #    role=<The IAM role name>,
        #    instance_type="ml.m5.xlarge",
        #    dependencies='./requirements.txt'
        #)

        #@remote(**settings)
        #def divide(x, y):
        #    return x / y

        #print(divide(2, 3.0))

        #with Run(experiment_name="test-exp", sagemaker_session=self.sagemaker_session) as run:
        #    run.log_parameter(name="param1", value=0.5)
            

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
