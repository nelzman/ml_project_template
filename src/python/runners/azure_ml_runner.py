import argparse

from src.python.azure_ml.azure_ml_experiment import AMLExperiment
import src.python.utils as utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AML-Runner-BOF", add_help=True)
    parser.add_argument(
        "--ide",
        metavar="ide",
        help="used ide for development",
        type=str,
        choices=["pycharm", "vscode"],
        required=True,
    )
    parser.add_argument(
        "--runner",
        metavar="runner",
        help="name of the runner to run. None refers to a choice-window-popup to choose the runner",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--env_version",
        metavar="env_version",
        help="choose the environment-version: "
        "- 'update' adds a new version to aml based on your infrastructure/environment.yml. "
        "- 'latest' uses the latest environment version"
        "- number as string: uses this version number",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--use_compute_target",
        metavar="use_compute_target",
        help="use compute target for calculation",
        type=str,
        choices=["True", "False"],
        required=True,
        default="False",
    )
    args = parser.parse_args()

    # load config:

    config = utils.load_yaml_config("infrastructure/config.yml")

    aml_experiment = AMLExperiment(config)
    aml_experiment.run(
        ide=args.ide,
        runner=args.runner,
        env_version=args.env_version,
        use_compute_target=args.use_compute_target,
    )
