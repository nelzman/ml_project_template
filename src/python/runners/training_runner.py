import argparse
import gc
import os
import pandas as pd

import mlflow
from src.python.azure_ml.mlflow_tracker import MLFlowTracker
from src.python.azure_ml.azure_ml_utils import AMLUtils
from src.python.preprocess.preprocessor import Preprocessor
from src.python.ingress.ingressor import DataIngressor
from src.python.training.training_workflow import TrainingWorkflow
import src.python.utils as utils

if __name__ == "__main__":

    gc.enable()  # Enable Garbage Collection for cope with Memory Leak
    # parameters for training:
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--save_files_for_tests",
        metavar="save_files_for_tests",
        type=str,
        nargs="?",
        choices=["True", "False"],
        default="False",
        help="Save data for Tests [True, False]",
    )
    parser.add_argument(
        "--use_autolog",
        type=str,
        help="options=[True, False]",
        choices=["True", "False"],
        nargs="?",
        default="False",
    )
    parser.add_argument(
        "--save_models",
        metavar="save_models",
        type=str,
        nargs="?",
        choices=["True", "False"],
        default="False",
        help="Save models [True, False]",
    )
    parser.add_argument(
        "--use_azure_ml",
        type=str,
        help="options=[True, False]",
        choices=["True", "False"],
        nargs="?",
        default="False",
    )
    args = parser.parse_args()

    save_files_for_tests = True if args.save_files_for_tests == "True" else False
    use_autolog = True if args.use_autolog == "True" else False
    save_models = True if args.save_models == "True" else False
    use_azure_ml = True if args.use_azure_ml == "True" else False

    config = utils.load_yaml_config("infrastructure/config.yml")

    logger = utils.setup_logging(filename="training")

    mlflow_logger = MLFlowTracker(config, logger, "training", use_azure_ml)
    aml_utils = AMLUtils(config, mlflow_logger.workspace)

    # run data ingress:
    data_ingressor = DataIngressor(config, logger)
    data = data_ingressor.ingress(location="local")

    # run preprocessing:
    logger.info("Run Preprocess!")
    preprocess_runner = Preprocessor(
        config=config,
        logger=logger,
        save_data_for_tests=False,
        use_azure_ml=use_azure_ml,
        workspace=mlflow_logger.workspace,
    )
    data = preprocess_runner.preprocess_data(data=data)

    if use_autolog:
        mlflow.sklearn.autolog(log_models=False, log_post_training_metrics=False)

    with mlflow.start_run() as run:
        # run training:
        train_pipeline = TrainingWorkflow(
            config,
            data=data,
            logger=logger,
            save_files_for_tests=save_files_for_tests,
        )
        model_tree = train_pipeline.train(
            config["training"]["algorithms"], ensemble_weights=config["training"]["ensemble_weights"], save_trees=save_models
        )

        mlflow_logger.log_training_runner(model_tree=model_tree, save_trees=save_models, update_model_registry=False)
