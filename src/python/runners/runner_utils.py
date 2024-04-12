import os
import xmltodict
import easygui

from src.python import utils


class RunnerUtils:

    def __init__(self, config: dict):
        self._config = config

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
