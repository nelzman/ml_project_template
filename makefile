python_local_environment_name := local_dev
make_file_path := $(abspath $(lastword $(MAKEFILE_LIST)))
cwd := $(dir $(make_file_path))

# Python:

new_py_environment:
	conda env create --file infrastructure/environment.yml

update_py_environment:
	conda env update --name ml_project_template --file environment.yml

export_py_environment:
	conda env export --name ml_project_template > environment.yml
	
remove_py_environment:
	conda env remove --name ml_project_template

lint:
	pylint ./src/python

isort:
	isort . -l 130

black:
	black . -l 130 -t py38

pr_check:
	make black
	make isort
	make lint

wheel:
	python -m build . --wheel

unit_tests:
	pytest tests/unit -vv

unit_tests_cov:
	pytest tests/unit -vv --cov-config=.coveragerc --cov-report term-missing --cov=src

mlflow_ui:
	mlflow ui --backend-store-uri artifacts/.mlruns --port 5001

# R:
