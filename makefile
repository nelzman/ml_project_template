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

isort:
	isort . -l 130

black:
	black . -l 130 -t py38

pr_check:
	make black
	pylint .

wheel:
	python setup.py sdist bdist_wheel egg_info

# R:
