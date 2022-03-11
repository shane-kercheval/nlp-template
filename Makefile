#################################################################################
# File adapted from https://github.com/drivendata/cookiecutter-data-science
#################################################################################
.PHONY: clean_python clean_r clean environment_python environment_r environment tests_python tests_r tests data_extract data_transform data_training_test data exploration_python exploration_r exploration experiments experiments_eval final_model final_eval all

#################################################################################
# GLOBALS
#################################################################################
PYTHON_VERSION := 3.9
PYTHON_VERSION_SHORT := $(subst .,,$(PYTHON_VERSION))
PYTHON_INTERPRETER := python$(PYTHON_VERSION)
SNOWFLAKE_VERSION := 2.7.4

# PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


FORMAT_MESSAGE =  "\n[MAKE "$(1)"] >>>" $(2)

#################################################################################
# Project-specific Commands
#################################################################################
tests_python: environment_python
	@echo $(call FORMAT_MESSAGE,"tests_python", "Running python unit tests.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m unittest discover source/tests

tests_r: environment_r
	@echo $(call FORMAT_MESSAGE,"tests_r","Running R unit tests.")
	R --quiet -e "testthat::test_dir('source/tests')"

tests: tests_python tests_r
	@echo $(call FORMAT_MESSAGE,"tests","Finished running unit tests.")

## Make Dataset
data_extract: environment_python
	@echo $(call FORMAT_MESSAGE,"data_extract","Extracting data.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/executables/etl.py extract

data_transform: environment_python
	@echo $(call FORMAT_MESSAGE,"data_transform","Transforming data.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/executables/etl.py transform

data_training_test: environment_python
	@echo $(call FORMAT_MESSAGE,"data_training_test","Creating training & test sets.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/executables/etl.py create-training-test

data: data_extract data_transform data_training_test
	@echo $(call FORMAT_MESSAGE,"data","Finished running local ETL.")

exploration_python: environment_python data_training_test
	@echo $(call FORMAT_MESSAGE,"exploration_python","Running exploratory jupyter notebooks and converting to .html files.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/executables/data-profile.ipynb
	mv source/executables/data-profile.html docs/data/data-profile.html

exploration_r: environment_r
	@echo $(call FORMAT_MESSAGE,"exploration_r","Running exploratory RMarkdown notebooks and converting to .md files.")
	Rscript -e "rmarkdown::render('source/executables/r-markdown-template.Rmd')"
	rm -rf docs/data/r-markdown-template_files/
	mv source/executables/r-markdown-template.md docs/data/r-markdown-template.md
	mv source/executables/r-markdown-template_files/ docs/data/

exploration: exploration_python exploration_r
	@echo $(call FORMAT_MESSAGE,"exploration","Finished running exploration notebooks.")

experiments: environment_python
	@echo $(call FORMAT_MESSAGE,"experiments","Running Hyper-parameters experiments based on BayesianSearchCV.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/executables/experiments.py

experiments_eval: artifacts/models/experiments/new_results.txt
	@echo $(call FORMAT_MESSAGE,"experiments_eval","Running Evaluation of experiments")
	@echo $(call FORMAT_MESSAGE,"experiments_eval","Copying experiments template (experiment-template.ipynb) to /artifacts/models/experiments directory.")
	cp source/executables/templates/experiment-template.ipynb source/executables/$(shell cat artifacts/models/experiments/new_results.txt).ipynb
	@echo $(call FORMAT_MESSAGE,"experiments_eval","Setting the experiments yaml file name within the ipynb file.")
	sed -i '' 's/XXXXXXXXXXXXXXXX/$(shell cat artifacts/models/experiments/new_results.txt)/g' source/executables/$(shell cat artifacts/models/experiments/new_results.txt).ipynb
	@echo $(call FORMAT_MESSAGE,"experiments_eval","Running the notebook and creating html.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/executables/$(shell cat artifacts/models/experiments/new_results.txt).ipynb
	mv source/executables/$(shell cat artifacts/models/experiments/new_results.txt).html docs/models/experiments/$(shell cat artifacts/models/experiments/new_results.txt).html
	rm -f artifacts/models/experiments/new_results.txt

final_model: environment
	@echo $(call FORMAT_MESSAGE,"final_model","Building final model from best model in experiment.")

final_eval: environment
	@echo $(call FORMAT_MESSAGE,"final_eval","Running evaluation of final model on test set.")

## Run entire workflow.
all: environment tests data exploration experiments experiments_eval final_model final_eval
	@echo $(call FORMAT_MESSAGE,"all","Finished running entire workflow.")

## Delete all generated files (e.g. virtual environment)
clean: clean_python clean_r
	@echo $(call FORMAT_MESSAGE,"clean","Cleaning project files.")
	rm -f artifacts/data/raw/*.pkl
	rm -f artifacts/data/raw/*.csv
	rm -f artifacts/data/processed/*

#################################################################################
# Generic Commands
#################################################################################
clean_python:
	@echo $(call FORMAT_MESSAGE,"clean_python","Cleaning Python files.")
	rm -rf .venv
	find . \( -name __pycache__ \) -prune -exec rm -rf {} +
	find . \( -name .ipynb_checkpoints \) -prune -exec rm -rf {} +

clean_r:
	@echo $(call FORMAT_MESSAGE,"clean_r","Cleaning R files.")
	rm -rf renv
	rm -f .Rprofile

environment_python:
ifneq ($(wildcard .venv/.*),)
	@echo $(call FORMAT_MESSAGE,"environment_python","Found .venv directory. Skipping virtual environment creation.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Activating virtual environment.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo $(call FORMAT_MESSAGE,"environment_python","Did not find .venv directory. Creating virtual environment.")
	python -m pip install --upgrade pip
	python -m pip install -q virtualenv
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing virtualenv.")
	virtualenv .venv --python=$(PYTHON_INTERPRETER)
	@echo $(call FORMAT_MESSAGE,"environment_python","NOTE: Creating environment at .venv.")
	@echo $(call FORMAT_MESSAGE,"environment_python","NOTE: Run this command to activate virtual environment: 'source .venv/bin/activate'.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Activating virtual environment.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing snowflake packages.")
	. .venv/bin/activate && pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v$(SNOWFLAKE_VERSION)/tested_requirements/requirements_$(PYTHON_VERSION_SHORT).reqs
	. .venv/bin/activate && pip install snowflake-connector-python==v$(SNOWFLAKE_VERSION)
	. .venv/bin/activate && brew install libomp
endif

environment_r:
ifneq ($(wildcard renv/.*),)
	@echo $(call FORMAT_MESSAGE,"environment_r","Found renv directory. Skipping virtual environment creation.")
else
	@echo $(call FORMAT_MESSAGE,"environment_r","Did not find renv directory. Creating virtual environment.")
	R --quiet -e 'install.packages("renv", repos = "http://cran.us.r-project.org")'
	# Creates `.Rprofile` file, and `renv` folder
	R --quiet -e 'renv::init(bare = TRUE)'
	R --quiet -e 'renv::install()'
endif

## Set up python/R virtual environments and install dependencies
environment: environment_python environment_r
	@echo $(call FORMAT_MESSAGE,"environment","Finished setting up environment.")

#################################################################################
# Self Documenting Commands
#################################################################################
.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
