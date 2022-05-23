#################################################################################
# File adapted from https://github.com/drivendata/cookiecutter-data-science
#################################################################################
.PHONY: clean_python clean environment_python environment tests data_extract data_transform data exploration_basic topics_models exploration all

#################################################################################
# GLOBALS
#################################################################################
UNAME_M := $(shell uname -m)
APPLE_M1 = arm64
PYTHON_VERSION := 3.9
PYTHON_VERSION_SHORT := $(subst .,,$(PYTHON_VERSION))
PYTHON_INTERPRETER := python$(PYTHON_VERSION)

# select which type of pipeline spaCy pipeline https://spacy.io/usage
SPACY_PIPELINE_TYPE := en_core_web_sm  # optimize for efficiency
#SPACY_PIPELINE_TYPE := en_core_web_trf  # optimize for accuracy


# PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


FORMAT_MESSAGE =  "\n[MAKE "$(1)"] >>>" $(2)

#################################################################################
# Project-specific Commands
#################################################################################
## Run unit-tests.
tests: environment_python
	@echo $(call FORMAT_MESSAGE,"tests", "Running python unit tests.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m unittest discover source/tests

## Extract Data
data_extract: environment_python
	@echo $(call FORMAT_MESSAGE,"data_extract","Extracting data.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/etl.py extract

## Clean and pre-process text data.
data_transform: environment_python
	@echo $(call FORMAT_MESSAGE,"data_transform","Transforming data.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/etl.py transform

## Extract data and clean/pre-process.
data: data_extract data_transform
	@echo $(call FORMAT_MESSAGE,"data","Finished running local ETL.")

## Run the basic exploration notebook(s).
exploration_basic: environment_python
	@echo $(call FORMAT_MESSAGE,"exploration_basic","Running exploratory jupyter notebooks and converting to .html files.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/text_eda_un_debates.ipynb
	mv source/notebooks/text_eda_un_debates.html docs/data/text_eda_un_debates.html
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/text_eda_reddit.ipynb
	mv source/notebooks/text_eda_reddit.html docs/data/text_eda_reddit.html

## Run all the NLP notebooks.
exploration: exploration_basic
	@echo $(call FORMAT_MESSAGE,"exploration","Finished running exploration notebooks.")

## Run topic modeling with n-grams 1-3
topics_1_3: environment_python
	@echo $(call FORMAT_MESSAGE,"topics","Running NMF and LDA Models with n-grams 1-3")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/topic_modeling.py nmf -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/topic_modeling.py lda -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/topic_modeling.py k-means -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	cp source/notebooks/templates/text_topic_modeling_template.ipynb source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	# set values ngrams_how and ngrams_low in notebook
	sed -i '' 's/XXXXXXXXXXXXXXXX/1/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	sed -i '' 's/YYYYYYYYYYYYYYYY/3/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	sed -i '' 's/ZZZZZZZZZZZZZZZZ/10/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	mv source/notebooks/text_topic_modeling_10_ngrams_1_3.html docs/models/topics/text_topic_modeling_10_ngrams_1_3.html

## Run all topic-modeling
topics: topics_1_3
	@echo $(call FORMAT_MESSAGE,"topics","Finished NMF and LDA Models")

## Run entire workflow.
all: environment tests data exploration topics
	@echo $(call FORMAT_MESSAGE,"all","Finished running entire workflow.")

## Delete all generated files (e.g. virtual environment)
clean: clean_python
	@echo $(call FORMAT_MESSAGE,"clean","Cleaning project files.")
	rm -f artifacts/data/raw/*.pkl
	rm -f artifacts/data/raw/*.csv
	rm -f artifacts/data/processed/*
	rm -f artifacts/models/topics/*
	rm -f docs/data/*
	rm -f docs/models/topics/*

#################################################################################
# Generic Commands
#################################################################################
clean_python:
	@echo $(call FORMAT_MESSAGE,"clean_python","Cleaning Python files.")
	rm -rf .venv
	find . \( -name __pycache__ \) -prune -exec rm -rf {} +
	find . \( -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
	rm -rf source/resources/lid.176.ftz

environment_python:
ifneq ($(wildcard .venv/.*),)
	@echo $(call FORMAT_MESSAGE,"environment_python","Found .venv directory. Skipping virtual environment creation.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Activating virtual environment.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo $(call FORMAT_MESSAGE,"environment_python","Did not find .venv directory. Creating virtual environment.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing virtualenv.")
	python3 -m pip install --upgrade pip
	python3 -m pip install -q virtualenv
	@echo $(call FORMAT_MESSAGE,"environment_python","NOTE: Creating environment at .venv.")
	@echo $(call FORMAT_MESSAGE,"environment_python","NOTE: Run this command to activate virtual environment: 'source .venv/bin/activate'.")
	virtualenv .venv --python=$(PYTHON_INTERPRETER)
	@echo $(call FORMAT_MESSAGE,"environment_python","Activating virtual environment.")

	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade pip
	. .venv/bin/activate && pip install -U pip setuptools wheel
ifneq ($(UNAME_M), $(APPLE_M1))
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing cython and cytoolz for non-m1 macs")
	. .venv/bin/activate && pip install cython
	. .venv/bin/activate && ARCHFLAGS="-arch x86_64" pip3 install cytoolz
	. .venv/bin/activate && ARCHFLAGS="-arch x86_64" pip3 install wordcloud
endif
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && brew install libomp

	@echo $(call FORMAT_MESSAGE,"environment_python","Installing NLTK (https://www.nltk.org) corpora packages which will create a folder in your home directory 'nltk_data/corpora'.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/nltk_download.py

	@echo $(call FORMAT_MESSAGE,"environment_python","Installing spaCy - https://spacy.io/usage")
ifeq ($(UNAME_M), $(APPLE_M1))
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing spaCy for Apple M1")
	# command if apple m1
	. .venv/bin/activate && pip install -U 'spacy[apple]'
else
	# command if not apple m1
	. .venv/bin/activate && pip install -U spacy
endif
	. .venv/bin/activate && python3 -m spacy download $(SPACY_PIPELINE_TYPE)

	@echo $(call FORMAT_MESSAGE,"environment_python","Installing fasttext - https://fasttext.cc/docs/en/language-identification.html")
	brew install wget
	wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
	mv lid.176.ftz source/resources/lid.176.ftz

ifeq ($(UNAME_M), $(APPLE_M1))
	@echo "\n[MAKE environment_python] >>> Installing snowflake packages for Apple M1."
	#https://github.com/snowflakedb/snowflake-connector-python/issues/986
	. .venv/bin/activate && pip install 'snowflake-connector-python[pandas,secure-local-storage]' --no-binary snowflake-connector-python
else
	@echo "\n[MAKE environment_python] >>> Installing snowflake packages for non-Apple-M1."
	. .venv/bin/activate && pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v$(SNOWFLAKE_VERSION)/tested_requirements/requirements_$(PYTHON_VERSION_SHORT).reqs
	. .venv/bin/activate && pip install snowflake-connector-python==v$(SNOWFLAKE_VERSION)
endif

	. .venv/bin/activate && brew install libomp
	. .venv/bin/activate && brew install lightgbm

endif

## Set up python/R virtual environments and install dependencies
environment: environment_python
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
