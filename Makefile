
.PHONY: tests

docker_build:
	docker compose -f docker-compose.yml build

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_rebuild:
	docker compose -f docker-compose.yml build --no-cache

docker_bash:
	docker compose -f docker-compose.yml up --build bash

docker_open: notebook mlflow_ui zsh

notebook:
	open 'http://127.0.0.1:8888/?token=d4484563805c48c9b55f75eb8b28b3797c6757ad4871776d'

zsh:
	docker exec -it nlp-template-bash-1 /bin/zsh

#################################################################################
# Project-specific Commands
#################################################################################
linting:
	flake8 --max-line-length 99 source/scripts
	flake8 --max-line-length 99 source/library
	flake8 --max-line-length 99 tests

## Run unit-tests.
tests:
	# show the slowest 20 tests among those that take at least 1 second
	coverage run -m pytest --durations=20 --durations-min=1.0 tests
	coverage html

## Extract Data
data_extract:
	python source/scripts/commands.py extract

## Clean and pre-process text data.
data_transform:
	python source/scripts/commands.py transform

## Extract data and clean/pre-process.
data: data_extract data_transform

## Run the basic exploration notebook(s).
exploration_basic:
	jupyter nbconvert --execute --to html source/notebooks/text_eda_reddit.ipynb
	mv source/notebooks/text_eda_reddit.html output/data/text_eda_reddit.html
	# jupyter nbconvert --execute --to html source/notebooks/text_eda_un_debates.ipynb
	# mv source/notebooks/text_eda_un_debates.html output/data/text_eda_un_debates.html

## Run all the NLP notebooks.
exploration: exploration_basic

## Run topic modeling with n-grams 1-3
topics_1_3:
	python source/scripts/commands.py nmf -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	python source/scripts/commands.py lda -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	python source/scripts/commands.py k-means -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	cp source/notebooks/templates/text_topic_modeling_template.ipynb source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	# set values ngrams_how and ngrams_low in notebook
	sed -i 's/XXXXXXXXXXXXXXXX/1/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	sed -i 's/YYYYYYYYYYYYYYYY/3/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	sed -i 's/ZZZZZZZZZZZZZZZZ/10/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	jupyter nbconvert --execute --to html source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	mv source/notebooks/text_topic_modeling_10_ngrams_1_3.html output/models/topics/text_topic_modeling_10_ngrams_1_3.html

## Run all topic-modeling
topics: topics_1_3

cosine_sim:
	jupyter nbconvert --execute --to html source/notebooks/cosine_similarity.ipynb
	mv source/notebooks/cosine_similarity.html output/models/topics/cosine_similarity.html

## Run entire workflow.
all: tests data exploration topics cosine_sim

## Delete all generated files
clean: clean_python
	rm -f artifacts/data/raw/*.pkl
	rm -f artifacts/data/raw/*.csv
	rm -f artifacts/data/processed/*
	rm -f artifacts/models/topics/*
	rm -f output/data/*
	rm -f output/models/topics/*

#################################################################################
# Generic Commands
#################################################################################
clean_python:
	find . \( -name __pycache__ \) -prune -exec rm -rf {} +
	find . \( -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
