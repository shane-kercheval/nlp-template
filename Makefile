
.PHONY: tests

compose:
	docker compose -f docker-compose.yml up --build

docker_run_container:
	docker run -it shanekercheval/python:nlp

docker_run: docker_jupyter docker_zsh

docker_jupyter:
	open 'http://127.0.0.1:8888/?token=d4484563805c48c9b55f75eb8b28b3797c6757ad4871776d'

docker_zsh:
	docker exec -it nlp-template-bash-1 /bin/zsh

#################################################################################
# Project-specific Commands
#################################################################################
## Run unit-tests.
tests:
	python -m unittest discover source/tests

## Extract Data
data_extract: environment_python
	python source/scripts/etl.py extract

## Clean and pre-process text data.
data_transform: environment_python
	python source/scripts/etl.py transform

## Extract data and clean/pre-process.
data: data_extract data_transform

## Run the basic exploration notebook(s).
exploration_basic: environment_python
	jupyter nbconvert --execute --to html source/notebooks/text_eda_un_debates.ipynb
	mv source/notebooks/text_eda_un_debates.html docs/data/text_eda_un_debates.html
	jupyter nbconvert --execute --to html source/notebooks/text_eda_reddit.ipynb
	mv source/notebooks/text_eda_reddit.html docs/data/text_eda_reddit.html

## Run all the NLP notebooks.
exploration: exploration_basic

## Run topic modeling with n-grams 1-3
topics_1_3: environment_python
	python source/scripts/topic_modeling.py nmf -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	python source/scripts/topic_modeling.py lda -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	python source/scripts/topic_modeling.py k-means -num_topics=10 -ngrams_low=1 -ngrams_high=3 -num_samples=5000
	cp source/notebooks/templates/text_topic_modeling_template.ipynb source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	# set values ngrams_how and ngrams_low in notebook
	sed -i '' 's/XXXXXXXXXXXXXXXX/1/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	sed -i '' 's/YYYYYYYYYYYYYYYY/3/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	sed -i '' 's/ZZZZZZZZZZZZZZZZ/10/g' source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	jupyter nbconvert --execute --to html source/notebooks/text_topic_modeling_10_ngrams_1_3.ipynb
	mv source/notebooks/text_topic_modeling_10_ngrams_1_3.html docs/models/topics/text_topic_modeling_10_ngrams_1_3.html

## Run all topic-modeling
topics: topics_1_3

## Run entire workflow.
all: environment tests data exploration topics

## Delete all generated files (e.g. virtual environment)
clean: clean_python
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
	find . \( -name __pycache__ \) -prune -exec rm -rf {} +
	find . \( -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
