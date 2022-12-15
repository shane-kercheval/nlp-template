# nlp-template

This repo contains a template for natural language processing (NLP) Projects. The structure was generated from this template https://github.com/shane-kercheval/data-science-template.

Much of the code and examples are copied/modified from 

> Blueprints for Text Analytics Using Python by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler (O'Reilly, 2021), 978-1-492-07408-3.
>
> https://github.com/blueprints-for-text-analytics-python/blueprints-text

---

This project contains python that mimics a slimmed down version of a NLP project.

The `Makefile` runs all components of the project. You can think of it as containing the implicit DAG, or recipe, of the project.

The project runs in a Docker container. To create and run the docker container, use the command `make docker_run` from the terminal in this directory. See the Makefile for more commands related to running the Docker container.

Common commands available from the Makefile are:

- `make all`: The entire project can be built/ran with the simple command `make all` from the project directory, which runs all components (build virtual environments, run tests, run scripts, generate output, etc.)
- `make tests`: Run all of the unit-tests in teh project.
- `make data`: Runs ETL scripts
- `make exploration`: Runs exploration notebooks and generate html/md documents.
- `make topics`: Runs topic modeling scripts and notebooks.

---

NOTE: Including `jinja2==3.0.3` in `requirements.txt` because of 

- https://github.com/microsoft/vscode-jupyter/issues/9468
- https://github.com/jupyter/nbconvert/issues/1736

---

See `Makefile` for additional commands and implicit project DAG.

This project requires Python 3.9 (but the python version can be configured in the Makefile) and is currently ran with R version 4.X.

---

- Text Processing is done in the etl.py script. 

`python source/scripts/etl.py extract`
`make data_extract` which also gets ran with `make data`

`python source/scripts/etl.py transform`
`make data_transform` which also gets ran with `make data`

- `source/notebooks/text_eda.ipynb`
    - basic exploratory analysis 
        - descriptive statistics & trends
        - word frequencies
        - tf-idf
        - word-clouds
        - context search

## Dependencies

- spaCy
    - used for tokenization and part-of-speech recognitiion
    - `make environment` installs this and has logic to install specific package for Apple M1 if detected
- NLTK (https://www.nltk.org)
    - primary used for stop-words
    - `make environment` will create a folder in your home directory 'nltk_data/corpora'
- fasttext (https://fasttext.cc/docs/en/language-identification.html)
    - used for language recognitiion
    - installed by `make environment`

- language codes in the `source/resources/lanugage_codes.csv` was copied from https://raw.githubusercontent.com/haliaeetus/iso-639/master/data/iso_639-1.csv

---

To activate virtual environment run `source .venv/bin/activate`; for example:

```commandline
source .venv/bin/activate
jupyter notebook
```
