# nlp-template

This repo contains a template for natural language processing (NLP) Projects. The structure was generated from this template https://github.com/shane-kercheval/data-science-template.

Much of the code and examples are copied/modified from 

> Blueprints for Text Analytics Using Python by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler (O'Reilly, 2021), 978-1-492-07408-3.
>
> https://github.com/blueprints-for-text-analytics-python/blueprints-text

---

This project contains python that mimics a slimmed down version of a NLP project.

The `Makefile` runs all components of the project. You can think of it as containing the implicit DAG, or recipe, of the project.

Common commands available from the Makefile are:



TBD


- `make all`: The entire project can be built/ran with the simple command `make all` from the project directory, which runs all components (build virtual environments, run tests, run scripts, generate output, etc.)
- `make clean`: Removes all virtual environments, python/R generated/hidden folders, and interim/processed data.
- `make environment`: Creates python/R virtual environments and install packages from the requirements.txt/DESCRIPTION files
- `make data`: Runs ETL scripts
- `make exploration`: Runs exploration notebooks and generate html/md documents.
- `make experiments`: Runs scripts which use BayesSearchCV over several models.
- `make experiments_eval`: Runs notebooks which evaluates the performance of the BayesSearchCV and produces an html report.
- `make final_model`: Retrains the best model from the most recent experiments on all data, and predict on test/holdout set. (not implemented yet)
- `make final_eval`: Runs the notebook which shows the performance of the final model and produces an html report. (not implemented yet)









See `Makefile` for additional commands and implicit project DAG.

This project requires Python 3.9 (but the python version can be configured in the Makefile) and is currently ran with R version 4.X.

---

- Text Processing is done in the etl.py script. 

`python source/executables/etl.py extract`
`make data_extract` which also gets ran with `make data`

`python source/executables/etl.py transform`
`make data_transform` which also gets ran with `make data`




- `source/executables/text_eda.ipynb`
    - basic exploratory analysis 
        - descriptive statistics & trends
        - word frequencies
        - tf-idf
        - word-clouds
        - context search


---

To activate virtual environment run `source .venv/bin/activate`; for example:

```commandline
source .venv/bin/activate
jupyter notebook
```

---

## Repo structure 

```
├── README.md                  <- You are here
├── Makefile                   <- Makefile, which runs all components of the project with commands like `make all`, `make environment`, `make data`, etc.
├── requirements.txt           <- Python package dependencies
│
│
├── artifacts/                 <- All non-code/document artifacts (e.g. data, models, etc.).
│   ├── data/                  <- Folder that contains data (used or generated).
│       ├── external/          <- Data from third party sources.
│       ├── interim/           <- Intermediate data that has been transformed. (This directory is excluded via .gitignore)
│       ├── processed/         <- The final, canonical data sets for modeling. (This directory is excluded via .gitignore)
│       ├── raw/               <- The original, immutable data dump. (This directory is excluded via .gitignore)
│   ├── models/                <- Trained model objects (TMOs), model predictions, and/or model summaries
│       ├── archive/           <- Folder that contains old models.
│       ├── current/           <- The current model being used by the project.
│       ├── experiments/       <- Contains experimentss and experiments output (e.g. yaml/html showing performance of experiments.)
│
├── source/                    <- All source-code (e.g. SQL, python scripts, notebooks, unit-tests, etc.)
│   ├── config/                <- Directory for yaml configuration files for model training, scoring, etc
│   ├── executables/           <- Notebooks, as well as command-line programs that execute the project tasks (e.g. etl & data processing, experiments, model-building, etc.). They typically have outputs that are artifacts (e.g. .pkl models or data).
│       ├── helpers/           <- Supporting source-code that promotes code reusability and unit-testing. Clients that use this code are notebooks, executables, and tests.
│       ├── templates/             <- Template notebooks for analysis with useful imports and helper functions. 
│   ├── tests/                 <- Files necessary for running model tests (see documentation below) 
│       ├── test_files/        <- Files that help run unit tests, e.g. mock yaml files.
│       ├── test_file.py       <- python unit-test script
│
├── docs/                      <- All documentation, data dictionaries, manuals, and final reports and deliverables.
│   ├── data/                  <- Location to place documents describing results of data exploration, data dictionaries, etc.
│   ├── deliverables/          <- All generated and sharable deliverables.
│   ├── models/                <- Model documentation 
│       ├── archive/
│       ├── baseline_model/
│       ├── current_model/
│       ├── experiments/
│   ├── project/               <- Project documentation, including project charter, and results.
│   ├── figures/               <- Centralized location for all figures and diagrams in project except for those embedded in notebooks.
│           ├── archive/
│           ├── data/
│           ├── models/
```

---

# Project Details

- see [docs/project/Charter.md](./docs/project/Charter.md) for project description.
- see [docs/project/Exit-Report.md](./docs/project/Exit-Report.md) for project results.
