import datetime
import pickle

import helpsk as hlp
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV  # noqa

from helpers.utilities import get_logger, Timer
import helpers.classification_search_space as css


def main():
    """
    Logic For Running Experiments.
    """
    with open('artifacts/data/processed/X_train.pkl', 'rb') as handle:
        x_train = pickle.load(handle)

    with open('artifacts/data/processed/y_train.pkl', 'rb') as handle:
        y_train = pickle.load(handle)

    logger = get_logger()
    with Timer("Running Model Experiments (BayesSearchCV)", logger):
        bayes_search = BayesSearchCV(
            estimator=css.create_pipeline(data=x_train),
            search_spaces=css.create_search_space(iterations=5),
            cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=42),
            scoring='roc_auc',
            refit=True,
            return_train_score=False,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        bayes_search.fit(x_train, y_train)
        results = hlp.sklearn_eval.MLExperimentResults.from_sklearn_search_cv(
            searcher=bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV',
            parameter_name_mappings=css.get_search_space_mappings(),
        )

    timestamp = f'{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
    file_name = f'multi-model-BayesSearchCV-{timestamp}'

    yaml_file_name = f'artifacts/models/experiments/{file_name}.yaml'
    model_file_name = f'artifacts/models/experiments/{file_name}_best_estimator.pkl'

    logger.info(f"Saving results of BayesSearchCV to: `{yaml_file_name}`")
    results.to_yaml_file(yaml_file_name=yaml_file_name)

    logger.info(f"Saving the best_estimator of BayesSearchCV to: `{model_file_name}`")
    with open(model_file_name, 'wb') as handle:
        pickle.dump(bayes_search.best_estimator_, handle)

    logger.info(f"Best Score: {bayes_search.best_score_}")
    logger.info(f"Best Params: {bayes_search.best_params_}")

    with open('artifacts/models/experiments/new_results.txt', 'w') as text_file:
        text_file.writelines(file_name)


if __name__ == '__main__':
    main()
