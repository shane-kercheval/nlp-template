import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from skopt.space import Categorical, Real, Integer  # noqa
from xgboost import XGBClassifier
import helpsk as hlp

from helpsk.sklearn_pipeline import CustomOrdinalEncoder


def create_pipeline(data: pd.DataFrame) -> Pipeline:
    numeric_column_names = hlp.pandas.get_numeric_columns(data)
    non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)

    numeric_pipeline = Pipeline([
        # tune how we want to impute values
        # e.g. whether or not we want to impute (and how) or simply remove rows with missing values
        ('imputer', hlp.sklearn_pipeline.TransformerChooser()),
        # tune how we want to scale values
        # e.g. MinMax/Normalization/None
        ('scaler', hlp.sklearn_pipeline.TransformerChooser()),
        ('pca', hlp.sklearn_pipeline.TransformerChooser()),
    ])
    non_numeric_pipeline = Pipeline([
        # tune how we handle categoric values
        # e.g. One Hot, Custom-OrdinalEncoder
        ('encoder', hlp.sklearn_pipeline.TransformerChooser()),
    ])
    # associate numeric/non-numeric columns with corresponding pipeline
    transformations_pipeline = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_column_names),
        ('non_numeric', non_numeric_pipeline, non_numeric_column_names)
    ])
    full_pipeline = Pipeline([
        ('prep', transformations_pipeline),
        ('model', DummyClassifier())
    ])

    return full_pipeline


def create_search_space(iterations=50, random_state=42) -> list:
    return [
        (
            {
                'model':
                    Categorical(categories=[LogisticRegression(max_iter=1000, random_state=random_state)]),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=[SimpleImputer()]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[StandardScaler()]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=[OneHotEncoder(handle_unknown='ignore')])
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=[LogisticRegression(max_iter=1000, random_state=random_state)]),
                'model__C':
                    Real(low=1e-06, high=100, prior='log-uniform', transform='identity'),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=[
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')
                        ],
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[StandardScaler(), MinMaxScaler()], prior=[0.65, 0.35]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None, PCA(n_components='mle')]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=[
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ],
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=[LinearSVC(random_state=random_state)]),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=[SimpleImputer()]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[StandardScaler()]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=[OneHotEncoder(handle_unknown='ignore')])
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=[LinearSVC(random_state=random_state)]),
                'model__C':
                    Real(low=1e-06, high=100, prior='log-uniform', transform='identity'),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=[
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')
                        ],
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[StandardScaler(), MinMaxScaler()], prior=[0.65, 0.35]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None, PCA(n_components='mle')]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=[
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ],
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=[
                        ExtraTreesClassifier(bootstrap=True, n_estimators=500, random_state=random_state)
                    ]),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=[SimpleImputer()]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[None]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=[OneHotEncoder(handle_unknown='ignore')])
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=[
                        ExtraTreesClassifier(bootstrap=True, n_estimators=500, random_state=random_state)
                    ]),
                'model__max_features':
                    Real(low=0.01, high=0.95, prior='uniform', transform='identity'),
                'model__max_depth':
                    Integer(low=1, high=100, prior='uniform', transform='identity'),
                'model__n_estimators':
                    Integer(low=500, high=2000, prior='uniform', transform='identity'),
                'model__min_samples_split':
                    Integer(low=2, high=50, prior='uniform', transform='identity'),
                'model__min_samples_leaf':
                    Integer(low=1, high=50, prior='uniform', transform='identity'),
                'model__max_samples':
                    Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
                'model__criterion':
                    Categorical(categories=['gini', 'entropy']),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=[
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')
                        ],
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[None]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None, PCA(n_components='mle')]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=[
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ],
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=[RandomForestClassifier(n_estimators=500, random_state=random_state)]),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=[SimpleImputer()]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[None]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=[OneHotEncoder(handle_unknown='ignore')])
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=[RandomForestClassifier(n_estimators=500, random_state=random_state)]),
                'model__max_features':
                    Real(low=0.01, high=0.95, prior='uniform', transform='identity'),
                'model__max_depth':
                    Integer(low=1, high=100, prior='uniform', transform='identity'),
                'model__n_estimators':
                    Integer(low=500, high=2000, prior='uniform', transform='identity'),
                'model__min_samples_split':
                    Integer(low=2, high=50, prior='uniform', transform='identity'),
                'model__min_samples_leaf':
                    Integer(low=1, high=50, prior='uniform', transform='identity'),
                'model__max_samples':
                    Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
                'model__criterion':
                    Categorical(categories=['gini', 'entropy']),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=[
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')],
                        prior=[0.5, 0.25, 0.25]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[None]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None, PCA(n_components='mle')]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=[
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ],
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=[
                        XGBClassifier(
                            n_estimators=500,
                            eval_metric='logloss',
                            use_label_encoder=False,
                            random_state=random_state,
                        )
                    ]),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=[SimpleImputer()]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[None]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=[OneHotEncoder(handle_unknown='ignore')])
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=[
                        XGBClassifier(
                            n_estimators=500,
                            eval_metric='logloss',
                            use_label_encoder=False,
                            random_state=random_state,
                        )
                    ]),
                'model__max_depth':
                    Integer(low=1, high=20, prior='log-uniform', transform='identity'),
                'model__learning_rate':
                    Real(low=0.01, high=0.3, prior='log-uniform', transform='identity'),
                'model__n_estimators':
                    Integer(low=500, high=2000, prior='uniform', transform='identity'),
                'model__min_child_weight':
                    Integer(low=1, high=50, prior='log-uniform', transform='identity'),
                'model__subsample':
                    Real(low=0.5, high=1, prior='uniform', transform='identity'),
                'model__colsample_bytree':
                    Real(low=0.5, high=1, prior='uniform', transform='identity'),
                'model__colsample_bylevel':
                    Real(low=0.5, high=1, prior='uniform', transform='identity'),
                'model__reg_alpha':
                    Real(low=0.0001, high=1, prior='log-uniform', transform='identity'),
                'model__reg_lambda':
                    Real(low=1, high=4, prior='log-uniform', transform='identity'),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=[
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')],
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=[None]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=[None, PCA(n_components='mle')]),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=[
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()],
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
    ]


def get_search_space_mappings():
    """
    This function returns a dictionary, with the keys being the paths from the `sklearn.pipeline.Pipeline`
    returned by the `create_pipeline()` function (e.g. "prep__numeric__imputer") and transforms the
    path into a 'friendlier' value (e.g. "imputer"), returned as the value in the dictionary.

    The dictionary returned by this function can be used, for example, by passing it to the
    `parameter_name_mappings` parameter in the `MLExperimentResults.from_sklearn_search_cv()` function.
    This will allow the `MLExperimentResults` to use the friendlier names in the output (e.g. tables and
    graphs) and will make the output more readable.
    """
    mappings = {}
    for space in create_search_space():
        params = list(space[0].keys())
        for param in params:
            if param not in mappings:
                if param == 'model':
                    mappings[param] = param
                elif param.startswith('model__'):
                    mappings[param] = param.removeprefix('model__')
                elif param.endswith('__transformer'):
                    mappings[param] = param. \
                        removesuffix('__transformer'). \
                        removeprefix('prep__numeric__'). \
                        removeprefix('prep__non_numeric__')
                else:
                    mappings[param] = param

    ordered_mappings = {key: value for key, value in mappings.items() if not key.startswith('prep__')}
    ordered_mappings.update({key: value for key, value in mappings.items() if key.startswith('prep__')})
    return ordered_mappings
