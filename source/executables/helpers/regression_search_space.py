import numpy as np
import pandas as pd
from helpsk.sklearn_pipeline import CustomOrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from skopt.space import Categorical, Real, Integer
from xgboost import XGBRegressor


def create_pipeline(data: pd.DataFrame):
    pass

def create_search_space():
    return [({
'model': Categorical(categories=(ElasticNet(random_state=42),), prior=None),
'model__alpha': Real(low=1e-05, high=10, prior='log-uniform', transform='identity'),
'model__l1_ratio': Real(low=0, high=1, prior='uniform', transform='identity'),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent')), prior=[0.5, 0.25, 0.25]),
'prep__numeric__scaler__transformer': Categorical(categories=(StandardScaler(), MinMaxScaler()), prior=[0.65, 0.35]),
'prep__numeric__pca__transformer': Categorical(categories=(None, PCA(n_components='mle')), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'), CustomOrdinalEncoder()), prior=[0.65, 0.35])
}, 45),
({
'model': Categorical(categories=(ElasticNet(random_state=42),), prior=None),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(),), prior=None),
'prep__numeric__scaler__transformer': Categorical(categories=(StandardScaler(),), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None,), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'),), prior=None)
}, 1),
({
'model': Categorical(categories=(ExtraTreesRegressor(bootstrap=True, n_estimators=500, random_state=42),), prior=None),
'model__max_features': Real(low=0.01, high=0.95, prior='uniform', transform='identity'),
'model__max_depth': Integer(low=1, high=100, prior='uniform', transform='identity'),
'model__n_estimators': Integer(low=500, high=2000, prior='uniform', transform='identity'),
'model__min_samples_split': Integer(low=2, high=50, prior='uniform', transform='identity'),
'model__min_samples_leaf': Integer(low=1, high=50, prior='uniform', transform='identity'),
'model__max_samples': Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
'model__criterion': Categorical(categories=('squared_error',), prior=None),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent')), prior=[0.5, 0.25, 0.25]),
'prep__numeric__scaler__transformer': Categorical(categories=(None,), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None, PCA(n_components='mle')), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'), CustomOrdinalEncoder()), prior=[0.65, 0.35])
}, 45),
({
'model': Categorical(categories=(ExtraTreesRegressor(bootstrap=True, n_estimators=500, random_state=42),), prior=None),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(),), prior=None),
'prep__numeric__scaler__transformer': Categorical(categories=(None,), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None,), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'),), prior=None)
}, 1),
({
'model': Categorical(categories=(RandomForestRegressor(n_estimators=500, random_state=42),), prior=None),
'model__max_features': Real(low=0.01, high=0.95, prior='uniform', transform='identity'),
'model__max_depth': Integer(low=1, high=100, prior='uniform', transform='identity'),
'model__n_estimators': Integer(low=500, high=2000, prior='uniform', transform='identity'),
'model__min_samples_split': Integer(low=2, high=50, prior='uniform', transform='identity'),
'model__min_samples_leaf': Integer(low=1, high=50, prior='uniform', transform='identity'),
'model__max_samples': Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
'model__criterion': Categorical(categories=('squared_error',), prior=None),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent')), prior=[0.5, 0.25, 0.25]),
'prep__numeric__scaler__transformer': Categorical(categories=(None,), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None, PCA(n_components='mle')), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'), CustomOrdinalEncoder()), prior=[0.65, 0.35])
}, 45),
({
'model': Categorical(categories=(RandomForestRegressor(n_estimators=500, random_state=42),), prior=None),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(),), prior=None),
'prep__numeric__scaler__transformer': Categorical(categories=(None,), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None,), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'),), prior=None)
}, 1),
({
'model': Categorical(categories=(XGBRegressor(base_score=None, booster=None, colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=None,
             enable_categorical=False, eval_metric='rmse', gamma=None,
             gpu_id=None, importance_type=None, interaction_constraints=None,
             learning_rate=None, max_delta_step=None, max_depth=None,
             min_child_weight=None, missing=np.nan, monotone_constraints=None,
             n_estimators=500, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=42, reg_alpha=None, reg_lambda=None,
             scale_pos_weight=None, subsample=None, tree_method=None,
             use_label_encoder=False, validate_parameters=None, verbosity=None),), prior=None),
'model__max_depth': Integer(low=1, high=20, prior='log-uniform', transform='identity'),
'model__learning_rate': Real(low=0.01, high=0.3, prior='log-uniform', transform='identity'),
'model__n_estimators': Integer(low=500, high=2000, prior='uniform', transform='identity'),
'model__min_child_weight': Integer(low=1, high=50, prior='log-uniform', transform='identity'),
'model__subsample': Real(low=0.5, high=1, prior='uniform', transform='identity'),
'model__colsample_bytree': Real(low=0.5, high=1, prior='uniform', transform='identity'),
'model__colsample_bylevel': Real(low=0.5, high=1, prior='uniform', transform='identity'),
'model__reg_alpha': Real(low=0.0001, high=1, prior='log-uniform', transform='identity'),
'model__reg_lambda': Real(low=1, high=4, prior='log-uniform', transform='identity'),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent')), prior=[0.5, 0.25, 0.25]),
'prep__numeric__scaler__transformer': Categorical(categories=(None,), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None, PCA(n_components='mle')), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'), CustomOrdinalEncoder()), prior=[0.65, 0.35])
}, 45),
({
'model': Categorical(categories=(XGBRegressor(base_score=None, booster=None, colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=None,
             enable_categorical=False, eval_metric='rmse', gamma=None,
             gpu_id=None, importance_type=None, interaction_constraints=None,
             learning_rate=None, max_delta_step=None, max_depth=None,
             min_child_weight=None, missing=np.nan, monotone_constraints=None,
             n_estimators=500, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=42, reg_alpha=None, reg_lambda=None,
             scale_pos_weight=None, subsample=None, tree_method=None,
             use_label_encoder=False, validate_parameters=None, verbosity=None),), prior=None),
'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(),), prior=None),
'prep__numeric__scaler__transformer': Categorical(categories=(None,), prior=None),
'prep__numeric__pca__transformer': Categorical(categories=(None,), prior=None),
'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'),), prior=None)
}, 1)]
