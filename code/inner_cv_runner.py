import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import optuna
from typing import Callable, Union
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from config import Config
from util import ShuffledGroupKFold
from metric import quadratic_weighted_kappa
from preprocess_tools import FeatureSelector

config = Config()

class InnerCVRunner:
    def __init__(self, tuning_seed: int) -> None:
        self.n_repeats = 1
        self.n_splits = 2
        self.tuning_seed = tuning_seed
        self.selector = FeatureSelector()

    def objective(self, trial, model_type: str, tr_x: pd.DataFrame, tr_y: pd.Series, va_x: pd.DataFrame, va_y: pd.Series) -> float:
        if model_type == 'lightgbm':
            params_range = {
                'learning_rate': trial.suggest_float('lightgbm_learning_rate', 0.0001, 0.001, log=True),
                'reg_alpha': trial.suggest_float('lightgbm_reg_alpha', 1e-1, 100, log=True),
                'reg_lambda': trial.suggest_float('lightgbm_reg_labmda', 1e-1, 100, log=True),
                'num_leaves': trial.suggest_int('lightgbm_num_leaves', 4, 32),
                'colsample_bytree': trial.suggest_float('lightgbm_colsample_bytree', 0.6, 0.9),
                'subsample': trial.suggest_float('lightgbm_subsample', 0.6, 0.9),
                'subsample_freq': trial.suggest_int('lightgbm_subsample_freq', 3, 7),
                'min_child_samples': trial.suggest_int('lightgbm_min_child_samples', 100, 300),
                'max_depth': trial.suggest_int('lightgbm_max_depth', 2, 6),
                # 'device': 'gpu'
            }
            model_pipe = self.build_model(is_pipeline=True, params_range=params_range)
            ### pipelineで完結したいが、eval_setを使う場合は別で適用する必要がありそう。将来的に改善したい。
            # 前処理部分のみを先にfit
            preprocessor = model_pipe.named_steps['preprocessor']
            preprocessor.fit(tr_x)

            # 前処理を適用（eval_setでva_xを利用するため）
            tr_x_transformed = preprocessor.transform(tr_x)
            va_x_transformed = preprocessor.transform(va_x)

            model = model_pipe.named_steps['model']
            model.fit(
                 tr_x_transformed
                ,tr_y
                ,eval_set=[(tr_x_transformed, tr_y), (va_x_transformed, va_y)]
                ,eval_names=['train', 'valid']
                ,eval_metric='rmse'
                ,callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False)
                ]
            )


        elif model_type == 'xgboost':
            params_range = {
                'learning_rate': trial.suggest_float('xgboost_learning_rate', 0.001, 0.01, log=True),
                'max_depth': trial.suggest_int('xgboost_max_depth', 3, 5),
                'n_estimators': trial.suggest_int('xgboost_n_estimators', 100, 200),
                'subsample': trial.suggest_float('xgboost_subsample', 0.6, 0.8),
                'colsample_bytree': trial.suggest_float('xgboost_colsample_bytree', 0.6, 0.8),
                'reg_alpha': trial.suggest_float('xgboost_reg_alpha', 10, 30),
                'reg_lambda': trial.suggest_float('xgboost_reg_lambda', 10, 30),
                'random_state': self.tuning_seed
                # 'tree_method': 'gpu_hist',
            }
            model = XGBRegressor(**params_range)
            model.fit(tr_x, tr_y)
        elif model_type == 'catboost':
            params_range = {
                'learning_rate': trial.suggest_float('catboost_learning_rate', 0.001, 0.01, log=True),
                'depth': trial.suggest_int('catboost_depth', 3, 5),
                'iterations': trial.suggest_int('catboost_iterations', 100, 150),
                'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 10, 20),
                'random_seed': self.tuning_seed,
                'verbose': 0
                # 'task_type': 'GPU'
            }
            model = CatBoostRegressor(**params_range)
            model.fit(tr_x, tr_y)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        va_y_pred = model_pipe.predict(va_x)
        
        rmse = mean_squared_error(va_y, va_y_pred, squared=False)
        # qwk = quadratic_weighted_kappa(va_y.round().astype(int), va_y_pred.round().astype(int))

        # return -qwk
        return rmse
    
    def parameter_tuning(self, all_x: pd.DataFrame, all_y: pd.Series, all_group: pd.Series, n_trials: int = 100):
        # FeatureSelectorをfitする
        self.selector.fit(all_x)
        
        model_types = ['lightgbm']
        best_params_all = {}

        for model_type in model_types:
            score_list = []
            best_params_list = []

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.tuning_seed))

            if config.group_column is None:
                kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.tuning_seed)
                for i_fold, (tr_idx, va_idx) in enumerate(kfold.split(all_x, all_y)):
                    tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
                    va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

                    # Optunaでのハイパーパラメータチューニング
                    study.optimize(lambda trial: self.objective(trial, model_type, tr_x, tr_y, va_x, va_y), n_trials=n_trials)

                    # 各フォールドのスコアとパラメータを記録
                    score_list.append(study.best_trial.value)
                    best_params_list.append(study.best_params)
                
                # 最適なパラメータを保存
                best_index = np.argmin(score_list)
                best_params_all[model_type] = best_params_list[best_index]
            else:
                kfold = ShuffledGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.tuning_seed)
                for i_fold in range(self.n_splits):
                    tr_idx, va_idx = list(kfold.split(all_x, all_y, all_group))[i_fold]
                    tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
                    va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

                    study.optimize(lambda trial: self.objective(trial, model_type, tr_x, tr_y, va_x, va_y), n_trials=n_trials)
                    
                    score_list.append(study.best_trial.value)
                    best_params_list.append(study.best_params)

                best_index = np.argmin(score_list)
                best_params_all[model_type] = best_params_list[best_index]

        return best_params_all

    def build_model(self, is_pipeline: bool, params_range: dict):
        """
        クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :param params: チューニングされたパラメータ
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        if is_pipeline:
            # model = self.build_pipeline(self.run_name, fold_name, params)
            lgb_model = LGBMRegressor(
                 **params_range
                ,random_state = self.tuning_seed
                ,verbose = -1
                ,n_estimators = 5000
                # ,device = 'gpu'
                # ,gpu_device_id = 0
                ,num_threads = 4
            )

            # カラムの型に応じて異なる変換を適用するColumnTransformer
            numeric_features = self.selector.get_feature_names_out(feature_types=['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'int8', 'float8'])
            categorical_features = self.selector.get_feature_names_out(feature_types=['category', 'object'])
            boolean_features = self.selector.get_feature_names_out(feature_types=['bool'])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), numeric_features),
                    ('cat', Pipeline([
                        ('to_string', FunctionTransformer(lambda x: x.astype(str))),
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                    ]), categorical_features),
                    ('bool', Pipeline([
                        ('to_int', FunctionTransformer(lambda x: x.astype(float))),
                        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                        ('to_bool', FunctionTransformer(lambda x: x.astype(int)))
                    ]), boolean_features)
                ],
                remainder='passthrough'
            )

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', clone(lgb_model))
            ])
            return pipeline
        else:
            # model = self.model_cls(self.run_name, fold_name, params)
            lgb_model = LGBMRegressor(
                 **params_range
                ,random_state = self.tuning_seed
                ,verbose = -1
                ,n_estimators = 5000
                # ,device = 'gpu'
                # ,gpu_device_id = 0
                ,num_threads = 4
            )

            return lgb_model