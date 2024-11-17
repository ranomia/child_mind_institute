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
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from config import Config
from util import ShuffledGroupKFold
from metric import quadratic_weighted_kappa

config = Config()

class InnerCVRunner:
    def __init__(self) -> None:
        self.n_repeats = 1
        self.n_splits = 3
        self.seed = config.tuning_seed

    def objective(self, trial, model_type: str, tr_x: pd.DataFrame, tr_y: pd.Series, va_x: pd.DataFrame, va_y: pd.Series) -> float:
        if model_type == 'lightgbm':
            params_range = {
                'learning_rate': trial.suggest_float('lightgbm_learning_rate', 0.01, 0.05, log=True),
                'max_depth': trial.suggest_int('lightgbm_max_depth', 3, 10),
                'num_leaves': trial.suggest_int('lightgbm_num_leaves', 31, 256),
                'min_data_in_leaf': trial.suggest_int('lightgbm_min_data_in_leaf', 20, 100),
                'feature_fraction': trial.suggest_float('lightgbm_feature_fraction', 0.5, 0.8),
                'bagging_fraction': trial.suggest_float('lightgbm_bagging_fraction', 0.5, 0.8),
                'bagging_freq': trial.suggest_int('lightgbm_bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lightgbm_lambda_l1', 0, 100),
                'lambda_l2': trial.suggest_float('lightgbm_lambda_l2', 0, 100),
                'random_state': self.seed,
                'verbose': -1,
                'n_estimators': 300
                # 'device': 'gpu'
            }
            model = LGBMRegressor(**params_range)
            model.fit(
                 tr_x
                ,tr_y
                ,eval_set=[(va_x, va_y)]
                ,eval_metric=lambda y_true, y_pred: ('qwk', quadratic_weighted_kappa(y_true, y_pred), True)
                ,callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ]
            )


        va_y_pred = model.predict(va_x)
        
        # rmse = mean_squared_error(va_y, va_y_pred, squared=False)
        qwk = quadratic_weighted_kappa(va_y.round().astype(int), va_y_pred.round().astype(int))

        return -qwk
    
    def parameter_tuning(self, all_x: pd.DataFrame, all_y: pd.Series, all_group: pd.Series, n_trials: int = 100):
        model_types = ['lightgbm']
        best_params_all = {}

        for model_type in model_types:
            score_list = []
            best_params_list = []

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))

            if config.group_column is None:
                kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=config.tuning_seed)
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
                kfold = ShuffledGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=config.tuning_seed)
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