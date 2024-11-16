import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import torch
import optuna
from typing import Callable, Union
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
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

    def objective_lgb(self, trial, tr_x: pd.DataFrame, tr_y: pd.Series, va_x: pd.DataFrame, va_y: pd.Series) -> float:
        lgb_params_range = {
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 31, 512),
            'min_data_in_leaf': trial.suggest_int('lgb_min_data_in_leaf', 1, 50),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lgb_lambda_l1', 0, 50),
            'lambda_l2': trial.suggest_float('lgb_lambda_l2', 0, 50)
            # 'device': 'gpu'
        }

        xgb_params_range = {
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 10),
            'random_state': config.cv_seed
            # 'tree_method': 'gpu_hist',
        }
    
        cat_params_range = {
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('cat_depth', 2, 10),
            'iterations': trial.suggest_int('cat_iterations', 100, 500),
            'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 0, 10),
            'random_seed': config.cv_seed,
            'verbose': 0
            # 'task_type': 'GPU'
        }
        
        lgb_model = LGBMRegressor(**lgb_params_range, random_state=config.cv_seed, verbose=-1, n_estimators=300)
        xgb_model = XGBRegressor(**xgb_params_range)
        cat_model = CatBoostRegressor(**cat_params_range)

        voting_model = VotingRegressor(
                estimators=[
                     ('lightgbm', lgb_model)
                    ,('xgboost', xgb_model)
                    ,('catboost', cat_model)
                ]
            )

        voting_model.fit(tr_x, tr_y)
        va_y_pred = voting_model.predict(va_x)
        
        # rmse = mean_squared_error(va_y, va_y_pred, squared=False)
        qwk = quadratic_weighted_kappa(va_y.round().astype(int), va_y_pred.round().astype(int))

        return -qwk
    
    def parameter_tuning(self, all_x: pd.DataFrame, all_y: pd.Series, all_group: pd.Series, n_trials: int = 100):
        rmse_score_list = []
        best_params_list = []

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))

        if config.group_column is None:
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=config.tuning_seed)
            for i_fold, (tr_idx, va_idx) in enumerate(kfold.split(all_x, all_y)):
                tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
                va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

                # Optunaでのハイパーパラメータチューニング
                study.optimize(lambda trial: self.objective_lgb(trial, tr_x, tr_y, va_x, va_y), n_trials=n_trials)

                # 各フォールドのスコアとパラメータを記録
                rmse_score_list.append(study.best_trial.value)
                best_params_list.append(study.best_params)
        else:
            kfold = ShuffledGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=config.tuning_seed)
            for i_fold in range(self.n_splits):
                tr_idx, va_idx = list(kfold.split(all_x, all_y, all_group))[i_fold]
                tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
                va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

                study.optimize(lambda trial: self.objective_lgb(trial, tr_x, tr_y, va_x, va_y), n_trials=n_trials)
                rmse_score_list.append(study.best_trial.value)
                best_params_list.append(study.best_params)
        
        best_trial = np.argmin(rmse_score_list)
        best_params = best_params_list[best_trial]

        return best_params