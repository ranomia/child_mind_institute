import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import torch
import optuna
from typing import Callable, Union

from model import Model
from model_lgb import ModelLGB, ModelLGBWrapper
from model_mlp import ModelMLP, ModelMLPWrapper
from config import Config
from util import ShuffledGroupKFold

config = Config()

class InnerCVRunner:
    def __init__(self, model_cls: Callable[[str, str, dict], Model]) -> None:
        self.n_repeats = 1
        self.n_splits = 3
        self.seed = config.tuning_seed
        self.model_cls = model_cls

    # パイプライン構築関数
    def build_pipeline(self, run_name: str, fold_name: str, params: dict, input_size=None, output_size=None) -> Pipeline:
        """
        標準化とモデルのパイプラインを作成する

        :param model: モデル
        :param params: モデルに適用するパラメータ
        :return: 標準化とモデルのパイプライン
        """
        if self.model_cls == ModelLGB:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # 標準化
                ('model', ModelLGBWrapper(model_cls=self.model_cls, run_name=run_name, fold_name=fold_name, params=params))  # モデル
            ])
        elif self.model_cls == ModelMLP:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # 標準化
                ('model', ModelMLPWrapper(model_cls=self.model_cls, run_name=run_name, fold_name=fold_name, params=params, input_size=input_size, output_size=output_size))  # モデル
            ])
        else:
            raise ValueError("model_cls must be ModelLGB or ModelMLP.")
        return pipeline

    def objective_lgb(self, trial, tr_x: pd.DataFrame, tr_y: pd.Series, va_x: pd.DataFrame, va_y: pd.Series) -> float:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 20.0, log=True),
            'num_threads': 4,
            'verbose': -1
        }
        model_pipe = self.build_pipeline('', '', params)
        
        model_pipe.fit(tr_x, tr_y)
        va_y_pred = model_pipe.predict(va_x)
        
        rmse = mean_squared_error(va_y, va_y_pred, squared=False)

        return rmse
    
    def objective_mlp(self, trial, tr_x: pd.DataFrame, tr_y: pd.Series, va_x: pd.DataFrame, va_y: pd.Series, input_size=None, output_size=None) -> float:
        params = {
            # 'objective': 'regression',
            # 'metric': 'rmse',
            # 'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'n_epochs': trial.suggest_int('n_epochs', 10, 100),
            'n_layers': trial.suggest_int('n_layers', 2, 3),
            'hidden_size': trial.suggest_int('hidden_size', 16, 512),
            'weight_init': trial.suggest_categorical('weight_init', ['xavier', 'he']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'MomentumSGD']),
            'scheduler': trial.suggest_categorical('scheduler', ['step', 'cosine', 'none']),
            'verbose': -1
        }
        # modelの初期化
        input_size = tr_x.shape[1]
        output_size = 1
        model_pipe = self.build_pipeline('', '', params, input_size, output_size)
        
        model_pipe.fit(tr_x.reset_index(drop=True).values, tr_y.reset_index(drop=True).values)
        va_y_pred = model_pipe.predict(va_x.reset_index(drop=True).values)
        
        rmse = mean_squared_error(va_y, va_y_pred, squared=False)

        return rmse
    
    def parameter_tuning(self, all_x: pd.DataFrame, all_y: pd.Series, all_group: pd.Series, n_trials: int = 100):
        rmse_score_list = []
        best_params_list = []

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))

        kfold = ShuffledGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=config.tuning_seed)
        for i_fold in range(self.n_splits):
            tr_idx, va_idx = list(kfold.split(all_x, all_y, all_group))[i_fold]
            tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
            va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

            if self.model_cls == ModelLGB:
                study.optimize(lambda trial: self.objective_lgb(trial, tr_x, tr_y, va_x, va_y), n_trials=n_trials)
            elif self.model_cls == ModelMLP:
                study.optimize(lambda trial: self.objective_mlp(trial, tr_x, tr_y, va_x, va_y), n_trials=n_trials)
            else:
                raise ValueError("model_cls must be ModelLGB or ModelMLP.")
            rmse_score_list.append(study.best_trial.value)
            best_params_list.append(study.best_params)
        
        best_trial = np.argmin(rmse_score_list)
        best_params = best_params_list[best_trial]

        return best_params

    def build_model(self, is_pipeline: bool, i_fold: Union[int, str], params: dict, input_size=None, output_size=None) -> Union[Model, Pipeline]:
        """
        クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :param params: チューニングされたパラメータ
        :return: モデルのインスタンス
        """
        fold_name = str(i_fold)
        # ラン名、fold、モデルのクラスからモデルを作成する
        if is_pipeline:
            model = self.build_pipeline(self.run_name, fold_name, params, input_size, output_size)
        else:
            model = self.model_cls('', '', params, input_size, output_size)
        return model