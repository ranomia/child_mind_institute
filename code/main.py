import numpy as np
import pandas as pd

from model_lgb import ModelLGB
from model_mlp import ModelMLP
from outer_cv_runner import OuterCVRunner
from inner_cv_runner import InnerCVRunner
from preprocess import Preprocess
from util import Submission
from config import Config

if __name__ == '__main__':
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 30,
        'max_depth': 10,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'num_threads': 4
    }

    # 前処理
    preprocess = Preprocess()
    preprocess.forward()

    config = Config()
    input_size = pd.read_pickle(config.train_preprocessed_file_path).drop(config.target_column, axis=1).shape[1]

    params_mlp = {
        'input_size': input_size,
        'hidden_size': 100,
        'output_size': 1,
        'learning_rate': 0.001
    }

    # NLPによる学習・予測
    outer_runner = OuterCVRunner('mlp', ModelMLP, params_mlp)
    outer_runner.run_train_cv()
    outer_runner.run_predict_cv()

    # lightGBMによる学習・予測
    # outer_runner = OuterCVRunner('lgb', ModelLGB, params_lgb)
    # outer_runner.run_train_cv()
    # outer_runner.run_predict_cv()
    # Submission.create_submission('lgb1')