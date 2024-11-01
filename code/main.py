import numpy as np
import pandas as pd

from outer_cv_runner import OuterCVRunner
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
    # preprocess = Preprocess()
    # preprocess.forward()

    config = Config()

    # NLPによる学習・予測
    # outer_runner = OuterCVRunner('mlp', ModelMLP, params_mlp)
    # outer_runner.run_train_cv()
    # outer_runner.run_predict_cv()

    # lightGBMによる学習・予測
    outer_runner = OuterCVRunner('lgb', None, None)
    outer_runner.run_train_cv()
    outer_runner.run_predict_cv()
    # Submission.create_submission('lgb1')