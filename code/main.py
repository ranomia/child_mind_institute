import numpy as np
import pandas as pd

from outer_cv_runner import OuterCVRunner
from preprocess import Preprocess
from util import Submission
from config import Config

if __name__ == '__main__':

    # 前処理
    # preprocess = Preprocess()
    # preprocess.forward()

    config = Config()

    # NLPによる学習・予測
    # outer_runner = OuterCVRunner('mlp', ModelMLP, params_mlp)
    # outer_runner.run_train_cv()
    # outer_runner.run_predict_cv()

    # lightGBMによる学習・予測
    for run_i in range(100):
        outer_runner = OuterCVRunner('lgb'+str(run_i), None, None, config.cv_seed, config.tuning_seed)
        outer_runner.run_train_cv()
        # outer_runner.run_predict_cv()

        config.cv_seed_step_forward()
        config.tuning_seed_step_forward()
    # Submission.create_submission('lgb1')