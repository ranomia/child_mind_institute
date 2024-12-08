import numpy as np
import pandas as pd

from outer_cv_runner import OuterCVRunner
from preprocess import Preprocess
from user_column_extractor import UseColumnExtractor
from util import Submission
from config import Config

if __name__ == '__main__':

    config = Config()

    # 利用カラムの抽出
    # extractor = UseColumnExtractor(
    #      input_path = '../data/input/mixture_data_20240627_confirmed.xlsx'
    #     ,output_dir = '../data/input/'
    # )
    # extractor.forward()

    # 前処理
    # preprocess = Preprocess(
    #      tr_input_path = config.train_file_path
    #     ,tr_output_path = config.train_preprocessed_file_path
    #     ,te_input_path = config.test_file_path
    #     ,te_output_path = config.test_preprocessed_file_path
    #     ,is_trainonly = True
    # )
    # preprocess.forward()

    # lightGBMによる学習・予測
    for run_i in range(1):
        outer_runner = OuterCVRunner(
             run_name = 'lgb'+str(run_i)
            ,model_cls = None
            ,params_dict = {}
            ,cv_seed = config.cv_seed
            ,tuning_seed = config.tuning_seed
        )
        outer_runner.run_train_cv()
    #     outer_runner.run_predict_cv()

    #     config.cv_seed_step_forward()
    #     config.tuning_seed_step_forward()