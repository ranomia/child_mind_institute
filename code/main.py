import numpy as np
import pandas as pd
import os

from outer_cv_runner import OuterCVRunner
from preprocess import Preprocess
from feature_engineering import FeatureEngineering
from user_column_extractor import UseColumnExtractor
from util import Submission
from config import Config
from missing_value_handler import MissingValueHandler
from preprocess_tools import ColumnCleaner

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

    # 特徴量作成
    # feature_engineering = FeatureEngineering(
    #      input_path = config.train_preprocessed_file_path
    #     ,output_path = config.train_feature_engineering_file_path
    # )
    # feature_engineering.forward()

    # 欠損値処理
    # missing_value_handler = MissingValueHandler(
    #      input_path = config.train_feature_engineering_file_path
    #     ,output_path = config.train_missing_value_handled_file_path
    # )
    # missing_value_handler.forward()

    # 不要カラムの削除
    # columncleaner = ColumnCleaner(
    #      input_path = config.train_missing_value_handled_file_path
    #     ,output_path = config.train_column_cleaned_file_path
    # )
    # columncleaner.forward()

    # lightGBMによる学習・予測
    for run_i in range(1):
        run_name = f'lgb_{datetime.datetime.now().strftime("%Y%m%d%H%M")}_{run_i}'
        
        # モデルディレクトリの作成
        model_dir = f'../model/{run_name}'
        os.makedirs(model_dir, exist_ok=True)
        
        outer_runner = OuterCVRunner(
             run_name = run_name
            ,model_cls = None
            ,params_dict = {}
            ,cv_seed = config.cv_seed
            ,tuning_seed = config.tuning_seed
            ,model_dir = model_dir
        )
        outer_runner.run_train_cv()
    #     outer_runner.run_predict_cv()

    #     config.cv_seed_step_forward()
    #     config.tuning_seed_step_forward()