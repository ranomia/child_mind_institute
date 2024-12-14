import os
import platform
import matplotlib as mpl
import pandas as pd

class Config:
    def __init__(self):
        self.set_font()

        self._tuning_seed = 10
        self._cv_seed = 150
        self._train_file_path = '../data/input/train_extracted_密度_かさ.jsonl'
        self._test_file_path = '../data/input/train_extracted_密度_かさ.jsonl'
        self._train_preprocessed_file_path = '../data/intermediate/train.jsonl'
        self._test_preprocessed_file_path = '../data/intermediate/test.jsonl'
        self._train_feature_engineering_file_path = '../data/intermediate/train_fe.jsonl'
        self._test_feature_engineering_file_path = '../data/intermediate/test_fe.jsonl'
        self._train_missing_value_handled_file_path = '../data/intermediate/train_mvh.jsonl'
        self._test_missing_value_handled_file_path = '../data/intermediate/test_mvh.jsonl'
        self._train_column_cleaned_file_path = '../data/intermediate/train.parquet'
        self._test_column_cleaned_file_path = '../data/intermediate/test.parquet'
        self._target_column = 'sii'
        self._group_column = None # group無の場合はNoneを指定

    @staticmethod
    def set_font():
        os_name = platform.system()
        if os_name == 'Windows':  # Windows
            mpl.rcParams['font.family'] = 'BIZ UDGothic'
        elif os_name == 'Darwin':  # macOS
            mpl.rcParams['font.family'] = 'Helvetica'  # Mac用の適切なフォントを指定
        elif os_name == 'Linux':  # Linux
            mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
        else:
            raise EnvironmentError(f'Unsupported OS: {os_name}')
            
    @property
    def tuning_seed(self):
        return self._tuning_seed
    
    @property
    def cv_seed(self):
        return self._cv_seed
    
    @property
    def train_file_path(self):
        return self._train_file_path
    
    @property
    def test_file_path(self):
        return self._test_file_path
    
    @property
    def train_preprocessed_file_path(self):
        return self._train_preprocessed_file_path
    
    @property
    def test_preprocessed_file_path(self):
        return self._test_preprocessed_file_path
    
    @property
    def train_feature_engineering_file_path(self):
        return self._train_feature_engineering_file_path
    
    @property
    def test_feature_engineering_file_path(self):
        return self._test_feature_engineering_file_path
    
    @property
    def train_missing_value_handled_file_path(self):
        return self._train_missing_value_handled_file_path
    
    @property
    def test_missing_value_handled_file_path(self):
        return self._test_missing_value_handled_file_path
    
    @property
    def train_column_cleaned_file_path(self):
        return self._train_column_cleaned_file_path
    
    @property
    def test_column_cleaned_file_path(self):
        return self._test_column_cleaned_file_path
    
    @property
    def target_column(self):
        return self._target_column
    
    @property
    def group_column(self):
        return self._group_column
    
    def cv_seed_step_forward(self):
        self._cv_seed += 1
    
    def tuning_seed_step_forward(self):
        self._tuning_seed += 1