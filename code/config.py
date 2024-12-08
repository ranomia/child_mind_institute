import os
import platform
import matplotlib as mpl
import pandas as pd

class Config:
    def __init__(self):
        self.set_font()

        self._tuning_seed = 10
        self._cv_seed = 150
        self._train_file_path = '../data/input/train_extracted_密度_かさ.jsol'
        self._test_file_path = '../data/input/train_extracted_密度_かさ.jsol'
        self._train_preprocessed_file_path = '../data/intermediate/train.jsonl'
        self._test_preprocessed_file_path = '../data/intermediate/test.jsonl'
        self._target_column = '密度_かさ'
        self._group_column = 'no__md使用_' # group無の場合はNoneを指定

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
    def target_column(self):
        return self._target_column
    
    @property
    def group_column(self):
        return self._group_column
    
    def cv_seed_step_forward(self):
        self._cv_seed += 1
    
    def tuning_seed_step_forward(self):
        self._tuning_seed += 1