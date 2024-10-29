import os
import matplotlib as mpl
import pandas as pd

class Config:
    def __init__(self):
        self.set_font()

        self._tuning_seed = 10
        self._cv_seed = 150
        self._train_file_path = '../data/input/train.csv'
        self._test_file_path = '../data/input/test.csv'
        self._train_preprocessed_file_path = '../data/intermediate/train_preprocessed.pickle'
        self._test_preprocessed_file_path = '../data/intermediate/test_preprocessed.pickle'
        self._target_column = '密度_かさ'
        self._group_column = 'no' # group無の場合は''を指定

    @staticmethod
    def set_font():
        if os.name == 'nt':  # Windows
            mpl.rcParams['font.family'] = 'BIZ UDGothic'
        elif os.name == 'posix':  # Linux
            mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
        else:
            raise EnvironmentError('Unsupported OS')
            
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
    
