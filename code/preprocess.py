import numpy as np
import pandas as pd
import unicodedata
import re

from config import Config
from preprocess_tools import ColumnCleaner, NumericConverter
from util import Logger

config = Config()
column_cleaner = ColumnCleaner()
numeric_converter = NumericConverter()
logger = Logger()

class Preprocess:
    def __init__(self) -> None:
        pass

    def normalize_string_of_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームのカラム名と値を正規化し、日本語文字を含むUTF-8で読み込み可能な形式に変更。
        数値データについてはそのまま数値として扱う。

        :param df: 修正前のデータフレーム

        :return: 修正後のデータフレーム
        """
        def normalize_string(raw_string: str) -> str:
            # 正規化し、空白や改行をアンダースコアに変換し、小文字に変換
            normalized_string = unicodedata.normalize('NFKC', raw_string)
            normalized_string = re.sub(r'[^\w]', '_', normalized_string).lower()
            return normalized_string
        
        # カラム名を修正
        df.columns = [normalize_string(col) for col in df.columns]

        # 各セルの値を修正
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype): # オブジェクト型/カテゴリー型の列のみ処理
                df[col] = df[col].apply(lambda x: normalize_string(x) if isinstance(x, str) else x)
        
        return df
    
    def load_train(self) -> pd.DataFrame:
        """
        学習データを読み込む

        :return: 学習データ
        """
        # 学習データを読み込む
        if config.train_file_path.split('.')[-1] == 'pkl' or config.train_file_path.split('.')[-1] == 'pickle':
            train = pd.read_pickle(config.train_file_path)
        elif config.train_file_path.split('.')[-1] == 'csv':
            train = pd.read_csv(config.train_file_path)
        elif config.train_file_path.split('.')[-1] == 'excel':
            train = pd.read_excel(config.train_file_path)
            
        return train
    
    def load_test(self) -> pd.DataFrame:
        """
        テストデータを読み込む

        :return: テストデータ
        """
        # 学習データを読み込む
        if config.test_file_path.split('.')[-1] == 'pkl' or config.test_file_path.split('.')[-1] == 'pickle':
            test = pd.read_pickle(config.test_file_path)
        elif config.test_file_path.split('.')[-1] == 'csv':
            test = pd.read_csv(config.test_file_path)
        elif config.test_file_path.split('.')[-1] == 'excel':
            test = pd.read_excel(config.test_file_path)
            
        return test
    
    def save_train(self, train: pd.DataFrame) -> None:
        """
        学習データを保存する
        """
        train.to_pickle(config.train_preprocessed_file_path)

    def save_test(self, test: pd.DataFrame) -> None:
        """
        学習データを保存する
        """
        test.to_pickle(config.test_preprocessed_file_path)

    def forward(self) -> None:
        logger.info(f'preprocess start')
        # ファイルの読込
        df_train = self.load_train()
        df_test = self.load_test()

        tr_x, tr_y = df_train.drop(config.target_column, axis=1), df_train[config.target_column]
        te_x = df_test
        
        # 値やカラム名の文字列を正規化（特徴量のみに適用）
        tr_x = self.normalize_string_of_df(tr_x)
        te_x = self.normalize_string_of_df(te_x)
        
        # 不要なカラムを削除（特徴量のみに適用）
        tr_x = column_cleaner.fit_transform(tr_x)
        te_x = column_cleaner.transform(te_x)

        # 全データの数値化
        tr_x = numeric_converter.fit_transform(tr_x)
        te_x = numeric_converter.transform(te_x)

        # ファイルの書き出し
        self.save_train(pd.merge(tr_x, tr_y, how='inner', left_index=True, right_index=True))
        self.save_test(te_x)
        logger.info(f'preprocess end')
