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
    def __init__(self, tr_input_path: str, tr_output_path: str, te_input_path: str, te_output_path: str, is_trainonly: bool = False) -> None:
        self.tr_input_path = tr_input_path
        self.tr_output_path = tr_output_path
        self.te_input_path = te_input_path
        self.te_output_path = te_output_path
        self.is_trainonly = is_trainonly

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
    
    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        データを読み込む

        :return: データフレーム
        """
        # ファイル拡張子を取得
        file_extension = input_path.split('.')[-1].lower()

        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            df = pd.read_pickle(input_path)
        elif file_extension == 'csv':
            df = pd.read_csv(input_path)
        elif file_extension == 'xlsx':
            df = pd.read_excel(input_path)
        elif file_extension == 'parquet':
            df = pd.read_parquet(input_path)
        elif file_extension == 'jsol':
            df = pd.read_json(input_path, lines=True)
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet.")

        return df
    
    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        データフレームをjson linesで保存する
        """
        df.to_json(output_path, force_ascii=False, lines=True, orient='records')

    def forward(self) -> None:
        logger.info(f'preprocess start')

        # ファイルの読込
        df_train = self.load_data(self.tr_input_path)
        tr_x, tr_y = df_train.drop(config.target_column, axis=1), df_train[config.target_column]
        
        # 値やカラム名の文字列を正規化（特徴量のみに適用）
        tr_x = self.normalize_string_of_df(tr_x)
        
        # 不要なカラムを削除（特徴量のみに適用）
        tr_x = column_cleaner.fit_transform(tr_x)

        # 全データの数値化
        tr_x = numeric_converter.fit_transform(tr_x)

        # 訓練データの特徴量と目的変数結合
        tr_xy = pd.merge(tr_x, tr_y, how='inner', left_index=True, right_index=True)

        # ファイルの書き出し
        self.save_data(tr_xy, self.tr_output_path)

        if self.is_trainonly == False:
            # ファイルの読込
            df_test = self.load_data(self.te_input_path)
            te_x = df_test
            
            # 値やカラム名の文字列を正規化（特徴量のみに適用）
            te_x = self.normalize_string_of_df(te_x)
            
            # 不要なカラムを削除（特徴量のみに適用）
            te_x = column_cleaner.transform(te_x)

            # 全データの数値化
            te_x = numeric_converter.transform(te_x)

            # ファイルの書き出し
            self.save_data(te_x, self.te_output_path)

        logger.info(f'preprocess end')
