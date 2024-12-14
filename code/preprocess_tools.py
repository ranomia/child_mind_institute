import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Union

from config import Config

config = Config()

class ColumnCleaner:
    def __init__(self, input_path: str, output_path: str, columns_to_keep: list = []):
        self.input_path = input_path
        self.output_path = output_path
        self.columns_to_remove = []
        self.columns_to_keep = columns_to_keep

    def fit(self, df: pd.DataFrame) -> None:
        """
        学習データを基に不要なカラムを特定する
        """
        # 1種類の値しかないカラムを削除
        nunique = df.nunique()
        unique_cols = nunique[nunique == 1].index.tolist()

        # 完全に重複しているカラムを削除
        duplicate_cols = df.columns[df.T.duplicated()].tolist()

        # 相関係数が1のカラムを削除（数値データのみ）
        numeric_data = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper.columns if any(upper[column] == 1)]

        # ラベルを変えたら同じになるカラムを削除
        factorized_data = df.apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x)
        factorized_cols = df.columns[factorized_data.T.duplicated()].tolist()

        # 削除対象のカラムを記録
        self.columns_to_remove = list(set(unique_cols) | set(duplicate_cols) | set(high_corr_cols) | set(factorized_cols))
        
        # columns_to_keepに含まれるカラムは削除対象から除外
        self.columns_to_remove = [col for col in self.columns_to_remove if col not in self.columns_to_keep]

        # 削除対象のカラムを残した学習データのカラムを記録
        self.remaining_columns = df.drop(columns=self.columns_to_remove).columns.tolist()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fitで特定した不要なカラムを削除する
        """
        # 不要なカラムを削除し、残っているカラムのみをデータフレームとして返す
        return df[self.remaining_columns]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習データに対して一度にfitとtransformを行う
        """
        self.fit(df)
        return self.transform(df)
    
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
        elif file_extension == 'jsonl':
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
        """
        欠損値処理を実行
        """
        df = self.load_data(self.input_path)
        df_x, df_y = df.drop(columns=[config.target_column]), df[config.target_column]

        df_x = self.fit_transform(df_x)

        df = pd.concat([df_x, df_y], axis=1)

        self.save_data(df, self.output_path)
    
class NumericConverter:
    def __init__(self):
        self.label_encoders = {}  # 各列のLabelEncoderを格納する辞書

    def fit(self, df):
        """
        訓練データに対してfit処理を行い、カテゴリ列に対するLabelEncoderを作成する
        """
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                # 数値型データの場合は特に何もしない
                pass
            else:
                # カテゴリ型データの場合はLabelEncoderを作成してfit
                le = LabelEncoder()
                le.fit(df[column].astype(str))  # nanを考慮して文字列として変換
                self.label_encoders[column] = le

    def transform(self, df):
        """
        データに対してtransform処理を行い、数値列はゼロ埋め、カテゴリ列はLabelEncodingする
        """
        df_transformed = df.copy()  # 元のデータを変えないようにコピー
        
        for column in df_transformed.columns:
            if pd.api.types.is_numeric_dtype(df_transformed[column]):
                # 数値型データの場合はゼロ埋め処理
                df_transformed[column] = df_transformed[column].fillna(0)
            else:
                # カテゴリ型データの場合はLabelEncodingを適用
                if column in self.label_encoders:
                    le = self.label_encoders[column]
                    df_transformed[column] = le.transform(df_transformed[column].astype(str))
                else:
                    raise ValueError(f"{column}に対応するLabelEncoderがfitされていません。")

        return df_transformed

    def fit_transform(self, df):
        """
        訓練データに対してfitとtransformを同時に実行
        """
        self.fit(df)
        return self.transform(df)

class FeatureSelector:
    def __init__(self):
        self._feature_names = None
        self._feature_types = None

    def fit(self, X: pd.DataFrame) -> 'FeatureSelector':
        """
        データフレームの特徴量名と型を記録します
        
        Args:
            X: 入力データフレーム
        """
        self._feature_names = X.columns.tolist()
        self._feature_types = X.dtypes
        return self
    
    def get_feature_names_out(self, feature_types: Union[str, List[str]] = None) -> List[str]:
        """
        指定されたデータ型の特徴量名のリストを返します
        
        Args:
            feature_types: 取得したい特徴量の型（'int64', 'float64', 'category', 'object', 'bool'など）
            
        Returns:
            指定された型の特徴量名のリスト
        """
        if self._feature_names is None:
            raise ValueError("fit メソッドを先に実行してください")
            
        if feature_types is None:
            return self._feature_names
            
        if isinstance(feature_types, str):
            feature_types = [feature_types]
            
        selected_features = []
        for col in self._feature_names:
            if self._feature_types[col].name in feature_types:
                selected_features.append(col)
                
        return selected_features

    def select_dtypes(self, include=None, exclude=None) -> List[str]:
        """
        pandas.DataFrameのselect_dtypesと同様の機能を提供します
        
        Args:
            include: 含めたいデータ型
            exclude: 除外したいデータ型
            
        Returns:
            条件に合致する特徴量名のリスト
        """
        if self._feature_names is None:
            raise ValueError("fit メソッドを先に実行してください")
            
        temp_df = pd.DataFrame(columns=self._feature_names).astype(dict(zip(self._feature_names, self._feature_types)))
        return temp_df.select_dtypes(include=include, exclude=exclude).columns.tolist()