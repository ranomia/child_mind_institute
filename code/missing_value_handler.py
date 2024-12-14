from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from feature_engine.imputation import AddMissingIndicator
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from typing import Dict, List, Optional, Union

from config import Config

config = Config()

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """欠損値を処理するためのカスタムトランスフォーマー
    
    欠損値の割合が高い特徴量の削除、欠損値の補完、欠損フラグの追加を行う
    """
    def __init__(
        self,
        input_path: str,
        output_path: str,
        high_missing_threshold: float = 0.5,
        normality_pvalue_threshold: float = 0.05
    ) -> None:
        """
        Args:
            high_missing_threshold: 削除する特徴量の欠損値割合の閾値
            normality_pvalue_threshold: 正規性検定のp値の閾値
        """
        self.input_path = input_path
        self.output_path = output_path
        self.high_missing_threshold = high_missing_threshold
        self.normality_pvalue_threshold = normality_pvalue_threshold
        self.imputers: Dict[str, SimpleImputer] = {}
        self.indicator: Optional[AddMissingIndicator] = None
        self.columns_to_drop: List[str] = []

    def _is_normal_distribution(self, data: pd.Series) -> bool:
        """Shapiro-Wilk検定を使用してデータが正規分布に従うかチェック

        Args:
            data: 検定対象のデータ

        Returns:
            bool: 正規分布に従う場合True
        """
        _, p_value = shapiro(data)
        return p_value > self.normality_pvalue_threshold

    def fit(self, tr_x: Union[pd.DataFrame, np.ndarray]) -> 'MissingValueHandler':
        """欠損値処理のパラメータを学習

        Args:
            train_x: 学習データ
            train_y: 目的変数(未使用)

        Returns:
            self: 学習済みのインスタンス
        """
        tr_x = pd.DataFrame(tr_x)
        self.columns_to_drop = []
        self.imputers = {}
        self.original_columns = tr_x.columns.tolist()  # 追加：元の列名を保存

        # 各特徴量に対して欠損値処理方法を決定
        for column in self.original_columns:  # 変更：元の列名を使用
            missing_ratio = tr_x[column].isnull().mean()

            # 欠損値が多い特徴量は削除
            if missing_ratio > self.high_missing_threshold:
                self.columns_to_drop.append(column)
            else:
                # 数値型の場合は正規性に応じて平均値/中央値で補完
                if pd.api.types.is_numeric_dtype(tr_x[column]):
                    if self._is_normal_distribution(tr_x[column].dropna()):
                        self.imputers[column] = SimpleImputer(strategy="mean")
                    else:
                        self.imputers[column] = SimpleImputer(strategy="median")
                # カテゴリ型の場合は最頻値で補完
                else:
                    self.imputers[column] = SimpleImputer(strategy="most_frequent")

                self.imputers[column].fit(tr_x[[column]])

        # 欠損フラグを追加するための処理を学習
        columns_for_indicator = [col for col in self.original_columns if col not in self.columns_to_drop]  # 変更：削除される列を除外
        self.indicator = AddMissingIndicator(variables=columns_for_indicator)
        self.indicator.fit(tr_x)

        return self

    def transform(self, tr_x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """欠損値処理を適用

        Args:
            tr_x: 変換対象のデータ

        Returns:
            pd.DataFrame: 変換後のデータ
        """
        tr_x = pd.DataFrame(tr_x).copy()
        
        # 元の列が全て存在することを確認
        missing_cols = set(self.original_columns) - set(tr_x.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in transform: {missing_cols}")

        # 欠損フラグを追加
        tr_x = self.indicator.transform(tr_x)

        # 欠損値が多い特徴量を削除
        tr_x.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

        # 欠損値を補完
        for column, imputer in self.imputers.items():
            tr_x[column] = imputer.transform(tr_x[[column]]).ravel()

        return tr_x
    
    def fit_transform(self, tr_x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        欠損値処理を実行
        """
        self.fit(tr_x)
        return self.transform(tr_x)

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

