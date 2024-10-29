import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ColumnCleaner:
    def __init__(self):
        self.columns_to_remove = []

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
