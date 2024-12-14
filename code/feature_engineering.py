import re
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from config import Config

config = Config()

ASPHALT_CATEGORY_L_DICT = {
    "アスファルト60/80": "新アスファルト",
    "アスファルト80/100": "新アスファルト",
    "ストレートアスファルト40/60": "新アスファルト",
    "ストレートアスファルト60/80": "新アスファルト",
    "ストレートアスファルト80/100": "新アスファルト",
    "ストレートアスファルト150/200": "新アスファルト",
    "ストレートアスファルト200/300": "新アスファルト",
    "ラバーフィックス": "新アスファルト",
    "ガムファルトS": "新アスファルト",
    "ポリファルトS": "新アスファルト",
    "グランファルトR": "新アスファルト",
    "ポリマー改質Ⅰ型": "新アスファルト",
    "ポリマー改質アスファルトⅠ型": "新アスファルト",
    "ロードスター": "新アスファルト",
    "ニッシールGS": "新アスファルト",
    "ロードエース": "新アスファルト",
    "ポリマー改質Ⅱ型": "新アスファルト",
    "ポリファルトSS": "新アスファルト",
    "ポリマー改質アスファルトⅡ型": "新アスファルト",
    "ポリマー改質Ⅱ型ポリファルトSS": "新アスファルト",
    "ポリマー改質Ⅱ型レジフィックス": "新アスファルト",
    "レジフィックス": "新アスファルト",
    "レジフィッックス": "新アスファルト",
    "ポリマー改質Ⅱ型グランファルトART": "新アスファルト",
    "グランファルトART": "新アスファルト",
    "ポリマー改質Ⅱ型グランファルトHD": "新アスファルト",
    "ポリマー改質Ⅱ型ｸﾞﾗﾝﾌｧﾙﾄHD": "新アスファルト",
    "グランファルトHD": "新アスファルト",
    "キャリメックスHD": "新アスファルト",
    "スーパーバインダー": "新アスファルト",
    "エポックファルトD": "新アスファルト",
    "HRバインダー": "新アスファルト",
    "スーパーバインダ": "新アスファルト",
    "JSRローデックスU-Ⅱ": "新アスファルト",
    "改質Ⅲ型": "新アスファルト",
    "ニチファルトHDS": "新アスファルト",
    "HRスーパー": "新アスファルト",
    "レキファルトスーパー": "新アスファルト",
    "セナファルト": "新アスファルト",
    "タフファルトスーパー": "新アスファルト",
    "ドレイニッジECO": "新アスファルト",
    "パーミバインダー": "新アスファルト",
    "ポリマー改質アスファルトH型": "新アスファルト",
    "改質アスファルトH型": "新アスファルト",
    "ポリマー改質H型": "新アスファルト",
    "エバーフィックス": "新アスファルト",
    "シナヤカファルト": "新アスファルト",
    "カラークスSSーS": "新アスファルト",
    "脱色アスファルトⅡ型": "新アスファルト",
    "再生アスファルト40/60": "再生アスファルト",
    "再生アスファルト60/80": "再生アスファルト",
    "再生改質アスファルトⅠ型ガムファルトS": "再生アスファルト",
    "再生改質ｱｽﾌｧﾙﾄⅠ型ｶﾞﾑﾌｧﾙﾄS": "再生アスファルト",
    "再生ポリマー改質Ⅱ型": "再生アスファルト",
    "再生改質Ⅱ型": "再生アスファルト",
    "再生ポリマー改質Ⅱ型キャリメックスHD": "再生アスファルト",
    "再生改質Ⅱ型 RCアスファルト": "再生アスファルト",
    "再生改質Ⅱ型RCアスファルト": "再生アスファルト",
    "再生改質Ⅱ型ＲＣアスファルト": "再生アスファルト",
    "RCアスファルト": "再生アスファルト",
    "再生用改質アスファルトRCアスファルト": "再生アスファルト",
    "再生用改質アスファルトAFファルトH": "再生アスファルト",
    "AFファルトH": "再生アスファルト",
}
ASPHALT_CATEGORY_M_DICT = {
    "アスファルト60/80": "ストレートアスファルト",
    "アスファルト80/100": "ストレートアスファルト",
    "ストレートアスファルト40/60": "ストレートアスファルト",
    "ストレートアスファルト60/80": "ストレートアスファルト",
    "ストレートアスファルト80/100": "ストレートアスファルト",
    "ストレートアスファルト150/200": "ストレートアスファルト",
    "ストレートアスファルト200/300": "ストレートアスファルト",
    "ラバーフィックス": "改質アスファルト",
    "ガムファルトS": "改質アスファルト",
    "ポリファルトS": "改質アスファルト",
    "グランファルトR": "改質アスファルト",
    "ポリマー改質Ⅰ型": "改質アスファルト",
    "ポリマー改質アスファルトⅠ型": "改質アスファルト",
    "ロードスター": "改質アスファルト",
    "ニッシールGS": "改質アスファルト",
    "ロードエース": "改質アスファルト",
    "ポリマー改質Ⅱ型": "改質アスファルト",
    "ポリファルトSS": "改質アスファルト",
    "ポリマー改質アスファルトⅡ型": "改質アスファルト",
    "ポリマー改質Ⅱ型ポリファルトSS": "改質アスファルト",
    "ポリマー改質Ⅱ型レジフィックス": "改質アスファルト",
    "レジフィックス": "改質アスファルト",
    "レジフィッックス": "改質アスファルト",
    "ポリマー改質Ⅱ型グランファルトART": "改質アスファルト",
    "グランファルトART": "改質アスファルト",
    "ポリマー改質Ⅱ型グランファルトHD": "改質アスファルト",
    "ポリマー改質Ⅱ型ｸﾞﾗﾝﾌｧﾙﾄHD": "改質アスファルト",
    "グランファルトHD": "改質アスファルト",
    "キャリメックスHD": "改質アスファルト",
    "スーパーバインダー": "改質アスファルト",
    "エポックファルトD": "改質アスファルト",
    "HRバインダー": "改質アスファルト",
    "スーパーバインダ": "改質アスファルト",
    "JSRローデックスU-Ⅱ": "改質アスファルト",
    "改質Ⅲ型": "改質アスファルト",
    "ニチファルトHDS": "改質アスファルト",
    "HRスーパー": "改質アスファルト",
    "レキファルトスーパー": "改質アスファルト",
    "セナファルト": "改質アスファルト",
    "タフファルトスーパー": "改質アスファルト",
    "ドレイニッジECO": "改質アスファルト",
    "パーミバインダー": "改質アスファルト",
    "ポリマー改質アスファルトH型": "改質アスファルト",
    "改質アスファルトH型": "改質アスファルト",
    "ポリマー改質H型": "改質アスファルト",
    "エバーフィックス": "改質アスファルト",
    "シナヤカファルト": "改質アスファルト",
    "カラークスSSーS": "明色バインダ",
    "脱色アスファルトⅡ型": "明色バインダ",
    "再生アスファルト40/60": "再生ストレートアスファルト",
    "再生アスファルト60/80": "再生ストレートアスファルト",
    "再生改質アスファルトⅠ型ガムファルトS": "再生改質アスファルト",
    "再生改質ｱｽﾌｧﾙﾄⅠ型ｶﾞﾑﾌｧﾙﾄS": "再生改質アスファルト",
    "再生ポリマー改質Ⅱ型": "再生改質アスファルト",
    "再生改質Ⅱ型": "再生改質アスファルト",
    "再生ポリマー改質Ⅱ型キャリメックスHD": "再生改質アスファルト",
    "再生改質Ⅱ型 RCアスファルト": "再生改質アスファルト",
    "再生改質Ⅱ型RCアスファルト": "再生改質アスファルト",
    "再生改質Ⅱ型ＲＣアスファルト": "再生改質アスファルト",
    "RCアスファルト": "再生改質アスファルト",
    "再生用改質アスファルトRCアスファルト": "再生改質アスファルト",
    "再生用改質アスファルトAFファルトH": "再生改質アスファルト",
    "AFファルトH": "再生改質アスファルト",
}
ASPHALT_CATEGORY_S_DICT = {
    "アスファルト60/80": "ストレートアスファルト",
    "アスファルト80/100": "ストレートアスファルト",
    "ストレートアスファルト40/60": "ストレートアスファルト",
    "ストレートアスファルト60/80": "ストレートアスファルト",
    "ストレートアスファルト80/100": "ストレートアスファルト",
    "ストレートアスファルト150/200": "ストレートアスファルト",
    "ストレートアスファルト200/300": "ストレートアスファルト",
    "ラバーフィックス": "ポリマー改質アスファルトⅠ型",
    "ガムファルトS": "ポリマー改質アスファルトⅠ型",
    "ポリファルトS": "ポリマー改質アスファルトⅠ型",
    "グランファルトR": "ポリマー改質アスファルトⅠ型",
    "ポリマー改質Ⅰ型": "ポリマー改質アスファルトⅠ型",
    "ポリマー改質アスファルトⅠ型": "ポリマー改質アスファルトⅠ型",
    "ロードスター": "ポリマー改質アスファルトⅠ型",
    "ニッシールGS": "ポリマー改質アスファルトⅠ型",
    "ロードエース": "ポリマー改質アスファルトⅠ型",
    "ポリマー改質Ⅱ型": "ポリマー改質アスファルトⅡ型",
    "ポリファルトSS": "ポリマー改質アスファルトⅡ型",
    "ポリマー改質アスファルトⅡ型": "ポリマー改質アスファルトⅡ型",
    "ポリマー改質Ⅱ型ポリファルトSS": "ポリマー改質アスファルトⅡ型",
    "ポリマー改質Ⅱ型レジフィックス": "ポリマー改質アスファルトⅡ型",
    "レジフィックス": "ポリマー改質アスファルトⅡ型",
    "レジフィッックス": "ポリマー改質アスファルトⅡ型",
    "ポリマー改質Ⅱ型グランファルトART": "ポリマー改質アスファルトⅡ型",
    "グランファルトART": "ポリマー改質アスファルトⅡ型",
    "ポリマー改質Ⅱ型グランファルトHD": "ポリマー改質アスファルトⅡ型",
    "ポリマー改質Ⅱ型ｸﾞﾗﾝﾌｧﾙﾄHD": "ポリマー改質アスファルトⅡ型",
    "グランファルトHD": "ポリマー改質アスファルトⅡ型",
    "キャリメックスHD": "ポリマー改質アスファルトⅡ型",
    "スーパーバインダー": "ポリマー改質アスファルトⅡ型",
    "エポックファルトD": "ポリマー改質アスファルトⅡ型",
    "HRバインダー": "ポリマー改質アスファルトⅡ型",
    "スーパーバインダ": "ポリマー改質アスファルトⅡ型",
    "JSRローデックスU-Ⅱ": "ポリマー改質アスファルトⅡ型",
    "改質Ⅲ型": "ポリマー改質アスファルトⅢ型",
    "ニチファルトHDS": "ポリマー改質アスファルトⅢ型",
    "HRスーパー": "ポリマー改質アスファルトⅢ型",
    "レキファルトスーパー": "ポリマー改質アスファルトⅢ型-W",
    "セナファルト": "ポリマー改質アスファルトH型",
    "タフファルトスーパー": "ポリマー改質アスファルトH型",
    "ドレイニッジECO": "ポリマー改質アスファルトH型",
    "パーミバインダー": "ポリマー改質アスファルトH型",
    "ポリマー改質アスファルトH型": "ポリマー改質アスファルトH型",
    "改質アスファルトH型": "ポリマー改質アスファルトH型",
    "ポリマー改質H型": "ポリマー改質アスファルトH型",
    "エバーフィックス": "ポリマー改質アスファルトH型",
    "シナヤカファルト": "ポリマー改質アスファルト",
    "カラークスSSーS": "明色バインダ",
    "脱色アスファルトⅡ型": "脱色アスファルトⅡ型",
    "再生アスファルト40/60": "再生ストレートアスファルト",
    "再生アスファルト60/80": "再生ストレートアスファルト",
    "再生改質アスファルトⅠ型ガムファルトS": "再生ポリマー改質アスファルトⅠ型",
    "再生改質ｱｽﾌｧﾙﾄⅠ型ｶﾞﾑﾌｧﾙﾄS": "再生ポリマー改質アスファルトⅠ型",
    "再生ポリマー改質Ⅱ型": "再生ポリマー改質アスファルトⅡ型",
    "再生改質Ⅱ型": "再生ポリマー改質アスファルトⅡ型",
    "再生ポリマー改質Ⅱ型キャリメックスHD": "再生ポリマー改質アスファルトⅡ型",
    "再生改質Ⅱ型 RCアスファルト": "再生ポリマー改質アスファルトⅡ型",
    "再生改質Ⅱ型RCアスファルト": "再生ポリマー改質アスファルトⅡ型",
    "再生改質Ⅱ型ＲＣアスファルト": "再生ポリマー改質アスファルトⅡ型",
    "RCアスファルト": "再生ポリマー改質アスファルトⅡ型",
    "再生用改質アスファルトRCアスファルト": "再生ポリマー改質アスファルトⅡ型",
    "再生用改質アスファルトAFファルトH": "再生ポリマー改質アスファルトⅡ型",
    "AFファルトH": "再生ポリマー改質アスファルトⅡ型",
}
MATERIAL_CLASSIFICATION_DICT = {
    "その他": ["他産業再生資材"],
    "フィラー": ["フィラー", "石粉"],
    "再生骨材": ["再生骨材r20_13", "再生骨材r13_5", "再生骨材r13_0", "再生骨材r5_0"],
    "砂": ["砕砂", "粗砂", "細砂", "特砕砂", "スクリーニングス"],
    "砂その他": ["砂その他"],
    "砕石": [
        "4号砕石",
        "5号砕石",
        "特5号砕石",
        "6号砕石",
        "特6号砕石",
        "7号砕石",
        "特7号砕石",
    ],
}
MATERIAL_SPHERE_SIZE_DICT = {  # 単位はmm
    "再生骨材r20_13": 20,  # 不明
    "再生骨材r13_0": 13,  # 不明
    "再生骨材r13_5": 13,  # 不明
    "再生骨材r5_0": 5,  # 不明
    "4号砕石": 25,  # 20-30
    "5号砕石": 16.5,  # 13-20
    "6号砕石": 9,  # 5-13
    "7号砕石": 3.75,  # 2.5-5
    "砕砂": 1.75,  # 1.0-2.5
    "粗砂": 0.75,  # 0.5 -1.0
    "細砂": 0.375,  # 0.25 - 0.5
    "特5号砕石": 25,
    "特6号砕石": 16.5,
    "特7号砕石": 9,
    "特砕砂": 1.75,
    "砂その他": 0.5,
    "スクリーニングス": 1.75,
    "他産業再生資材": 10,  # 不明
    "フィラー": 0.05,
    "石粉": 0.05,
}


class FeatureEngineering:

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.material_column_list_without_all = [
            "再生骨材r20_13",
            "再生骨材r13_0",
            "再生骨材r13_5",
            "再生骨材r5_0",
            "4号砕石",
            "5号砕石",
            "6号砕石",
            "7号砕石",
            "特5号砕石",
            "特6号砕石",
            "特7号砕石",
            "砕砂",
            "粗砂",
            "細砂",
            "特砕砂",
            "砂その他",
            "スクリーニングス",
            "他産業再生資材",
            "フィラー",
            "石粉",
        ]

    # ============= Asphalt ================
    def _add_asphalt_category(self, df):
        asphalt_name_column_list = [
            "再生アスファルト_材料の種類",
            "新アスファルト1_材料の種類",
            "新アスファルト2_材料の種類",
        ]
        for column in asphalt_name_column_list:
            df[f'{column.replace("_材料の種類", "")}_大分類'] = df[column].map(
                ASPHALT_CATEGORY_L_DICT
            )
            df[f'{column.replace("_材料の種類", "")}_中分類'] = df[column].map(
                ASPHALT_CATEGORY_M_DICT
            )
            df[f'{column.replace("_材料の種類", "")}_小分類'] = df[column].map(
                ASPHALT_CATEGORY_S_DICT
            )
        return df

    # FIXME : for pred, we have to use the training df results
    def _calc_new_asphalt_features(self, df):
        def __get_penetration_degree(x):
            if x != x:
                return np.nan
            elif 40 < x <= 60:
                return "60-80"
            elif 60 < x <= 80:
                return "60-80"
            elif 80 < x <= 100:
                return "80-100"
            elif 100 < x <= 120:
                return "100-120"
            elif 120 < x <= 150:
                return "120-150"
            elif 150 < x <= 200:
                return "150-200"
            elif 200 < x <= 300:
                return "200-300"
            else:
                return "others"

        def __get_extended_degree(x):
            if x != x:
                return np.nan
            elif 10 <= x < 100:
                return "10-100"
            else:
                return "others"

        df["fe_アスファルトの体積割合"] = (
            df["アスファルト量_内割_"] / df["アスファルトの密度_比重_"]
        )
        asphalt_column_list = ["新アスファルト1", "新アスファルト2"]
        for column in asphalt_column_list:
            density_col = f"{column}_密度_1_5_c_"
            result_col = f"fe_{column}_group_density"
            mean = df.groupby(f"{column}_小分類", as_index=False).agg(
                **{result_col: (density_col, "mean")}
            )
            df = pd.merge(df, mean, how="left", on=f"{column}_小分類")
            df[f"fe_{column}_針入度"] = (
                df[f"{column}_針入度_2_5_c_"].astype(float).map(__get_penetration_degree)
            )
            df[f"fe_{column}_伸度"] = (
                df[f"{column}_伸度_1_5_c_"]
                .map(
                    lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0]) if isinstance(x, str) and re.findall(r'\d+\.?\d*', str(x)) else (float(x) if pd.notnull(x) else 0)
                )
                .map(__get_extended_degree)
            )
        return df

    # ============= Material ================
    def _calc_material_ratio_by_category(self, df):
        for key in MATERIAL_CLASSIFICATION_DICT:
            ratio_column_name_list = [
                f"{i}_旧アス含む配合割合" for i in MATERIAL_CLASSIFICATION_DICT[key]
            ]
            volume_column_name_list = [
                f"{i}_材料_理論密度" for i in MATERIAL_CLASSIFICATION_DICT[key]
            ]
            density_column_name_list = [
                f"{i}_計算に用いる密度" for i in MATERIAL_CLASSIFICATION_DICT[key]
            ]
            df[f"fe_配合割合_{key}_sum"] = df.loc[:, ratio_column_name_list].sum(axis=1)
            df[f"fe_理論密度_{key}_mean"] = df.loc[:, volume_column_name_list].mean(
                axis=1
            )
            df[f"fe_理論密度_{key}_sum"] = df.loc[:, volume_column_name_list].sum(
                axis=1
            )
            df[f"fe_密度_{key}_mean"] = df.loc[:, density_column_name_list].mean(axis=1)
        return df

    def _calc_mass_flow_rate_feature(self, df):

        def __calculate_basic_gradation_features(df: pd.DataFrame) -> pd.DataFrame:
            """
            高速化された特徴量計算関数
            全値がNaNの行は計算せず、全特徴量にNaNを返す
            """

            def __create_bounded_interpolation(
                passing, sieve_sizes, min_bound=0, max_bound=100
            ):
                """制約付き補間関数の作成"""
                f_raw = interp1d(
                    passing[::-1],
                    sieve_sizes[::-1],
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                def bounded_interp(x):
                    x_bounded = np.clip(x, 0, max_bound)
                    y = f_raw(x_bounded)
                    return np.maximum(y, min_bound)

                return bounded_interp

            def __calculate_mean_size(sieve_sizes, diff_distribution):
                # 隣接するふるい目サイズの平均を計算
                sieve_sizes_arr = np.array(sieve_sizes)
                interval_means = (sieve_sizes_arr[1:] + sieve_sizes_arr[:-1]) / 2

                # 差分分布を確率分布として正規化
                prob_distribution = diff_distribution / np.sum(diff_distribution)

                # 期待値の計算
                mean_size = np.sum(interval_means * prob_distribution)
                return mean_size

            def __process_single_sample(passing, sieve_sizes):
                """1つのサンプルの特徴量を計算"""
                # 補間関数の作成
                f = __create_bounded_interpolation(passing, sieve_sizes)

                # 差分分布の計算
                diff_distribution = -np.diff(passing)

                # 1. 基本的なD値と関連指標
                d_values = {
                    "passing_d10": f(10),
                    "passing_d30": f(30),
                    "passing_d50": f(50),
                    "passing_d60": f(60),
                    "passing_d90": f(90),
                }

                # 基本的な係数の計算
                coefficient_features = {
                    "uniformity_coef": (
                        d_values["passing_d60"] / d_values["passing_d10"]
                        if d_values["passing_d10"] > 0
                        else np.nan
                    ),
                    "curvature_coef": (
                        (d_values["passing_d30"] ** 2)
                        / (d_values["passing_d60"] * d_values["passing_d10"])
                        if (d_values["passing_d10"] > 0 and d_values["passing_d60"] > 0)
                        else np.nan
                    ),
                }

                # 2. 累積分布の特徴
                cumulative_features = {
                    # 基本統計量
                    "passing_mean": np.mean(passing),
                    "passing_std": np.std(passing),
                    "passing_skew": stats.skew(passing),
                    "passing_kurtosis": stats.kurtosis(passing),
                    # 形状特徴
                    "slope_mean": np.mean(
                        np.abs(np.diff(passing) / np.diff(sieve_sizes))
                    ),
                    "slope_std": np.std(np.diff(passing) / np.diff(sieve_sizes)),
                    "curvature_mean": np.mean(np.diff(np.diff(passing))),
                    "auc": np.trapz(passing, sieve_sizes),  # 累積曲線の面積
                    "symmetry": (
                        d_values["passing_d90"]
                        + d_values["passing_d10"]
                        - 2 * d_values["passing_d50"]
                    )
                    / (d_values["passing_d90"] - d_values["passing_d10"]),
                }

                # 3. 差分分布の特徴
                diff_features = {
                    # 基本統計量
                    "diff_max": np.max(diff_distribution),
                    "diff_mean": np.mean(diff_distribution),
                    "diff_std": np.std(diff_distribution),
                    "diff_median": np.median(diff_distribution),
                    "diff_max_sieve_size": sieve_sizes[np.argmax(diff_distribution)],
                    # ピーク特性
                    "peak_count": len(
                        find_peaks(diff_distribution)[0]
                    ),  # scipy.signalのfind_peaksを使用
                    "peak_height": np.max(diff_distribution),
                    "peak_width": np.sum(
                        diff_distribution > np.mean(diff_distribution)
                    ),
                    # 分布の特徴
                    "diff_skew": stats.skew(diff_distribution),
                    "diff_kurtosis": stats.kurtosis(diff_distribution),
                    "diff_entropy": stats.entropy(
                        diff_distribution / np.sum(diff_distribution)
                    ),
                    "diff_cv": (
                        np.std(diff_distribution) / np.mean(diff_distribution)
                        if np.mean(diff_distribution) != 0
                        else np.nan
                    ),
                }

                # 4. 粒度構成の特徴
                composition_features = {
                    "fine_content": passing[-1],  # 最小ふるい通過率
                    "coarse_content": 100 - passing[0],  # 最大ふるい残留率
                    "mid_content": passing[len(passing) // 2],  # 中間粒度の通過率
                    "fine_peak_ratio": np.max(
                        diff_distribution[-len(diff_distribution) // 3 :]
                    )
                    / np.max(diff_distribution),
                    "coarse_peak_ratio": np.max(
                        diff_distribution[: len(diff_distribution) // 3]
                    )
                    / np.max(diff_distribution),
                    "mid_retention": np.mean(
                        diff_distribution[
                            len(diff_distribution)
                            // 3 : 2
                            * len(diff_distribution)
                            // 3
                        ]
                    ),
                    "mean_particle_size": __calculate_mean_size(
                        sieve_sizes, diff_distribution
                    ),
                }

                # 5. 連続性と分布関係
                relationship_features = {
                    "continuity": np.mean(
                        np.abs(np.diff(diff_distribution))
                    ),  # 隣接粒径での変化
                    "local_concentration": np.max(diff_distribution)
                    / np.mean(diff_distribution),  # 局所的な集中度
                    "distribution_balance": np.sum(
                        diff_distribution[: len(diff_distribution) // 2]
                    )
                    / np.sum(
                        diff_distribution[len(diff_distribution) // 2 :]
                    ),  # 粗粒/細粒バランス
                    "change_point_count": len(
                        find_peaks(np.abs(np.diff(passing)))[0]
                    ),  # 変化点の数
                }

                # すべての特徴量を結合
                result = {
                    **d_values,
                    **coefficient_features,
                    **cumulative_features,
                    **diff_features,
                    **composition_features,
                    **relationship_features,
                }
                return result

            sieve_sizes = [
                53.0,
                37.5,
                31.5,
                26.5,
                19.0,
                13.2,
                9.5,
                4.75,
                2.36,
                1.18,
                0.6,
                0.3,
                0.15,
                0.075,
            ]  # mm単位
            results = []

            # 適当な配列（生成するのは時間がかかるためあらかじめ定義しているsieve_sizesを流用）からfeature_namesを生成。特徴量が増えても対応可能に。
            feature_names = list(
                __process_single_sample(sieve_sizes, sieve_sizes).keys()
            )
            for idx, row in df.iterrows():
                # 行がすべてNaNかチェック
                if row.isna().all():
                    # すべてNaNの場合は、すべての特徴量をNaNとして返す
                    results.append({feature: np.nan for feature in feature_names})
                    continue

                # NaNでない場合は通常の計算を実行
                result = __process_single_sample(row.values, sieve_sizes)
                results.append(result)

            # 結果をDataFrameに変換
            features = pd.DataFrame(results, index=df.index)
            return features

        def __fillna_method(df):
            df = df.T.ffill()

            def fill_column(col):
                return col.fillna(col.max())

            return df.apply(fill_column, axis=0)

        material_column_list = self.material_column_list_without_all + ["合成"]
        for material in tqdm(material_column_list):

            flow_rate_column_list = [
                i
                for i in df.columns
                if re.fullmatch(rf"{material}_通過質量百分率.*", i)
            ]
            df[flow_rate_column_list] = __fillna_method(df[flow_rate_column_list]).T
            pp_features_df = __calculate_basic_gradation_features(
                df[flow_rate_column_list]
            ).astype(float)
            pp_features_df.columns = [
                f"fe_{material}_pp_{i}" for i in pp_features_df.columns
            ]

            mixture_flow_rate_column_list = [
                i
                for i in df.columns
                if re.fullmatch(rf"{material}_配合率_通過質量百分率.*", i)
            ]
            # 配合率を加味した場合 blending passing pct
            df[mixture_flow_rate_column_list] = __fillna_method(
                df[mixture_flow_rate_column_list]
            ).T
            bpp_features_df = __calculate_basic_gradation_features(
                df[mixture_flow_rate_column_list]
            ).astype(float)
            bpp_features_df.columns = [
                f"fe_{material}_bpp_{i}" for i in bpp_features_df.columns
            ]

            df = pd.concat(
                [
                    df,
                    pp_features_df,
                    bpp_features_df,
                ],
                axis=1,
            ).drop(flow_rate_column_list + mixture_flow_rate_column_list, axis=1)
        return df

    def _calc_material_volume_feature(self, df):
        material_column_list = self.material_column_list_without_all
        mix_ratio_col_list = [f"{i}_旧アス含む配合割合" for i in material_column_list]
        material_density_col_list = [
            f"{i}_計算に用いる密度" for i in material_column_list
        ]
        # 供試体1000gあたりの各材料の質量
        result_div = (
            df[mix_ratio_col_list].div(df[material_density_col_list].values).fillna(0)
            * 10
        )
        result_div.columns = [f"fe_{i}_体積" for i in material_column_list]

        # 体積パーセントへ変換
        result_div_norm = result_div.div(result_div.sum(axis=1), axis=0) * 100
        result_div_norm.columns = [f"fe_{i}_体積pct" for i in material_column_list]

        result_div["供試体体積"] = result_div.sum(axis=1)

        df = pd.concat([df, result_div, result_div_norm], axis=1)
        return df

    def _calc_material_sphere_size_feature(self, df):

        def __calc_sphere_volume(r):
            return (4 / 3) * np.pi * r**3

        material_column_list = self.material_column_list_without_all
        df["fe_全粒径の加重平均"] = 0
        for material in material_column_list:
            df[f"fe_{material}_骨材1つの体積"] = (
                df[f"fe_{material}_pp_mean_particle_size"] / 20
            ).map(__calc_sphere_volume)
            df[f"fe_{material}_骨材の含有個数"] = (
                df[f"fe_{material}_体積pct"] / df[f"fe_{material}_骨材1つの体積"]
            ).fillna(0)

            # 粒径の加重平均を計算(最初に骨材体積そのままで重みづけし最後に供試体質量で割って割合化)
            df["fe_全粒径の加重平均"] += (
                df[f"fe_{material}_pp_mean_particle_size"]
                * df[f"fe_{material}_体積pct"]
            )
        df["fe_全粒径の加重平均"] = df["fe_全粒径の加重平均"] / 100

        return df.copy()

    # ============= Mixture ================
    # FIXME : for pred, we have to use the training df results
    def _calc_fuctory_features(self, df):
        feature_name_columns = ["計算に用いる密度", "材料_理論密度"]
        material_column_list = self.material_column_list_without_all + ["合成"]
        for feature_name in feature_name_columns:
            feature_column_list = [f"{i}_{feature_name}" for i in material_column_list]
            df_group = df.groupby("合材工場", as_index=False)[
                feature_column_list
            ].mean()
            df_group.columns = [
                f"fe_作成会社別_{i}" if i != "合材工場" else i for i in df_group.columns
            ]
            df = pd.merge(df, df_group, how="left", on="合材工場")
        return df

    def _extract_max_particle_size(self, df, column_name):
        def __extract_number(text):
            match = re.search(r"\((\d+)", text)
            if match:
                return match.group(1)
            else:
                return np.nan

        def __delete_useless_number(text):
            if text is np.nan:
                return np.nan
            text = re.sub(r"【[^】]*】", "", text)
            text = re.sub(r"\[[^\]]*\]", "", text)
            return text.replace("(ガラス10%)", "")

        def __zenkaku_to_hankaku(text):
            if text is np.nan:
                return np.nan
            return text.translate(
                str.maketrans("０１２３４５６７８９（）", "0123456789()")
            ).replace(" ", "")

        def __extract_numbers(text):
            match = re.search(r"\d+", text)
            if match:
                return float(match.group(0))
            else:
                return np.nan

        series = df[column_name].apply(__zenkaku_to_hankaku)
        series = series.apply(__delete_useless_number)
        processed_series = series.apply(__extract_numbers)
        return processed_series

    # FIXME : for pred, we have to use the training df results
    # WARNING : It does not work because of Kmens. So we do not use it.
    # def _get_mix_ratio_info(self, df):
    #     ratio_col_list = [
    #         "再生骨材R20-13_骨材のみの配合割合",
    #         "再生骨材R13-0_骨材のみの配合割合",
    #         "再生骨材R13-5_骨材のみの配合割合",
    #         "再生骨材R5-0_骨材のみの配合割合",
    #         "4号砕石_骨材のみの配合割合",
    #         "5号砕石_骨材のみの配合割合",
    #         "6号砕石_骨材のみの配合割合",
    #         "7号砕石_骨材のみの配合割合",
    #         "砕砂_骨材のみの配合割合",
    #         "粗砂_骨材のみの配合割合",
    #         "細砂_骨材のみの配合割合",
    #         "特5号砕石_骨材のみの配合割合",
    #         "特6号砕石_骨材のみの配合割合",
    #         "特7号砕石_骨材のみの配合割合",
    #         "特砕砂_骨材のみの配合割合",
    #         "砂その他_骨材のみの配合割合",
    #         "スクリーニングス_骨材のみの配合割合",
    #         "他産業再生資材_骨材のみの配合割合",
    #         "フィラー_骨材のみの配合割合",
    #         "石粉_骨材のみの配合割合",
    #     ]
    #     df.loc[:, ratio_col_list] = df.loc[:, ratio_col_list].fillna(0)
    #     df["fe_mix_ratio_cat"] = (
    #         df[ratio_col_list]
    #         .astype(int)
    #         .apply(lambda x: x.astype(str).str.zfill(2))
    #         .sum(axis=1)
    #     )
    #     # 本番学習用
    #     # kmeans = KMeans(n_clusters=len(df.大分類.unique()))
    #     # df['fe_mix_ratio_cluster'] = kmeans.fit_predict(df[ratio_col_list])
    #     # df['fe_mix_ratio_cluster_with_asphalt'] = kmeans.fit_predict(df[ratio_col_list+["アスファルト量（内割）"]])
    #     # 確認用
    #     loaded_kmeans = joblib.load("kmeans_model_without_asphalt.pkl")
    #     df["fe_mix_ratio_cluster"] = loaded_kmeans.predict(df[ratio_col_list])
    #     loaded_kmeans = joblib.load("kmeans_model_with_asphalt.pkl")
    #     df["fe_mix_ratio_cluster_with_asphalt"] = loaded_kmeans.predict(
    #         df[ratio_col_list + ["アスファルト量（内割）"]]
    #     )

    #     fe_mix_ratio_cluster_group = df.groupby("fe_mix_ratio_cluster", as_index=False)[
    #         "fe_全粒径の加重平均"
    #     ].mean()
    #     fe_mix_ratio_cluster_group.columns = [
    #         "fe_mix_ratio_cluster",
    #         "fe_全粒径の加重平均_mean_c1",
    #     ]
    #     fe_mix_ratio_cluster_with_asphalt_group = df.groupby(
    #         "fe_mix_ratio_cluster_with_asphalt", as_index=False
    #     )["fe_全粒径の加重平均"].mean()
    #     fe_mix_ratio_cluster_with_asphalt_group.columns = [
    #         "fe_mix_ratio_cluster_with_asphalt",
    #         "fe_全粒径の加重平均_mean_c2",
    #     ]
    #     df = pd.merge(
    #         df, fe_mix_ratio_cluster_group, how="left", on="fe_mix_ratio_cluster"
    #     )
    #     df = pd.merge(
    #         df,
    #         fe_mix_ratio_cluster_with_asphalt_group,
    #         how="left",
    #         on="fe_mix_ratio_cluster_with_asphalt",
    #     )
    #     return df

    # FIXME : for pred, we have to use the training df results
    def _encode_category_with_target_enc(self, df):
        def __binning(series, bins, labels=None):
            if isinstance(bins, int):
                # Create equal-width bins
                binned_series = pd.cut(series, bins=bins, labels=labels)
            elif isinstance(bins, list):
                # Create bins with specified edges
                binned_series = pd.cut(
                    series, bins=bins, labels=labels, include_lowest=True
                )
            else:
                raise ValueError("bins should be either an int or a list of bin edges")

            return binned_series

        def __target_encoder(df, target_column, encoding_column, drop_used_column):
            # いったん一番簡単に実装。

            if "アスファルト量_内割_" in encoding_column:
                df["アスファルト量_内割__binned"] = __binning(
                    df["アスファルト量_内割_"],
                    bins=[
                        1.92,
                        2.5,
                        3.0,
                        3.5,
                        4.0,
                        4.5,
                        5.0,
                        5.5,
                        6.0,
                        6.5,
                        7.0,
                        7.5,
                        8.0,
                        8.5,
                        9.0,
                        9.5,
                    ],
                )
                if type(encoding_column) != list:
                    encoding_column = "アスファルト量_内割__binned"
                else:
                    encoding_column = [
                        i.replace(
                            "アスファルト量_内割_", "アスファルト量_内割__binned"
                        )
                        for i in encoding_column
                    ]

            group = df.groupby(encoding_column, as_index=False)[target_column].mean()

            if type(encoding_column) != list:
                group.columns = [
                    encoding_column,
                    f"fe_{encoding_column}_{target_column}_target_enc",
                ]
            else:
                c_name = ""
                for i in encoding_column:
                    c_name += i + "_"
                group.columns = encoding_column + [
                    f"fe_{c_name}{target_column}_target_enc"
                ]

            df = pd.merge(df, group, how="left", on=encoding_column)
            if drop_used_column:
                df = df.drop(encoding_column, axis=1)

            return df

        target_column = config.target_column
        df = __target_encoder(
            df, target_column, "合材工場", drop_used_column=False
        )
        df = __target_encoder(
            df, target_column, "アスファルト量_内割_", drop_used_column=False
        )
        df = __target_encoder(
            df, target_column, "突固め回数", drop_used_column=False
        )
        df = __target_encoder(
            df, target_column, "大分類", drop_used_column=False
        )
        df = __target_encoder(
            df, target_column, ["大分類", "突固め回数"], drop_used_column=False
        )
        return df

    def check_output_df(self):
        df = self.df.copy()
        processed_df = self._add_asphalt_category(df)
        processed_df = self._calc_material_ratio_by_category(processed_df)
        processed_df = self._calc_fuctory_features(processed_df)
        processed_df = self._calc_new_asphalt_features(processed_df)
        processed_df = self._calc_mass_flow_rate_feature(processed_df)
        processed_df = self._calc_material_volume_feature(processed_df)
        processed_df = self._calc_material_sphere_size_feature(processed_df)
        # processed_df = self._get_mix_ratio_info(processed_df)
        processed_df = self._encode_category_with_target_enc(processed_df)
        processed_df["fe_particle_size"] = self._extract_max_particle_size(
            df, "混合物種類"
        )

        add_columns = sorted(set(processed_df.columns) - set(self.df.columns))

        print(
            f"Shpae {self.df.shape} -> {processed_df.shape}, ADD : {len(add_columns)} columns. detail↓"
        )
        # print(add_columns)
        # display(processed_df.loc[:, add_columns].info())
        return processed_df, add_columns

    def forward(self):
        # データ読み込み
        df = self.load_data(self.input_path)

        # 特徴量作成
        processed_df = self._add_asphalt_category(df)
        processed_df = self._calc_material_ratio_by_category(processed_df)
        processed_df = self._calc_fuctory_features(processed_df)
        processed_df = self._calc_new_asphalt_features(processed_df)
        processed_df = self._calc_mass_flow_rate_feature(processed_df)
        processed_df = self._calc_material_volume_feature(processed_df)
        processed_df = self._calc_material_sphere_size_feature(processed_df)
        # processed_df = self._get_mix_ratio_info(processed_df)
        processed_df = self._encode_category_with_target_enc(processed_df)
        processed_df["fe_particle_size"] = self._extract_max_particle_size(
            df, "混合物種類"
        )

        # 中間テーブルの作成
        add_columns = sorted(set(processed_df.columns) - set(df.columns))
        asphalt_name_column_list = [
            "再生アスファルト_材料の種類",
            "新アスファルト1_材料の種類",
            "新アスファルト2_材料の種類",
        ]
        fuctory_column_list = ["合材工場"]
        target_enc_column_list = [
            "アスファルト量_内割_",
            "アスファルト量_内割__binned",
            "突固め回数",
            "大分類",
        ]
        base_columns = (
            asphalt_name_column_list + fuctory_column_list + target_enc_column_list
        )
        need_columns = list(set(base_columns + add_columns))
        agg_df = (
            processed_df.loc[:, need_columns]
            .drop_duplicates(subset=base_columns)
            .reset_index(drop=True)
        )

        processed_df = processed_df.drop(
            ["合材工場", "アスファルト量_内割__binned", "突固め回数", "大分類"],
            axis=1,
        )

        # データ保存
        self.save_data(processed_df, self.output_path)

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