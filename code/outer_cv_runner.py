import numpy as np
import pandas as pd
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from preprocess_tools import FeatureSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap

from typing import Callable, List, Optional, Tuple, Union

from inner_cv_runner import InnerCVRunner
from util import Logger, Util, ShuffledGroupKFold
from config import Config
from params import Params
from metric import quadratic_weighted_kappa

logger = Logger()
config = Config()
params_cls = Params()

class OuterCVRunner:
    def __init__(self, run_name: str, model_cls: None, params_dict: None, cv_seed: int, tuning_seed: int, model_dir: str):
        """
        コンストラクタ

        :param run_name: ランの名前
        :param model_cls: モデルのクラス
        :param params: ハイパーパラメータ
        :param n_fold: fold数
        :param dtype_dict: データ型の定義（trainに合わせてtestをロード）
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.params_dict = params_dict
        self.n_fold = 5
        self.dtype_dict = {}
        self.cv_seed = cv_seed
        self.tuning_seed = tuning_seed
        self.selector = FeatureSelector()
        self.model_dir = model_dir  # モデルディレクトリのパスを保存

    def train_fold(self, i_fold: Union[int, str], cv_results: dict):
        """
        クロスバリデーションにおける特定のfoldの学習・評価を行う

        他メソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号(全量で学習するときは'all'とする)
        :return: (モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア)のタプル
        """
        
        is_validation = i_fold != 'all'
        
        # 学習データの読込
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        # groupありの場合はgroup_columnを特徴量から削除
        if config.group_column is not None:
            train_group = train_x[config.group_column]
            train_x = train_x.drop(config.group_column, axis=1)

        # FeatureSelectorをfitする
        self.selector.fit(train_x)

        if is_validation:
            # 学習データ・バリデーションデータをセットする
            tr_idx, va_idx = self.load_index_fold_inner(i_fold)
            if config.group_column is None:
                tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
                va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
            else:
                tr_x, tr_y, tr_g = train_x.iloc[tr_idx], train_y.iloc[tr_idx], train_group.iloc[tr_idx]
                va_x, va_y, va_g = train_x.iloc[va_idx], train_y.iloc[va_idx], train_group.iloc[va_idx]

            # ハイパーパラメータのチューニングを行う
            if config.group_column is None:
                inner_runner = InnerCVRunner(tuning_seed=self.tuning_seed)
                tuned_params_dict_flatten = inner_runner.parameter_tuning(tr_x, tr_y, None, n_trials=100)
            else:
                inner_runner = InnerCVRunner(tuning_seed=self.tuning_seed)
                tuned_params_dict_flatten = inner_runner.parameter_tuning(tr_x, tr_y, tr_g, n_trials=100)
            
            tuned_params_dict = {
                'lightgbm': {},
                'xgboost': {},
                'catboost': {}
            }

            for model_type in tuned_params_dict_flatten.keys():
                for param_name, param_value in tuned_params_dict_flatten[model_type].items():
                    tuned_params_dict[model_type][param_name.replace(model_type+'_', '')] = param_value

            # params_dictの更新
            self.params_dict = self.update_params_dict(params_dict=self.params_dict, tuned_params_dict=tuned_params_dict)

            with open(f"{self.model_dir}/params_dict_{i_fold}.json", "w") as f:
                json.dump(self.params_dict, f)

            model_pipe = self.build_model(is_pipeline=True, params_dict=self.params_dict)
            
            ### pipelineで完結したいが、eval_setを使う場合は別で適用する必要がありそう。将来的に改善したい。
            # 前処理部分のみを先にfit
            preprocessor = model_pipe.named_steps['preprocessor']
            preprocessor.fit(tr_x)

            # 前処理を適用（eval_setでva_xを利用するため）
            tr_x_transformed = preprocessor.transform(tr_x)
            va_x_transformed = preprocessor.transform(va_x)

            model = model_pipe.named_steps['model']
            model.fit(
                 tr_x_transformed
                ,tr_y
                ,eval_set=[(tr_x_transformed, tr_y), (va_x_transformed, va_y)]
                ,eval_names=['train', 'valid']
                ,eval_metric='rmse'
                ,callbacks=[
                     lgb.early_stopping(stopping_rounds=30, verbose=False)
                ]
            )

            # 学習データ・バリデーションデータへの予測・評価を行う
            tr_y_pred = model_pipe.predict(tr_x)
            va_y_pred = model_pipe.predict(va_x)

            # 評価指標の算出
            cv_results['tr_rmse'].append(np.sqrt(mean_squared_error(tr_y, tr_y_pred)))
            cv_results['va_rmse'].append(np.sqrt(mean_squared_error(va_y, va_y_pred)))
            cv_results['tr_mae'].append(mean_absolute_error(tr_y, tr_y_pred))
            cv_results['va_mae'].append(mean_absolute_error(va_y, va_y_pred))
            cv_results['tr_qwk'].append(quadratic_weighted_kappa(tr_y, tr_y_pred.round(0).astype(int)))
            cv_results['va_qwk'].append(quadratic_weighted_kappa(va_y, va_y_pred.round(0).astype(int)))
            
            # 実験条件・結果の保存
            cv_results['tr_idx'].append(tr_idx)
            cv_results['va_idx'].append(va_idx)
            cv_results['tr_y'].append(tr_y)
            cv_results['va_y'].append(va_y)
            cv_results['tr_y_pred'].append(tr_y_pred)
            cv_results['va_y_pred'].append(va_y_pred)
            # cv_results['params'].append(best_params)

            # モデル、インデックス、予測値、評価を返す
            return model_pipe, cv_results
        else:
            # 学習データ全てで学習を行う
            model_pipe = self.build_model(is_pipeline=False, i_fold=i_fold, params=self.params)
            model_pipe.train_model(train_x, train_y)

            # モデルを返す
            return model_pipe, None, None, None

    def run_train_cv(self) -> None:
        """
        クロスバリデーションでの学習・評価を行う
        """
        # 学習データの読込（shap出力のため）
        train_x = self.load_x_train()

        # groupありの場合はgroup_columnを特徴量から削除
        if config.group_column is not None:
            train_group = train_x[config.group_column]
            train_x = train_x.drop(config.group_column, axis=1)

        logger.info(f'{self.run_name} - start training outer cv')

        cv_results = {
            'tr_idx': [],       # 各foldの学習データのインデックス
            'va_idx': [],       # 各foldの検証データのインデックス
            'group': [],             # 分析用のグループ
            'tr_rmse': [],       # 各foldの学習データに対するRMSE
            'va_rmse': [],       # 各foldの検証データに対するRMSE
            'tr_mae': [],        # 各foldの学習データに対するMAE
            'va_mae': [],        # 各foldの検証データに対するMAE
            'tr_qwk': [],        # 各foldの学習データに対するquadratic weighted kappa
            'va_qwk': [],        # 各foldの検証データに対するquadratic weighted kappa
            'tr_y': [],           # 各foldの学習データに対する予測値
            'va_y': [],           # 各foldの学習データの正解値
            'tr_y_pred': [],           # 各foldの検証データに対する予測値
            'va_y_pred': [],           # 各foldの検証データの正解値
            'params': []            # 各foldのモデルのハイパーパラメータ
        }

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, cv_results = self.train_fold(i_fold, cv_results)
            logger.info(f'{self.run_name} fold {i_fold} - end training - rmse score {cv_results["va_rmse"][i_fold]}')

            # モデルを保存する
            model.named_steps['model'].booster_.save_model(
                f'{self.model_dir}/model_{self.run_name}_{i_fold}.txt'
            )
        
            # 学習曲線のプロット
            eval_results = model.named_steps['model'].evals_result_
            plt.figure(figsize=(10, 6))
            plt.plot(eval_results['train']['rmse'], label='Training Loss')
            plt.plot(eval_results['valid']['rmse'], label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Learning Curve')
            plt.ylim([0, 1.0])
            plt.legend()
            plt.savefig(f'{self.model_dir}/lr_{self.run_name}_{i_fold}.png')

            # 残差のプロット
            tr_res = cv_results['tr_y'][i_fold] - cv_results['tr_y_pred'][i_fold]
            va_res = cv_results['va_y'][i_fold] - cv_results['va_y_pred'][i_fold]

            # y軸の範囲を計算
            res_min = min(tr_res.min(), va_res.min())
            res_max = max(tr_res.max(), va_res.max())

            # 残差プロット
            plt.figure(figsize=(14, 6))
            # tr
            plt.subplot(1, 2, 1)
            plt.scatter(cv_results['tr_y'][i_fold], tr_res, alpha=0.5)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.ylim(res_min*1.2, res_max*1.2)
            plt.xlabel('Real Values (Train)')
            plt.ylabel('Residuals (Train)')
            plt.title('Residual Plot (Train)')
            # va
            plt.subplot(1, 2, 2)
            plt.scatter(cv_results['va_y'][i_fold], va_res, alpha=0.5)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.ylim(res_min*1.2, res_max*1.2)
            plt.xlabel('Real Values (Validation)')
            plt.ylabel('Residuals (Validation)')
            plt.title('Residual Plot (Validation)')
            
            plt.tight_layout()
            plt.savefig(f'{self.model_dir}/res_{self.run_name}_{i_fold}.png')

            # shap
            train_data = train_x.iloc[cv_results['tr_idx'][i_fold]]
            valid_data = train_x.iloc[cv_results['va_idx'][i_fold]]
            
            # パイプラインの前処理を適用
            train_processed = model.named_steps['preprocessor'].transform(train_data)
            valid_processed = model.named_steps['preprocessor'].transform(valid_data)
            
            # 特徴量名を取得
            numeric_features = self.selector.get_feature_names_out(feature_types=['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'int8', 'float8'])
            categorical_features = self.selector.get_feature_names_out(feature_types=['category', 'object'])
            boolean_features = self.selector.get_feature_names_out(feature_types=['bool'])
            
            feature_names = (
                numeric_features +
                [f"{f}_encoded" for f in categorical_features] +
                [f"{f}_bool" for f in boolean_features]
            )
            
            # SHAPの計算
            explainer = shap.TreeExplainer(
                model.named_steps['model'],
                feature_names=feature_names,
                feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(valid_processed)
            
            # SHAPプロット
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values=shap_values,
                features=valid_processed,
                feature_names=feature_names,
                show=False
            )
            plt.title(f'SHAP Summary Plot for {self.run_name}_{i_fold}')
            plt.tight_layout()
            plt.savefig(f'{self.model_dir}/shap_{self.run_name}_{i_fold}.png')
            plt.close()

        # 各foldの結果をまとめる
        # va_idxes = np.concatenate(va_idxes)
        # order = np.argsort(va_idxes)
        # preds = np.concatenate(preds, axis=0)
        # preds = preds[order]

        logger.info(f'{self.run_name} - end training outer cv - rmse score {np.mean(cv_results["va_rmse"])}')
        logger.log_fold_scores('tr_rmse', cv_results['tr_rmse'])
        logger.log_fold_scores('va_rmse', cv_results['va_rmse'])
        logger.log_fold_scores('tr_qwk', cv_results['tr_qwk'])
        logger.log_fold_scores('va_qwk', cv_results['va_qwk'])

        # 予測結果の保存
        # Util.dump(preds, f'../model/{self.run_name}/pred/train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, cv_results['va_rmse'])

    def run_predict_cv(self) -> None:
        """
        クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction outer cv')

        test_x = self.load_x_test()

        # groupありの場合はgroup_columnを特徴量から削除
        if config.group_column is not None:
            test_x = test_x.drop(config.group_column, axis=1)

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold: {i_fold}')
            model = self.build_model(i_fold=i_fold, params={})
            model.load_model()
            pred = model.predict_model(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold: {i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f'../model/{self.run_name}/pred/test.pkl')

        logger.info(f'{self.run_name} - end prediction outer cv')

    def run_train_all(self) -> None:
        """学習データ全てで学習し、そのモデルを保存する"""
        logger.info(f'{self.run_name} - start training all')

        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """
        学習データ全てで学習したモデルにより、テストデータの予測を行う

        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 学習データすべてで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold=i_fold, params={})
        model.load_model()
        pred = model.predict_model(test_x)

        # 予測結果の保存
        Util.dump(pred, f'../model/{self.run_name}/pred/test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, is_pipeline: bool, params_dict: dict):
        """
        クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param params: チューニングされたパラメータ
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        if is_pipeline:
            lgb_model = LGBMRegressor(
                 **params_dict['lightgbm']
                ,random_state = self.cv_seed
                ,verbose = -1
                ,n_estimators = 5000
                # ,device = 'gpu'
                # ,gpu_device_id = 0
                ,num_threads = 4
            )

            # カラムの型に応じて異なる変換を適用するColumnTransformer
            numeric_features = self.selector.get_feature_names_out(feature_types=['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'int8', 'float8'])
            categorical_features = self.selector.get_feature_names_out(feature_types=['category', 'object'])
            boolean_features = self.selector.get_feature_names_out(feature_types=['bool'])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), numeric_features),
                    ('cat', Pipeline([
                        ('to_string', FunctionTransformer(lambda x: x.astype(str))),
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                    ]), categorical_features),
                    ('bool', Pipeline([
                        ('to_int', FunctionTransformer(lambda x: x.astype(float))),
                        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                        ('to_bool', FunctionTransformer(lambda x: x.astype(int)))
                    ]), boolean_features)
                ],
                remainder='passthrough'
            )

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', clone(lgb_model))
            ])
            return pipeline
        else:
            lgb_model = LGBMRegressor(
                 **params_dict['lightgbm']
                ,random_state = self.cv_seed
                ,verbose = -1
                ,n_estimators = 5000
                # ,device = 'gpu'
                # ,gpu_device_id = 0
                ,num_threads = 4
            )

            return lgb_model
    
    def load_x_train(self) -> pd.DataFrame:
        """
        学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = config.train_column_cleaned_file_path.split('.')[-1].lower()
        
        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            x_train = pd.read_pickle(config.train_column_cleaned_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'csv':
            x_train = pd.read_csv(config.train_column_cleaned_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'xlsx':
            x_train = pd.read_excel(config.train_column_cleaned_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'parquet':
            x_train = pd.read_parquet(config.train_column_cleaned_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'jsonl':
            x_train = pd.read_json(config.train_column_cleaned_file_path, lines=True).drop(config.target_column, axis=1)
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        # int -> float -> objectの順で変換を試行
        for column in x_train.columns:
            col_data = x_train[column]
            
            # まずbool型への変換を試みる
            unique_values = col_data.dropna().unique()
            if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False, '0', '1', 'True', 'False', 'true', 'false'}):
                x_train[column] = col_data.map({'True': True, 'true': True, '1': True, 1: True,
                                              'False': False, 'false': False, '0': False, 0: False}).astype(bool)
                continue

            # Noneやnanを含むかチェック
            has_null = col_data.isna().any()
            
            if not has_null:
                # Nullを含まない場合のみint変換を試みる
                try:
                    x_train[column] = pd.to_numeric(col_data, downcast='integer')
                    continue
                except (ValueError, TypeError):
                    pass
            
            # float変換を試みる
            try:
                x_train[column] = pd.to_numeric(col_data, downcast='float')
                continue
            except (ValueError, TypeError):
                pass

            # 上記の変換が全て失敗した場合はcategory型として扱う
            x_train[column] = col_data.astype('category')

        # group_columnをcategory型に変換
        if config.group_column is not None:
            x_train[config.group_column] = x_train[config.group_column].astype(str)

        self.dtype_dict = x_train.dtypes.to_dict()

        # 特殊文字を置き換えるコード例
        x_train.columns = x_train.columns.str.replace('"', '').str.replace("'", "").str.replace('-', '_').str.replace(' ', '').str.replace(',', '_').str.replace('.', '')
            
        return x_train
    
    def load_y_train(self) -> pd.Series:
        """
        学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = config.train_column_cleaned_file_path.split('.')[-1].lower()

        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            y_train = pd.read_pickle(config.train_column_cleaned_file_path)[config.target_column]
        elif file_extension == 'csv':
            y_train = pd.read_csv(config.train_column_cleaned_file_path)[config.target_column]
        elif file_extension == 'xlsx':
            y_train = pd.read_excel(config.train_column_cleaned_file_path)[config.target_column]
        elif file_extension == 'parquet':
            y_train = pd.read_parquet(config.train_column_cleaned_file_path)[config.target_column]
        elif file_extension == 'jsonl':
            y_train = pd.read_json(config.train_column_cleaned_file_path, lines=True)[config.target_column]
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        return y_train
    
    def load_x_test(self) -> pd.DataFrame:
        """
        テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = config.test_preprocessed_file_path.split('.')[-1].lower()

        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            x_test = pd.read_pickle(config.test_preprocessed_file_path)
        elif file_extension == 'csv':
            x_test = pd.read_csv(config.test_preprocessed_file_path)
        elif file_extension == 'xlsx':
            x_test = pd.read_excel(config.test_preprocessed_file_path)
        elif file_extension == 'parquet':
            x_test = pd.read_parquet(config.test_preprocessed_file_path)
        elif file_extension == 'jsonl':
            x_test = pd.read_json(config.test_preprocessed_file_path, lines=True)
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        # 訓練データのデータ型に合わせる
        for column, dtype in self.dtype_dict.items():
            if column in x_test.columns:
                x_test[column] = x_test[column].astype(dtype)
        
        # object型のカラムをcategory型に変換
        for col in x_test.select_dtypes(include=['object']).columns:
            x_test[col] = x_test[col].astype('category')

        # group_columnをcategory型に変換
        if config.group_column is not None:
            x_test[config.group_column] = x_test[config.group_column].astype(str)

        # 特殊文字を置き換えるコード例
        x_test.columns = x_test.columns.str.replace('"', '').str.replace("'", "").str.replace('-', '_').str.replace(' ', '').str.replace(',', '_').str.replace('.', '')

        return x_test

    def load_index_fold(self, i_fold: int) -> np.array:
        """
        クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if config.group_column == '':
            kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            idx_thisfold = list(kfold.split(train_x, train_y))[i_fold]
        else:
            kfold = ShuffledGroupKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            idx_thisfold = list(kfold.split(train_x, train_y, train_x[config.group_column]))[i_fold]
        return idx_thisfold
    
    def load_index_fold_inner(self, i_fold: int) -> Tuple[np.array, np.array, np.array]:
        """
        クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: fold_iにおけるtrain, tuning, validationのインデックスのタプル
        """
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if config.group_column is None:
            # tr+tuとvaの分割
            kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            tr_idx, va_idx = list(kfold.split(train_x, train_y))[i_fold]
        else:
            # tr+tuとvaの分割
            kfold = ShuffledGroupKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            tr_idx, va_idx = list(kfold.split(train_x, train_y, train_x[config.group_column]))[i_fold]

        return tr_idx, va_idx
    
    def update_params_dict(self, params_dict: dict, tuned_params_dict: dict) -> dict:
        """
        InnerCVRunnerでチューニングされたパラメータを上書きするためのプログラム
        元のparams_dictに定義されていない場合は、新規に追加する
        :param params_dict: 元のパラメータ
        :param tuned_params_dict: InnerCVRunnerでチューニングされたパラメータ
        :return: 更新されたパラメータ
        """

        
        for key, value in tuned_params_dict.items():
            # キーが存在しない場合は初期化
            if key not in params_dict:
                params_dict[key] = {}
            # params_dict を tuned_params_dict の内容で更新
            params_dict[key].update(value)
        
        return params_dict
