import numpy as np
import pandas as pd
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
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
    def __init__(self, run_name: str, model_cls: None, params_dict: None, cv_seed: int, tuning_seed: int):
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
        self.n_fold = 10
        self.dtype_dict = {}
        self.cv_seed = cv_seed
        self.tuning_seed = tuning_seed

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

        if is_validation:
            # 学習データ・バリデーションデータをセットする
            tr_idx, tu_idx, va_idx = self.load_index_fold_inner(i_fold)
            if config.group_column is None:
                tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
                tu_x, tu_y = train_x.iloc[tu_idx], train_y.iloc[tu_idx]
                va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
            else:
                tr_x, tr_y, tr_g = train_x.iloc[tr_idx], train_y.iloc[tr_idx], train_group.iloc[tr_idx]
                tu_x, tu_y, tu_g = train_x.iloc[tu_idx], train_y.iloc[tu_idx], train_group.iloc[tu_idx]
                va_x, va_y, va_g = train_x.iloc[va_idx], train_y.iloc[va_idx], train_group.iloc[va_idx]

            # ハイパーパラメータのチューニングを行う
            if config.group_column is None:
                inner_runner = InnerCVRunner(tuning_seed=self.tuning_seed)
                tuned_params_dict_flatten = inner_runner.parameter_tuning(tu_x, tu_y, None, n_trials=10)
            else:
                inner_runner = InnerCVRunner(tuning_seed=self.tuning_seed)
                tuned_params_dict_flatten = inner_runner.parameter_tuning(tu_x, tu_y, tu_g, n_trials=100)
            
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

            with open(f"../model/params_dict_{i_fold}.json", "w") as f:
                json.dump(self.params_dict, f)

            model = self.build_model(is_pipeline=False, i_fold=i_fold, params_dict=self.params_dict)
            # model.fit(tr_x, tr_y)
            model.fit(
                 tr_x
                ,tr_y
                ,eval_set=[(tr_x, tr_y), (va_x, va_y)]
                # ,eval_metric=lambda y_true, y_pred: ('qwk', quadratic_weighted_kappa(y_true, y_pred), True)
                ,eval_metric='rmse'
                ,callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ]
            )

            # 学習データ・バリデーションデータへの予測・評価を行う
            tr_y_pred = model.predict(tr_x)
            va_y_pred = model.predict(va_x)

            # 評価指標の算出
            cv_results['tr_rmse'].append(root_mean_squared_error(tr_y, tr_y_pred))
            cv_results['va_rmse'].append(root_mean_squared_error(va_y, va_y_pred))
            cv_results['tr_mae'].append(mean_absolute_error(tr_y, tr_y_pred))
            cv_results['va_mae'].append(mean_absolute_error(va_y, va_y_pred))
            cv_results['tr_qwk'].append(quadratic_weighted_kappa(tr_y, tr_y_pred.round(0).astype(int)))
            cv_results['va_qwk'].append(quadratic_weighted_kappa(va_y, va_y_pred.round(0).astype(int)))
            
            # 実験条件・結果の保存
            cv_results['tr_idx'].append(tr_idx)
            cv_results['tu_idx'].append(tu_idx)
            cv_results['va_idx'].append(va_idx)
            cv_results['tr_y'].append(tr_y)
            cv_results['va_y'].append(va_y)
            cv_results['tr_y_pred'].append(tr_y_pred)
            cv_results['va_y_pred'].append(va_y_pred)
            # cv_results['params'].append(best_params)

            # モデル、インデックス、予測値、評価を返す
            return model, cv_results
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
            'tu_idx': [],       # 各foldのチューニングデータのインデックス
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
            model.booster_.save_model(f'../model/model_{self.run_name}_{i_fold}.txt')
        
            # 学習曲線のプロット
            eval_results = model.evals_result_
            plt.figure(figsize=(10, 6))
            plt.plot(eval_results['training']['rmse'], label='Training Loss')
            plt.plot(eval_results['valid_1']['rmse'], label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Learning Curve')
            plt.ylim([0, 1.0])
            plt.legend()
            plt.savefig(f'../model/lr_{self.run_name}_{i_fold}.png')

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
            plt.savefig(f'../model/res_{self.run_name}_{i_fold}.png')

            # shap
            explainer = shap.Explainer(model, train_x.iloc[cv_results['tr_idx'][i_fold]])
            shap_values = explainer.shap_values(train_x.iloc[cv_results['va_idx'][i_fold]])
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                 shap_values=shap_values
                ,features=train_x.iloc[cv_results['va_idx'][i_fold]]
                ,feature_names=train_x.columns
                ,show=False
            )
            plt.title(f'SHAP Summary Plot for {self.run_name}_{i_fold}')
            plt.savefig(f'../model/shap_{self.run_name}_{i_fold}.png')
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

    def build_model(self, is_pipeline: bool, i_fold: Union[int, str], params_dict: dict):
        """
        クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :param params: チューニングされたパラメータ
        :return: モデルのインスタンス
        """
        fold_name = str(i_fold)
        # ラン名、fold、モデルのクラスからモデルを作成する
        if is_pipeline:
            # model = self.build_pipeline(self.run_name, fold_name, params)
            lgb_model = LGBMRegressor(**params_dict['lightgbm'], random_state=self.cv_seed, verbose=-1, n_estimators=5000)

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', clone(lgb_model))
            ])
            return pipeline
        else:
            # model = self.model_cls(self.run_name, fold_name, params)
            lgb_model = LGBMRegressor(**params_dict['lightgbm'], random_state=self.cv_seed, verbose=-1, n_estimators=5000)

            return lgb_model
    
    def load_x_train(self) -> pd.DataFrame:
        """
        学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = config.train_preprocessed_file_path.split('.')[-1].lower()
        
        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            x_train = pd.read_pickle(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'csv':
            x_train = pd.read_csv(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'xlsx':
            x_train = pd.read_excel(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'parquet':
            x_train = pd.read_parquet(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'jsonl':
            x_train = pd.read_json(config.train_preprocessed_file_path, lines=True).drop(config.target_column, axis=1)
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        # int -> float -> objectの順で変換を試行
        for column in x_train.columns:
            col_data = x_train[column]
            
            # Try converting to integer
            try:
                x_train[column] = col_data.astype(int)
            except ValueError:
                # If integer conversion fails, try converting to float
                try:
                    x_train[column] = col_data.astype(float)
                except ValueError:
                    # If float conversion fails, keep as object
                    x_train[column] = col_data.astype(object)
        
        # object型のカラムをcategory型に変換
        for col in x_train.select_dtypes(include=['object']).columns:
            x_train[col] = x_train[col].astype('category')

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
        file_extension = config.train_preprocessed_file_path.split('.')[-1].lower()

        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            y_train = pd.read_pickle(config.train_preprocessed_file_path)[config.target_column]
        elif file_extension == 'csv':
            y_train = pd.read_csv(config.train_preprocessed_file_path)[config.target_column]
        elif file_extension == 'xlsx':
            y_train = pd.read_excel(config.train_preprocessed_file_path)[config.target_column]
        elif file_extension == 'parquet':
            y_train = pd.read_parquet(config.train_preprocessed_file_path)[config.target_column]
        elif file_extension == 'jsonl':
            y_train = pd.read_json(config.train_preprocessed_file_path, lines=True)[config.target_column]
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
            trtu_idx, va_idx = list(kfold.split(train_x, train_y))[i_fold]
            # trとtuの分割
            tr_idx, tu_idx = train_test_split(trtu_idx, test_size=0.2, random_state=55)
        else:
            # tr+tuとvaの分割
            kfold = ShuffledGroupKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            trtu_idx, va_idx = list(kfold.split(train_x, train_y, train_x[config.group_column]))[i_fold]
            # trとtuの分割
            tr_group, tu_group = train_test_split(train_x.iloc[trtu_idx][config.group_column].unique(), test_size=0.2, random_state=55)
            tr_idx = np.array(train_x[train_x[config.group_column].isin(tr_group)].index)
            tu_idx = np.array(train_x[train_x[config.group_column].isin(tu_group)].index)

        return tr_idx, tu_idx, va_idx
    
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
