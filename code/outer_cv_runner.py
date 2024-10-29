import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from typing import Callable, List, Optional, Tuple, Union

from model import Model
from inner_cv_runner import InnerCVRunner
from util import quadratic_weighted_kappa, evaluate_predictions
from util import Logger, Util, ShuffledGroupKFold
from config import Config
from params import Params

logger = Logger()
config = Config()
params = Params()

class OuterCVRunner:
    def __init__(self, run_name: str, model_cls: Callable[[str, str, dict], Model], params: dict):
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
        self.params = params
        self.n_fold = 3
        self.dtype_dict = {}
    
    # パイプライン構築関数
    def build_pipeline(self, run_name: str, fold_name: str, params: dict):
        """
        標準化とモデルのパイプラインを作成する

        :param model: モデル
        :param params: モデルに適用するパラメータ
        :return: 標準化とモデルのパイプライン
        """
        if self.model_cls == ModelLGB:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # 標準化
                ('model', ModelLGBWrapper(model_cls=self.model_cls, run_name=run_name, fold_name=fold_name, params=params))  # モデル
            ])
        elif self.model_cls == ModelMLP:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # 標準化
                ('model', ModelMLPWrapper(model_cls=self.model_cls, run_name=run_name, fold_name=fold_name, params=params))  # モデル
            ])
        else:
            raise ValueError("model_cls must be ModelLGB or ModelMLP.")
        return pipeline

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
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
            tr_x, tr_y, tr_g = train_x.iloc[tr_idx], train_y.iloc[tr_idx], train_group.iloc[tr_idx]
            tu_x, tu_y, tu_g = train_x.iloc[tu_idx], train_y.iloc[tu_idx], train_group.iloc[tu_idx]
            va_x, va_y, va_g = train_x.iloc[va_idx], train_y.iloc[va_idx], train_group.iloc[va_idx]

            # ハイパーパラメータのチューニングを行う
            # inner_runner = InnerCVRunner(self.model_cls)
            # best_params = inner_runner.parameter_tuning(tu_x, tu_y, tu_g, n_trials=100)

            # 学習を行う
            model = self.build_model(is_pipeline=False, i_fold=i_fold, params=best_params)
            model.fit(tr_x, tr_y)

            # 学習データ・バリデーションデータへの予測・評価を行う
            tr_y_pred = model.predict(tr_x)
            va_y_pred = model.predict(va_x)

            # 評価指標の算出
            score_dict = {}
            score_dict['tr_rmse'] = root_mean_squared_error(tr_y, tr_y_pred)
            score_dict['va_rmse'] = root_mean_squared_error(va_y, va_y_pred)
            
            # モデル、インデックス、予測値、評価を返す
            return model, va_idx, va_y_pred, score_dict
        else:
            # 学習データ全てで学習を行う
            model_pipe = self.build_model(is_pipeline=True, i_fold=i_fold, params=self.params)
            model_pipe.train_model(train_x, train_y)

            # モデルを返す
            return model_pipe, None, None, None

    def run_train_cv(self) -> None:
        """
        クロスバリデーションでの学習・評価を行う
        """
        logger.info(f'{self.run_name} - start training outer cv')

        va_idxes = []
        va_preds = []
        scores = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, va_idx, va_pred, score_dict = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score_dict['va_rmse']}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する
            va_idxes.append(va_idx)
            va_preds.append(va_pred)
            scores.append(score_dict)
        
        # 各foldの結果をまとめる
        va_idxes = np.concatenate(va_idxes)
        order = np.argsort(va_idxes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training outer cv - score {np.mean(scores)}')

        # 予測結果の保存
        Util.dump(preds, f'../model/{self.run_name}/pred/train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

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

    def build_model(self, is_pipeline: bool, i_fold: Union[int, str], params: dict) -> Union[Model, Pipeline]:
        """
        クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :param params: チューニングされたパラメータ
        :return: モデルのインスタンス
        """
        fold_name = str(i_fold)
        # ラン名、fold、モデルのクラスからモデルを作成する
        if is_pipeline:
            model = self.build_pipeline(self.run_name, fold_name, params)
        else:
            # model = self.model_cls(self.run_name, fold_name, params)
            lgb_ins = LGBMRegressor(params.lgb_params, random_state=config.cv_seed, verbose=-1, n_estimators=300)
            xgb_ins = XGBRegressor(params.xgb_params)
            cat_ins = CatBoostRegressor(params.cat_params)

            voting_model = VotingRegressor(
                estimators=[
                     ('lightgbm', lgb_ins)
                    ,('xgboost', xgb_ins)
                    ,('catboost', cat_ins)
                ]
            )
        return voting_model
    
    def load_x_train(self) -> pd.DataFrame:
        """
        学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データを読み込む
        if config.train_preprocessed_file_path.split('.')[-1] == 'pkl' or config.train_preprocessed_file_path.split('.')[-1] == 'pickle':
            x_train = pd.read_pickle(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        elif config.train_file_path.split('.')[-1] == 'csv':
            x_train = pd.read_csv(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        elif config.train_file_path.split('.')[-1] == 'excel':
            x_train = pd.read_excel(config.train_preprocessed_file_path).drop(config.target_column, axis=1)
        
        # object型のカラムをcategory型に変換
        for col in x_train.select_dtypes(include=['object']).columns:
            x_train[col] = x_train[col].astype('category')

        # group_columnをcategory型に変換
        x_train[config.group_column] = x_train[config.group_column].astype(str)

        self.dtype_dict = x_train.dtypes.to_dict()
            
        return x_train
    
    def load_y_train(self) -> pd.Series:
        """
        学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 学習データを読み込む
        if config.train_preprocessed_file_path.split('.')[-1] == 'pkl' or config.train_preprocessed_file_path.split('.')[-1] == 'pickle':
            y_train = pd.read_pickle(config.train_preprocessed_file_path)[config.target_column]
        elif config.train_file_path.split('.')[-1] == 'csv':
            y_train = pd.read_csv(config.train_preprocessed_file_path)[config.target_column]
        elif config.train_file_path.split('.')[-1] == 'excel':
            y_train = pd.read_excel(config.train_preprocessed_file_path)[config.target_column]

        return y_train
    
    def load_x_test(self) -> pd.DataFrame:
        """
        テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        # テストデータを読み込む
        if config.test_preprocessed_file_path.split('.')[-1] == 'pkl' or config.test_preprocessed_file_path.split('.')[-1] == 'pickle':
            x_test = pd.read_pickle(config.test_preprocessed_file_path)
            # 訓練データのデータ型に合わせる
            for column, dtype in self.dtype_dict.items():
                if column in x_test.columns:
                    x_test[column] = x_test[column].astype(dtype)
        
        elif config.test_preprocessed_file_path.split('.')[-1] == 'csv':
            x_test = pd.read_csv(config.test_preprocessed_file_path, dtype=self.dtype_dict)
        elif config.test_preprocessed_file_path.split('.')[-1] == 'excel':
            x_test = pd.read_excel(config.test_preprocessed_file_path)
            # 訓練データのデータ型に合わせる
            for column, dtype in self.dtype_dict.items():
                if column in x_test.columns:
                    x_test[column] = x_test[column].astype(dtype)
        
        # object型のカラムをcategory型に変換
        for col in x_test.select_dtypes(include=['object']).columns:
            x_test[col] = x_test[col].astype('category')

        # group_columnをcategory型に変換
        x_test[config.group_column] = x_test[config.group_column].astype(str)

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
            kfold = KFold(n_splits=self.n_fold, shuffle=True, random_state=config.cv_seed)
            idx_thisfold = list(kfold.split(train_x, train_y))[i_fold]
        else:
            kfold = ShuffledGroupKFold(n_splits=self.n_fold, shuffle=True, random_state=config.cv_seed)
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

        if config.group_column == '':
            # tr+tuとvaの分割
            kfold = KFold(n_splits=self.n_fold, shuffle=True, random_state=config.cv_seed)
            trtu_idx, va_idx = list(kfold.split(train_x, train_y))[i_fold]
            # trとtuの分割
            tr_idx, tu_idx = train_test_split(trtu_idx, test_size=0.2, random_state=55)
        else:
            # tr+tuとvaの分割
            kfold = ShuffledGroupKFold(n_splits=self.n_fold, shuffle=True, random_state=config.cv_seed)
            trtu_idx, va_idx = list(kfold.split(train_x, train_y, train_x[config.group_column]))[i_fold]
            # trとtuの分割
            tr_group, tu_group = train_test_split(train_x.iloc[trtu_idx]['no'].unique(), test_size=0.2, random_state=55)
            tr_idx = np.array(train_x[train_x['no'].isin(tr_group)].index)
            tu_idx = np.array(train_x[train_x['no'].isin(tu_group)].index)

        return tr_idx, tu_idx, va_idx