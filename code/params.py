from config import Config

config = Config()

class Params:
    def __init__(self):
        self._lgb_params = {
            'learning_rate': 0.046,
            'max_depth': 12,
            'num_leaves': 478,
            'min_data_in_leaf': 13,
            'feature_fraction': 0.893,
            'bagging_fraction': 0.784,
            'bagging_freq': 4,
            'lambda_l1': 10,  # Increased from 6.59
            'lambda_l2': 0.01  # Increased from 2.68e-06
            # 'device': 'gpu'
        }

        self._xgb_params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,  # Increased from 0.1
            'reg_lambda': 5,  # Increased from 1
            'random_state': config.cv_seed
            # 'tree_method': 'gpu_hist',
        }
        
        self._cat_params = {
            'learning_rate': 0.05,
            'depth': 6,
            'iterations': 200,
            'random_seed': config.cv_seed,
            'verbose': 0,
            'l2_leaf_reg': 10  # Increase this value
            # 'task_type': 'GPU'
        }

    def get_lgb_params_range(self, trial):
        return {
            'learning_rate': trial.suggest_loguniform('lgb_learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 31, 512),
            'min_data_in_leaf': trial.suggest_int('lgb_min_data_in_leaf', 1, 50),
            'feature_fraction': trial.suggest_uniform('lgb_feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('lgb_bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_uniform('lgb_lambda_l1', 0, 50),
            'lambda_l2': trial.suggest_uniform('lgb_lambda_l2', 0, 50)
            # 'device': 'gpu'
        }

    def get_xgb_params_range(self, trial):
        return {
            'learning_rate': trial.suggest_loguniform('xgb_learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
            'subsample': trial.suggest_uniform('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('xgb_colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_uniform('xgb_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_uniform('xgb_reg_lambda', 0, 10),
            'random_state': config.cv_seed
            # 'tree_method': 'gpu_hist',
        }
    
    def get_cat_params_range(trial):
        return {
            'learning_rate': trial.suggest_loguniform('cat_learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('cat_depth', 2, 10),
            'iterations': trial.suggest_int('cat_iterations', 100, 500),
            'l2_leaf_reg': trial.suggest_uniform('cat_l2_leaf_reg', 0, 10),
            'random_seed': config.cv_seed,
            'verbose': 0
            # 'task_type': 'GPU'
        }

    @property
    def lgb_params(self):
        return self._lgb_params

    @property
    def xgb_params(self):
        return self._xgb_params
    
    @property
    def cat_params(self):
        return self._cat_params
