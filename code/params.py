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
            'lambda_l2': 0.01,  # Increased from 2.68e-06
            'device': 'gpu'
        }

        self._xgb_params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,  # Increased from 0.1
            'reg_lambda': 5,  # Increased from 1
            'random_state': config.cv_seed,
            'tree_method': 'gpu_hist',
        }
        
        self._cat_params = {
            'learning_rate': 0.05,
            'depth': 6,
            'iterations': 200,
            'random_seed': config.cv_seed,
            'verbose': 0,
            'l2_leaf_reg': 10,  # Increase this value
            'task_type': 'GPU'
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
