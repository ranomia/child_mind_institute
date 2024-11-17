import numpy as np
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_true, y_pred):
    y_true_rounded = np.round(y_true).astype(int)
    y_pred_rounded = np.round(y_pred).astype(int)
    return cohen_kappa_score(y_true_rounded, y_pred_rounded, weights='quadratic')