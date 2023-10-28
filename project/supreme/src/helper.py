import numpy as np
import torch

# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import ExtraTreeRegressor


def ratio(new_x: torch) -> int:
    shape = new_x.shape[0]
    train_idx = round(shape * 0.75)
    return [train_idx, shape - train_idx]


def masking_indexes(data, indexes):
    return np.array([i in set(indexes) for i in range(data.x.shape[0])])


def load_missing_method(imputer_name):
    if imputer_name == "KNNImputer":
        return KNNImputer(n_neighbors=2)  # , keep_empty_features=True)
    elif imputer_name == "IterativeImputer":
        return IterativeImputer(
            max_iter=2, estimator=ExtraTreeRegressor(), keep_empty_features=True
        )
    return None
