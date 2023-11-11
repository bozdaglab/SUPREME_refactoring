import os
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from boruta import BorutaPy
from scipy.stats import pearsonr, spearmanr
from settings import DATA, SIMILARITY

# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import ExtraTreeRegressor
from torch import Tensor


def ratio(new_x: torch) -> List[int]:
    """
    This function defines the ratio for train test splitting

    Parameters:
    -----------
    new_x:
        Concatenated features from different omics file
    Return:
        A list of two values that shows the ratio between
        the train (The first value) and test (the second one)

    """
    shape = new_x.shape[0]
    train_idx = round(shape * 0.75)
    return [train_idx, shape - train_idx]


def random_split(new_x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths

    Parameters:
    -----------
    new_x:
         Concatenated features from different omics file

    Return:
        New datasets
    """
    return torch.utils.data.random_split(new_x, ratio(new_x))


def masking_indexes(data, indexes):
    return np.array([i in set(indexes) for i in range(data.x.shape[0])])


def load_missing_method(imputer_name):
    if imputer_name == "KNNImputer":
        return KNNImputer(n_neighbors=2)
    elif imputer_name == "IterativeImputer":
        return IterativeImputer(
            max_iter=2, estimator=ExtraTreeRegressor(), keep_empty_features=True
        )
    return None


def select_boruta(X: pd.DataFrame, y: pd.DataFrame) -> List[str]:
    model = xgb.XGBClassifier()
    features = BorutaPy(
        estimator=model,
        n_estimators="auto",
        verbose=2,
        random_state=42,
        max_iter=40,
    )
    features.fit(np.array(X), np.array(y))
    selected_features = features.support_
    return [X.columns[i] for i in range(len(selected_features)) if selected_features[i]]


def get_stat_methos(stat_method: str):
    factory = {"spearman": spearmanr, "pearson": pearsonr}
    try:
        return factory[stat_method]
    except KeyError:
        raise KeyError("Please check your stat model")


def similarity():
    if os.path.exists(SIMILARITY):
        return
    os.mkdir(SIMILARITY)
    stat_method = "pearson"
    for file in os.listdir(DATA):
        correlation_dictionary = defaultdict()
        data = pd.read_csv(DATA / file)
        # data = data.dropna(inplace=True)
        stat_model = get_stat_methos(stat_method)
        for ind_i, patient_1 in data.iterrows():
            for ind_j, patient_2 in data[ind_i + 1 :].iterrows():
                correlation_dictionary[f"{ind_i}_{ind_j}"] = {
                    "Patient_1": ind_i,
                    "Patient_2": ind_j,
                    "Similarity Score": stat_model(
                        patient_1.values, patient_2.values
                    ).statistic,
                }
        pd.DataFrame(
            correlation_dictionary.values(),
            columns=["Patient_1", "Patient_2", "Similarity Score"],
        ).to_csv(SIMILARITY / f"similarity_{file}")
