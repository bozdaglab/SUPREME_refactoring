import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from boruta import BorutaPy
from pre_processings import pre_processing
from scipy.stats import pearsonr, spearmanr
from settings import EDGES, STAT_METHOD
from torch import Tensor


def search_dictionary(methods_features: Dict, thr: int = 2) -> List[str]:
    count_features = Counter()
    for features in methods_features.values():
        count_features.update(features)
    return [feat for feat, val in count_features.items() if val > thr]


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


def select_boruta(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    boruta = BorutaPy(
        estimator=xgb.XGBClassifier(),
        n_estimators="auto",
        verbose=2,
        random_state=42,
        max_iter=40,
    )
    boruta.fit(X.values, y.values)
    return X[X.columns[boruta.support_]]


def get_stat_methos(stat_method: str):
    factory = {"spearman": spearmanr, "pearson": pearsonr}
    try:
        return factory[stat_method]
    except KeyError:
        raise KeyError("Please check your stat model")


def set_same_users(sample_data: Dict, users: Dict) -> Dict:
    new_dataset = defaultdict()
    shared_users = search_dictionary(users, len(users) - 1)
    shared_users = sorted(shared_users)[0:100]
    for file_name, data in sample_data.items():
        new_dataset[file_name] = data[data.index.isin(shared_users)]
    return new_dataset


def similarity_matrix_generation(new_dataset: Dict) -> Dict:
    # parqua dataset, parallel
    final_correlation = defaultdict()
    if not os.path.exists(EDGES):
        os.mkdir(EDGES)
    file_names = os.listdir(EDGES)
    if file_names:
        for file in file_names:
            final_correlation[file] = pd.read_pickle(f"{EDGES}/{file}")
        return final_correlation

    for file_name, data in new_dataset.items():
        correlation_dictionary = defaultdict()
        if sum(data.isna().sum()):
            data = pre_processing(data=data)
        stat_model = get_stat_methos(STAT_METHOD)
        for ind_i, patient_1 in enumerate(data.iloc):
            for ind_j, patient_2 in enumerate(data[ind_i + 1 :].iloc):
                correlation_dictionary[f"{ind_i}_{ind_j}"] = {
                    "Patient_1": ind_i,
                    "Patient_2": ind_j,
                    "Similarity Score": stat_model(
                        patient_1.values, patient_2.values
                    ).statistic,
                }
        final_correlation[file_name] = correlation_dictionary

        pd.DataFrame(
            correlation_dictionary.values(),
            columns=["Patient_1", "Patient_2", "Similarity Score"],
        ).to_pickle(EDGES / f"similarity_{file_name}")
    return final_correlation
