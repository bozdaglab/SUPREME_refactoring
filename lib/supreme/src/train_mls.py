import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from helper import random_split
from learning_types import LearningTypes
from ml_models import ClusteringModels, MLModels
from node_generation import add_row_features
from settings import (
    ADD_RAW_FEAT,
    INT_MOTHOD_CLASSIFICATION,
    INT_MOTHOD_CLUSTERING,
    X_TIME2,
)

DEVICE = torch.device("cpu")


def train_ml_model(
    ml_type: str,
    trial_combs: List[List[int]],
    trials: int,
    labels: pd.DataFrame,
    dir_name: str,
) -> Dict:
    files = os.listdir(dir_name)
    NODE_NETWORKS2 = [files[i] for i in trial_combs[trials]]
    if len(NODE_NETWORKS2) == 1:
        emb = torch.tensor(
            pd.read_pickle(f"{dir_name}/{NODE_NETWORKS2[0]}").values, device=DEVICE
        )
    else:
        is_first = True
        for netw_base in NODE_NETWORKS2:
            cur_emb = pd.read_pickle(f"{dir_name}/{netw_base}").values
            if is_first:
                emb = torch.tensor(cur_emb, device=DEVICE)
                is_first = False
            else:
                emb = torch.cat(
                    (emb, torch.tensor(cur_emb, device=DEVICE).float()), dim=1
                )
    if ADD_RAW_FEAT is True:
        emb = add_row_features(emb=emb)
    train_valid_idx, test_idx = random_split(emb)
    X_train = emb[train_valid_idx.indices].numpy()
    X_test = emb[test_idx.indices].numpy()
    try:
        y_train = labels[train_valid_idx.indices].ravel()
        y_test = labels[test_idx.indices].ravel()
    except KeyError:
        y_train = labels.values[train_valid_idx.indices].ravel()
        y_test = labels.values[test_idx.indices].ravel()

    if ml_type == LearningTypes.clustering.name:
        all_results = defaultdict()
        for clustering_model in INT_MOTHOD_CLUSTERING:
            ml_model = ClusteringModels(
                model=clustering_model, x_train=X_train, y_train=y_train
            )
            all_results[clustering_model] = get_ml_result(ml_model=ml_model)
        return all_results
    all_results = defaultdict()
    for classification_model in INT_MOTHOD_CLASSIFICATION:
        ml_model = MLModels(
            model=classification_model,
            x_train=X_train,
            y_train=y_train,
        )
        all_results[classification_model] = get_ml_result(
            ml_model=ml_model, X_test=X_test, y_test=y_test
        )
    return all_results


def get_ml_result(
    ml_model: Union[ClusteringModels, MLModels],
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.DataFrame] = None,
) -> Dict:

    model = ml_model.train_classifier()
    results = defaultdict(list)
    for _ in range(X_TIME2):
        ml_model.get_result(model=model, results=results, X_test=X_test, y_test=y_test)

    final_result = generate_final_result(results=results)
    # final_result["best_parameters"] = search.best_params_
    return final_result


def generate_final_result(results: Dict) -> Dict:
    return {key: calculate_result(results.get(key)) for key in results.keys()}


def calculate_result(inp: List[float]) -> str:
    try:
        return f"{round(statistics.median(inp), 3)}+-{round(statistics.stdev(inp), 3)}"
    except AssertionError:
        return f"{round(statistics.median(inp), 3)}+-{round(np.std(inp), 3)}"
