import os
import statistics
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import torch
from learning_types import LearningTypes
from ml_models import MLModels
from node_generation import add_row_features
from settings import (
    ADD_RAW_FEAT,
    EMBEDDINGS,
    INT_MOTHOD_CLASSIFICATION,
    INT_MOTHOD_CLUSTERING,
    LEARNING,
    X_TIME2,
)
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from torch import Tensor

DEVICE = torch.device("cpu")


def train_ml_model(
    ml_type: str,
    trial_name: str,
    trial_combs: List[List[int]],
    trials: int,
    labels: pd.DataFrame,
    train_valid_idx: Tensor,
    test_idx: Tensor,
    embeddings: Dict,
):
    files = os.listdir(EMBEDDINGS / ml_type / trial_name)
    NODE_NETWORKS2 = [files[i] for i in trial_combs[trials]]
    if len(NODE_NETWORKS2) == 1:
        emb = embeddings[trial_name][files.index(NODE_NETWORKS2[0])]
    else:
        for netw_base in NODE_NETWORKS2:
            emb = pd.DataFrame()
            cur_emb = pd.read_csv(f"{EMBEDDINGS}/{LEARNING}/{netw_base}")
            emb = emb.append(cur_emb)
    emb = torch.tensor(emb.values, device=DEVICE)
    if ADD_RAW_FEAT is True:
        emb = add_row_features(emb=emb)

    X_train = emb[train_valid_idx.indices].numpy()
    X_test = emb[test_idx.indices].numpy()
    try:
        y_train = labels[train_valid_idx.indices].ravel()
        y_test = labels[test_idx.indices].ravel()
    except KeyError:
        y_train = labels.values[train_valid_idx.indices].ravel()
        y_test = labels.values[test_idx.indices].ravel()

    clustering = False
    if LEARNING is LearningTypes.clustering.name:
        ml_model = MLModels(model=INT_MOTHOD_CLUSTERING, x_train=X_train)
        clustering = True
    else:
        ml_model = MLModels(
            model=INT_MOTHOD_CLASSIFICATION, x_train=X_train, y_train=y_train
        )

    model, search = ml_model.train_ml_model_factory()
    results = defaultdict(list)
    for _ in range(X_TIME2):
        if clustering:
            model.fit(X_train)
            predictions = model.predict(X_train)
            results[
                "homogeneity_completeness_v_measure"
            ] = homogeneity_completeness_v_measure(y_train, predictions)
            results["homogeneity"] = homogeneity_score(y_train, predictions)
            results["Completeness"] = completeness_score(y_train, predictions)
            results["v_measure"] = v_measure_score(y_train, predictions)
            results["adjusted_rand"] = adjusted_rand_score(y_train, predictions)
            results["silhouette"] = silhouette_score(X_train, predictions)

        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            train_pred = model.predict(X_train)
            results["test_accuracy"].append(round(accuracy_score(y_test, y_pred), 3))
            results["test_weighted_f1"].append(
                round(f1_score(y_test, y_pred, average="weighted"), 3)
            )
            results["test_macro_f1"].append(
                round(f1_score(y_test, y_pred, average="macro"), 3)
            )
            results["train_accuracy"].append(
                round(accuracy_score(y_train, train_pred), 3)
            )
            results["train_weighted_f1"].append(
                round(f1_score(y_train, train_pred, average="weighted"), 3)
            )
            results["train_macro_f1"].append(
                round(f1_score(y_train, train_pred, average="macro"), 3)
            )

    final_result = generate_final_result(results=results)
    final_result["best_parameters"] = search.best_params_
    return final_result


def generate_final_result(results: Dict) -> Dict:
    return {key: calculate_result(results.get(key)) for key in results.keys()}


def calculate_result(inp: List[float]) -> str:
    return f"{round(statistics.median(inp), 3)}+-{round(statistics.stdev(inp), 3)}"
