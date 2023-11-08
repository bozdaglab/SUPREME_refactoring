import os
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from learning_types import LearningTypes
from ml_models import MLModels
from settings import (ADD_RAW_FEAT, EMBEDDINGS, INT_MOTHOD, LEARNING,
                      OPTIONAL_FEATURE_SELECTION, X_TIME2)
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             completeness_score, f1_score,
                             homogeneity_completeness_v_measure,
                             homogeneity_score, silhouette_score,
                             v_measure_score)
from torch_geometric.data import Data

DEVICE = torch.device("cpu")


def ml(trial_combs, trials, labels, train_valid_idx, test_idx):
    NODE_NETWORKS2 = [os.listdir(EMBEDDINGS / LEARNING)[i] for i in trial_combs[trials]]
    if len(NODE_NETWORKS2) == 1:
        emb = pd.read_csv(f"{EMBEDDINGS}/{LEARNING}/{NODE_NETWORKS2[0]}")
    else:
        for netw_base in NODE_NETWORKS2:
            emb = pd.DataFrame()
            cur_emb = pd.read_csv(f"{EMBEDDINGS}/{LEARNING}/{netw_base}")
            emb = emb.append(cur_emb)
    emb = torch.tensor(emb.values, device=DEVICE)
    if ADD_RAW_FEAT is True:
        is_first = 0
        for addFeatures in os.listdir(EMBEDDINGS / LEARNING):
            features = pd.read_csv(f"{EMBEDDINGS}/{LEARNING}/{addFeatures}")

            if is_first == 0:
                allx = torch.tensor(features.values, device=DEVICE).float()
                is_first = 1
            else:
                allx = torch.cat(
                    (allx, torch.tensor(features.values, device=DEVICE).float()), dim=1
                )

        if OPTIONAL_FEATURE_SELECTION is True:
            pass
        else:
            emb = torch.cat((emb, allx), dim=1)

    data = Data(x=emb, y=labels)
    train_mask = np.array(
        [i in set(train_valid_idx.indices) for i in range(data.x.shape[0])]
    )
    data.train_mask = torch.tensor(train_mask, device=DEVICE)
    train_mask = np.array(
        [i in set(train_valid_idx.indices) for i in range(data.x.shape[0])]
    )
    test_mask = np.array([i in set(test_idx.indices) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=DEVICE)
    X_train = pd.DataFrame(data.x[data.train_mask].numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].numpy())

    try:
        y_train = pd.DataFrame(data.y[data.train_mask].numpy()).values.ravel()
        y_test = pd.DataFrame(data.y[data.test_mask].numpy()).values.ravel()
    except:
        try:
            y_train = pd.DataFrame(data.y.values[data.train_mask]).values.ravel()
            y_test = pd.DataFrame(data.y.values[data.test_mask]).values.ravel()
        except:
            pass

    clustering = False
    if LEARNING is LearningTypes.clustering.name:
        m = MLModels(model=INT_MOTHOD, x_train=X_train)
        clustering = True
    else:
        m = MLModels(model=INT_MOTHOD, x_train=X_train, y_train=y_train)

    model, search = m.train_ml_model_factory()
    results = defaultdict(list)

    for _ in range(X_TIME2):
        final_result = defaultdict()
        if clustering:
            model.fit(X_train)
            predictions = model.predict(X_train)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        if clustering:
            final_result[
                "homogeneity_completeness_v_measure"
            ] = homogeneity_completeness_v_measure(y_train, model.labels_)
            final_result["homogeneity"] = homogeneity_score(y_train, model.labels_)
            final_result["Completeness"] = completeness_score(y_train, model.labels_)
            final_result["v_measure"] = v_measure_score(y_train, model.labels_)
            final_result["adjusted_rand"] = adjusted_rand_score(y_train, model.labels_)
            final_result["silhouette"] = silhouette_score(X_train, model.labels_)

        else:
            y_pred = [round(value) for value in predictions]
            # preds = model.predict(pd.DataFrame(data.x.numpy()))
            results["av_result_acc"].append(round(accuracy_score(y_test, y_pred), 3))
            results["av_result_wf1"].append(
                round(f1_score(y_test, y_pred, average="weighted"), 3)
            )
            results["av_result_mf1"].append(
                round(f1_score(y_test, y_pred, average="macro"), 3)
            )
            tr_predictions = model.predict(X_train)
            tr_pred = [round(value) for value in tr_predictions]
            results["av_tr_result_acc"].append(
                round(accuracy_score(y_train, tr_pred), 3)
            )
            results["av_tr_result_wf1"].append(
                round(f1_score(y_train, tr_pred, average="weighted"), 3)
            )
            results["av_tr_result_mf1"].append(
                round(f1_score(y_train, tr_pred, average="macro"), 3)
            )

            if X_TIME2 == 1:
                results["av_result_acc"].append(
                    round(accuracy_score(y_test, y_pred), 3)
                )
                results["av_result_wf1"].append(
                    round(f1_score(y_test, y_pred, average="weighted"), 3)
                )
                results["av_result_mf1"].append(
                    round(f1_score(y_test, y_pred, average="macro"), 3)
                )
                results["av_tr_result_acc"].append(
                    round(accuracy_score(y_train, tr_pred), 3)
                )
                results["av_tr_result_wf1"].append(
                    round(f1_score(y_train, tr_pred, average="weighted"), 3)
                )
                results["av_tr_result_mf1"].append(
                    round(f1_score(y_train, tr_pred, average="macro"), 3)
                )

            final_result["result_acc"] = calculate_result(results["av_result_acc"])
            final_result["result_wf1"] = calculate_result(results["av_result_wf1"])
            final_result["result_mf1"] = calculate_result(results["av_result_mf1"])
            final_result["tr_result_acc"] = calculate_result(
                results["av_tr_result_acc"]
            )
            final_result["tr_result_wf1"] = calculate_result(
                results["av_tr_result_wf1"]
            )
            final_result["tr_result_mf1"] = calculate_result(
                results["av_tr_result_mf1"]
            )
            final_result["best_parameters"] = search.best_params_
    return final_result


def calculate_result(inp):
    return f"{round(statistics.median(inp), 3)}+-{round(statistics.stdev(inp), 3)}"
