import pickle
import re
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data

from lib.supreme.src.ml_models import MLModels
from project.supreme.src.settings import (
    ADD_RAW_FEAT,
    BORUTA_RUNS,
    BORUTA_TOP_FEATURES,
    FEATURE_NETWORKS_INTEGRATION,
    INT_MOTHOD,
    NODE_NETWORKS,
    OPTIONAL_FEATURE_SELECTION,
    X_TIME2,
)

# Running Machine Learning for each possible combination of input network
# Input for Machine Learning algorithm is the concatanation of node embeddings
#  (specific to each combination) and node features (if node feature integration is True)
DEVICE = torch.device("cpu")


def ml(save_path, dataset_name, trial_combs, trials, labels, train_valid_idx, test_idx):
    NODE_NETWORKS2 = [NODE_NETWORKS[i] for i in trial_combs[trials]]
    netw_base = NODE_NETWORKS2[0]
    emb_file = save_path / f"Emb_{NODE_NETWORKS2[0]}.pkl"
    with open(emb_file, "rb") as f:
        emb = pickle.load(f)

    if len(NODE_NETWORKS2) > 1:
        for netw_base in NODE_NETWORKS2[1:]:
            emb_file = save_path + "Emb_" + netw_base + ".pkl"
            with open(emb_file, "rb") as f:
                cur_emb = pickle.load(f)
            emb = torch.cat((emb, cur_emb), dim=1)

    if ADD_RAW_FEAT is True:
        is_first = 0
        addFeatures = FEATURE_NETWORKS_INTEGRATION
        for netw in addFeatures:
            file = "" + "data/" + dataset_name + "/" + netw + ".pkl"
            with open(file, "rb") as f:
                feat = pickle.load(f)
            if is_first == 0:
                allx = torch.tensor(feat.values, device=DEVICE).float()
                is_first = 1
            else:
                allx = torch.cat(
                    (allx, torch.tensor(feat.values, device=DEVICE).float()), dim=1
                )

        if OPTIONAL_FEATURE_SELECTION is True:
            allx_flat = [item for sublist in allx.tolist() for item in sublist]
            allx_temp = robjects.FloatVector(allx_flat)
            robjects.globalenv["allx_matrix"] = robjects.r("matrix")(allx_temp)
            robjects.globalenv["allx_x"] = robjects.IntVector(allx.shape)
            robjects.globalenv["labels_vector"] = robjects.IntVector(labels.tolist())
            robjects.globalenv["top"] = BORUTA_TOP_FEATURES
            robjects.globalenv["maxBorutaRuns"] = BORUTA_RUNS
            robjects.r(
                """
                require(rFerns)
                require(Boruta)
                labels_vector = as.factor(labels_vector)
                allx_matrix <- Reshape(allx_matrix, allx_x[1])
                allx_data = data.frame(allx_matrix)
                colnames(allx_data) <- 1:allx_x[2]
                allx_data <- allx_data %>%
                    mutate('Labels' = labels_vector)
                boruta.train <- Boruta(allx_data$Labels ~ ., data= allx_data, doTrace = 0,
                  getImp=getImpFerns, holdHistory = T, maxRuns = maxBorutaRuns)
                thr = sort(attStats(boruta.train)$medianImp, decreasing = T)[top]
                boruta_signif = rownames(attStats(boruta.train)[attStats(boruta.train)$medianImp >= thr,])
                    """
            )
            boruta_signif = robjects.globalenv["boruta_signif"]
            robjects.r.rm("allx_matrix")
            robjects.r.rm("labels_vector")
            robjects.r.rm("allx_data")
            robjects.r.rm("boruta_signif")
            robjects.r.rm("thr")
            topx = []
            for index in boruta_signif:
                t_index = re.sub("`", "", index)
                topx.append((np.array(allx).T)[int(t_index) - 1])
            topx = np.array(topx)
            emb = torch.cat((emb, torch.tensor(topx.T, device=DEVICE)), dim=1)
            print("Top " + str(BORUTA_TOP_FEATURES) + " features have been selected.")
        else:
            emb = torch.cat((emb, allx), dim=1)

    data = Data(x=emb, y=labels)
    train_mask = np.array([i in set(train_valid_idx) for i in range(data.x.shape[0])])
    data.train_mask = torch.tensor(train_mask, device=DEVICE)
    test_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
    data.test_mask = torch.tensor(test_mask, device=DEVICE)
    X_train = pd.DataFrame(data.x[data.train_mask].numpy())
    X_test = pd.DataFrame(data.x[data.test_mask].numpy())
    y_train = pd.DataFrame(data.y[data.train_mask].numpy()).values.ravel()
    y_test = pd.DataFrame(data.y[data.test_mask].numpy()).values.ravel()

    m = MLModels(model=INT_MOTHOD, x_train=X_train, y_train=y_train)
    model, search = m.train_ml_model_factory()
    results = defaultdict(list)

    for ii in range(X_TIME2):
        model.fit(X_train, y_train)  # ?
        predictions = model.predict(X_test)
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
        results["av_tr_result_acc"].append(round(accuracy_score(y_train, tr_pred), 3))
        results["av_tr_result_wf1"].append(
            round(f1_score(y_train, tr_pred, average="weighted"), 3)
        )
        results["av_tr_result_mf1"].append(
            round(f1_score(y_train, tr_pred, average="macro"), 3)
        )
    # ?
    if X_TIME2 == 1:
        results["av_result_acc"].append(round(accuracy_score(y_test, y_pred), 3))
        results["av_result_wf1"].append(
            round(f1_score(y_test, y_pred, average="weighted"), 3)
        )
        results["av_result_mf1"].append(
            round(f1_score(y_test, y_pred, average="macro"), 3)
        )
        results["av_tr_result_acc"].append(round(accuracy_score(y_train, tr_pred), 3))
        results["av_tr_result_wf1"].append(
            round(f1_score(y_train, tr_pred, average="weighted"), 3)
        )
        results["av_tr_result_mf1"].append(
            round(f1_score(y_train, tr_pred, average="macro"), 3)
        )

    final_result = defaultdict()
    final_result["result_acc"] = calculate_result(results["av_result_acc"])
    final_result["result_wf1"] = calculate_result(results["av_result_wf1"])
    final_result["result_mf1"] = calculate_result(results["av_result_mf1"])
    final_result["tr_result_acc"] = calculate_result(results["av_tr_result_acc"])
    final_result["tr_result_wf1"] = calculate_result(results["av_tr_result_wf1"])
    final_result["tr_result_mf1"] = calculate_result(results["av_tr_result_mf1"])
    final_result["best_parameters"] = search.best_params_
    return final_result


def calculate_result(inp):
    return f"{round(statistics.median(inp), 3)}+-{round(statistics.stdev(inp), 3)}"
