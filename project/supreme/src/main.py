import argparse
import errno
import logging
import os
import statistics
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle5 as pickle
import torch
from set_logging import set_log_config
from settings import (
    ADD_RAW_FEAT,
    BORUTA_RUNS,
    BORUTA_TOP_FEATURES,
    FEATURE_NETWORKS_INTEGRATION,
    FEATURE_SELECTION_PER_NETWORK,
    HIDDEN_SIZE,
    INT_MOTHOD,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    NODE_NETWORKS,
    OPTIONAL_FEATURE_SELECTION,
    PATH,
    PATIENCE,
    TOP_FEATURES_PER_NETWORK,
    X_TIME,
    X_TIME2,
)

from lib.supreme.src.new_tensors import create_new_x

SAMPLE_PATH = PATH / "sample_data"

set_log_config()
logger = logging.getLogger()

from torch_geometric.data import Data

from lib.supreme.src.ml_models import MLModels
from lib.supreme.src.module import Net

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
BASE_PATH = ""
random_state = 404


if (True in FEATURE_SELECTION_PER_NETWORK) or (OPTIONAL_FEATURE_SELECTION == True):
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    utils = importr("utils")
    rFerns = importr("rFerns")
    Boruta = importr("Boruta")
    pracma = importr("pracma")
    dplyr = importr("dplyr")
    import re

# Parser
parser = argparse.ArgumentParser(
    description="""An integrative node classification framework, called SUPREME
(a subtype prediction methodology), that utilizes graph convolutions on multiple datatype-specific networks that are annotated with multiomics datasets as node features.
This framework is model-agnostic and could be applied to any classification problem with properly processed datatypes and networks."""
)
parser.add_argument("-data", "--data_location", nargs=1, default=["sample_data"])

args = parser.parse_args()
dataset_name = args.data_location[0]


if not os.path.exists(SAMPLE_PATH):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

device = torch.device("cpu")


def train():
    model.train()
    optimizer.zero_grad()
    out, emb1 = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return emb1


def validate():
    model.eval()
    with torch.no_grad():
        out, emb2 = model(data)
        pred = out.argmax(dim=1)
        loss = criterion(out[data.valid_mask], data.y[data.valid_mask])
    return loss, emb2


criterion = torch.nn.CrossEntropyLoss()

data_path_node = PATH / dataset_name  # SAMPLE_PATH
run_name = "SUPREME_" + dataset_name + "_results"
save_path = PATH / f"SUPREME_{dataset_name}_results"

if not os.path.exists(save_path):
    os.makedirs(save_path)

file = PATH / "sample_data" / "labels.pkl"  # SAMPLE_PATH
with open(file, "rb") as f:
    labels = pickle.load(f)

file = PATH / "sample_data" / "mask_values.pkl"  # SAMPLE_PATH
if os.path.exists(file):
    with open(file, "rb") as f:
        train_valid_idx, test_idx = pickle.load(f)
else:
    train_valid_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.20, shuffle=True, stratify=labels
    )

# from cProfile import Profile
# with Profile() as profile:
start = time.time()
# for i in range(0, 100):
#     print (i)
is_first = 0

logger.info("SUPREME is running..")
# Node feature generation - Concatenating node features from all the input datatypes
create_new_x()
# Node embedding generation using GCN for each input network with hyperparameter tuning
for n in range(len(NODE_NETWORKS)):
    netw_base = NODE_NETWORKS[n]
    with open(data_path_node + "edges_" + netw_base + ".pkl", "rb") as f:
        edge_index = pickle.load(f)
    best_ValidLoss = np.Inf

    for learning_rate in LEARNING_RATE:
        for hid_size in HIDDEN_SIZE:
            av_valid_losses = list()

            for ii in range(X_TIME2):
                data = Data(
                    x=new_x,
                    edge_index=torch.tensor(
                        edge_index[edge_index.columns[0:2]].transpose().values,
                        device=device,
                    ).long(),
                    edge_attr=torch.tensor(
                        edge_index[edge_index.columns[2]].transpose().values,
                        device=device,
                    ).float(),
                    y=labels,
                )
                X = data.x[train_valid_idx]
                y = data.y[train_valid_idx]
                rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)

                for train_part, valid_part in rskf.split(X, y):
                    train_idx = train_valid_idx[train_part]
                    valid_idx = train_valid_idx[valid_part]
                    break

                train_mask = np.array(
                    [i in set(train_idx) for i in range(data.x.shape[0])]
                )
                valid_mask = np.array(
                    [i in set(valid_idx) for i in range(data.x.shape[0])]
                )
                data.valid_mask = torch.tensor(valid_mask, device=device)
                data.train_mask = torch.tensor(train_mask, device=device)
                test_mask = np.array(
                    [i in set(test_idx) for i in range(data.x.shape[0])]
                )
                data.test_mask = torch.tensor(test_mask, device=device)

                in_size = data.x.shape[1]
                out_size = torch.unique(data.y).shape[0]
                model = Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                min_valid_loss = np.Inf
                patience_count = 0

                for epoch in range(MAX_EPOCHS):
                    emb = train()
                    this_valid_loss, emb = validate()

                    if this_valid_loss < min_valid_loss:
                        min_valid_loss = this_valid_loss
                        patience_count = 0
                        this_emb = emb
                    else:
                        patience_count += 1

                    if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
                        break

                av_valid_losses.append(min_valid_loss.item())

            av_valid_loss = round(statistics.median(av_valid_losses), 3)

            if av_valid_loss < best_ValidLoss:
                best_ValidLoss = av_valid_loss
                best_emb_lr = learning_rate
                best_emb_hs = hid_size
                selected_emb = this_emb

    emb_file = save_path + "Emb_" + netw_base + ".pkl"
    with open(emb_file, "wb") as f:
        pickle.dump(selected_emb, f)
        pd.DataFrame(selected_emb).to_csv(emb_file[:-4] + ".csv")

start2 = time.time()

print(
    "It took "
    + str(round(start2 - start, 1))
    + " seconds for node embedding generation ("
    + str(len(LEARNING_RATE) * len(HIDDEN_SIZE))
    + " trials for "
    + str(len(NODE_NETWORKS))
    + " seperate GCNs)."
)

logger.info("SUPREME is integrating the embeddings..")
""
from lib.supreme.src.train_mls import ml

# Running Machine Learning for each possible combination of input network
# Input for Machine Learning algorithm is the concatanation of node embeddings (specific to each combination) and node features (if node feature integration is True)

addFeatures = []
t = range(len(NODE_NETWORKS))
trial_combs = []
for r in range(1, len(t) + 1):
    trial_combs.extend([list(x) for x in itertools.combinations(t, r)])

for trials in range(len(trial_combs)):
    model = ml(save_path, dataset_name)
    results = defaultdict(list)

    for ii in range(X_TIME2):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        y_pred = [round(value) for value in predictions]
        preds = model.predict(pd.DataFrame(data.x.numpy()))
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
    final_result["result_acc"] = calculate_result(result["av_result_acc"])
    final_result["result_wf1"] = calculate_result(result["av_result_wf1"])
    final_result["result_mf1"] = calculate_result(result["av_result_mf1"])
    final_result["tr_result_acc"] = calculate_result(result["av_tr_result_acc"])
    final_result["tr_result_wf1"] = calculate_result(result["av_tr_result_wf1"])
    final_result["tr_result_mf1"] = calculate_result(result["av_tr_result_mf1"])

    def calculate_result(inp):
        return (
            str(round(statistics.median(inp), 3))
            + "+-"
            + str(round(statistics.stdev(inp), 3))
        )

    print(
        "Combination "
        + str(trials)
        + " "
        + str(NODE_NETWORKS2)
        + " >  selected parameters = "
        + str(search.best_params_)
        + ", train accuracy = "
        + str(tr_result_acc)
        + ", train weighted-f1 = "
        + str(tr_result_wf1)
        + ", train macro-f1 = "
        + str(tr_result_mf1)
        + ", test accuracy = "
        + str(result_acc)
        + ", test weighted-f1 = "
        + str(result_wf1)
        + ", test macro-f1 = "
        + str(result_mf1)
    )


end = time.time()
print("It took " + str(round(end - start, 1)) + " seconds in total.")
print("SUPREME is done.")
