import statistics

# import rpy2.robjects as robjects
import numpy as np
import pandas as pd
import pickle
import torch
from module import criterion, train, validate
from sklearn.model_selection import RepeatedStratifiedKFold
from torch_geometric.data import Data
from feature_extraction import FeatureALgo
from module import Net
from settings import (
    FEATURE_SELECTION_PER_NETWORK,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    NODE_NETWORKS,
    PATIENCE,
    TOP_FEATURES_PER_NETWORK,
    X_TIME2,
)

DEVICE = torch.device("cpu")


def node_feature_generation(SAMPLE_PATH):
    is_first = 0
    for netw in NODE_NETWORKS:
        file = SAMPLE_PATH / f"{netw}.pkl"
        with open(file, "rb") as f:
            feat = pickle.load(f)
            if not any(FEATURE_SELECTION_PER_NETWORK): # any does not make sense. We need it seperate for each dataset
                values = feat.values
            else:
                if (
                    TOP_FEATURES_PER_NETWORK[NODE_NETWORKS.index(netw)]
                    < feat.values.shape[1]
                ):
                    topx = FeatureALgo().select_boruta("pass x and y")
                    topx = np.array(topx)
                    values = torch.tensor(topx.T, device=DEVICE)
                elif (
                    TOP_FEATURES_PER_NETWORK[NODE_NETWORKS.index(netw)]
                    >= feat.values.shape[1]
                ):
                    values = feat.values

        if is_first == 0:
            new_x = torch.tensor(values, device=DEVICE).float()
            is_first = 1
        else:
            new_x = torch.cat(
                (new_x, torch.tensor(values, device=DEVICE).float()), dim=1
            )
    return new_x



def node_embedding_generation(SAMPLE_PATH, new_x, train_valid_idx, labels, test_idx, save_path):
    for n in range(len(NODE_NETWORKS)):
        netw_base = SAMPLE_PATH / f"edges_{NODE_NETWORKS[n]}.pkl"
        with open(netw_base, "rb") as f:
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
                            device=DEVICE,
                        ).long(),
                        edge_attr=torch.tensor(
                            edge_index[edge_index.columns[2]].transpose().values,
                            device=DEVICE,
                        ).float(),
                        y=labels,
                    )
                    X = data.x[train_valid_idx]
                    y = data.y.values[train_valid_idx]
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
                    data.valid_mask = torch.tensor(valid_mask, device=DEVICE)
                    data.train_mask = torch.tensor(train_mask, device=DEVICE)
                    test_mask = np.array(
                        [i in set(test_idx) for i in range(data.x.shape[0])]
                    )
                    data.test_mask = torch.tensor(test_mask, device=DEVICE)

                    in_size = data.x.shape[1]
                    out_size = torch.unique(torch.tensor(data.y.values)).shape[0]
                    model = Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    min_valid_loss = np.Inf
                    patience_count = 0

                    for epoch in range(MAX_EPOCHS):
                        emb = train(model, optimizer, data, criterion)
                        this_valid_loss, emb = validate(model, criterion, data)

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

        embedding_path =  save_path / f"Emb_{NODE_NETWORKS[n]}"
        with open(f"{embedding_path}.pkl", "wb") as f:
            pickle.dump(selected_emb, f)
            pd.DataFrame(selected_emb).to_csv(f"{embedding_path}.csv")
