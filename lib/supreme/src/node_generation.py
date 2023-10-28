import os
import statistics
from itertools import product

import numpy as np
import pandas as pd
import torch
from feature_extraction import FeatureALgo
from helper import masking_indexes
from module import Net, criterion, train, validate
from settings import (
    EDGES,
    EMBEDDINGS,
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
from sklearn.model_selection import RepeatedStratifiedKFold
from torch_geometric.data import Data

DEVICE = torch.device("cpu")


def node_feature_generation(BASE_DATAPATH):
    is_first = True
    for file in NODE_NETWORKS:
        feat = pd.read_csv(f"{BASE_DATAPATH}/{file}")
        feat = feat.drop("Unnamed: 0", axis=1)
        if not any(
            FEATURE_SELECTION_PER_NETWORK
        ):  # any does not make sense. We need it seperate for each dataset
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

        if is_first:
            new_x = torch.tensor(values, device=DEVICE).float()
            is_first = False
        else:
            new_x = torch.cat(
                (new_x, torch.tensor(values, device=DEVICE).float()), dim=1
            )
    return new_x


def node_embedding_generation(new_x, train_valid_idx, labels, test_idx):
    for edge_file in os.listdir(EDGES):
        edge_index = pd.read_csv(f"{EDGES}/{edge_file}")
        best_ValidLoss = np.Inf
        # here we dont need the y anymore

        for col in labels.iloc[:, 0:3]:
            for learning_rate, hid_size in product(LEARNING_RATE, HIDDEN_SIZE):
                av_valid_losses = []
                for _ in range(X_TIME2):
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
                        y=torch.tensor(labels[col].values, dtype=torch.float32),
                    )
                    X = data.x[train_valid_idx.indices]
                    y = data.y[train_valid_idx.indices]
                    y_ph = labels.iloc[:, 3][train_valid_idx.indices]

                    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)
                    for train_part, valid_part in rskf.split(X, y_ph):
                        train_idx = np.array(train_valid_idx.indices)[train_part]
                        valid_idx = np.array(train_valid_idx.indices)[valid_part]
                        break

                    train_mask = masking_indexes(data=data, indexes=train_idx)
                    valid_mask = masking_indexes(data=data, indexes=valid_idx)
                    test_mask = masking_indexes(data=data, indexes=test_idx)

                    data.valid_mask = torch.tensor(valid_mask, device=DEVICE)
                    data.train_mask = torch.tensor(train_mask, device=DEVICE)
                    data.test_mask = torch.tensor(test_mask, device=DEVICE)

                    in_size = data.x.shape[1]
                    out_size = 1  # torch.tensor(data.y).shape[0]
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
                    # best_emb_lr = learning_rate
                    # best_emb_hs = hid_size
                    selected_emb = this_emb

            embedding_path = f"{EMBEDDINGS}/{edge_file.split('.csv')[0]}"
            pd.DataFrame(selected_emb).to_csv(
                f"{embedding_path}_{col}.csv", index=False
            )
