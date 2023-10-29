import os
import statistics
from itertools import product
from selected_models import load_model

import numpy as np
import pandas as pd
import torch
from module import cluster
from helper import masking_indexes, select_boruta
from module import Net
from settings import (
    DATA,
    EDGES,
    EMBEDDINGS,
    FEATURE_SELECTION_PER_NETWORK,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PATIENCE,
    X_TIME2,
)
from sklearn.model_selection import RepeatedStratifiedKFold

DEVICE = torch.device("cpu")


def node_feature_generation(labels):
    is_first = True
    for file in os.listdir(DATA):
        feat = pd.read_csv(f"{DATA}/{file}")
        feat = feat.drop("Unnamed: 0", axis=1)
        if not any(
            FEATURE_SELECTION_PER_NETWORK
        ):  # any does not make sense. We need it seperate for each dataset
            values = feat.values
        else:
            topx = select_boruta(feat, labels)
            topx = np.array(topx)
            values = torch.tensor(topx.T, device=DEVICE)
        if is_first:
            new_x = torch.tensor(values, device=DEVICE).float()
            is_first = False
        else:
            new_x = torch.cat(
                (new_x, torch.tensor(values, device=DEVICE).float()), dim=1
            )
    return new_x


def node_embedding_generation(new_x, train_valid_idx, labels, test_idx, learning):
    learning_model = load_model(learning)
    for edge_file in os.listdir(EDGES):
        edge_index = pd.read_csv(f"{EDGES}/{edge_file}")
        best_ValidLoss = np.Inf
        # here we dont need the y anymore

        for col in labels.iloc[:, 0:3]:
            for learning_rate, hid_size in product(LEARNING_RATE, HIDDEN_SIZE):
                av_valid_losses = []
                for _ in range(X_TIME2):
                    data = learning_model.prepare_data(new_x, edge_index, labels, col)
                    X = data.x[train_valid_idx.indices]
                    # y = data.y[train_valid_idx.indices]
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
                    criterion, out_size = learning_model.select_model()
                    in_size = data.x.shape[1]
                    model = Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    min_valid_loss = np.Inf
                    patience_count = 0

                    for epoch in range(MAX_EPOCHS):
                        emb = learning_model.train(model, optimizer, data, criterion)
                        this_valid_loss, emb = learning_model.validate(model, criterion, data)

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
