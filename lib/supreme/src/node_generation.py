import os
import statistics
from itertools import product

import numpy as np
import pandas as pd
import torch
from helper import select_boruta
from module import Net
from selected_models import load_model, select_optimizer
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
    LEARNING
)
from typing import Optional

DEVICE = torch.device("cpu")


def node_feature_generation(labels: Optional[pd.DataFrame]):
    is_first = True
    for file in os.listdir(DATA):
        feat = pd.read_csv(f"{DATA}/{file}")
        if "Unnamed: 0" in feat.columns:
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


def node_embedding_generation(
        new_x: torch, 
        labels: Optional[pd.DataFrame], 
        ) -> None:
    
    if not os.path.exists(EMBEDDINGS/LEARNING):
        os.mkdir(EMBEDDINGS/LEARNING)
    learning_model = load_model(labels=labels)
    for edge_file in os.listdir(EDGES):
        edge_index = pd.read_csv(f"{EDGES}/{edge_file}")
        if "Unnamed: 0" in edge_index.columns:
            edge_index.drop("Unnamed: 0", axis=1, inplace=True)
        best_ValidLoss = np.Inf
        for learning_rate, hid_size in product(LEARNING_RATE, HIDDEN_SIZE):
            av_valid_losses = []
            for _ in range(X_TIME2):
                data = learning_model.prepare_data(
                    new_x=new_x, edge_index=edge_index
                    )
                criterion, out_size = learning_model.select_model()
                in_size = data.x.shape[1]
                model = Net(in_size=in_size, hid_size=hid_size, out_size=out_size)
                optimizer = select_optimizer("adam", model, learning_rate)
                min_valid_loss = np.Inf
                patience_count = 0
                for epoch in range(MAX_EPOCHS):
                    emb = learning_model.train(
                        model=model, optimizer=optimizer, data=data, criterion=criterion)
                    this_valid_loss, emb = learning_model.validate(
                        model=model, criterion=criterion, data=data
                    )
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

                selected_emb = this_emb

        embedding_path = f"{EMBEDDINGS}/{LEARNING}/{edge_file.split('.csv')[0]}"
        pd.DataFrame(selected_emb).to_csv(f"{embedding_path}.csv", index=False)
