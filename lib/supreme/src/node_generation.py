import os
import statistics
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import torch
from helper import select_boruta
from selected_models import load_model, select_model, select_optimizer
from settings import (
    DATA,
    EDGES,
    EMBEDDINGS,
    FEATURE_SELECTION_PER_NETWORK,
    HIDDEN_SIZE,
    LEARNING,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    OPTIM,
    PATIENCE,
    UNNAMED,
    X_TIME2,
)
from torch import Tensor

DEVICE = torch.device("cpu")


def node_feature_generation(labels: Optional[pd.DataFrame]) -> Tensor:
    """
    Load features from each omic separately, apply feature selection if needed,
    and contact them together

    Parameters:
    ----------
    labels:
        Dataset labels in case we want to apply feature selection algorithems


    Return:
        Concatenated features from different omics file
    """
    is_first = True
    for file in os.listdir(DATA):
        feat = pd.read_csv(f"{DATA}/{file}")
        if UNNAMED in feat.columns:
            feat = feat.drop(UNNAMED, axis=1)
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
    new_x: Tensor,
    labels: Optional[pd.DataFrame],
) -> None:
    """
    This function loads edges, turns SUPREME to supervised or unsupervised
    and generates embeddings for each omic

    Parameters:
    ----------
    new_x:
        Concatenated features from different omics file
    labels:
        Dataset labels

    Return:
        Generate embeddings for each omic
    """
    if not os.path.exists(EMBEDDINGS / LEARNING):
        os.mkdir(EMBEDDINGS / LEARNING)
    learning_model = load_model(new_x=new_x, labels=labels)
    for edge_file in os.listdir(EDGES):
        edge_index = pd.read_csv(f"{EDGES}/{edge_file}")
        if UNNAMED in edge_index.columns:
            edge_index.drop(UNNAMED, axis=1, inplace=True)
        best_ValidLoss = np.Inf
        for learning_rate, hid_size in product(LEARNING_RATE, HIDDEN_SIZE):
            av_valid_losses = []
            for _ in range(X_TIME2):
                data = learning_model.prepare_data(edge_index=edge_index)
                criterion, out_size = learning_model.model_loss_output()
                in_size = data.x.shape[1]
                model = select_model(
                    in_size=in_size, hid_size=hid_size, out_size=out_size
                )
                optimizer = select_optimizer(OPTIM, model, learning_rate)
                min_valid_loss = np.Inf
                patience_count = 0
                for epoch in range(MAX_EPOCHS):
                    # emb = model.encode(
                    #     model=model, optimizer=optimizer, data=data, criterion=criterion
                    # )
                    model.train()
                    optimizer.encoder_loss.zero_grad()
                    emb = model.encode(data.x, data.edge_index, data.edge_attr)
                    for i in range(5):
                        optimizer.decoder_loss.zero_grad()
                        discriminator_loss = model.discriminator_loss(emb)
                        discriminator_loss.backward()
                        optimizer.decoder_loss.step()
                    loss = model.recon_loss(emb, data.pos_edge_labels)
                    loss = loss + model.reg_loss(emb)
                    loss = loss + (1 / data.num_nodes) * model.kl_loss()
                    loss.backward()
                    optimizer.encoder_loss.step()

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
