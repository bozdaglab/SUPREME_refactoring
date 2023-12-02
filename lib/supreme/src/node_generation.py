import os
import statistics
from itertools import product
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from feature_selections import select_features
from helper import nan_checker, row_col_ratio
from learning_types import LearningTypes
from pre_processings import pre_processing
from selected_models import (
    GCNSupervised,
    GCNUnsupervised,
    load_model,
    select_model,
    select_optimizer,
)
from settings import (
    EMBEDDINGS,
    HIDDEN_SIZE,
    LEARNING,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    OPTIM,
    OPTIONAL_FEATURE_SELECTION,
    PATIENCE,
    X_TIME2,
)
from torch import Tensor

DEVICE = torch.device("cpu")


def node_feature_generation(
    new_dataset: Dict, labels: Dict, feature_type: Optional[str] = None
) -> Tensor:
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
    for _, feat in new_dataset.items():
        if row_col_ratio(feat):
            feat = feat[feat.columns[0:300]]
            if nan_checker(feat):
                feat = pre_processing(feat)
            feat = select_features(
                application_train=feat, labels=labels, feature_type=feature_type
            )
            values = torch.tensor(feat.values, device=DEVICE)
        else:
            values = feat.values
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
    final_correlation: Dict,
    stat: str,
    feature_type: Optional[str] = None,
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

    for model_choice in LEARNING:
        # emb_path = EMBEDDINGS / model_choice / stat
        # if not os.path.exists(emb_path):
        #     os.makedirs(emb_path)
        if isinstance(feature_type, list):
            feature_type = "_".join(feature_type)
        learning_model = load_model(new_x=new_x, labels=labels, model=model_choice)
        for name, edge_index in final_correlation.items():
            if model_choice == LearningTypes.clustering.name:
                dir_path = f"{EMBEDDINGS}/{model_choice}/{stat}/{feature_type}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                list_dir = os.listdir(dir_path)
                name_ = f"{name}.pkl"
                name_dir = f"{dir_path}/{name_}"
                if list_dir and name_ in list_dir:
                    continue
                train_steps(
                    learning_model=learning_model,
                    edge_index=edge_index,
                    name=name_dir,
                    model_choice=model_choice,
                )
            else:
                train_steps(
                    learning_model=learning_model,
                    edge_index=edge_index,
                    name=name,
                    model_choice=model_choice,
                )


def add_row_features(emb: Tensor, is_first: bool = True) -> Tensor:
    """
    This function adds row features and performs feature selection
    on the row features and concat it to the embeddings

    Parameters:
    -----------

    emb:
        Embeddings of the trial(s)
    is_first:
        True, features from the first file, False, features other files

    Return:
        Concatenation of row features and embeddings
    """
    for addFeatures in os.listdir(EMBEDDINGS / LEARNING):
        features = pd.read_pickle(f"{EMBEDDINGS}/{LEARNING}/{addFeatures}")

        if is_first:
            allx = torch.tensor(features.values, device=DEVICE).float()
            is_first = False
        else:
            allx = torch.cat(
                (allx, torch.tensor(features.values, device=DEVICE).float()), dim=1
            )

    if OPTIONAL_FEATURE_SELECTION is True:
        pass
    else:
        emb = torch.cat((emb, allx), dim=1)

    return emb


def train_steps(
    learning_model: Union[GCNUnsupervised, GCNSupervised],
    edge_index: pd.DataFrame,
    name: str,
    model_choice: str,
):

    """
    This function craete the loss funciton, train and validate the model
    """

    data = learning_model.prepare_data(edge_index=edge_index)
    best_ValidLoss = np.Inf
    out_size = learning_model.model_loss_output(model_choice=model_choice)
    in_size = data.x.shape[1]
    for learning_rate, hid_size in product(LEARNING_RATE, HIDDEN_SIZE):
        model = select_model(
            in_size=in_size,
            hid_size=hid_size,
            out_size=out_size,
            super_unsuper_model=model_choice,
        )
        av_valid_losses = []
        for _ in range(X_TIME2):
            optimizer = select_optimizer(OPTIM, model, learning_rate)
            min_valid_loss = np.Inf
            patience_count = 0
            for epoch in range(MAX_EPOCHS):
                model.train(optimizer, data)
                this_valid_loss, emb = model.validate(data=data)
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
    pd.DataFrame(selected_emb).to_pickle(f"{name}")
