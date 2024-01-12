import logging
import os
from collections import defaultdict

# import statistics
from functools import partial
from itertools import product
from typing import Dict, Optional

import numpy as np
import pandas as pd
import ray
import torch
from feature_selections import select_features
from helper import row_col_ratio
from learning_types import LearningTypes  # SuperUnsuperModel
from ray import train, tune
from ray.train import report
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from selected_models import select_model, select_optimizer
from set_logging import set_log_config
from settings import (  # X_TIME2,
    EMBEDDINGS,
    GRAPH_DATA,
    LEARNING,
    MAX_EPOCHS,
    MIN_EPOCHS,
    OPTIM,
    OPTIONAL_FEATURE_SELECTION,
    PATIENCE,
    UNSUPERVISED_MODELS,
)
from torch import Tensor

set_log_config()
logger = logging.getLogger()
DEVICE = torch.device("cpu")

config = {
    "hidden_size": tune.choice([2**i for i in range(4, 5)]),
    "lr": tune.loguniform(1e-4, 1e-3),
    # "batch_size": tune.choice([2, 4, 8, 16])
}
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=20,
    grace_period=20,
    reduction_factor=2,
)


@ray.remote(num_cpus=os.cpu_count())
def node_feature_generation(
    new_dataset: Dict,
    labels: Dict,
    feature_type: Optional[str] = None,
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
    selected_features = defaultdict(list)
    selected_features["feature_type"] = feature_type
    result = [
        features_selection.remote(feat, selected_features, labels, feature_type)
        for _, feat in new_dataset.items()
    ]

    return ray.get(result)


@ray.remote(num_cpus=os.cpu_count())
def features_selection(feat, selected_features, labels, feature_type):
    if row_col_ratio(feat):
        feat, final_features = select_features(
            application_train=feat, labels=labels, feature_type=feature_type
        )
        selected_features["features"].extend(final_features)
        if any(feat):
            values = torch.tensor(feat.values, device=DEVICE)
    else:
        selected_features["features"].extend(feat.columns.tolist())
        values = feat.values
    selected_features["tensors"].append(torch.tensor(values, device=DEVICE).float())

    return selected_features


def node_embedding_generation() -> None:
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
    ready_data = [
        os.path.join(direc, file)
        for direc, _, files in os.walk(GRAPH_DATA)
        if files
        for file in files
    ]
    for model_choice in LEARNING:
        emb_path = EMBEDDINGS / model_choice
        if model_choice == LearningTypes.clustering.name:
            for data_gen_types, unsupervised_model in product(
                ready_data, UNSUPERVISED_MODELS
            ):
                base_path = data_gen_types.split("graph_data/")[1]
                folder_path, file_path = base_path.split("/")
                name = f"{folder_path}_{unsupervised_model}"
                final_path = emb_path / name
                if not os.path.exists(final_path):
                    os.makedirs(final_path)
                list_dir = os.listdir(final_path)
                name_ = file_path.replace(".pt", ".pkl")
                name_dir = f"{final_path}/{name_}"
                if list_dir and name_ in list_dir:
                    continue

                result = tune.run(
                    partial(
                        train_steps,
                        data_generation_types=data_gen_types,
                        model_choice=model_choice,
                        super_unsuper_model=unsupervised_model,
                    ),
                    resources_per_trial={"cpu": 5, "gpu": 0},
                    config=config,
                    num_samples=10,
                    scheduler=scheduler,
                )
                best_trial = result.get_best_trial("loss", "min", "avg")
                logger.info(f"Best trial config: {best_trial.config}")
                logger.info(
                    f"Best trial final validation loss: {best_trial.last_result['loss']}"
                )
                selected_emb = torch.from_numpy(best_trial.last_result["embeddings"])
                logger.info(f"Ready to save {name} embeddings ....")
                pd.DataFrame(selected_emb).to_pickle(f"{name_dir}")

        else:
            tune.run(
                partial(
                    train_steps,
                    name=name,
                    model_choice=model_choice,
                ),
                resources_per_trial={"cpu": os.cpu_count(), "gpu": 0},
                config=config,
                num_samples=10,
                scheduler=scheduler,
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
    config: Dict,
    model_choice: str,
    data_generation_types: Optional[str] = None,
    super_unsuper_model: Optional[str] = None,
) -> ExperimentAnalysis:
    """
    This function craete the loss funciton, train and validate the model
    """
    if not super_unsuper_model:
        super_unsuper_model = model_choice
    data = torch.load(data_generation_types)
    # best_ValidLoss = np.Inf
    out_size = 32  # learning_model.model_loss_output(model_choice=model_choice)
    in_size = data.x.shape[1]
    min_valid_loss = np.Inf
    this_emb = None
    # model_accuracy = 0
    # if super_unsuper_model == SuperUnsuperModel.entireinput.name:
    #     metric_1 = "r2_square"
    #     metric_2 = "mean_square_error"
    # else:
    #     metric_1 = "auc"
    #     metric_2 = "ap"
    model = select_model(
        in_size=in_size,
        hid_size=config["hidden_size"],
        out_size=out_size,
        super_unsuper_model=super_unsuper_model,
    )
    # av_valid_losses = []
    optimizer = select_optimizer(
        optimizer_type=OPTIM, model=model, learning_rate=config["lr"]
    )  # add OPTIM to the actual function
    # scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    checkpoint = train.get_checkpoint()

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            epoch = checkpoint_dict["epoch"] + 1
            # model.load_state_dict(checkpoint_dict["model_state_dict"])
    for epoch in range(MAX_EPOCHS):
        loss, emb = model.train(optimizer, data)
        val_loss = model.validate(data=data)
        logger.info(
            f"Epoch: {epoch}, train_loss: {loss}, val_loss: {val_loss}",
        )
        # scheduler_lr.step()
        if loss < min_valid_loss:  # and auc > model_accuracy:
            # model_accuracy = auc
            min_valid_loss = loss
            patience_count = 0
            this_emb = emb
        else:
            patience_count += 1
        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            break
        metrics = {
            "loss": min_valid_loss,
            "epoch": epoch,
            "embeddings": this_emb.detach().cpu().numpy(),
        }
        report(
            metrics=metrics,
        )

    # av_valid_losses.append(min_valid_loss)

    # av_valid_loss = round(statistics.median(av_valid_losses), 3)
    # if av_valid_loss < best_ValidLoss:
    #     best_ValidLoss = av_valid_loss
    #     selected_emb = this_emb
    #     selected_emb = selected_emb.detach()
