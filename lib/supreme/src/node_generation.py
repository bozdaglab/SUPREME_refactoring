import logging
import os
import statistics
from functools import partial
from itertools import product
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import ray
import torch
from dataset import process_data

# from feature_selections import select_features
from helper import edge_index_from_dict, nan_checker, row_col_ratio
from learning_types import LearningTypes
from pre_processings import pre_processing
from ray import tune

# from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from selected_models import (
    GCNSupervised,
    GCNUnsupervised,
    load_model,
    select_model,
    select_optimizer,
)
from set_logging import set_log_config
from settings import (  # HIDDEN_SIZE,; LEARNING_RATE,
    EMBEDDINGS,
    HIDDEN_SIZE,
    LEARNING,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    OPTIM,
    OPTIONAL_FEATURE_SELECTION,
    PATIENCE,
    POS_NEG_MODELS,
    UNSUPERVISED_MODELS,
    X_TIME2,
)
from torch import Tensor

set_log_config()
logger = logging.getLogger()
DEVICE = torch.device("cpu")


config = {
    "hidden_size": tune.choice([2**i for i in range(4, 9)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    # "batch_size": tune.choice([2, 4, 8, 16])
}
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=20,
    grace_period=1,
    reduction_factor=2,
)


@ray.remote(num_cpus=os.cpu_count())
def node_feature_generation(
    new_dataset: Dict,
    labels: Dict,
    path_features: str,
    path_embeggings: str,
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
    is_first = True
    # selected_features = []
    for _, feat in new_dataset.items():
        if row_col_ratio(feat):
            if nan_checker(feat):
                feat = pre_processing(feat)
            # add an inner remote function and use get to get the result of the inner one before proceding
            # feat, final_features = select_features(
            #     application_train=feat, labels=labels, feature_type=feature_type
            # )
            # selected_features.extend(final_features)
            if not any(feat):
                continue
            values = torch.tensor(feat.values, device=DEVICE)
        else:
            # selected_features.extend(feat.columns)
            values = feat.values
        if is_first:
            new_x = torch.tensor(values, device=DEVICE).float()
            is_first = False
        else:
            new_x = torch.cat(
                (new_x, torch.tensor(values, device=DEVICE).float()), dim=1
            )
    # if not os.path.exists(path_features):
    #     os.makedirs(path_features)
    # pd.DataFrame(selected_features).to_pickle(
    #     path_features / f"selected_features_{feature_type}.pkl"
    # )
    if not os.path.exists(path_embeggings):
        os.makedirs(path_embeggings)
    pd.DataFrame(new_x).to_pickle(path_embeggings / f"embeddings_{feature_type}.pkl")


def node_embedding_generation(
    stat: str,
    new_x: Tensor,
    labels: Optional[pd.DataFrame],
    final_correlation: Dict,
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
        emb_path = EMBEDDINGS / model_choice / stat
        if not os.path.exists(emb_path):
            os.makedirs(emb_path)
        # embeddings_file = os.listdir(emb_path)
        # if embeddings_file:
        #     for name in embeddings_file:
        #         path_dir = f"{emb_path}/{name}"
        #         for plk_file in os.listdir(path_dir):
        #             embeddings[name].append(pd.read_pickle(f"{path_dir}/{plk_file}"))
        #     return embeddings
        if isinstance(feature_type, list):
            feature_type = "_".join(feature_type)
        learning_model = load_model(new_x=new_x, labels=labels, model=model_choice)
        for name, edge_index in final_correlation.items():
            # d = edge_index.set_index(edge_index.columns[0])
            # edge_index = edge_index_from_dict(d.to_dict()["related"])
            if model_choice == LearningTypes.clustering.name:
                for data_gen_types, unsupervised_model in product(
                    POS_NEG_MODELS, UNSUPERVISED_MODELS
                ):
                    dir_path = f"{emb_path}/{data_gen_types}_{unsupervised_model}/{feature_type}"
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    list_dir = os.listdir(dir_path)
                    name_ = f"{name}.pkl"
                    name_dir = f"{dir_path}/{name_}"
                    if list_dir and name_ in list_dir:
                        continue

                    # result = tune.run(
                    #     partial(
                    #         train_steps,
                    #         data_generation_types=data_gen_types,
                    #         learning_model=learning_model,
                    #         edge_index=edge_index,
                    #         name=name_dir,
                    #         model_choice=model_choice,
                    #         super_unsuper_model=unsupervised_model,
                    #     ),
                    #     resources_per_trial={"cpu": 2, "gpu": 0},
                    #     config=config,
                    #     num_samples=10,
                    #     scheduler=scheduler,
                    # )
                    # best_trial = result.get_best_trial("loss", "min", "last")
                    # print(f"Best trial config: {best_trial.config}")
                    # print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
                    # print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

                    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
                    # device = "cpu"
                    # if torch.cuda.is_available():
                    #     device = "cuda:0"
                    #     if gpus_per_trial > 1:
                    #         best_trained_model = nn.DataParallel(best_trained_model)
                    # best_trained_model.to(device)

                    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
                    # best_checkpoint_data = best_checkpoint.to_dict()

                    # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

                    # test_acc = test_accuracy(best_trained_model, device)
                    # print("Best trial test set accuracy: {}".format(test_acc))
                    train_steps(
                        data_generation_types=data_gen_types,
                        learning_model=learning_model,
                        edge_index=edge_index,
                        name=name_dir,
                        model_choice=model_choice,
                        super_unsuper_model=unsupervised_model,
                    )
            else:
                tune.run(
                    partial(
                        train_steps,
                        learning_model=learning_model,
                        edge_index=edge_index,
                        name=name,
                        model_choice=model_choice,
                    ),
                    resources_per_trial={"cpu": os.cpu_count(), "gpu": 0},
                    config=config,
                    num_samples=10,
                    scheduler=scheduler,
                )
                # train_steps(
                #     config=config,
                #     learning_model=learning_model,
                #     edge_index=edge_index,
                #     name=name,
                #     model_choice=model_choice,
                # )


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
    # config: Dict,
    learning_model: Union[GCNUnsupervised, GCNSupervised],
    edge_index: pd.DataFrame,
    name: str,
    model_choice: str,
    data_generation_types: Optional[str] = None,
    super_unsuper_model: Optional[str] = None,
):
    """
    This function craete the loss funciton, train and validate the model
    """
    if not super_unsuper_model:
        super_unsuper_model = model_choice
    data = learning_model.prepare_data(
        data_generation_types=data_generation_types, edge_index=edge_index
    )
    process_data(edge_index=edge_index)
    best_ValidLoss = np.Inf
    out_size = 32  # learning_model.model_loss_output(model_choice=model_choice)
    in_size = data.x.shape[1]
    min_valid_loss = np.Inf
    this_emb = None
    model_accuracy = 0
    for learning_rate, hid_size in product(LEARNING_RATE, HIDDEN_SIZE):
        # out_size = hid_size
        model = select_model(
            in_size=in_size,
            hid_size=hid_size,
            out_size=out_size,
            super_unsuper_model=super_unsuper_model,
        )
        av_valid_losses = []
        optimizer = select_optimizer(
            optimizer_type=OPTIM, model=model, learning_rate=learning_rate
        )  # add OPTIM to the actual function
        # checkpoint = session.get_checkpoint()

        # if checkpoint:
        #     checkpoint_state = checkpoint.to_dict()
        #     # start_epoch = checkpoint_state["epoch"]
        #     model.load_state_dict(checkpoint_state["net_state_dict"])
        #     optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        # else:
        #     start_epoch = 0
        for x_times in range(X_TIME2):
            patience_count = 0
            for epoch in range(MAX_EPOCHS):
                loss, emb = model.train(optimizer, data)
                auc, ap, val_loss = model.validate(data=data)

                logger.info(
                    f"Number of times: {x_times}, epoch: {epoch}, train_loss: {loss}, auc: {auc}, ap: {ap}",
                )
                if loss < min_valid_loss and auc > model_accuracy:
                    model_accuracy = auc
                    min_valid_loss = loss
                    patience_count = 0
                    this_emb = emb
                else:
                    patience_count += 1
                if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
                    break

            # checkpoint_data = {
            #     "epoch": epoch,
            #     "net_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            # }
            # checkpoint = Checkpoint.from_dict(checkpoint_data)

            # session.report(
            #     {
            #         "loss": min_valid_loss,
            #         "accuracy": min_valid_loss,
            #         "embeddings": this_emb,
            #     },
            #     checkpoint=checkpoint,
            # )

        av_valid_losses.append(min_valid_loss)

    # print("Finished Training")
    # av_valid_loss = round(statistics.median(av_valid_losses), 3)
    # if av_valid_loss < best_ValidLoss:
    #     best_ValidLoss = av_valid_loss
    #     selected_emb = this_emb
    # pd.DataFrame(selected_emb).to_pickle(f"{name}")
