import logging
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations, product
from typing import List

import pandas as pd
import ray
import torch
from dotenv import find_dotenv, load_dotenv
from helper import random_split, set_same_users, similarity_matrix_generation
from node_generation import node_embedding_generation, node_feature_generation
from set_logging import set_log_config
from settings import (
    BASE_DATAPATH,
    DATA,
    EDGES,
    EMBEDDINGS,
    LABELS,
    LEARNING,
    SELECTION_METHOD,
    STAT_METHOD,
)
from sklearn.preprocessing import LabelEncoder
from train_mls import train_ml_model

load_dotenv(find_dotenv())
LIB = os.environ.get("LIB")
PROJECT = os.environ.get("PROJECT")

sys.path.append(LIB)
sys.path.append(PROJECT)
set_log_config()
logger = logging.getLogger()
warnings.filterwarnings("ignore", category=FutureWarning)


if not os.path.exists(BASE_DATAPATH):
    raise FileNotFoundError(f"no such a director {BASE_DATAPATH}")


def combine_trails(base_path: str) -> List[List[int]]:
    final_trial_combs = defaultdict()
    for file in os.listdir(base_path):
        t = range(len(os.listdir(base_path / file)))
        trial_combs = []
        for r in range(1, len(t) + 1):
            trial_combs.extend([list(x) for x in combinations(t, r)])
        final_trial_combs[file] = trial_combs
    return final_trial_combs
    # return [combinations(NODE_NETWORKS, i) for i in range(1, len(NODE_NETWORKS)+1)]


def txt_to_pickle():
    """
    Search in the given repository, and load pickle/txt files, and generate
    labels, and input data
    """
    encoder = LabelEncoder()
    sample_data = defaultdict()
    labels = ""
    users = defaultdict()
    list_files = os.listdir(BASE_DATAPATH)
    if all(file_name in os.listdir(BASE_DATAPATH) for file_name in ["data", "labels"]):
        for file in os.listdir(DATA):
            data = pd.read_pickle(f"{DATA}/{file}")
            sample_data[file] = data
            if "PATIENT_ID" in data.columns:
                patient_id = data["PATIENT_ID"]
            else:
                patient_id = data.index
            users[file.split(".")[0]] = patient_id
        for file in os.listdir(LABELS):
            labels = pd.read_pickle(f"{LABELS}/{file}")
        return sample_data, users, labels

    for file in list_files:
        if file.endswith(".pkl"):
            with open(f"{BASE_DATAPATH}/{file}", "rb") as pkl_file:
                data = pickle.load(pkl_file)
        if file.endswith(".txt"):
            with open(f"{BASE_DATAPATH}/{file}", "rb") as txt_file:
                data = pd.read_csv(txt_file, sep="\t")
        else:
            continue
        if "data_cna" in file or "data_mrna" in file:
            data.drop("Entrez_Gene_Id", axis=1, inplace=True)
        if "edges_" in file:
            to_save_folder = EDGES
        elif "labels." in file:
            to_save_folder = LABELS
        else:
            to_save_folder = DATA
        if not os.path.exists(to_save_folder):
            os.mkdir(to_save_folder)
        if "data_clinical_patient" in file:
            patient_id = data["PATIENT_ID"]
            data.drop("PATIENT_ID", axis=1, inplace=True)
            if not os.path.exists(LABELS):
                os.mkdir(LABELS)

            data = data.apply(encoder.fit_transform)
            data["PATIENT_ID"] = patient_id
            data = data.set_index("PATIENT_ID")
            data["CLAUDIN_SUBTYPE"].to_pickle(f"{LABELS}/labels.pkl")
            labels = data["CLAUDIN_SUBTYPE"]
        else:
            hugo_symbol = data["Hugo_Symbol"]
            data.drop("Hugo_Symbol", axis=1, inplace=True)
            data = data.T
            data.columns = hugo_symbol.values
            patient_id = data.index
        file_name = file.split(".")[0]
        users[file_name] = patient_id
        sample_data[file_name] = data
        data.to_pickle(f"{to_save_folder}/{file_name}.pkl")
    return sample_data, users, labels


# do preprocessing here
sample_data, users, labels = txt_to_pickle()


new_dataset, labels = set_same_users(
    sample_data=sample_data, users=users, labels=labels
)

# ray.init()


# @ray.remote(num_cpus=os.cpu_count())
# def compute_similarity(new_dataset: Dict, stat: str):
#     similarity_matrix_generation.remote(new_dataset=new_dataset, stat=stat)


if os.path.exists(EDGES):
    pass
else:
    similarity_result_ray = [
        similarity_matrix_generation(new_dataset, stat) for stat in STAT_METHOD
    ]
    # ray.wait(similarity_result_ray)

logger.info("SUPREME is running..")
path_features = DATA.parent / "selected_features"
path_embeggings = DATA.parent / "selected_features_embeddings"

if os.path.exists(path_embeggings):
    pass
else:
    embeddings_result_ray = [
        node_feature_generation.remote(
            new_dataset=new_dataset,
            labels=labels,
            feature_type=feature_type,
            path_features=path_features,
            path_embeggings=path_embeggings,
        )
        for feature_type in SELECTION_METHOD
    ]

    ray.wait(embeddings_result_ray)

if not os.path.exists(EMBEDDINGS):
    for stat in os.listdir(EDGES):
        final_correlation = defaultdict()
        for file in os.listdir(EDGES / stat):
            final_correlation[file] = pd.read_pickle(EDGES / stat / file)
        for idx, feature_type in enumerate(os.listdir(path_embeggings)):
            new_x = pd.read_pickle(path_embeggings / feature_type)
            if isinstance(new_x, pd.DataFrame):
                new_x = torch.tensor(new_x.values, dtype=torch.float32)
            train_valid_idx, test_idx = random_split(new_x)
            node_embedding_generation(
                stat=stat,
                new_x=new_x,
                labels=labels,
                final_correlation=final_correlation,
                feature_type=feature_type,
            )
    # else:
    #     new_x = node_feature_generation(new_dataset=new_dataset, labels=labels)
    #     train_valid_idx, test_idx = random_split(new_x)
    #     node_embedding_generation(
    #         new_x=new_x, labels=labels, final_correlation=final_correlation
    #     )

start2 = time.time()

# logger.info(
#     f"It took {str(round(start2 - start, 1))} seconds for node embedding generation"
# )
logger.info("SUPREME is integrating the embeddings..")
for ml_type, stat in product(LEARNING, STAT_METHOD):
    dir = EMBEDDINGS / ml_type / stat
    for models_type in os.listdir(dir):
        trial_combs = combine_trails(base_path=dir / models_type)
        for trial_name, trial in trial_combs.items():
            path_to_files = f"{dir}/{models_type}/{trial_name}"
            result_path = f"{EMBEDDINGS}/result/{ml_type}/{stat}/{trial_name}"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            logger.info(f"Similarity stat: {stat}")
            for idx, trials in enumerate(trial):
                final_result = train_ml_model(
                    ml_type=ml_type,
                    trial_combs=trial,
                    trials=idx,
                    labels=labels,
                    train_valid_idx=train_valid_idx,
                    test_idx=test_idx,
                    dir_name=path_to_files,
                )

                with open(
                    f"{result_path}/{'_'.join(str(i) for i in trials)}_result.txt", "a"
                ) as file:
                    logger.info(f"Combination {trials}, selected parameters:")
                    for key, res in final_result.items():
                        logger.info(f"{key}: {res}")
                        file.write(f"{key}: {res}\n")
                    file.write("\n\n")
                    logger.info("\n\n")

# end = time.time()
# logger.info(f"It took {round(end - start, 1)} seconds in total.")
# logger.info("SUPREME is done.")
