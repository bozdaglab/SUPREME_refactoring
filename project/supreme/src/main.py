import logging
import os
import pickle
import time
import warnings
from collections import defaultdict
from itertools import combinations
from typing import List

import pandas as pd
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
)
from sklearn.preprocessing import LabelEncoder
from train_mls import train_ml_model

load_dotenv(find_dotenv())
set_log_config()
logger = logging.getLogger()
warnings.filterwarnings("ignore", category=FutureWarning)


if not os.path.exists(BASE_DATAPATH):
    raise FileNotFoundError(f"no such a director {BASE_DATAPATH}")


def combine_trails(ml_type: str) -> List[List[int]]:
    final_trial_combs = defaultdict()
    base_path = EMBEDDINGS / ml_type
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


start = time.time()
new_dataset, labels = set_same_users(
    sample_data=sample_data, users=users, labels=labels
)
final_correlation = similarity_matrix_generation(new_dataset=new_dataset)
logger.info("SUPREME is running..")
if isinstance(labels, pd.Series):
    for feature_type in SELECTION_METHOD:
        new_x = node_feature_generation(
            new_dataset=new_dataset, labels=labels, feature_type=feature_type
        )
        train_valid_idx, test_idx = random_split(new_x)
        node_embedding_generation(
            new_x=new_x,
            labels=labels,
            final_correlation=final_correlation,
            feature_type=feature_type,
        )
else:
    new_x = node_feature_generation(new_dataset=new_dataset, labels=labels)
    train_valid_idx, test_idx = random_split(new_x)
    node_embedding_generation(
        new_x=new_x, labels=labels, final_correlation=final_correlation
    )
start2 = time.time()

logger.info(
    f"It took {str(round(start2 - start, 1))} seconds for node embedding generation"
)
logger.info("SUPREME is integrating the embeddings..")
for ml_type in LEARNING:
    trial_combs = combine_trails(ml_type=ml_type)
    for trial_name, trial in trial_combs.items():
        path_name = f"{EMBEDDINGS}/{ml_type}/{trial_name}"
        for feature_selection_types in os.listdir(path_name):
            result_path = f"{EMBEDDINGS}/result/{feature_selection_types}"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            for trials in range(len(trial)):
                path_to_files = f"{path_name}/{feature_selection_types}"
                final_result = train_ml_model(
                    trial_combs=trial,
                    trials=trials,
                    labels=labels,
                    train_valid_idx=train_valid_idx,
                    test_idx=test_idx,
                    dir_name=f"{path_name}/{feature_selection_types}",
                )
                with open(f"{result_path}/{trial_name}_result.txt", "a") as file:
                    logger.info(f"Combination {trials}, selected parameters:")
                    for key, res in final_result.items():
                        logger.info(f"{key}: {res}")
                        file.write(f"{key}: {res}\n")
                    file.write("\n\n")
                    logger.info("\n\n")

end = time.time()
logger.info(f"It took {round(end - start, 1)} seconds in total.")
logger.info("SUPREME is done.")
