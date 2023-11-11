import logging
import os
import pickle
import time
import warnings
from itertools import combinations
from typing import List

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from helper import random_split, similarity_matrix_generation
from node_generation import node_embedding_generation, node_feature_generation
from set_logging import set_log_config
from settings import BASE_DATAPATH, DATA, EDGES, EMBEDDINGS, LABELS, LEARNING, UNNAMED
from train_mls import train_ml_model

load_dotenv(find_dotenv())
set_log_config()
logger = logging.getLogger()
warnings.filterwarnings("ignore", category=FutureWarning)


if not os.path.exists(BASE_DATAPATH):
    raise FileNotFoundError(f"no such a director {BASE_DATAPATH}")

if not os.path.exists(EMBEDDINGS):
    os.makedirs(EMBEDDINGS)


def combine_trails() -> List[List[int]]:
    t = range(len(os.listdir(EMBEDDINGS / LEARNING)))
    trial_combs = []
    for r in range(1, len(t) + 1):
        trial_combs.extend([list(x) for x in combinations(t, r)])
    return trial_combs
    # return [combinations(NODE_NETWORKS, i) for i in range(1, len(NODE_NETWORKS)+1)]


def pickle_to_csv():
    """
    Search in the given repository, and load pickle files, if exist,
    and convert them to csv with the same name
    """
    for file in os.listdir(BASE_DATAPATH):
        if file.endswith(".pkl"):
            with open(f"{BASE_DATAPATH}/{file}", "rb") as pkl_file:
                data = pickle.load(pkl_file)
            if "edges_" in file:
                to_save_folder = EDGES
            elif "labels." in file:
                to_save_folder = LABELS
            else:
                to_save_folder = DATA
            pd.DataFrame(data).to_csv(f"{to_save_folder}/{file.split('.')[0]}.csv")


labels = None
for file in os.listdir(LABELS):
    if not file:
        break
    labels = pd.read_csv(f"{LABELS}/{file}")
    if UNNAMED in labels:
        labels = labels.drop(UNNAMED, axis=1)


start = time.time()
pickle_to_csv()
similarity_matrix_generation()
logger.info("SUPREME is running..")
new_x = node_feature_generation(labels=labels)
train_valid_idx, test_idx = random_split(new_x)
node_embedding_generation(new_x=new_x, labels=labels)
start2 = time.time()

logger.info(
    f"It took {str(round(start2 - start, 1))} seconds for node embedding generation"
)
logger.info("SUPREME is integrating the embeddings..")

trial_combs = combine_trails()

for trials in range(len(trial_combs)):
    final_result = train_ml_model(
        trial_combs=trial_combs,
        trials=trials,
        labels=labels,
        train_valid_idx=train_valid_idx,
        test_idx=test_idx,
    )
    logger.info(f"Combination {trials}, selected parameters:")
    for key, res in final_result.items():
        logger.info(f"{key}: {res}")
    logger.info("Done\n")


end = time.time()
logger.info(f"It took {round(end - start, 1)} seconds in total.")
logger.info("SUPREME is done.")
