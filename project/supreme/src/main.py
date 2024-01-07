import logging
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations, product
from typing import List

import pandas as pd
import torch
from dataset import BioDataset
from dotenv import find_dotenv, load_dotenv
from helper import random_split
from node_generation import node_embedding_generation
from set_logging import set_log_config
from settings import (
    BASE_DATAPATH,
    EDGES,
    EMBEDDINGS,
    LABELS,
    LEARNING,
    PATH_EMBEDDIGS,
    STAT_METHOD,
)
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


logger.info("SUPREME is running..")

BioDataset(root="data/sample_data/", file_name=os.listdir(f"{BASE_DATAPATH}/raw"))

labels = pd.read_pickle(LABELS / os.listdir(LABELS)[0])
if not os.path.exists(EMBEDDINGS):
    for stat in os.listdir(EDGES):
        final_correlation = defaultdict()
        for file in os.listdir(EDGES / stat):
            final_correlation[file] = pd.read_pickle(EDGES / stat / file)
        for idx, feature_type in enumerate(os.listdir(PATH_EMBEDDIGS)):
            new_x = pd.read_pickle(PATH_EMBEDDIGS / feature_type)
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
