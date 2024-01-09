import logging
import os
import sys
import warnings
from itertools import combinations
from typing import List

import pandas as pd
from dataset import BioDataset
from dotenv import find_dotenv, load_dotenv
from node_generation import node_embedding_generation
from set_logging import set_log_config
from settings import BASE_DATAPATH, EMBEDDINGS, LABELS, LEARNING
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
    t = range(len(os.listdir(base_path)))
    trial_combs = []
    for r in range(1, len(t) + 1):
        trial_combs.extend([list(x) for x in combinations(t, r)])
    return trial_combs


logger.info("SUPREME is running..")

BioDataset(
    root="data/sample_data/",
    raw_directories=["graph_data"],
    file_name=os.listdir(f"{BASE_DATAPATH}/raw"),
)

node_embedding_generation()

labels = pd.read_pickle(LABELS / os.listdir(LABELS)[0])["CLAUDIN_SUBTYPE"]
logger.info("SUPREME is integrating the embeddings..")
for ml_type in LEARNING:
    dir = EMBEDDINGS / ml_type
    for models_type in os.listdir(dir):
        trial_combs = combine_trails(base_path=dir / models_type)
        path_to_files = os.path.join(dir, models_type)
        result_path = os.path.join(EMBEDDINGS, "result", ml_type, models_type)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for idx, trials in enumerate(trial_combs):
            final_result = train_ml_model(
                ml_type=ml_type,
                trial_combs=trial_combs,
                trials=idx,
                labels=labels,
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
