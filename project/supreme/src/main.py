import logging
import os
import time
import warnings
from itertools import combinations

import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from learning_types import LearningTypes
from node_generation import node_embedding_generation, node_feature_generation
from set_logging import set_log_config
from settings import (
    BASE_DATAPATH,
    EMBEDDINGS,
    HIDDEN_SIZE,
    LABELS,
    LEARNING,
    LEARNING_RATE,
)
from train_mls import ml

load_dotenv(find_dotenv())
set_log_config()
logger = logging.getLogger()
warnings.filterwarnings("ignore", category=FutureWarning)


if not os.path.exists(BASE_DATAPATH):
    raise FileNotFoundError(f"no such a director {BASE_DATAPATH}")

if not os.path.exists(EMBEDDINGS):
    os.makedirs(EMBEDDINGS)


def combine_trails():
    t = range(len(os.listdir(EMBEDDINGS)))
    trial_combs = []
    for r in range(1, len(t) + 1):
        trial_combs.extend([list(x) for x in combinations(t, r)])
    return trial_combs
    # return [combinations(NODE_NETWORKS, i) for i in range(1, len(NODE_NETWORKS)+1)]


labels = None
if LEARNING in [LearningTypes.regression.name, LearningTypes.classification.name]:
    for file in os.listdir(LABELS):
        labels = pd.read_csv(f"{LABELS}/{file}")
        labels = labels.drop("Unnamed: 0", axis=1)


start = time.time()

logger.info("SUPREME is running..")
new_x = node_feature_generation(labels)

node_embedding_generation(new_x, labels, learning=LEARNING)
start2 = time.time()

logger.info(
    f"It took {str(round(start2 - start, 1))}"
    f"seconds for node embedding generation "
    f"({str(len(LEARNING_RATE) * len(HIDDEN_SIZE))} trials for {str(len(os.listdir(EMBEDDINGS)))} seperate GCNs)."
)

logger.info("SUPREME is integrating the embeddings..")

trial_combs = combine_trails()

for trials in range(len(trial_combs)):
    final_result = ml(
        trial_combs=trial_combs,
        trials=trials,
        labels=labels,
        train_valid_idx=train_valid,
        test_idx=test,
    )

    print(
        f"Combination {trials}  {os.listdir(EMBEDDINGS)} >  selected parameters:\n"
        f"Best_params: {final_result['best_parameters']}\n"
        f"Train accuracy: {final_result['tr_result_acc']}\n"
        f"Train weighted-f1: {final_result['tr_result_wf1']}\n"
        f"Train macro-f1: {final_result['tr_result_mf1']}\n"
        f"Test accuracy: {final_result['result_acc']}\n"
        f"Test weighted-f1: {final_result['result_wf1']}\n"
        f"Test macro-f1: {final_result['result_mf1']}"
    )


end = time.time()
logger.info(f"It took {round(end - start, 1)} seconds in total.")
logger.info("SUPREME is done.")
