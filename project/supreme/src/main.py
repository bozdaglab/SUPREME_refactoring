import argparse
import errno
import itertools
import logging
import os
import time
import warnings

import numpy as np
import pickle5 as pickle
from node_generation import node_embedding_generation, node_feature_generation
from set_logging import set_log_config
from settings import (
    FEATURE_SELECTION_PER_NETWORK,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NODE_NETWORKS,
    OPTIONAL_FEATURE_SELECTION,
    PATH,
)
from sklearn.model_selection import train_test_split

from lib.supreme.src.train_mls import ml

set_log_config()
logger = logging.getLogger()


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
BASE_PATH = ""
random_state = 404


if (True in FEATURE_SELECTION_PER_NETWORK) or (OPTIONAL_FEATURE_SELECTION is True):

    from rpy2.robjects.packages import importr

    utils = importr("utils")
    rFerns = importr("rFerns")
    Boruta = importr("Boruta")
    pracma = importr("pracma")
    dplyr = importr("dplyr")

# Parser
parser = argparse.ArgumentParser(
    description="""An integrative node
classification framework, called SUPREME
(a subtype prediction methodology), that
utilizes graph convolutions on multiple
datatype-specific networks that are annotated
with multiomics datasets as node features.
This framework is model-agnostic and could be
applied to any classification problem with
properly processed datatypes and networks."""
)
parser.add_argument("-data", "--data_location", nargs=1, default=["sample_data"])

args = parser.parse_args()
dataset_name = args.data_location[0]

SAMPLE_PATH = PATH / dataset_name


if not os.path.exists(SAMPLE_PATH):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), SAMPLE_PATH)


save_path = PATH / f"SUPREME_{dataset_name}_results"

if not os.path.exists(save_path):
    os.makedirs(save_path)

file = SAMPLE_PATH / "labels.pkl"
with open(file, "rb") as f:
    labels = pickle.load(f)

file = SAMPLE_PATH / "mask_values.pkl" # csv file
if os.path.exists(file):
    with open(file, "rb") as f:
        train_valid_idx, test_idx = pickle.load(f)
else:
    train_valid_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.20, shuffle=True, stratify=labels
    )

start = time.time()

logger.info("SUPREME is running..")
new_x = node_feature_generation(SAMPLE_PATH)
node_embedding_generation(
    SAMPLE_PATH, new_x, train_valid_idx, labels, test_idx, save_path
)
start2 = time.time()

logger.info(
    f"It took {str(round(start2 - start, 1))}"
    f"seconds for node embedding generation "
    f"({str(len(LEARNING_RATE) * len(HIDDEN_SIZE))} trials for {str(len(NODE_NETWORKS))} seperate GCNs)."
)

logger.info("SUPREME is integrating the embeddings..")

addFeatures = []
t = range(len(NODE_NETWORKS))
trial_combs = []
for r in range(1, len(t) + 1):
    trial_combs.extend([list(x) for x in itertools.combinations(t, r)])

for trials in range(len(trial_combs)):
    final_result = ml(
        save_path, dataset_name, trial_combs, trials, labels, train_valid_idx, test_idx
    )

    print(
        f"Combination {trials}  {NODE_NETWORKS} >  selected parameters:\n"
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
