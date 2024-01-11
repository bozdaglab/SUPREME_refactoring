import json
import os
from distutils.util import strtobool
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


ADD_RAW_FEAT = bool(strtobool(os.environ.get("ADD_RAW_FEAT")))
LEARNING = json.loads(os.environ.get("LEARNING"))
OPTIONAL_FEATURE_SELECTION = bool(
    strtobool(os.environ.get("OPTIONAL_FEATURE_SELECTION"))
)
ROOT = os.environ.get("ROOT")
INT_MOTHOD_CLUSTERING = json.loads(os.environ.get("INT_MOTHOD_CLUSTERING"))
INT_MOTHOD_CLASSIFICATION = json.loads(os.environ.get("INT_MOTHOD_CLASSIFICATION"))
NUMBER_FEATURES = [50]
X_ITER = 30
SELECTION_METHOD = [
    "lasso",
    "RFE",
    "SelectFromModel",
    "BorutaPy",
    "pearson",
    "SelectBySingleFeaturePerformance",
    "SelectByShuffling",
    "GeneticSelectionCV",
    "SequentialFeatureSelector",
    ["BorutaPy", "pearson", "RFE", "SelectFromModel", "lasso"],
]
METHYLATION_P = float(os.environ.get("METHYLATION_P"))
METHYLATION_S = float(os.environ.get("METHYLATION_S"))
MICRO = float(os.environ.get("MICRO"))
CNA = float(os.environ.get("CNA"))
MODELS_B = ["RF"]
UNSUPERVISED_MODELS = json.loads(os.environ.get("UNSUPERVISED_MODELS"))
POS_NEG_MODELS = json.loads(os.environ.get("POS_NEG_MODELS"))
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL")
IMPUTER_NAME_SUBSET = os.environ.get("IMPUTER_NAME_SUBSET")
IMPUTER_NAME_WHOLE = os.environ.get("IMPUTER_NAME_WHOLE")
CLASS_NAME = os.environ.get("CLASS_NAME")
GROUPBY_COLUMNS = json.loads(os.environ.get("GROUPBY_COLUMNS"))
FEATURE_TO_DROP = json.loads(os.environ.get("FEATURE_TO_DROP"))
OPTIM = os.environ.get("OPTIM")
STAT_METHOD = json.loads(os.environ.get("STAT_METHOD"))
LEARNING_RATE = json.loads(os.environ.get("LEARNING_RATE"))
HIDDEN_SIZE = json.loads(os.environ.get("HIDDEN_SIZE"))
X_TIME = int(os.environ.get("X_TIME"))
DISCRIMINATOR = bool(os.environ.get("DISCRIMINATOR"))
NODE2VEC = bool(os.environ.get("NODE2VEC"))
SIMILARITY_BASED = bool(os.environ.get("SIMILARITY_BASED"))
TRAIN_TEST = bool(os.environ.get("TRAIN_TEST"))
MASKING = bool(os.environ.get("MASKING"))
LINKPREDICTION = bool(os.environ.get("LINKPREDICTION"))
ENCODERINPRODUCT = bool(os.environ.get("ENCODERINPRODUCT"))
UNNAMED = os.environ.get("UNNAMED")
BORUTA_RUNS = int(os.environ.get("BORUTA_RUNS"))
BORUTA_TOP_FEATURES = int(os.environ.get("BORUTA_TOP_FEATURES"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM"))
WALK_LENGHT = int(os.environ.get("WALK_LENGHT"))
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE"))
WALK_PER_NODE = int(os.environ.get("WALK_PER_NODE"))
P = int(os.environ.get("P"))
Q = int(os.environ.get("Q"))
SPARSE = bool(os.environ.get("SPARSE"))

FORMAT = "%(asctime)s.%(msecs)03d %(name)-8s %(levelname)-4s %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"
BASE_DATAPATH = Path(__file__).parent.parent.parent.parent / "data" / "sample_data"
EMBEDDINGS = BASE_DATAPATH / "embeddings"
EDGES = BASE_DATAPATH / "edges"
DATA = BASE_DATAPATH / "data"
LABELS = BASE_DATAPATH / "labels"
X_TIME2 = int(os.environ.get("X_TIME2"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS"))
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS"))
PATIENCE = int(os.environ.get("PATIENCE"))
GRAPH_DATA = BASE_DATAPATH / "processed" / "graph_data"
FEATURE_NETWORKS_INTEGRATION = [
    i for i in os.listdir(BASE_DATAPATH) if i.endswith(".csv")
]
PATH_FEATURES = DATA.parent / "selected_features"
PATH_EMBEDDIGS = DATA.parent / "selected_features_embeddings"

NODE_NETWORKS = FEATURE_NETWORKS_INTEGRATION.copy()
FEATURE_SELECTION_PER_NETWORK = [False, False, False]
TOP_FEATURES_PER_NETWORK = [50, 50, 50]
