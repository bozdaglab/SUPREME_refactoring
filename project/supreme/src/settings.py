import os
from distutils.util import strtobool
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


INPUT_SIZE = os.environ.get("INPUT_SIZE")
HIDDEN_SIZE = os.environ.get("HIDDEN_SIZE")
OUT_SIZE = os.environ.get("OUT_SIZE")


FORMAT = "%(asctime)s.%(msecs)03d %(name)-8s %(levelname)-4s %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"
BASE_DATAPATH = Path(__file__).parent.parent.parent.parent / "data" / "sample_data"
EMBEDDINGS = BASE_DATAPATH / "embeddings"
EDGES = BASE_DATAPATH / "edges"
DATA = BASE_DATAPATH / "data"
LABELS = BASE_DATAPATH / "labels"
IMPUTER_NAME_SUBSET = os.environ.get("IMPUTER_NAME_SUBSET")
IMPUTER_NAME_WHOLE = os.environ.get("IMPUTER_NAME_WHOLE")
CLASS_NAME = os.environ.get("CLASS_NAME")
GROUPBY_COLUMNS = ["CDR_Sum", "ID_Gender"]
FEATURE_NETWORKS_INTEGRATION = [
    i for i in os.listdir(BASE_DATAPATH) if i.endswith(".csv")
]
FEATURE_TO_DROP = ["Med_ID", "Visit_ID", "CDR_Sum"]
NODE_NETWORKS = FEATURE_NETWORKS_INTEGRATION.copy()
LEARNING = "clustering"
OPTIM = "adam"
LEARNING_RATE = [0.01]
HIDDEN_SIZE = [32]
X_TIME = 50
FEATURE_SELECTION_PER_NETWORK = [False, False, False]
TOP_FEATURES_PER_NETWORK = [50, 50, 50]

POS_NEG = True  # bool(os.environ.get("NODE2VEC"))
ONLY_POS = False
DISCRIMINATOR = True
NODE2VEC = True  # bool(os.environ.get("NODE2VEC"))
MASKING = False  # bool(os.environ.get("MASKING"))

UNNAMED = "Unnamed: 0"
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL")
ADD_RAW_FEAT = bool(strtobool(os.environ.get("ADD_RAW_FEAT")))
if LEARNING == "clustering":
    INT_MOTHOD = os.environ.get("INT_MOTHOD_CLUSTERING")
else:
    INT_MOTHOD = os.environ.get("INT_MOTHOD_CLASSIFICATION")
# X_TIME2 = os.environ.get("X_TIME2")
X_TIME2 = 2
# MAX_EPOCHS = os.environ.get("MAX_EPOCHS")
# MAX_EPOCHS = os.environ.get("MAX_EPOCHS")
# MAX_EPOCHS = os.environ.get("MAX_EPOCHS")
MAX_EPOCHS = 2
MIN_EPOCHS = 1
PATIENCE = 0
OPTIONAL_FEATURE_SELECTION = bool(
    strtobool(os.environ.get("OPTIONAL_FEATURE_SELECTION"))
)
# BORUTA_RUNS = os.environ.get("BORUTA_RUNS")
BORUTA_RUNS = 100
# BORUTA_TOP_FEATURES = os.environ.get("BORUTA_TOP_FEATURES")
BORUTA_TOP_FEATURES = 50
EMBEDDING_DIM = 128
WALK_LENGHT = 6
CONTEXT_SIZE = 2
WALK_PER_NODE = 3
P = 1.0
Q = 1.0
SPARSE = True
