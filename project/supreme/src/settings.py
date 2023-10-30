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
LEARNING_RATE = [0.01, 0.001, 0.0001]
HIDDEN_SIZE = [32, 64, 128, 256]
X_TIME = 50
FEATURE_SELECTION_PER_NETWORK = [False, False, False]
TOP_FEATURES_PER_NETWORK = [50, 50, 50]

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL")
ADD_RAW_FEAT = bool(strtobool(os.environ.get("ADD_RAW_FEAT")))
INT_MOTHOD = os.environ.get("INT_MOTHOD")
# X_TIME2 = os.environ.get("X_TIME2")
X_TIME2 = 10
# MAX_EPOCHS = os.environ.get("MAX_EPOCHS")
# MAX_EPOCHS = os.environ.get("MAX_EPOCHS")
# MAX_EPOCHS = os.environ.get("MAX_EPOCHS")
MAX_EPOCHS = 500
MIN_EPOCHS = 200
PATIENCE = 30
OPTIONAL_FEATURE_SELECTION = bool(
    strtobool(os.environ.get("OPTIONAL_FEATURE_SELECTION"))
)
# BORUTA_RUNS = os.environ.get("BORUTA_RUNS")
BORUTA_RUNS = 100
# BORUTA_TOP_FEATURES = os.environ.get("BORUTA_TOP_FEATURES")
BORUTA_TOP_FEATURES = 50
