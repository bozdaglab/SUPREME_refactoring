import os
from distutils.util import strtobool
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


FORMAT = "%(asctime)s.%(msecs)03d %(name)-8s %(levelname)-4s %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"
FEATURE_NETWORKS_INTEGRATION = [
    "clinical",
    "cna",
    "exp",
]
NODE_NETWORKS = ["clinical", "cna", "exp"]
LEARNING_RATE = [0.01, 0.001, 0.0001]
HIDDEN_SIZE = [32, 64, 128, 256]
X_TIME = 50
FEATURE_SELECTION_PER_NETWORK = [False, False, False]
TOP_FEATURES_PER_NETWORK = [50, 50, 50]
PATH = Path(__file__).parent.parent.parent.parent / "data"

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL")
ADD_RAW_FEAT = bool(strtobool(os.environ.get("ADD_RAW_FEAT")))
INT_MOTHOD = os.environ.get("MLP")
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
