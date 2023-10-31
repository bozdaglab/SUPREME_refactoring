from enum import Enum, auto


class LearningTypes(Enum):
    classification = auto()
    regression = auto()
    clustering = auto()



class EmbeddingModel(Enum):
    gcn_ori = auto()
    gcn_encoder = auto()