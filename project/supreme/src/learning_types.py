from enum import Enum, auto


class LearningTypes(Enum):
    classification = auto()
    regression = auto()
    clustering = auto()


class OptimizerType(Enum):
    sgd = auto()
    adam = auto()
    sparse_adam = auto()


class FeatureSelectionType(Enum):
    RFE = auto()
    SelectFromModel = auto()
    SequentialFeatureSelector = auto()
    SelectBySingleFeaturePerformance = auto()
    SelectByShuffling = auto()
    GeneticSelectionCV = auto()
    BorutaPy = auto
