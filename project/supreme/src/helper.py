import os
from collections import Counter
from itertools import repeat
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from boruta import BorutaPy
from feature_engine.selection import SelectByShuffling, SelectBySingleFeaturePerformance
from learning_types import FeatureSelectionType
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
# from genetic_selection import GeneticSelectionCV
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import pearsonr, spearmanr
from settings import CNA, LABELS, METHYLATION, MICRO
from sklearn.feature_selection import RFE, SelectFromModel
from torch import Tensor
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.loader import NeighborLoader, DataLoader
DEVICE = torch.device("cpu")


def pos_neg_generator(pos_neg_data: Tensor, edges: Tensor):
    mask = torch.zeros_like(pos_neg_data, dtype=bool)
    for idx in range(len(pos_neg_data[0])):
        if all(torch.isin(dim[idx], edges) for dim in pos_neg_data):
            mask[:, idx] = True
    return mask

def chnage_connections_thr(file_name: str) -> float:
    if "data_methylation" in file_name:
        thr = METHYLATION
    elif "data_mrna" in file_name:
        thr = MICRO
    elif "data_cna" in file_name:
        thr = CNA
    return thr


def data_loader(data):
    if isinstance(data, list):
        train_data = DataLoader(dataset=data, batch_size=8, shuffle=True)
    else:
        train_data = NeighborLoader(data=data, batch_size=16, shuffle=True,
                        num_neighbors=[5, 10],
                        num_workers=6, persistent_workers=True)
    return train_data

def re_generate_pos_neg(batch_data):
    pos_bool = pos_neg_generator(batch_data.pos_edge_labels,batch_data.edge_index)
    filtered_pos = batch_data.pos_edge_labels[pos_bool]
    batch_data.pos_edge_labels = filtered_pos.view(pos_bool.shape[0], -1)
    neg_bool = pos_neg_generator(batch_data.neg_edge_labels,batch_data.edge_index)
    filtered_neg = batch_data.neg_edge_labels[neg_bool]
    batch_data.neg_edge_labels = filtered_neg.view(neg_bool.shape[0], -1)
    return batch_data.to(DEVICE)

def read_labels() -> pd.DataFrame:
    return pd.read_pickle(LABELS / os.listdir(LABELS)[0])["CLAUDIN_SUBTYPE"]


def search_dictionary(methods_features: Dict, thr: int = 2) -> List[str]:
    count_features = Counter()
    for features in methods_features.values():
        count_features.update(features)
    return [feat for feat, val in count_features.items() if val > thr]


def pos_neg(edge_index: pd.DataFrame, column: str, col_idx: int) -> Tensor:
    return torch.tensor(
        edge_index[edge_index[column] == col_idx].iloc[:, 0:2].values.T, device=DEVICE
    ).long()


def train_test_ratio(new_x: torch) -> List[int]:
    """
    This function defines the ratio for train test splitting

    Parameters:
    -----------
    new_x:
        Concatenated features from different omics file
    Return:
        A list of two values that shows the ratio between
        the train (The first value) and test (the second one)

    """
    shape = new_x.shape[0]
    train_idx = round(shape * 0.75)
    return [train_idx, shape - train_idx]


def features_ratio(number: int) -> int:
    return round(number * 70 / 100) - 1


def lower_upper_bound(new_alpha: float, mulitply: int = 2, range_value=6) -> np.array:
    large = new_alpha * mulitply
    lower_bound = abs(new_alpha - large)
    upper_bound = new_alpha + large
    return np.linspace(lower_bound, upper_bound, range_value)

def silhouette(emb: Tensor):
    emb = emb.detach().cpu().numpy()
    pred = KMeans(n_clusters=7).fit_predict(emb)
    return silhouette_score(emb, pred)


def random_split(new_x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths

    Parameters:
    -----------
    new_x:
         Concatenated features from different omics file

    Return:
        New datasets
    """
    return torch.utils.data.random_split(new_x, train_test_ratio(new_x))


def masking_indexes(data, indexes):
    return np.array([i in set(indexes) for i in range(data.x.shape[0])])


def select_boruta(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    boruta = BorutaPy(
        estimator=xgb.XGBClassifier(),
        n_estimators="auto",
        verbose=2,
        random_state=42,
        max_iter=40,
    )
    boruta.fit(X.values, y.values)
    return X[X.columns[boruta.support_]]


def get_stat_methos(stat_method: str) -> Union[spearmanr, pearsonr]:
    factory = {"spearman": spearmanr, "pearson": pearsonr}
    try:
        return factory[stat_method]
    except KeyError:
        raise KeyError("Please check your stat model")


def drop_rows(application_train: pd.DataFrame, gh: List[str]) -> pd.DataFrame:
    return application_train[gh].reset_index(drop=True)


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    num_nodes = maybe_num_nodes(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index


def load_models1(
    feature_selection_type: str,
    mlmodel: str,
    feature_number: int,
) -> [RFE, SelectFromModel, SequentialFeatureSelector]:
    if feature_selection_type == FeatureSelectionType.RFE.name:
        return RFE(
            estimator=mlmodel,
            n_features_to_select=feature_number,
            step=30,
            verbose=5,
        )

    elif feature_selection_type == FeatureSelectionType.SelectFromModel.name:
        return SelectFromModel(
            estimator=mlmodel,
            max_features=feature_number,
            threshold="1.25*median",
        )

    elif feature_selection_type == FeatureSelectionType.SequentialFeatureSelector.name:
        return SequentialFeatureSelector(
            estimator=mlmodel,
            k_features=feature_number,
            forward=True,
            verbose=2,
            cv=5,
            n_jobs=2,
            scoring="r2",
        )


def row_col_ratio(feat: pd.DataFrame) -> bool:
    row, columns = feat.shape
    return True if row / columns < 0.60 else False


def nan_checker(feat: pd.DataFrame) -> bool:
    return True if sum(feat.isna().sum()) > 0 else False


def load_models2(
    feature_selection_type: str, mlmodel: str
) -> [
    SelectBySingleFeaturePerformance,
    SelectByShuffling,
    # GeneticSelectionCV,
    BorutaPy,
]:
    if (
        feature_selection_type
        == FeatureSelectionType.SelectBySingleFeaturePerformance.name
    ):
        return SelectBySingleFeaturePerformance(
            estimator=mlmodel,
            scoring="roc_auc",
            threshold=0.6,
            cv=3,
        )

    elif feature_selection_type == FeatureSelectionType.SelectByShuffling.name:
        return SelectByShuffling(
            estimator=mlmodel,
            scoring="roc_auc",
            threshold=0.06,
            cv=3,
        )

    # elif feature_selection_type == FeatureSelectionType.GeneticSelectionCV.name:
    #     return GeneticSelectionCV(
    #         estimator=mlmodel,
    #         cv=5,
    #         scoring="accuracy",
    #         max_features=12,
    #         n_population=10,
    #         crossover_proba=0.5,
    #         mutation_proba=0.2,
    #         n_generations=10,
    #         crossover_independent_proba=0.5,
    #         mutation_independent_proba=0.05,
    #         n_gen_no_change=10,
    #         n_jobs=-1,
    #     )

    elif feature_selection_type == FeatureSelectionType.BorutaPy.name:
        return BorutaPy(
            estimator=mlmodel,
            n_estimators="auto",
            verbose=2,
            random_state=42,
            max_iter=50,
        )
