import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from boruta import BorutaPy
from feature_engine.selection import SelectByShuffling, SelectBySingleFeaturePerformance
from learning_types import FeatureSelectionType

# from genetic_selection import GeneticSelectionCV
from mlxtend.feature_selection import SequentialFeatureSelector
from pre_processings import pre_processing
from scipy.stats import pearsonr, spearmanr
from settings import EDGES
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

DEVICE = torch.device("cpu")


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


def get_stat_methos(stat_method: str):
    factory = {"spearman": spearmanr, "pearson": pearsonr}
    try:
        return factory[stat_method]
    except KeyError:
        raise KeyError("Please check your stat model")


def set_same_users(sample_data: Dict, users: Dict, labels: Dict) -> Dict:
    new_dataset = defaultdict()
    shared_users = search_dictionary(users, len(users) - 1)
    shared_users = sorted(shared_users)
    shared_users_encoded = LabelEncoder().fit_transform(shared_users)
    for file_name, data in sample_data.items():
        new_dataset[file_name] = data[data.index.isin(shared_users)].set_index(
            shared_users_encoded
        )
    return new_dataset, labels[shared_users]


def drop_rows(application_train: pd.DataFrame, gh: List[str]) -> pd.DataFrame:
    return application_train[gh].reset_index(drop=True)


def similarity_matrix_generation(
    new_dataset: Dict, stat: str, thr: float = 0.60
) -> Dict:
    # parqua dataset, parallel
    final_correlation = defaultdict()
    path_dir = EDGES / stat
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    file_names = os.listdir(path_dir)
    if file_names:
        for file in file_names:
            final_correlation[file.split(".")[0]] = pd.read_pickle(f"{path_dir}/{file}")
        return final_correlation

    for file_name, data in new_dataset.items():
        correlation_dictionary = defaultdict()
        if nan_checker(data):
            data = pre_processing(data=data)
        stat_model = get_stat_methos(stat)
        for ind_i, patient_1 in data.iterrows():
            for ind_j, patient_2 in data[ind_i + 1 :].iterrows():
                try:
                    similarity_score = stat_model(
                        patient_1.values, patient_2.values
                    ).statistic
                except AttributeError:
                    similarity_score = stat_model(patient_1.values, patient_2.values)[0]
                if similarity_score > 0.0:
                    correlation_dictionary[f"{ind_i}_{ind_j}"] = {
                        "Patient_1": ind_i,
                        "Patient_2": ind_j,
                        "Similarity Score": similarity_score,
                        "link": 1 if similarity_score > thr else 0,
                    }

        final_correlation[file_name] = correlation_dictionary

        pd.DataFrame(
            correlation_dictionary.values(),
            columns=list(correlation_dictionary.items())[0][1].keys(),
        ).to_pickle(path_dir / f"similarity_{file_name}.pkl")
    return final_correlation


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
            max_iter=10,
        )
