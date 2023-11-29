import logging

import pandas as pd
from set_logging import set_log_config
from settings import IMPUTER_NAME_SUBSET
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import ExtraTreeRegressor

set_log_config()
logger = logging.getLogger()


def pre_processing(data: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    # data = duplication(data=data)
    # data = drop_columns(data=data)
    data = handle_missing(data=data)
    if normalize:
        data = normalization(data=data)
    return data.reset_index(drop=True)


def duplication(data: pd.DataFrame) -> pd.DataFrame:
    duplicated_columns = data.columns[data.columns.duplicated()]
    if not any(duplicated_columns):
        return data
    for column in duplicated_columns:
        for idx, (col, _) in enumerate(data[column].items()):
            data.rename(columns={col: f"{col}_{idx}"}, inplace=True)
    return data


def drop_columns(data, thr: float = 0.90) -> pd.DataFrame:
    for col in data.columns:
        if (
            any(data[col].isna())
            and data[col].isna().value_counts().to_dict().get(True) / len(data[col])
            > thr
        ):
            data = data.drop(col, axis=1)
    return data


def normalization(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)


def load_missing_method(imputer_name):
    if imputer_name == "KNNImputer":
        return KNNImputer(n_neighbors=2)
    elif imputer_name == "IterativeImputer":
        return IterativeImputer(
            max_iter=2, estimator=ExtraTreeRegressor(), keep_empty_features=True
        )
    return None


def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns
    imputer_1 = load_missing_method(IMPUTER_NAME_SUBSET)
    return pd.DataFrame(imputer_1.fit_transform(data), columns=cols)
