import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from helper import load_missing_method
from set_logging import set_log_config
from settings import (
    CLASS_NAME,
    FEATURE_TO_DROP,
    GROUPBY_COLUMNS,
    IMPUTER_NAME_SUBSET,
    IMPUTER_NAME_WHOLE,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

set_log_config()
logger = logging.getLogger()


def pre_processing(all_cohorts: Dict) -> pd.DataFrame:
    data = drop_class(data=all_cohorts)
    data = drop_columns(data=data)
    data = handle_missing(data=data)
    data = normalization(data=data)
    return data.reset_index(drop=True)


def drop_columns(data, thr: float = 0.90) -> pd.DataFrame:
    for col in data.columns:
        if (
            any(data[col].isna())
            and data[col].isna().value_counts().to_dict().get(True) / len(data[col])
            > thr
        ):
            data = data.drop(col, axis=1)
    return data


def drop_class(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(data[data[CLASS_NAME] == np.nan].index).reset_index(drop=True)


def drop_rows(application_train: pd.DataFrame, gh: List[str]) -> pd.DataFrame:
    for i in application_train:
        if i not in gh:
            application_train = application_train.drop(i, axis=1)
    return application_train.reset_index(drop=True)


def value_count(data1: pd.DataFrame) -> bool:
    d = data1[CLASS_NAME].value_counts().keys()
    return 0 in d and 1 in d
    # return len(set(data1[CLASS_NAME].value_counts().keys())) > 1


def normalization(data: pd.DataFrame) -> pd.DataFrame:
    features = []
    features.extend([*FEATURE_TO_DROP, CLASS_NAME, "Age"])
    original_data = data[features]
    new_data = data.drop(features, axis=1)
    cols = new_data.columns
    scale_data = pd.DataFrame(MinMaxScaler().fit_transform(new_data), columns=cols)
    return scale_data.merge(original_data, left_index=True, right_index=True)


def label_encoding(all_cohorts: Dict) -> pd.DataFrame:
    for col in all_cohorts.columns:
        if all_cohorts[col].dtype.name == "object":
            original_data = all_cohorts[col]
            nal_values = all_cohorts[col].isnull()
            maximum_value = 0
            for ind, value in enumerate(all_cohorts[col]):
                if isinstance(value, int):
                    if value > maximum_value:
                        maximum_value = value
                    nal_values[ind] = True

            all_cohorts[col] = LabelEncoder().fit_transform(
                all_cohorts[col].astype(str)
            )
            all_cohorts[col] = all_cohorts[col] + maximum_value
            all_cohorts[col].where(~nal_values, original_data, inplace=True)
    return all_cohorts


def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns
    sub_dataset = pd.DataFrame(columns=cols)
    for _, sample_data in data.groupby(GROUPBY_COLUMNS):
        imputer_1 = load_missing_method(IMPUTER_NAME_SUBSET)
        if imputer_1:
            try:
                new_data = pd.DataFrame(
                    imputer_1.fit_transform(sample_data), columns=cols
                )
                sub_dataset = pd.concat([sub_dataset, new_data])
            except:
                sub_dataset = pd.concat([sub_dataset, sample_data])

    imputer_2 = load_missing_method(IMPUTER_NAME_WHOLE)
    return pd.DataFrame(imputer_2.fit_transform(sub_dataset), columns=cols)


def missing_value_stat(data: dict) -> pd.DataFrame:
    missing_values = data.isna().sum()
    total_cells = np.product(data.shape)
    total_missing_percentage = pd.Series(
        {"total_missing_percentage": (missing_values.sum() / total_cells) * 100}
    )
    missing_values.append(total_missing_percentage)
    return pd.DataFrame(missing_values)
