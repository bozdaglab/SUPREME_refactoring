import logging
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import pandas as pd
import ray
import torch
from helper import (
    chnage_connections_thr,
    get_stat_methos,
    nan_checker,
    search_dictionary,
)
from networkx.readwrite import json_graph
from pre_processings import pre_processing
from scipy.stats import pearsonr, spearmanr
from set_logging import set_log_config
from settings import EDGES, LABELS
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import remove_self_loops

logger = logging.getLogger()
set_log_config()
FUNC_NAME = "only_one"


def process_data(edge_index: Dict):
    graph = nx.DiGraph(json_graph.node_link_graph(edge_index))
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def prepare_data(raw_paths: List[str], processed_dir: Path) -> Tuple:
    sample_data = defaultdict()
    labels = ""
    users = defaultdict()
    for file in raw_paths:
        if file.endswith(".pkl"):
            with open(file, "rb") as pkl_file:
                data = pickle.load(pkl_file)
        if file.endswith(".txt"):
            with open(file, "rb") as txt_file:
                data = pd.read_csv(txt_file, sep="\t")
        else:
            continue
        if "data_cna" in file or "data_mrna" in file:
            data.drop("Entrez_Gene_Id", axis=1, inplace=True)
        to_save_folder = save_folder(file=file, processed_dir=processed_dir)
        make_directory(to_save_folder)
        if "data_clinical_patient" in file:
            patient_id = data["PATIENT_ID"]
            data.drop("PATIENT_ID", axis=1, inplace=True)
            data = data.apply(LabelEncoder().fit_transform)
            labels = data[["CLAUDIN_SUBTYPE"]]
            labels["PATIENT_ID"] = patient_id
        else:
            hugo_symbol = data["Hugo_Symbol"]
            data.drop("Hugo_Symbol", axis=1, inplace=True)
            patient_id = data.columns
            data = data.T
            data.columns = hugo_symbol.values
        file_name = file.split(".")[0].split("/")[-1]
        if nan_checker(data):
            logger.info(f"Start preprocessing {file_name}")
            data = pre_processing(data)
        data.insert(0, column="PATIENT_ID", value=patient_id)
        data = data.set_index("PATIENT_ID")
        users[file_name] = data.index
        sample_data[file_name] = data
    sample_data, labels = set_same_users(sample_data, users, labels)
    save_file(sample_data, labels, processed_dir)
    return sample_data, labels


def save_file(sample_data: Dict, labels: pd.DataFrame, processed_dir: Path) -> None:
    logger.info("Save file in a pickle format...")
    for name, featuers in sample_data.items():
        featuers.to_pickle(f"{processed_dir}/{name}.pkl")
    if not os.path.exists(LABELS):
        os.mkdir(LABELS)
    labels.to_pickle(f"{LABELS}/labels.pkl")


def make_directory(dir_name: Path) -> None:
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def make_directories(dir_name: Path) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_folder(file: str, processed_dir: Path) -> Path:
    to_save_folder = processed_dir
    if "edges_" in file:
        to_save_folder = EDGES
    elif "labels." in file:
        to_save_folder = LABELS
    return to_save_folder


def set_same_users(sample_data: Dict, users: Dict, labels: Dict) -> Dict:
    new_dataset = defaultdict()
    shared_users = search_dictionary(users, len(users) - 1)
    shared_users = sorted(shared_users)[0:50]
    shared_users_encoded = LabelEncoder().fit_transform(shared_users)
    for file_name, data in sample_data.items():
        new_dataset[file_name] = data[data.index.isin(shared_users)].set_index(
            shared_users_encoded
        )
    return new_dataset, labels[labels["PATIENT_ID"].isin(shared_users)].set_index(
        shared_users_encoded
    )


@ray.remote(num_cpus=os.cpu_count())
def compute_similarity(
    data: pd.DataFrame,
    file_name: str,
    stat: str,
    func_name: str,
    stat_model: Union[pearsonr, spearmanr],
    correlation_dictionary: Dict,
    path_dir: Path,
) -> Dict:
    logger.info(f"Start generating similarity matrix for {stat}, {file_name}...")
    thr = chnage_connections_thr(file_name, stat)
    func = select_generator(func_name, thr)
    correlation_dictionary["final_path"] = path_dir / file_name
    correlation_dictionary["func_name"] = func_name
    for ind_i, patient_1 in data.iterrows():
        for ind_j, patient_2 in data.iterrows():
            if ind_i == ind_j:
                continue
            try:
                similarity_score = stat_model(
                    patient_1.values, patient_2.values
                ).statistic
            except AttributeError:
                similarity_score = stat_model(patient_1.values, patient_2.values)[0]
            func(ind_i, ind_j, similarity_score, correlation_dictionary)
    return correlation_dictionary


@ray.remote(num_cpus=os.cpu_count())
def similarity_matrix_generation(new_dataset: Dict, stat, func_name=FUNC_NAME):
    stat_model = get_stat_methos(stat)
    path_dir = EDGES / stat
    make_directories(path_dir)
    correlation_dictionary = defaultdict(list)
    result = [
        compute_similarity.remote(
            data,
            file_name,
            stat,
            func_name,
            stat_model,
            correlation_dictionary,
            path_dir,
        )
        for file_name, data in new_dataset.items()
    ]
    return ray.get(result)


def save_pickle(final_result: List[Dict]) -> None:
    for result in final_result:
        func_name = result["func_name"]
        path = Path(result["final_path"]).parts
        file_name = path[-1]
        path_dir = Path(*path[:-1])
        result.pop("func_name")
        result.pop("final_path")
        if func_name == "only_one_nx":
            result["directed"] = False
            result["multigraph"] = False
            result["graph"] = {}
            with open(path_dir / f"similarity_{file_name}.pkl", "wb") as pickle_file:
                pickle.dump(result, pickle_file)
        elif any(result):
            pd.DataFrame(
                result.values(),
                columns=list(result.items())[0][1].keys(),
            ).to_pickle(path_dir / f"similarity_{file_name}.pkl")


def _similarity_only_one(
    thr: float,
    ind_i: int,
    ind_j: int,
    similarity_score: float,
    correlation_dictionary: Dict,
):
    if similarity_score > thr:
        correlation_dictionary[f"{ind_i}_{ind_j}"] = {
            "Patient_1": ind_i,
            "Patient_2": ind_j,
            "link": 1,
            "Similarity Score": similarity_score,
        }


def _similarity_zero_one(
    thr: float,
    ind_i: int,
    ind_j: int,
    similarity_score: float,
    correlation_dictionary: Dict,
):
    correlation_dictionary[f"{ind_i}_{ind_j}"] = {
        "Patient_1": ind_i,
        "Patient_2": ind_j,
        "link": 1 if similarity_score > thr else 0,
        "Similarity Score": similarity_score,
    }


def _similarity_only_one_nx(
    thr: float,
    ind_i: int,
    ind_j: int,
    similarity_score: float,
    correlation_dictionary: Dict,
):
    if similarity_score > thr:
        correlation_dictionary["nodes"].append({"id": ind_i})
        correlation_dictionary["links"].append({"source": ind_i, "target": ind_j})


def select_generator(
    func_name: str, thr: float
) -> Union[_similarity_only_one, _similarity_zero_one, _similarity_only_one_nx]:
    factory = {
        "only_one": _similarity_only_one,
        "zero_one": _similarity_zero_one,
        "only_one_nx": _similarity_only_one_nx,
    }
    return partial(factory[func_name], thr)
