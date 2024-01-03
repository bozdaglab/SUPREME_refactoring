import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Dict

import networkx as nx
import pandas as pd
import ray
from helper import (
    chnage_connections_thr,
    get_stat_methos,
    nan_checker,
    search_dictionary,
)
from networkx.readwrite import json_graph
from pre_processings import pre_processing
from settings import BASE_DATAPATH, DATA, EDGES, LABELS

# from torch import Tensor
from sklearn.preprocessing import LabelEncoder

# from torch_geometric.data import InMemoryDataset


def similarity_only_one(
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


def similarity_zero_one(
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


def similarity_only_one_nx(
    thr: float,
    ind_i: int,
    ind_j: int,
    similarity_score: float,
    correlation_dictionary: Dict,
):
    if similarity_score > thr:
        correlation_dictionary["nodes"].append({"id": ind_i})
        correlation_dictionary["links"].append({"source": ind_i, "target": ind_j})


def select_generator(func_name: str, thr: float):
    factory = {
        "only_one": similarity_only_one,
        "zero_one": similarity_zero_one,
        "only_one_nx": similarity_only_one_nx,
    }
    return partial(factory[func_name], thr)


@ray.remote(num_cpus=os.cpu_count())
def similarity_matrix_generation(new_dataset: Dict, stat: str, func_name="only_one_nx"):
    # parqua dataset, parallel
    stat_model = get_stat_methos(stat)
    path_dir = EDGES / stat
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    for file_name, data in new_dataset.items():
        correlation_dictionary = defaultdict(list)
        if nan_checker(data):
            data = pre_processing(data=data)
        thr = chnage_connections_thr(file_name, stat)
        correlation_dictionary["directed"] = False
        correlation_dictionary["multigraph"] = False
        correlation_dictionary["graph"] = {}
        func = select_generator(func_name, thr)
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
        if func_name == "only_one_nx":
            with open(path_dir / f"similarity_{file_name}", "wb") as pickle_file:
                pickle.dump(correlation_dictionary, pickle_file)
        else:
            pd.DataFrame(
                correlation_dictionary.values(),
                columns=list(correlation_dictionary.items())[0][1].keys(),
            ).to_pickle(path_dir / f"similarity_{file_name}")
            # pd.DataFrame(
            #     correlation_dictionary.items(), columns=["key", "related"]
            # ).to_pickle(path_dir / f"similarity_{file_name}.pkl")
    return None


# do preprocessing here
def txt_to_pickle():
    """
    Search in the given repository, and load pickle/txt files, and generate
    labels, and input data
    """
    encoder = LabelEncoder()
    sample_data = defaultdict()
    labels = ""
    users = defaultdict()
    list_files = os.listdir(BASE_DATAPATH)
    if all(file_name in os.listdir(BASE_DATAPATH) for file_name in ["data", "labels"]):
        for file in os.listdir(DATA):
            data = pd.read_pickle(f"{DATA}/{file}")
            sample_data[file] = data
            if "PATIENT_ID" in data.columns:
                patient_id = data["PATIENT_ID"]
            else:
                patient_id = data.index
            users[file.split(".")[0]] = patient_id
        for file in os.listdir(LABELS):
            labels = pd.read_pickle(f"{LABELS}/{file}")
        return sample_data, users, labels

    for file in list_files:
        if file.endswith(".pkl"):
            with open(f"{BASE_DATAPATH}/{file}", "rb") as pkl_file:
                data = pickle.load(pkl_file)
        if file.endswith(".txt"):
            with open(f"{BASE_DATAPATH}/{file}", "rb") as txt_file:
                data = pd.read_csv(txt_file, sep="\t")
        else:
            continue
        if "data_cna" in file or "data_mrna" in file:
            data.drop("Entrez_Gene_Id", axis=1, inplace=True)
        if "edges_" in file:
            to_save_folder = EDGES
        elif "labels." in file:
            to_save_folder = LABELS
        else:
            to_save_folder = DATA
        if not os.path.exists(to_save_folder):
            os.mkdir(to_save_folder)
        if "data_clinical_patient" in file:
            patient_id = data["PATIENT_ID"]
            data.drop("PATIENT_ID", axis=1, inplace=True)
            if not os.path.exists(LABELS):
                os.mkdir(LABELS)

            data = data.apply(encoder.fit_transform)
            data["PATIENT_ID"] = patient_id
            data = data.set_index("PATIENT_ID")
            data["CLAUDIN_SUBTYPE"].to_pickle(f"{LABELS}/labels.pkl")
            labels = data["CLAUDIN_SUBTYPE"]
        else:
            hugo_symbol = data["Hugo_Symbol"]
            data.drop("Hugo_Symbol", axis=1, inplace=True)
            data = data.T
            data.columns = hugo_symbol.values
            patient_id = data.index
        file_name = file.split(".")[0]
        users[file_name] = patient_id
        sample_data[file_name] = data
        data.to_pickle(f"{to_save_folder}/{file_name}.pkl")
    return sample_data, users, labels


def set_same_users(sample_data: Dict, users: Dict, labels: Dict) -> Dict:
    new_dataset = defaultdict()
    shared_users = search_dictionary(users, len(users) - 1)
    shared_users = sorted(shared_users)[0:100]
    shared_users_encoded = LabelEncoder().fit_transform(shared_users)
    for file_name, data in sample_data.items():
        new_dataset[file_name] = data[data.index.isin(shared_users)].set_index(
            shared_users_encoded
        )
    return new_dataset, labels[shared_users].reset_index(drop=True)


# class CreateData:
#     """ Main class to generate graph data"""
#     def __init__(self):
#         pass


def process_data(edge_index: Dict):
    graph = nx.DiGraph(json_graph.node_link_graph(edge_index))
    print(graph)
