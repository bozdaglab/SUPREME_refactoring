import logging
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import pandas as pd
import ray
import torch
from helper import (
    chnage_connections_thr,
    edge_index_from_dict,
    get_stat_methos,
    nan_checker,
    search_dictionary,
)
from networkx.readwrite import json_graph
from pre_processings import pre_processing
from scipy.stats import pearsonr, spearmanr
from set_logging import set_log_config
from settings import (
    CONTEXT_SIZE,
    EDGES,
    EMBEDDING_DIM,
    LABELS,
    PATH_EMBEDDIGS,
    PATH_FEATURES,
    SPARSE,
    WALK_LENGHT,
    WALK_PER_NODE,
    P,
    Q,
)
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import remove_self_loops

logger = logging.getLogger()
set_log_config()
FUNC_NAME = "only_one"
DEVICE = torch.device("cpu")


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
    stat_model: Union[pearsonr, spearmanr],
    correlation_dictionary: Dict,
    path_dir: Path,
) -> Dict:
    logger.info(f"Start generating similarity matrix for {stat}, {file_name}...")
    correlation_dictionary["final_path"] = path_dir / file_name
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
            correlation_dictionary[f"{ind_i}_{ind_j}"] = {
                "source": ind_i,
                "target": ind_j,
                "Similarity Score": similarity_score,
            }
    return correlation_dictionary


@ray.remote(num_cpus=os.cpu_count())
def similarity_matrix_generation(new_dataset: Dict, stat):
    stat_model = get_stat_methos(stat)
    path_dir = EDGES / stat
    make_directories(path_dir)
    correlation_dictionary = defaultdict(list)
    result = [
        compute_similarity.remote(
            data,
            file_name,
            stat,
            stat_model,
            correlation_dictionary,
            path_dir,
        )
        for file_name, data in new_dataset.items()
    ]
    return ray.get(result)


def save_pickle_features(final_result: List[Dict]):
    selected_features = []
    feature_type = final_result[0]["feature_type"]
    is_first = True
    for result in final_result:
        selected_features.extend(result["features"])
        tensors = result["tensors"][0]
        if is_first:
            new_x = torch.tensor(tensors, device=DEVICE).float()
            is_first = False
        else:
            new_x = torch.cat(
                (new_x, torch.tensor(tensors, device=DEVICE).float()), dim=1
            )
    if not os.path.exists(PATH_FEATURES):
        os.makedirs(PATH_FEATURES)
    pd.DataFrame(selected_features).to_pickle(
        PATH_FEATURES / f"selected_features_{feature_type}.pkl"
    )
    if not os.path.exists(PATH_EMBEDDIGS):
        os.makedirs(PATH_EMBEDDIGS)
    pd.DataFrame(new_x).to_pickle(PATH_EMBEDDIGS / f"embeddings_{feature_type}.pkl")


def save_pickle_embeddings(final_result: List[Dict], func_name=False) -> None:
    for result in final_result:
        path = Path(result["final_path"]).parts
        file_name = path[-1]
        result.pop("final_path")
        path_dir = Path(*path[:-1])
        final_data = pd.DataFrame(
            result.values(),
            columns=list(result.items())[0][1].keys(),
        )
        if func_name:
            thr = chnage_connections_thr(file_name)
        else:
            thr = final_data["Similarity Score"].quantile(0.65)
        final_data["link"] = [
            1 if val >= thr else 0 for val in final_data["Similarity Score"]
        ]
        final_data.to_pickle(path_dir / f"similarity_{file_name}.pkl")


def node2vec(data: Data):
    node2vec_res = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=EMBEDDING_DIM,
        walk_length=WALK_LENGHT,
        context_size=CONTEXT_SIZE,
        walks_per_node=WALK_PER_NODE,
        p=P,
        q=Q,
        sparse=SPARSE,
    ).to(DEVICE)
    pos, neg = node2vec_res.sample([10, 15])  # batch size
    data.pos_edge_labels = torch.tensor(pos.T, device=DEVICE).long()
    data.neg_edge_labels = torch.tensor(neg.T, device=DEVICE).long()
    return data


def random_walk_pos(data: Data) -> Tuple[Tensor, Tensor]:
    pos_neighbors = defaultdict()
    neg_neighbors = defaultdict()
    for node in data.edge_index[0]:
        pos_nodes = []
        for _ in range(WALK_PER_NODE):
            cur_node = node
            for _ in range(WALK_LENGHT):
                neighbor_nodes = data.edge_index[1][data.edge_index[0] == cur_node]
                next_node = random.choice(neighbor_nodes)
                if next_node != node and next_node not in pos_nodes:
                    pos_nodes.append(int(next_node))
                cur_node = next_node
        pos_neighbors[int(node)] = pos_nodes
        neg_neighbors[int(node)] = random_walk_neg(node, pos_nodes, data, cur_node)
    return edge_index_from_dict(pos_neighbors), edge_index_from_dict(neg_neighbors)


def random_walk_neg(
    node, pos_nodes: List, data: Data, cur_node, random_sample: int = 15
):
    walk_lenght_distance_nodes = set(
        data.edge_index[1][data.edge_index[0] == cur_node].numpy()
    )
    frontier = walk_lenght_distance_nodes
    neghibor = walk_lenght_distance_nodes
    for _ in range(12):
        cur_nodes = set()
        for cur_node in frontier:
            cur_nodes |= set(data.edge_index[1][data.edge_index[0] == cur_node].numpy())
        frontier = cur_nodes.difference(neghibor)
        neghibor |= cur_nodes
    walk_lenght_distance_nodes = set(
        data.edge_index[1][data.edge_index[0] == cur_node].numpy()
    )
    node_connections = set(data.edge_index[1][data.edge_index[0] == node].numpy())
    node_connections.add(int(node))
    final_neg_nodes = [
        val
        for val, key in Counter(
            (*walk_lenght_distance_nodes, *neghibor, *node_connections, *pos_nodes)
        ).items()
        if key == 1
    ]
    return random.sample(final_neg_nodes, random_sample)
