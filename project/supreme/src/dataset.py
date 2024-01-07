import logging
import os
import pickle
from collections import defaultdict
from functools import partial
from typing import Dict, Optional

import networkx as nx
import pandas as pd
import torch
from feature_selections import select_features
from helper import (
    chnage_connections_thr,
    get_stat_methos,
    nan_checker,
    row_col_ratio,
    search_dictionary,
)
from networkx.readwrite import json_graph
from pre_processings import pre_processing
from set_logging import set_log_config
from settings import (
    EDGES,
    LABELS,
    PATH_EMBEDDIGS,
    PATH_FEATURES,
    SELECTION_METHOD,
    STAT_METHOD,
)

# from torch import Tensor
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.utils import remove_self_loops

logger = logging.getLogger()
set_log_config()
FUNC_NAME = "only_one"
DEVICE = torch.device("cpu")

"""
Add NeighborLoader here to create batches with neighbor sampling
similar to what we see here https://mlabonne.github.io/blog/posts/2022-04-06-GraphSAGE.html
"""


class BioDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_name: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
    ):
        self.file_name = file_name
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        return self.file_name

    @property
    def processed_file_names(self):
        files = [file.replace(".txt", ".pkl") for file in self.file_name]
        return files

    def download(self):
        pass

    def process_data(self, edge_index: Dict):
        graph = nx.DiGraph(json_graph.node_link_graph(edge_index))
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        edge_index, _ = remove_self_loops(edge_index)
        return edge_index

    def read_files(self):
        sample_data = defaultdict()
        labels = ""
        users = defaultdict()
        for file in self.raw_paths:
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
            to_save_folder = self.save_folder(file=file)
            self.make_directory(to_save_folder)
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
        return sample_data, users, labels

    def save_file(self, sample_data, labels):
        logger.info("Save file in a pickle format...")
        for name, featuers in sample_data.items():
            featuers.to_pickle(f"{self.processed_dir}/{name}.pkl")
        if not os.path.exists(LABELS):
            os.mkdir(LABELS)
        labels.to_pickle(f"{LABELS}/labels.pkl")

    def make_directory(self, dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    def make_directories(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def save_folder(self, file):
        to_save_folder = self.processed_dir
        if "edges_" in file:
            to_save_folder = EDGES
        elif "labels." in file:
            to_save_folder = LABELS
        return to_save_folder

    def set_same_users(self, sample_data: Dict, users: Dict, labels: Dict) -> Dict:
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

    def process(self):
        sample_data, users, labels = self.read_files()
        sample_data, labels = self.set_same_users(sample_data, users, labels)
        self.save_file(sample_data, labels)
        for feature_type in SELECTION_METHOD:
            self.node_feature_generation(
                new_dataset=sample_data,
                labels=labels,
                feature_type=feature_type,
                path_features=PATH_FEATURES,
                path_embeggings=PATH_EMBEDDIGS,
            )
        for stat in STAT_METHOD:
            self.similarity_matrix_generation(new_dataset=sample_data, stat=stat)

    def node_feature_generation(
        self,
        new_dataset: Dict,
        labels: Dict,
        path_features: str,
        path_embeggings: str,
        feature_type: Optional[str] = None,
    ) -> Tensor:
        """
        Load features from each omic separately, apply feature selection if needed,
        and contact them together

        Parameters:
        ----------
        labels:
            Dataset labels in case we want to apply feature selection algorithems


        Return:
            Concatenated features from different omics file
        """
        is_first = True
        selected_features = []
        for _, feat in new_dataset.items():
            if row_col_ratio(feat):
                # add an inner remote function and use get to get the result of the inner one before proceding
                feat, final_features = select_features(
                    application_train=feat,
                    labels=labels["CLAUDIN_SUBTYPE"],
                    feature_type=feature_type,
                )
                selected_features.extend(final_features)
                if not any(feat):
                    continue
                values = torch.tensor(feat.values, device=DEVICE)
            else:
                selected_features.extend(feat.columns)
                values = feat.values
            if is_first:
                new_x = torch.tensor(values, device=DEVICE).float()
                is_first = False
            else:
                new_x = torch.cat(
                    (new_x, torch.tensor(values, device=DEVICE).float()), dim=1
                )
        self.make_directories(path_features)
        pd.DataFrame(selected_features).to_pickle(
            path_features / f"selected_features_{feature_type}.pkl"
        )
        self.make_directories(path_embeggings)
        pd.DataFrame(new_x).to_pickle(
            path_embeggings / f"embeddings_{feature_type}.pkl"
        )

    def similarity_matrix_generation(
        self, new_dataset: Dict, stat, func_name=FUNC_NAME
    ):
        logger.info("Start generating similarity matrix...")
        stat_model = get_stat_methos(stat)
        path_dir = EDGES / stat
        self.make_directories(path_dir)
        for file_name, data in new_dataset.items():
            correlation_dictionary = defaultdict(list)
            thr = chnage_connections_thr(file_name, stat)
            func = self.select_generator(func_name, thr)
            for ind_i, patient_1 in data.iterrows():
                for ind_j, patient_2 in data.iterrows():
                    if ind_i == ind_j:
                        continue
                    try:
                        similarity_score = stat_model(
                            patient_1.values, patient_2.values
                        ).statistic
                    except AttributeError:
                        similarity_score = stat_model(
                            patient_1.values, patient_2.values
                        )[0]
                    func(ind_i, ind_j, similarity_score, correlation_dictionary)
            if func_name == "only_one_nx":
                correlation_dictionary["directed"] = False
                correlation_dictionary["multigraph"] = False
                correlation_dictionary["graph"] = {}
                with open(
                    path_dir / f"similarity_{file_name}.pkl", "wb"
                ) as pickle_file:
                    pickle.dump(correlation_dictionary, pickle_file)
            elif any(correlation_dictionary):
                pd.DataFrame(
                    correlation_dictionary.values(),
                    columns=list(correlation_dictionary.items())[0][1].keys(),
                ).to_pickle(path_dir / f"similarity_{file_name}.pkl")

    def select_generator(self, func_name: str, thr: float):
        factory = {
            "only_one": self._similarity_only_one,
            "zero_one": self._similarity_zero_one,
            "only_one_nx": self._similarity_only_one_nx,
        }
        return partial(factory[func_name], thr)

    def _similarity_only_one(
        self,
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
        self,
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
        self,
        thr: float,
        ind_i: int,
        ind_j: int,
        similarity_score: float,
        correlation_dictionary: Dict,
    ):
        if similarity_score > thr:
            correlation_dictionary["nodes"].append({"id": ind_i})
            correlation_dictionary["links"].append({"source": ind_i, "target": ind_j})

    def len(self):
        pass

    def get(self):
        pass
