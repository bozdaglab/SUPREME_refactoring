import logging
import os
import os.path as osp
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Dict, List

import pandas as pd
import ray
import torch
from helper import pos_neg, random_split, read_labels
from learning_types import SelectModel
from node_generation import node_feature_generation
from pre_process_data import (
    node2vec,
    prepare_data,
    random_walk_pos,
    save_pickle_embeddings,
    save_pickle_features,
    similarity_matrix_generation,
)
from set_logging import set_log_config
from settings import (
    BASE_DATAPATH,
    DATA,
    EDGES,
    PATH_EMBEDDIGS,
    PATH_FEATURES,
    POS_NEG_MODELS,
    SELECTION_METHOD,
    STAT_METHOD,
)
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import (
    coalesce,
    negative_sampling,
    remove_self_loops,
    train_test_split_edges,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes

logger = logging.getLogger()
set_log_config()
DEVICE = torch.device("cpu")
"""
Add NeighborLoader here to create batches with neighbor sampling
similar to what we see here https://mlabonne.github.io/blog/posts/2022-04-06-GraphSAGE.html
"""


class BioDataset(Dataset):
    def __init__(
        self,
        root: str,
        file_name: List[str],
        raw_directories: List[str],
        loader: bool = False,
    ):
        self.file_name = file_name
        self.loader = loader
        self.raw_directories = raw_directories
        self._pre_process_run(root)
        super().__init__(root)

    @property
    def raw_file_names(self):
        return self.file_name

    @property
    def processed_file_names(self):
        return self.raw_directories

    def download(self):
        pass

    def _pre_process_run(self, root: str) -> None:
        self.root = root
        files = [file.replace(".txt", ".pkl") for file in self.file_name]
        if not self.file_exist(files):
            prepare_data(self.raw_paths, DATA)
        self.data_dir = [osp.join(DATA, file) for file in os.listdir(DATA)]
        ray.init(num_cpus=os.cpu_count())
        self.run_node_feature_generation()
        self.run_similarity_matrix_generation()

    def file_exist(self, files: List[str]) -> bool:
        return len(files) != 0 and all([osp.exists(DATA / file) for file in files])

    def run_node_feature_generation(self):
        if osp.exists(PATH_FEATURES) and len(SELECTION_METHOD) == len(
            os.listdir(PATH_FEATURES)
        ):
            return
        feature_selections = SELECTION_METHOD
        if osp.exists(PATH_FEATURES):
            feature_files = []
            for feature, file in product(feature_selections, os.listdir(PATH_FEATURES)):
                if file.endswith(f"{feature}.pkl"):
                    feature_files.append(file)
            if len(feature_selections) != len(feature_files):
                feature_selections = feature_files
            else:
                feature_selections = None
        if feature_selections:
            sample_data = self.read_sample_data()
            labels = read_labels()
            features_result_ray_not_done = [
                node_feature_generation.remote(
                    new_dataset=sample_data,
                    labels=labels,
                    feature_type=feature_type,
                )
                for feature_type in feature_selections
            ]
            while features_result_ray_not_done:
                features_result_ray_done, features_result_ray_not_done = ray.wait(
                    features_result_ray_not_done
                )
                final_result_features_selections = ray.get(features_result_ray_done)
                save_pickle_features(final_result=final_result_features_selections[0])

    def run_similarity_matrix_generation(self):
        if osp.exists(EDGES) and sum(
            [len(files) for _, _, files in os.walk(BASE_DATAPATH / EDGES)]
        ) == len(STAT_METHOD) * len(self.data_dir):
            return
        sample_data = self.read_sample_data()
        embedding_result_ray_not_done = [
            similarity_matrix_generation.remote(new_dataset=sample_data, stat=stat)
            for stat in STAT_METHOD
        ]
        while embedding_result_ray_not_done:
            embedding_result_ray_done, embedding_result_ray_not_done = ray.wait(
                embedding_result_ray_not_done
            )
            final_result_emb = ray.get(embedding_result_ray_done)
            save_pickle_embeddings(final_result=final_result_emb[0])

    def read_sample_data(self) -> Dict:
        sample_data = defaultdict()
        for file in self.data_dir:
            file_name = file.split("/")[-1].split(".pkl")[0]
            sample_data[file_name] = pd.read_pickle(file)
        return sample_data

    def process(self):
        labels = read_labels()
        generator = self.data_generator_selector(labels)
        for stat, feature_type in product(
            os.listdir(EDGES), os.listdir(PATH_EMBEDDIGS)
        ):
            folder_path = f"{stat}_{feature_type.split('_')[1].split('.')[0]}"
            new_x = pd.read_pickle(PATH_EMBEDDIGS / feature_type)
            if isinstance(new_x, pd.DataFrame):
                new_x = torch.tensor(new_x.values, dtype=torch.float32)
            for file in os.listdir(EDGES / stat):
                file_path = f"{file.split('.')[0]}"
                dir = os.path.join(self.processed_dir, "graph_data", folder_path)
                edge_index = pd.read_pickle(EDGES / stat / file)
                generator(
                    new_x=new_x, edge_index=edge_index, dir=dir, file_path=file_path
                )

    def data_generator_selector(self, labels):
        if self.loader:
            return partial(self.create_with_loader, labels=labels)
        else:
            return partial(self.create_without_loader, labels=None)

    def create_without_loader(self, new_x, edge_index, dir, file_path, labels=None):
        for data_generation_types in POS_NEG_MODELS:
            self.create_data(
                new_x=new_x,
                data_generation_types=data_generation_types,
                edge_index=edge_index,
                dir=dir,
                file_path=file_path,
            )

    def create_with_loader(self, new_x, labels, edge_index, dir, file_path):
        graphs = []
        for idx, patient_feat in enumerate(new_x):
            node_features = patient_feat
            label = labels[labels.index == idx].values
            data = self.make_data(
                new_x=node_features, edge_index=edge_index, labels=label
            )
            graphs.append(data)
        train_valid_idx, test_idx = random_split(new_x=new_x)
        train_idx, valid_idx = train_test_split(train_valid_idx.indices, test_size=0.25)
        for indexs, name in zip(
            [train_idx, valid_idx, test_idx.indices],
            ["train_data", "valid_data", "test_data"],
        ):
            new_dir = osp.join(dir, file_path, name)
            if not osp.exists(new_dir):
                os.makedirs(new_dir)
            for idx in indexs:
                torch.save(graphs[idx], osp.join(new_dir, f"{idx}.pt"))

    def create_data(
        self,
        new_x,
        data_generation_types: str,
        edge_index: pd.DataFrame,
        dir,
        file_path,
    ) -> Data:
        """
        Create a data object by adding features, edge_index, edge_attr.
        For unsupervised GCN, this function
            1) creates positive and negative edges using Node2vec or
            2) make masking in the input data.

        Parameters:
        -----------
        edge_index:
            Adjacency matrix
        Return:
            A data object ready to pass to GCN
        """
        train_valid_idx, test_idx = random_split(new_x=new_x)
        train_idx, valid_idx = train_test_split(train_valid_idx.indices, test_size=0.25)
        if isinstance(edge_index, dict):
            edge_index = pd.DataFrame(edge_index).T
        for indexs, name in zip(
            [train_idx, valid_idx, test_idx.indices],
            ["train_data", "valid_data", "test_data"],
        ):
            edge_index_indexs = edge_index[
                edge_index["source"].isin(indexs) & edge_index["target"].isin(indexs)
            ].reset_index(drop=True)
            data = self.make_data(new_x=new_x, edge_index=edge_index_indexs)
            if data_generation_types == SelectModel.node2vec.name:
                data = node2vec(data)
            elif data_generation_types == SelectModel.randomwalk.name:
                data.pos_edge_labels = random_walk_pos(data)
                data.neg_edge_labels = negative_sampling(
                    data.pos_edge_labels, new_x.size(0)
                )
                # for short data, the negative might be empty
                # data.pos_edge_labels, data.neg_edge_labels = random_walk_pos(data)
            if data_generation_types == SelectModel.train_test.name:
                data.num_nodes = maybe_num_nodes(data.edge_index)
                edge_in = data.edge_index
                edge_attr = data.edge_attr
                data = train_test_split_edges(data=data)
                data.edge_index = edge_in
                data.edge_attr = edge_attr
            elif data_generation_types == SelectModel.similarity_based.name:
                data.pos_edge_labels = pos_neg(edge_index_indexs, "link", 1)
                data.neg_edge_labels = negative_sampling(
                    data.pos_edge_labels, new_x.size(0)
                )
            dir_update = osp.join(f"{dir}_{data_generation_types}", file_path, name)
            if not os.path.exists(dir_update):
                os.makedirs(dir_update)
            torch.save(data, osp.join(dir_update, f"{file_path}.pt"))

    def make_data(self, new_x: Tensor, edge_index: pd.DataFrame, labels=None) -> Data:
        """
        Generate a data object that holds node features, edge_index and edge_attr.

        Parameters:
        -----------
        new_x:
            Concatenated features from different omics file
        edge_index:
            Adjacency matrix

        Return:
            A data object
        """
        data = Data(
            x=new_x,
            edge_index=torch.tensor(
                edge_index[edge_index.columns[0:2]].transpose().values,
                device=DEVICE,
            ).long(),
            edge_attr=torch.tensor(
                edge_index[edge_index.columns[2]].transpose().values,
                device=DEVICE,
            ).float(),
        )
        if labels is not None:
            data.y = torch.tensor(labels)
        edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index, data.edge_attr = coalesce(
            edge_index=edge_index, edge_attr=data.edge_attr, num_nodes=new_x.shape[0]
        )
        return data

    def len(self):
        pass

    def get(self):
        pass
