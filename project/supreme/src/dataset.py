import logging
import os
import os.path as osp
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import ray
import torch
from helper import masking_indexes, pos_neg, random_split
from learning_types import SelectModel
from node_generation import node_feature_generation
from pre_process_data import prepare_data, save_pickle, similarity_matrix_generation
from set_logging import set_log_config
from settings import (
    BASE_DATAPATH,
    CONTEXT_SIZE,
    DATA,
    EDGES,
    EMBEDDING_DIM,
    LABELS,
    PATH_EMBEDDIGS,
    PATH_FEATURES,
    POS_NEG_MODELS,
    SELECTION_METHOD,
    SPARSE,
    STAT_METHOD,
    WALK_LENGHT,
    WALK_PER_NODE,
    P,
    Q,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import Node2Vec
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
        file_name: str,
        raw_directories: str,
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

    def _pre_process_run(self, root):
        self.root = root
        files = [file.replace(".txt", ".pkl") for file in self.file_name]
        if not self.file_exist(files):
            self.run_prepare_data()
        self.data_dir = [osp.join(DATA, file) for file in os.listdir(DATA)]
        ray.init(num_cpus=os.cpu_count())
        self.run_node_feature_generation()
        self.run_similarity_matrix_generation()

    def file_exist(self, files):
        return len(files) != 0 and all([osp.exists(DATA / file) for file in files])

    def run_prepare_data(self):
        prepare_data(self.raw_paths, DATA)

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
            labels = self.read_labels()
            for feature_type in feature_selections:
                # features_result_ray = [
                node_feature_generation(
                    new_dataset=sample_data,
                    labels=labels,
                    feature_type=feature_type,
                    path_features=PATH_FEATURES,
                    path_embeggings=PATH_EMBEDDIGS,
                )
            #     for feature_type in feature_selections
            # ]
            # ray.wait(features_result_ray)
            # ray.get(features_result_ray)

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
            final_result = ray.get(embedding_result_ray_done)
            save_pickle(final_result=final_result)

    def read_sample_data(self):
        sample_data = defaultdict()
        for file in self.data_dir:
            file_name = file.split("/")[-1].split(".pkl")[0]
            sample_data[file_name] = pd.read_pickle(file)
        return sample_data

    def read_labels(self):
        return pd.read_pickle(LABELS / os.listdir(LABELS)[0])

    def process(self):
        labels = self.read_labels()
        generator = self.data_generator_selector(labels)
        for stat, feature_type in product(
            os.listdir(EDGES), os.listdir(PATH_EMBEDDIGS)
        ):
            folder_path = f"{stat}_{feature_type.split('_')[1].split('.')[0]}"
            new_x = pd.read_pickle(PATH_EMBEDDIGS / feature_type)
            if isinstance(new_x, pd.DataFrame):
                new_x = torch.tensor(new_x.values, dtype=torch.float32)
            for file in os.listdir(EDGES / stat):
                file_path = f"{folder_path}_{file.split('.')[0]}"
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
            data = self.create_data(
                new_x=new_x,
                data_generation_types=data_generation_types,
                edge_index=edge_index,
            )
            dir_update = f"{dir}_{data_generation_types}"
            if not os.path.exists(dir_update):
                os.makedirs(dir_update)
            torch.save(data, osp.join(dir_update, f"{file_path}.pt"))

    def create_with_loader(self, new_x, labels, edge_index, dir, file_path):
        graphs = []
        for idx, patient in enumerate(new_x):
            node_features = patient
            label = labels["CLAUDIN_SUBTYPE"][labels.index == idx].values
            data = Data(x=node_features, edge_index=edge_index, y=label)
            graphs.append(data)
        data, slice = self.collate(data)
        torch.save((data, slice), dir / f"data_{idx}.pt")
        # create dataloader
        # for stat in os.listdir(EDGES):
        #     final_correlation = defaultdict()
        #     for file in os.listdir(EDGES / stat):
        #         final_correlation[file] = pd.read_pickle(EDGES / stat / file)
        #     new_x = pd.read_pickle(PATH_EMBEDDIGS / os.listdir(PATH_EMBEDDIGS)[0])
        #     if isinstance(new_x, pd.DataFrame):
        #         new_x = torch.tensor(new_x.values, dtype=torch.float32)
        #     lables = pd.read_pickle(LABELS / os.listdir(LABELS)[0])
        #     base_dir = BASE_DATAPATH / stat
        #     for file_name, edge_index in final_correlation.items():
        #         dataset_dir = base_dir / file_name
        #         self.make_directories(dataset_dir)
        #         graphs = []
        #         for idx, patient in enumerate(new_x):
        #             node_features = patient
        #             # edge_index = np.where(edges["Patient_1"] == idx)
        #             label = labels["CLAUDIN_SUBTYPE"][lables.index == idx].values
        #             data =  Data(x=node_features, edge_index=edge_index, y=label)
        #             graphs.append(data)
        #         data, slice = self.collate(data)
        #         torch.save((data, slice), dataset_dir / f"data_{idx}.pt")

    def create_data(
        self, new_x, data_generation_types: str, edge_index: pd.DataFrame
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
        if isinstance(edge_index, dict):
            edge_index = pd.DataFrame(edge_index).T
        # edge_index = process_data(edge_index=edge_index)
        # use index to mask inorder to generate the val, test, and train
        # index_to_mask (e.g, index_to_mask(train_index, size=y.size(0)))
        data = self.make_data(new_x=new_x, edge_index=edge_index)
        if data_generation_types == SelectModel.node2vec.name:
            node2vec = Node2Vec(
                edge_index=data.edge_index,
                embedding_dim=EMBEDDING_DIM,
                walk_length=WALK_LENGHT,
                context_size=CONTEXT_SIZE,
                walks_per_node=WALK_PER_NODE,
                p=P,
                q=Q,
                sparse=SPARSE,
            ).to(DEVICE)
            pos, neg = node2vec.sample([10, 15])  # batch size
            data.pos_edge_labels = torch.tensor(pos.T, device=DEVICE).long()
            data.neg_edge_labels = torch.tensor(neg.T, device=DEVICE).long()
        # elif data_generation_types == SelectModel.similarity_based.name:
        #     data.pos_edge_labels = pos_neg(edge_index, "link", 1)
        #     data.neg_edge_labels = pos_neg(edge_index, "link", 0)
        elif data_generation_types == SelectModel.train_test.name:
            data.num_nodes = maybe_num_nodes(data.edge_index)
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            data = train_test_split_edges(data=data)
            data.edge_index = edge_index
            data.edge_attr = edge_attr
        elif data_generation_types == SelectModel.similarity_based.name:
            data.pos_edge_labels = pos_neg(edge_index, "link", 1)
            data.neg_edge_labels = negative_sampling(
                data.pos_edge_labels, new_x.size(0)
            )
        return self.train_test_valid(
            data=data, train_valid_idx=train_valid_idx, test_idx=test_idx
        )

    def make_data(self, new_x: Tensor, edge_index: pd.DataFrame) -> Data:
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
                edge_index[edge_index.columns[3]].transpose().values,
                device=DEVICE,
            ).float(),
        )
        edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index = coalesce(edge_index=edge_index, num_nodes=new_x.shape[0])
        return data

    def train_test_valid(
        self,
        data: Data,
        train_valid_idx: Tensor,
        test_idx: Tensor,
        labels: Optional[pd.DataFrame] = None,
    ) -> Data:
        """
        This function adds train, test, and validation data to the data object.
        If a label is available, it applies a Repeated Stratified K-Fold cross-validator.
        Otherwise, split the tensor into random train and test subsets.

        Parameters:
        -----------
        data:
            Data object
        train_valid_idx:
            train validation indexes
        test_idx:
            test indexes
        labels:
            Dataset labels

        Return:
            A data object that holds train, test, and validation indexes
        """
        if labels is not None:
            try:
                X = data.x[train_valid_idx.indices]
                y = data.y[train_valid_idx.indices]
            except:
                X = data.x[train_valid_idx]
                y = data.y[train_valid_idx]
            # y_ph = labels.iloc[:, 3][train_valid_idx.indices]

            rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)
            for train_part, valid_part in rskf.split(X, y):
                try:
                    train_idx = np.array(train_valid_idx.indices)[train_part]
                    valid_idx = np.array(train_valid_idx.indices)[valid_part]
                except:
                    train_idx = np.array(train_valid_idx)[train_part]
                    valid_idx = np.array(train_valid_idx)[valid_part]
                break

        elif "val_pos_edge_index" in data.keys():
            return data
        else:
            train_idx, valid_idx = train_test_split(
                train_valid_idx.indices, test_size=0.25
            )

        data.valid_mask = torch.tensor(
            masking_indexes(data=data, indexes=valid_idx), device=DEVICE
        )
        data.train_mask = torch.tensor(
            masking_indexes(data=data, indexes=train_idx), device=DEVICE
        )
        try:
            data.test_mask = torch.tensor(
                masking_indexes(data=data, indexes=test_idx), device=DEVICE
            )
        except (KeyError, TypeError):
            data.test_mask = torch.tensor(
                masking_indexes(data=data, indexes=test_idx.indices), device=DEVICE
            )
        return data

    def len(self):
        pass

    def get(self):
        pass
