from collections import namedtuple
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from helper import masking_indexes, pos_neg, random_split
from learning_types import LearningTypes, OptimizerType
from module import (
    SUPREME,
    Discriminator,
    Encoder,
    EncoderDecoder,
    EncoderEntireInput,
    EncoderInnerProduct,
    SupremeClassification,
    SupremeClusteringLink,
)
from settings import (
    CONTEXT_SIZE,
    DISCRIMINATOR,
    EMBEDDING_DIM,
    ENCODERINPRODUCT,
    LEARNING,
    LINKPREDICTION,
    NODE2VEC,
    SIMILARITY_BASED,
    SPARSE,
    TRAIN_TEST,
    WALK_LENGHT,
    WALK_PER_NODE,
    P,
    Q,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch_geometric.utils.num_nodes import maybe_num_nodes

DEVICE = torch.device("cpu")
EPS = 1e-15


class GCNSupervised:
    def __init__(self, new_x: Tensor, labels) -> None:
        self.new_x = new_x
        self.labels = labels

    def prepare_data(
        self,
        edge_index: pd.DataFrame,
        col: Optional[str] = None,
        multi_labels: bool = False,
    ) -> Data:
        """
        Create a data object by adding features, edge_index, edge_attr.

        Parameters:
        -----------
        edge_index:
            Adjacency matrix
        col:
            For regressions (?)
        multi_labels:
            For regression (?)

        Return:
            A data object ready to pass to GCN
        """
        train_valid_idx, test_idx = random_split(new_x=self.new_x)
        data = make_data(new_x=self.new_x, edge_index=edge_index)
        if multi_labels:
            data.y = torch.tensor(self.labels[col].values, dtype=torch.float32)
        else:
            data.y = torch.tensor(self.labels.values.reshape(1, -1)[0]).long()
        return train_test_valid(
            data=data,
            train_valid_idx=train_valid_idx,
            test_idx=test_idx,
            labels=self.labels,
        )

    def model_loss_output(self) -> int:
        if LEARNING == LearningTypes.regression.name:
            out_size = 1
        elif LEARNING == LearningTypes.classification.name:
            # should be int. Check later
            out_size = torch.tensor(
                len(self.labels.value_counts().unique())
            )  # torch.tensor(self.labels).shape[0]
        return out_size


class GCNUnsupervised:
    def __init__(self, new_x: Tensor) -> None:
        self.new_x = new_x

    def prepare_data(self, edge_index: pd.DataFrame) -> Data:
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
        train_valid_idx, test_idx = random_split(new_x=self.new_x)
        data = make_data(new_x=self.new_x, edge_index=edge_index)
        if NODE2VEC:
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
            pos, neg = node2vec.sample([10, 15])
            data.pos_edge_labels = torch.tensor(pos.T, device=DEVICE).long()
            data.neg_edge_labels = torch.tensor(neg.T, device=DEVICE).long()
        elif SIMILARITY_BASED:
            # temporary
            edge_index["links"] = [
                1 if val > 0.60 else 0 for val in edge_index["Similarity Score"]
            ]
            data.pos_edge_labels = pos_neg(edge_index, "links", 1)
            data.neg_edge_labels = pos_neg(edge_index, "links", 0)
        elif TRAIN_TEST:
            data.num_nodes = maybe_num_nodes(data.edge_index)
            data = train_test_split_edges(data=data)
        else:
            data.neg_edge_labels = negative_sampling(
                data.edge_index, self.new_x.size(0)
            )
        return train_test_valid(
            data=data, train_valid_idx=train_valid_idx, test_idx=test_idx
        )

    def model_loss_output(self) -> int:
        """
        This function selects the output size

        """
        return self.new_x.shape[-1]


def make_data(new_x: Tensor, edge_index: pd.DataFrame) -> Data:
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
    return Data(
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


def train_test_valid(
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
    if isinstance(labels, pd.DataFrame):
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

    else:
        train_idx, valid_idx = train_test_split(train_valid_idx.indices, test_size=0.25)

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
    except KeyError:
        data.test_mask = torch.tensor(
            masking_indexes(data=data, indexes=test_idx.indices), device=DEVICE
        )
    return data


def load_model(
    new_x: Tensor, labels: Optional[pd.DataFrame]
) -> Union[GCNUnsupervised, GCNSupervised]:
    """
    Load supervised or unsupervised GCN
    Parameters:
    -----------
    new_x:
        Concatenated features from different omics file
    labels:
        Dataset labels
    Return:
        Supervised or unsupervised GCN
    """
    if LEARNING == LearningTypes.clustering.name:
        return GCNUnsupervised(new_x=new_x)
    return GCNSupervised(new_x=new_x, labels=labels)


def select_optimizer(
    optimizer_type: str, model: Module, learning_rate: float
) -> torch.optim:
    """
    This function selects the optimizer

    Parameters:
    -----------
    optimizer_type:
        Name of the optimizer
    model:
        Our model, supervised or unsupervised GCN
    learning_rate:
        Learning rate

    Return:
        Torch optimizer
    """
    if isinstance(model, EncoderDecoder):
        losses = namedtuple("losses", ["encoder_loss", "decoder_loss"])
        encoder_loss = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
        decoder_loss = torch.optim.Adam(
            model.discriminator.parameters(), lr=learning_rate
        )
        return losses(encoder_loss=encoder_loss, decoder_loss=decoder_loss)
    elif isinstance(model, EncoderInnerProduct):
        return torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)

    elif isinstance(model, EncoderEntireInput):
        losses = namedtuple("losses", ["encoder_loss", "decoder_loss"])
        encoder_loss = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
        decoder_loss = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)
        return losses(encoder_loss=encoder_loss, decoder_loss=decoder_loss)
    if optimizer_type == OptimizerType.sgd.name:
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9
        )
    elif optimizer_type == OptimizerType.adam.name:
        return torch.optim.Adam(model.model.parameters(), lr=learning_rate)
    elif optimizer_type == OptimizerType.sparse_adam.name:
        return torch.optim.SparseAdam(list(model.parameters()), lr=learning_rate)
    else:
        raise NotImplementedError


def select_model(
    in_size: int, hid_size: int, out_size: int
) -> Union[SupremeClassification, EncoderDecoder, EncoderInnerProduct]:
    """
    This function selects the return of the model

    Parameters:
    ----------
    in_size:
        Input size of the model
    hid_size:
        hidden size
    out_size:
        output size of the model

    Return:
        Models, whether original SUPREME, or encoder-decoder model
    """
    if LEARNING in [LearningTypes.classification.name, LearningTypes.regression.name]:
        model = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
        return SupremeClassification(model=model)
    else:
        if DISCRIMINATOR:
            encoder = Encoder(in_size=in_size, hid_size=hid_size, out_size=out_size)
            discriminator = Discriminator(
                in_size=in_size, hid_size=hid_size, out_size=out_size
            )
            return EncoderDecoder(encoder=encoder, discriminator=discriminator)
        elif LINKPREDICTION:
            model = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
            return SupremeClusteringLink(model=model)
        elif ENCODERINPRODUCT:
            encoder = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
            return EncoderInnerProduct(encoder=encoder)
        else:
            encoder = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
            decoder = Discriminator(
                in_size=in_size, hid_size=hid_size, out_size=out_size
            )
            return EncoderEntireInput(encoder=encoder, decoder=decoder)
