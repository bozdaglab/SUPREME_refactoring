from typing import Optional

import numpy as np
import pandas as pd
import torch
from helper import masking_indexes, ratio
from learning_types import LearningTypes
from module import Discriminator, Encoder, InnerProductDecoder

# from torch_geometric.nn import GAE, VGAE
from settings import (
    CONTEXT_SIZE,
    DISCRIMINATOR,
    EMBEDDING_DIM,
    LEARNING,
    MASKING,
    NODE2VEC,
    ONLY_POS,
    POS_NEG,
    SPARSE,
    WALK_LENGHT,
    WALK_PER_NODE,
    P,
    Q,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch import Tensor

# from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils.negative_sampling import negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes

DEVICE = torch.device("cpu")
EPS = 1e-15


class GCNSupervised:
    def __init__(self, new_x, labels) -> None:
        self.new_x = new_x
        self.labels = labels

    def prepare_data(
        self,
        edge_index: pd.DataFrame,
        col: Optional[str] = None,
        multi_labels: bool = False,
    ) -> Data:
        train_valid_idx, test_idx = torch.utils.data.random_split(
            self.new_x, ratio(new_x=self.new_x)
        )
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

    def select_model(self):
        if LEARNING == "regression":
            criterion = torch.nn.MSELoss()
            out_size = 1
        elif LEARNING == "classification":
            criterion = torch.nn.CrossEntropyLoss()
            out_size = torch.tensor(
                len(self.labels.value_counts().unique())
            )  # torch.tensor(self.labels).shape[0]
        return criterion, out_size

    def train(self, model, optimizer, data: Data, criterion):
        model.train()
        optimizer.zero_grad()
        out, emb1, _ = model(data)
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask],
        )
        loss.backward()
        optimizer.step()
        return emb1

    def validate(self, model, criterion, data):
        model.eval()
        with torch.no_grad():
            out, emb2, _ = model(data)
            loss = criterion(
                out[data.valid_mask],
                data.y[data.valid_mask],
            )
        return loss, emb2


class GCNUnsupervised:
    def __init__(self, new_x) -> None:
        self.new_x = new_x

    def prepare_data(self, edge_index: pd.DataFrame) -> Data:
        train_valid_idx, test_idx = torch.utils.data.random_split(
            self.new_x, ratio(self.new_x)
        )
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
            data = make_data(self.new_x, edge_index)
            data.pos_edge_labels = torch.tensor(pos, device=DEVICE).long()
            data.neg_edge_labels = torch.tensor(neg, device=DEVICE).long()
        elif MASKING:
            # mask some edges and used those as negative values
            num_nodes = maybe_num_nodes(data.edge_index)
            mask_edges = torch.rand(edge_index.size(1)) < 0.5
            non_mask_edges = ~mask_edges
            data.neg_edge_labels = data.edge_index.clone()
            data.neg_edge_labels[0:mask_edges] = torch.randint(
                num_nodes, (mask_edges.sum(),), device=DEVICE
            )
            data.neg_edge_labels[0:non_mask_edges] = torch.randint(
                num_nodes, (non_mask_edges.sum(),), device=DEVICE
            )
        return train_test_valid(
            data=data, train_valid_idx=train_valid_idx, test_idx=test_idx
        )

    def select_model(self):
        criterion = torch.nn.MSELoss()
        out_size = self.new_x.shape[-1]
        return criterion, out_size

    def train(self, model, optimizer, data, criterion):
        if DISCRIMINATOR:
            num_nodes = maybe_num_nodes(data.edge_index)
            # model = EncoderDecoder(encoder = Encoder(), decoder= Discriminator())
            encoder = Encoder()
            discriminator = Discriminator()
            mu, logstd = encoder(data)
            emb = mu + torch.randn_like(logstd) * torch.exp(logstd)
            loss = self.loss_discriminator(
                emb=emb, data=data.pos_edge_labels, discriminator=discriminator
            )
            loss += (1 / num_nodes) * discriminator.kl_loss(mu, logstd)
        else:
            model.train()
            optimizer.zero_grad()
            emb, out, prediction = model(data=data)
            if POS_NEG:
                """
                https://arxiv.org/abs/1607.00653,
                https://arxiv.org/abs/1611.0730,
                https://arxiv.org/abs/1706.02216
                """

                loss = self.loss_pos_neg(
                    emb, pos_rw=data.pos_edge_labels, neg_rw=data.neg_edge_labels
                )
            elif ONLY_POS:
                loss = self.loss_pos_only(emb=emb, pos_rw=data.pos_edge_labels)

            else:
                # encoder_decoder loss, criterion is MSE
                loss = criterion(out[data.train_mask], data.x[data.train_mask])
        loss.backward()
        optimizer.step()
        return out

    def compute(self, emb, start, rest, rw):
        h_start = emb[start].view(rw.size(0), 1, emb.size(1))
        h_rest = emb[rest.view(-1)].view(rw.size(0), -1, emb.size(1))
        return (h_start * h_rest).sum(dim=-1).view(-1)

    def loss_pos_neg(self, emb, pos_rw, neg_rw):
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        out = self.compute(emb, start, rest, pos_rw)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        out = self.compute(emb, start, rest, pos_rw)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def loss_pos_only(self, emb, pos_rw, neg_rw: Optional[Tensor] = None):
        pos_loss = -torch.log(
            InnerProductDecoder(emb, pos_rw, sigmoid=True) + EPS
        ).mean()
        if neg_rw is None:
            neg_rw = negative_sampling(pos_rw, emb.size(0))
        neg_loss = -torch.log(
            InnerProductDecoder(emb, neg_rw, sigmoid=True) + EPS
        ).mean()

        return pos_loss + neg_loss

    def loss_discriminator(self, emb, pos_rw, discriminator):
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        for _ in range(5):
            discriminator_optimizer.zero_grad()
            real = torch.sigmoid(discriminator(torch.randn_like(emb)))
            fake = torch.sigmoid(discriminator(emb.detach()))
            real_loss = -torch.log(real + EPS).mean()
            fake_loss = -torch.log(1 - fake + EPS).mean()
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

        loss = self.loss_pos_only(emb, pos_rw)
        real = torch.sigmoid(discriminator(emb))
        real_loss = -torch.log(real + EPS).mean()
        loss += real_loss
        return loss

    def validate(self, data, model, criterion):
        # model = GAE(model)
        model.eval()
        with torch.no_grad():
            z, emdb, _ = model(data=data)
            loss = criterion(z[data.valid_mask], data.x[data.valid_mask])
        return loss, emdb


def make_data(new_x: torch.tensor, edge_index: pd.DataFrame) -> Data:
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
    train_valid_idx: torch,
    test_idx: torch,
    labels: Optional[pd.DataFrame] = None,
) -> Data:
    if labels:
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
    data.test_mask = torch.tensor(
        masking_indexes(data=data, indexes=test_idx), device=DEVICE
    )

    return data


def load_model(
    new_x: torch, labels: Optional[pd.DataFrame]
) -> [GCNUnsupervised, GCNSupervised]:
    if LEARNING == LearningTypes.clustering.name:
        return GCNUnsupervised(new_x=new_x)
    return GCNSupervised(new_x=new_x, labels=labels)


def select_optimizer(optimizer_type: str, model, learning_rate: float):
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sparse_adam":
        return torch.optim.SparseAdam(list(model.parameters()), lr=learning_rate)
    else:
        raise NotImplementedError
