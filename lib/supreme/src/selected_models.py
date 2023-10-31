from typing import Optional

import numpy as np
import torch
from helper import masking_indexes, ratio
from learning_types import LearningTypes
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
# from torch_geometric.nn import GAE, VGAE
from settings import NODE2VEC


DEVICE = torch.device("cpu")
EPS = 1e-15


class GCNSupervised:
    def __init__(self, learning, new_x) -> None:
        self.learning = learning
        self.new_x = new_x

    def prepare_data(self, new_x, edge_index, labels, col):
        train_valid_idx, test_idx = torch.utils.data.random_split(
            self.new_x, ratio(self.new_x)
        )
        data = make_data(new_x, edge_index)
        data.y = torch.tensor(labels[col].values, dtype=torch.float32)
        return train_test_valid(data, train_valid_idx, test_idx)

    def select_model(self):
        if self.learning == "regression":
            criterion = torch.nn.MSELoss()
            out_size = 1
        elif self.learning == "classification":
            criterion = torch.nn.CrossEntropyLoss()
            out_size = torch.tensor(y).shape[0]
        return criterion, out_size

    def train(self, model, optimizer, data, criterion):
        model.train()
        optimizer.zero_grad()
        out, emb1 = model(data)
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
            out, emb2 = model(data)
            loss = criterion(
                out[data.valid_mask],
                data.y[data.valid_mask],
            )
        return loss, emb2


class GCNUnsupervised:
    def __init__(self, new_x) -> None:
        self.new_x = new_x

    def prepare_data(self, new_x, edge_index):
        train_valid_idx, test_idx = torch.utils.data.random_split(
            self.new_x, ratio(self.new_x)
        )
        data = make_data(new_x, edge_index)
        return train_test_valid(data, train_valid_idx, test_idx)

    def select_model(self):
        criterion = torch.nn.MSELoss()
        out_size = self.new_x.shape[-1]
        return criterion, out_size

    def train(self, model, optimizer, data, criterion):
        model.train()
        optimizer.zero_grad()
        out, emb = model(data, model)
        if NODE2VEC:
            node2vec = Node2Vec(
                data.edge_index,
                embedding_dim=128,
                walk_length=20,
                context_size=10,
                walks_per_node=10,
                num_negative_samples=1,
                p=1.0,
                q=1.0,
                sparse=True,
            ).to(DEVICE)
            pos, neg = node2vec.sample([10, 15])
            loss = self.loss(out,emb, pos_rw=pos, neg_rw=neg)
        else:
            loss = criterion(out[data.train_mask], data.x[data.train_mask])

        loss.backward()
        optimizer.step()
        return out

    def train_1(self, model, optimizer, data, criterion):
        # model = GAE(model)
        model.train()
        optimizer.zero_grad()
        out, _ = model(data, model)
        loss = criterion(out[data.train_mask], data.x[data.train_mask])
        loss.backward()
        optimizer.step()
        return out

    def compute(self, emb, start, rest, rw):
        h_start = emb[start].view(rw.size(0), 1, emb.size(1))
        h_rest = emb[rest.view(-1)].view(rw.size(0), -1, emb.size(1))
        return (h_start * h_rest).sum(dim=-1).view(-1)

    def loss(self, out,emb, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        out = self.compute(emb, start, rest, pos_rw)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        out = self.compute(emb, start, rest, pos_rw)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss
    

    def validate(self, data, model, criterion):
        # model = GAE(model)
        model.eval()
        with torch.no_grad():
            z, emdb = model(data, model)
            loss = criterion(z[data.valid_mask], data.x[data.valid_mask])
        return loss, emdb


def make_data(new_x, edge_index):
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


def train_test_valid(data, train_valid_idx, test_idx, labels: Optional = None):

    if labels:
        X = data.x[train_valid_idx.indices]
        # y = data.y[train_valid_idx.indices]
        y_ph = labels.iloc[:, 3][train_valid_idx.indices]

        rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)
        for train_part, valid_part in rskf.split(X, y_ph):
            train_idx = np.array(train_valid_idx.indices)[train_part]
            valid_idx = np.array(train_valid_idx.indices)[valid_part]
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


def load_model(learning, new_x):
    if learning in [LearningTypes.classification.name, LearningTypes.regression.name]:
        return GCNSupervised(learning=learning, new_x=new_x)
    return GCNUnsupervised(new_x=new_x)


def select_optimizer(optimizer_type, model, learning_rate):
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
