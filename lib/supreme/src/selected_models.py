
from torch_geometric.data import Data
import torch
from torch_geometric.nn import GAE, VGAE
DEVICE = torch.device("cpu")
from learning_types import LearningTypes
from module import Net
from helper import masking_indexes
from sklearn.model_selection import RepeatedStratifiedKFold
from helper import ratio
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split


class GCNSupervised:
    def __init__(self, learning, new_x) -> None:
        self.learning = learning
        self.new_x = new_x

    def prepare_data(self, new_x, edge_index, labels, col):
        train_valid_idx, test_idx = torch.utils.data.random_split(self.new_x, ratio(self.new_x))
        data = make_data(new_x, edge_index)
        data.y = torch.tensor(labels[col].values, dtype=torch.float32),
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
        train_valid_idx, test_idx = torch.utils.data.random_split(self.new_x, ratio(self.new_x))
        data = data = make_data(new_x, edge_index)
        data.pos_edge_label_index = torch.tensor(
                edge_index[edge_index.columns[0:2]].transpose().values,
                device=DEVICE,
            ).long()
        return train_test_valid(data, train_valid_idx, test_idx)
    
    
    def select_model(self):
        criterion = None
        out_size = 5
        return criterion, out_size

    def train(self, model, optimizer, data, criterion):
        model = GAE(model)
        model.train()
        optimizer.zero_grad()
        emb1, _ = model.encode(data)
        loss = model.recon_loss(
            emb1,
            data.edge_index,
        )
        loss.backward()
        optimizer.step()
        return emb1

    def validate(self, model, criterion, data):
        model = GAE(model)
        model.eval()
        z = model.encode(data)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)



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

    data.valid_mask = torch.tensor(masking_indexes(data=data, indexes=valid_idx), device=DEVICE)
    data.train_mask = torch.tensor(masking_indexes(data=data, indexes=train_idx), device=DEVICE)
    data.test_mask = torch.tensor(masking_indexes(data=data, indexes=test_idx), device=DEVICE)

    return data


def load_model(learning, new_x):
    if learning in [LearningTypes.classification.name, LearningTypes.regression.name]:
        return GCNSupervised(learning=learning, new_x=new_x)
    return GCNUnsupervised(new_x=new_x)



def select_optimizer(optimizer_type, model, learning_rate):
    if optimizer_type == "sgd":
        return  torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sparse_adam":
        return torch.optim.SparseAdam(list(model.parameters()), lr=learning_rate)
    else:
        raise NotImplementedError