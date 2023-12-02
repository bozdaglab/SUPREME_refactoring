import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from learning_types import LearningTypes
from torch.nn import Linear, Module
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

load_dotenv(find_dotenv())

DEVICE = torch.device("cuda")
MAX_LOGSTD = 10
EPS = 1e-15


# https://arxiv.org/abs/1607.00653,
# https://arxiv.org/abs/1611.0730,
# https://arxiv.org/abs/1706.02216


class SUPREME(Module):
    """
    Training SUPREME model
    """

    def __init__(self, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb


class Encoder(Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, hid_size)
        self.conv_mu = GCNConv(hid_size, out_size)
        self.conv_logstd = GCNConv(hid_size, out_size)

    def forward(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Decoder(Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.lin1 = Linear(in_size, hid_size)
        self.lin2 = Linear(hid_size, hid_size)
        self.lin3 = Linear(hid_size, out_size)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)


class SupremeClassification:
    def __init__(self, model: SUPREME, super_unsuper_model: str) -> None:
        self.model = model
        if super_unsuper_model == LearningTypes.regression.name:
            self.criterion = torch.nn.MSELoss()
        elif super_unsuper_model == LearningTypes.classification.name:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model(data)
        loss = self.criterion(emb[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.model.eval()
        emb, _ = self.model(data)
        loss = self.criterion(emb[data.valid_mask], data.y[data.valid_mask])
        return loss, emb


class SupremeClusteringLink:
    def __init__(self, model: SUPREME) -> None:
        self.model = model
        self.criterion_link = torch.nn.BCEWithLogitsLoss()

    def train(self, optimizer: torch.optim, data: Data):
        # GraphSAGE predict adhacency matrix
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model(data)
        h_src = emb[data.edge_index[0]]
        h_dst = emb[data.edge_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)
        loss = self.criterion_link(link_pred, data.edge_attr)
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.model.eval()
        emb, _ = self.model(data)
        h_src = emb[data.edge_index[0]]
        h_dst = emb[data.edge_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)
        loss = self.criterion_link(link_pred, data.edge_attr)
        return loss, emb
