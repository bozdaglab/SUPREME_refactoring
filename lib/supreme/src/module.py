import os

import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from torch_geometric.nn import GCNConv

load_dotenv(find_dotenv())
INPUT_SIZE = os.environ.get("INPUT_SIZE")
HIDDEN_SIZE = os.environ.get("HIDDEN_SIZE")
OUT_SIZE = os.environ.get("OUT_SIZE")


class Net(torch.nn.Module):
    """
    Training SUPREME model
    """

    def __init__(self, in_size=INPUT_SIZE, hid_size=HIDDEN_SIZE, out_size=OUT_SIZE):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb


def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    out, emb1 = model(data)
    loss = criterion(out[data.train_mask], torch.tensor(data.y.values.reshape(1,-1)[0])[data.train_mask])
    loss.backward()
    optimizer.step()
    return emb1


def validate(model, criterion, data):
    model.eval()
    with torch.no_grad():
        out, emb2 = model(data)
        pred = out.argmax(dim=1)
        loss = criterion(out[data.valid_mask], torch.tensor(data.y.values.reshape(1,-1)[0])[data.valid_mask])
    return loss, emb2

criterion = torch.nn.CrossEntropyLoss()
