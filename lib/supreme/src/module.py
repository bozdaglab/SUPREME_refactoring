import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from learning_types import EmbeddingModel
from settings import HIDDEN_SIZE, INPUT_SIZE, OUT_SIZE, SELECT_EMB_MODEL
from torch_geometric.nn import GCNConv

DEVICE = torch.device("cpu")
load_dotenv(find_dotenv())


class Net_ori(torch.nn.Module):
    """
    Training SUPREME model
    """

    def __init__(self, in_size=INPUT_SIZE, hid_size=HIDDEN_SIZE, out_size=OUT_SIZE):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, data, model):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb


class Net_encoder_decoder(torch.nn.Module):
    """
    Training SUPREME model
    """

    def __init__(self, in_size=INPUT_SIZE, hid_size=HIDDEN_SIZE, out_size=OUT_SIZE):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)
        self.conv3 = GCNConv(out_size, hid_size)
        self.conv4 = GCNConv(hid_size, in_size)

    def forward(self, data, model):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x_emb = self.conv3(x, edge_index, edge_weight)
        x_emb = F.relu(x_emb)
        x = self.conv4(x_emb, edge_index, edge_weight)
        x = F.relu(x)
        return x, x_emb


def select_clsuetr_model(in_size, hid_size, out_size):
    if SELECT_EMB_MODEL == EmbeddingModel.gcn_ori.name:
        return Net_ori(in_size, hid_size, out_size)
    return Net_encoder_decoder(in_size, hid_size, out_size)
