import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from learning_types import LearningTypes
from settings import HIDDEN_SIZE, INPUT_SIZE, LEARNING, OUT_SIZE
from torch_geometric.nn import GCNConv, Linear

DEVICE = torch.device("cpu")
load_dotenv(find_dotenv())


class Net(torch.nn.Module):
    """
    Training SUPREME model
    """

    def __init__(self, in_size=INPUT_SIZE, hid_size=HIDDEN_SIZE, out_size=OUT_SIZE):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)
        self.fc = Linear(out_size, out_size)

    def forward(self, data):
        predict = None
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if LearningTypes.clustering.name == LEARNING:
            predict = self.fc(x)
        return x, x_emb, predict
