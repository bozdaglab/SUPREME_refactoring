from typing import Optional, Union

import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from settings import HIDDEN_SIZE, INPUT_SIZE, OUT_SIZE
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.nn import GCNConv, ARGVA, GAE
from torch_geometric.data import Data

load_dotenv(find_dotenv())

DEVICE = torch.device("cpu")
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

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb

    def train(self, data: Data):
        self.model.train()
        
    @torch.no_grad()
    def validate():
        pass

class Encoder(torch.nn.Module):
    def __init__(self, in_size=INPUT_SIZE, hid_size=HIDDEN_SIZE, out_size=OUT_SIZE):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, hid_size)
        self.conv_mu = GCNConv(hid_size, out_size)
        self.conv_logstd = GCNConv(hid_size, out_size)

    def forward(self, x, edge_index, edge_weight):
        x_emb = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.lin1 = Linear(in_size, hid_size)
        self.lin2 = Linear(hid_size, hid_size)
        self.lin3 = Linear(hid_size, out_size)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)


class EncoderDecoder:
    def __init__(self,
                 encoder: Encoder,
                 discriminator: Discriminator):
        self.encoder = encoder
        self.discriminator = discriminator
        self.model = ARGVA(encoder=self.encoder, discriminator=self.discriminator).to(DEVICE)

    def train(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.encoder_loss.zero_grad()
        emb = self.model.encode(data.x, data.edge_index, data.edge_attr)
        for _ in range(5):
            optimizer.decoder_loss.zero_grad()
            discriminator_loss = self.model.discriminator_loss(emb)
            discriminator_loss.backward()
            optimizer.decoder_loss.step()
        loss = self.model.recon_loss(emb, data.pos_edge_labels)
        loss = loss + self.model.reg_loss(emb)
        loss = loss + (1 / data.num_nodes) * self.model.kl_loss()
        loss.backward()
        optimizer.encoder_loss.step()
        if not isinstance(loss, float):
            return float(loss)
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.model.eval()
        emb = self.model.encode(data.x, data.edge_index, data.edge_attr)
        return self.model.test(emb, data.pos_edge_labels, data.neg_edge_labels)

class EncoderInnerProduct:
    def __init__(self,encoder: SUPREME):
        self.encoder = encoder
        self.model = GAE(encoder=self.encoder)

    def train(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model.encode(data)
        loss = self.model.recon_loss(emb, data.pos_edge_labels)
        loss.backward()
        optimizer.step()
        if not isinstance(loss, float):
            return float(loss)
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.model.eval()
        emb, _ = self.model.encode(data)
        return self.model.test(emb, data.pos_edge_labels, data.neg_edge_labels)

