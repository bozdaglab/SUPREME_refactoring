import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from learning_types import LearningTypes
from settings import LEARNING, LINKPREDICTION
from torch.nn import Linear, Module
from torch_geometric.data import Data
from torch_geometric.nn import ARGVA, GAE, GCNConv

load_dotenv(find_dotenv())

DEVICE = torch.device("cpu")
MAX_LOGSTD = 10
EPS = 1e-15

if LEARNING == LearningTypes.regression.name:
    CRITERION = torch.nn.MSELoss()
elif LEARNING == LearningTypes.classification.name:
    CRITERION = torch.nn.CrossEntropyLoss()
elif LEARNING == LearningTypes.clustering.name:
    CRITERION = torch.nn.BCEWithLogitsLoss()

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


class Encoder(torch.nn.Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int):
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


class SupremeClassification:
    def __init__(self, model: SUPREME) -> None:
        self.model = model

    def train(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model(data)
        loss = CRITERION(emb[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.model.eval()
        emb, _ = self.model(data)
        loss = CRITERION(emb[data.valid_mask], data.y[data.valid_mask])
        return loss


class SupremeClusteringLink:
    def __init__(self, model: SUPREME) -> None:
        self.model = model

    def train(self, optimizer: torch.optim, data: Data):
        if LINKPREDICTION:
            return self.train_link_prediction(optimizer, data)
        else:
            return self.train_posneg(optimizer, data)

    def train_link_prediction(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.zero_grad()
        emb = self.model(data)
        h_src = emb[data.edge_label_index[0]]
        h_dst = emb[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)
        loss = CRITERION(link_pred, data.edge_index)
        loss.backward()
        optimizer.step()
        return loss

    def train_posneg(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model(data)

        # Positive loss.
        pos_rw = data.pos_edge_labels
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        out = self.compute(emb, start, rest, pos_rw)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        neg_rw = data.neg_edge_labels
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        out = self.compute(emb, start, rest, pos_rw)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        return pos_loss + neg_loss

    def compute(self, emb, start, rest, rw):
        h_start = emb[start].view(rw.size(0), 1, emb.size(1))
        h_rest = emb[rest.view(-1)].view(rw.size(0), -1, emb.size(1))
        return (h_start * h_rest).sum(dim=-1).view(-1)

    @torch.no_grad()
    def validation(self, data):
        self.model.eval()
        emb, _ = self.mdoel(data)
        # encoder_decoder loss, criterion is MSE
        # loss = criterion(out[data.train_mask], data.x[data.train_mask])


class EncoderDecoder:
    def __init__(self, encoder: Encoder, discriminator: Discriminator):
        self.encoder = encoder
        self.discriminator = discriminator
        self.model = ARGVA(encoder=self.encoder, discriminator=self.discriminator).to(
            DEVICE
        )

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
    def __init__(self, encoder: SUPREME):
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
