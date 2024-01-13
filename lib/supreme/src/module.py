# from typing import Union

import logging

# import numpy as np
import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from learning_types import LearningTypes
from set_logging import set_log_config
from torch.nn import Linear, Module
from torch_geometric.data import Data

# from sklearn.metrics import (  # mean_squared_error,; r2_score,
#     average_precision_score,
#     roc_auc_score,
# )
# from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GAE, GCNConv

# from torch_geometric.nn.models.autoencoder import InnerProductDecoder
set_log_config()
logger = logging.getLogger()
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

    def __init__(
        self,
        in_size: int,
        hid_size: int,
        out_size: int,
        activation_function: str = "relu",
        drop_out=True,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)
        self.activ_func = activation_function
        self.drop_out = drop_out

    def forward(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)

        if self.activ_func == "relu":
            active_func = F.relu
        elif self.activ_func == "sigmoid":
            active_func = F.sigmoid
        elif self.activ_func == "tanh":
            active_func = F.tanh

        x = active_func(x_emb)
        if self.drop_out:
            x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        return x, x_emb


class Discriminator(Module):
    def __init__(self, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.lin1 = Linear(out_size, hid_size)
        self.lin2 = Linear(hid_size, hid_size)
        self.lin3 = Linear(hid_size, in_size)

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
        # self.Q = 10  # defines the number of negative samples

    def train(self, optimizer: torch.optim, data: Data):
        # GraphSAGE predict adjacency matrix https://arxiv.org/abs/1706.02216
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model(data)
        if "neg_edge_labels" in data.keys():
            pos_data = data.pos_edge_labels
            neg_data = data.neg_edge_labels
        else:
            pos_data = data.test_pos_edge_index
            neg_data = data.test_neg_edge_index

        src_pos = pos_data[0]
        dst_pos = pos_data[1]
        src_neg = neg_data[0]
        dst_neg = neg_data[1]
        link_pred_p = (emb[src_pos] * emb[dst_pos]).sum(dim=-1)
        edge_label_p = torch.ones(src_pos.size(0))
        link_pred_n = (emb[src_neg] * emb[dst_neg]).sum(dim=-1)
        edge_label_n = torch.zeros(src_neg.size(0))
        link_pred = torch.cat((link_pred_p, link_pred_n), dim=0)
        edge_label = torch.cat((edge_label_p, edge_label_n), dim=0)
        loss = self.criterion_link(link_pred, edge_label)
        # this can be an alternative
        # node_score = []
        # for pos_node in src_pos:
        #     list_pos_node = np.where(src_pos == pos_node)
        #     pos_score = F.cosine_similarity(
        #         emb[data.pos_edge_labels[0][list_pos_node]],
        #         emb[data.pos_edge_labels[1][list_pos_node]],
        #     )
        #     link_pred_pos = torch.log(torch.sigmoid(pos_score))

        #     list_neg_node = np.where(src_neg == pos_node)
        #     neg_score = F.cosine_similarity(
        #         emb[data.neg_edge_labels[0][list_neg_node]],
        #         emb[data.neg_edge_labels[1][list_neg_node]],
        #     )
        #     link_pred_neg = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)))
        #     loss = torch.mean(-link_pred_pos - link_pred_neg).view(1, -1)
        #     if not bool(loss.isnan()): # take care of nan values
        #         node_score.append(loss)
        # loss = torch.mean(torch.cat(node_score, 0))
        loss.backward()
        optimizer.step()
        if not isinstance(loss, float):
            return float(loss), emb
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.model.eval()
        emb, _ = self.model(data)
        if "neg_edge_labels" in data.keys():
            pos_data = data.pos_edge_labels
            neg_data = data.neg_edge_labels
        else:
            pos_data = data.test_pos_edge_index
            neg_data = data.test_neg_edge_index
        pos_y = emb.new_ones(pos_data.size(1))
        neg_y = emb.new_zeros(neg_data.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = (emb[pos_data[0]] * emb[pos_data[1]]).sum(dim=-1)
        neg_pred = (emb[neg_data[0]] * emb[neg_data[1]]).sum(dim=-1)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        loss = self.criterion_link(y, pred)
        # y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        # return roc_auc_score(y, pred), average_precision_score(y, pred), float(loss)
        return float(loss)


class EncoderInnerProduct:
    # predict the link between two nodes. Actuallu predict the adjacency matrix
    def __init__(self, encoder: SUPREME):
        self.encoder = encoder
        self.model = GAE(encoder=self.encoder)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, optimizer: torch.optim, data: Data):
        self.model.train()
        optimizer.zero_grad()
        emb, _ = self.model.encode(data)
        if "neg_edge_labels" in data.keys():
            loss = self.model.recon_loss(
                z=emb,
                pos_edge_index=data.pos_edge_labels,
                neg_edge_index=data.neg_edge_labels,
            )
        else:
            loss = self.model.recon_loss(
                z=emb, pos_edge_index=data.train_pos_edge_index
            )
        loss.backward()
        optimizer.step()
        if not isinstance(loss, float):
            return float(loss), emb
        return loss, emb

    @torch.no_grad()
    def validate(self, data: Data):

        self.model.eval()
        emb, _ = self.model.encode(data)
        if "neg_edge_labels" in data.keys():
            pos_data = data.pos_edge_labels
            neg_data = data.neg_edge_labels
        else:
            pos_data = data.test_pos_edge_index
            neg_data = data.test_neg_edge_index
        pos_y = emb.new_ones(pos_data.size(1))
        neg_y = emb.new_zeros(neg_data.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(emb, pos_data, sigmoid=True)
        neg_pred = self.model.decoder(emb, neg_data, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        loss = self.criterion(y, pred)
        # y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        # return roc_auc_score(y, pred), average_precision_score(y, pred),
        return float(loss)


class EncoderEntireInput:
    def __init__(self, encoder: SUPREME, decoder: Discriminator):
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = torch.nn.MSELoss()

    def train(self, optimizer: torch.optim, data: Data):
        self.encoder.train()
        optimizer.encoder_loss.zero_grad()
        emb, _ = self.encoder(data)
        out_emb = self.decoder(emb)
        loss = self.criterion(out_emb, data.x)
        loss.backward()
        optimizer.encoder_loss.step()
        if not isinstance(loss, float):
            return float(loss), emb
        return loss

    @torch.no_grad()
    def validate(self, data: Data):
        self.encoder.eval()
        emb, _ = self.encoder(data)
        dec_out = self.decoder(emb)
        loss = self.criterion(dec_out, data.x)
        # return r2_score(dec_out, data.x), mean_squared_error(dec_out, data.x), loss
        return float(loss)
