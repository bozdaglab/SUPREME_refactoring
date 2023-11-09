from typing import Optional, Union

import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from settings import HIDDEN_SIZE, INPUT_SIZE, OUT_SIZE
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.nn import GCNConv
from torch_geometric.utils.negative_sampling import negative_sampling

# from dataclasses import dataclass
from torch_geometric.utils.num_nodes import maybe_num_nodes

load_dotenv(find_dotenv())

DEVICE = torch.device("cpu")
MAX_LOGSTD = 10
EPS = 1e-15


class SUPREME(Module):
    """
    Training SUPREME model
    """

    def __init__(self, in_size: int, hid_size: int, out_size: int):
        super().__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index)

    # def forward(self, data):
    #     predict = None
    #     x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    #     x_emb = self.conv1(x, edge_index, edge_weight)
    #     x = F.relu(x_emb)
    #     x = F.dropout(x, training=self.training)
    #     x = self.conv2(x, edge_index, edge_weight)
    #     return x, x_emb, predict

    # def compute(self, emb, start, rest, rw):
    #     h_start = emb[start].view(rw.size(0), 1, emb.size(1))
    #     h_rest = emb[rest.view(-1)].view(rw.size(0), -1, emb.size(1))
    #     return (h_start * h_rest).sum(dim=-1).view(-1)

    # def loss(self, emb, pos_rw, neg_rw):
    #     # Positive loss.
    #     start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
    #     out = self.compute(emb, start, rest, pos_rw)
    #     pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
    #     # Negative loss.
    #     start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
    #     out = self.compute(emb, start, rest, pos_rw)
    #     neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

    #     return pos_loss + neg_loss


# class InnerProductDecoder(Module):
#     r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
#     <https://arxiv.org/abs/1611.07308>`_ paper

#     .. math::
#         \sigma(\mathbf{Z}\mathbf{Z}^{\top})

#     where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
#     space produced by the encoder."""

#     def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
#         r"""Decodes the latent variables :obj:`z` into edge probabilities for
#         the given node-pairs :obj:`edge_index`.

#         Args:
#             z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
#             sigmoid (bool, optional): If set to :obj:`False`, does not apply
#                 the logistic sigmoid function to the output.
#                 (default: :obj:`True`)
#         """
#         value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
#         return torch.sigmoid(value) if sigmoid else value

# def loss(self, emb, pos_rw, neg_rw: Optional[Tensor] = None):
#     pos_loss = -torch.log(self.forward(emb, pos_rw, sigmoid=True) + EPS).mean()
#     if neg_rw is None:
#         neg_rw = negative_sampling(pos_rw, emb.size(0))
#     neg_loss = -torch.log(self.forward(emb, neg_rw, sigmoid=True) + EPS).mean()

#     return pos_loss + neg_loss


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
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.lin1 = Linear(in_size, hid_size)
        self.lin2 = Linear(hid_size, hid_size)
        self.lin3 = Linear(hid_size, out_size)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)

    # def kl_loss(
    #     self, mu: Optional[Tensor] = None, logstd: Optional[Tensor] = None
    # ) -> Tensor:
    #     r"""Computes the KL loss, either for the passed arguments :obj:`mu`
    #     and :obj:`logstd`, or based on latent variables from last encoding.

    #     Args:
    #         mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
    #             set to :obj:`None`, uses the last computation of :math:`\mu`.
    #             (default: :obj:`None`)
    #         logstd (torch.Tensor, optional): The latent space for
    #             :math:`\log\sigma`.  If set to :obj:`None`, uses the last
    #             computation of :math:`\log\sigma^2`. (default: :obj:`None`)
    #     """
    #     return -0.5 * torch.mean(
    #         torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1)
    #     )

    # def loss(self, emb, pos_rw, discriminator, mu, logstd, num_nodes):
    #     for _ in range(5):
    #         discriminator.zero_grad()
    #         real = torch.sigmoid(discriminator(torch.randn_like(emb)))
    #         fake = torch.sigmoid(discriminator(emb.detach()))
    #         real_loss = -torch.log(real + EPS).mean()
    #         fake_loss = -torch.log(1 - fake + EPS).mean()
    #         discriminator_loss = real_loss + fake_loss
    #         discriminator_loss.backward()
    #         discriminator.step()

    #     loss = self.loss_pos_only(emb, pos_rw)
    #     real = torch.sigmoid(discriminator(emb))
    #     real_loss = -torch.log(real + EPS).mean()
    #     loss += real_loss
    #     loss += (1 / num_nodes) * discriminator.kl_loss(mu, logstd)
    #     return loss


# class EncoderDecoder:
#     def __init__(self,
#                  encoder: Union[SUPREME, Encoder],
#                  decoder: Union[InnerProductDecoder, Discriminator]):
#         self.encoder = encoder
#         self.decoder = decoder

#     def train(self, model, optimizer, data, criterion):
#         mu, logstd = self.encoder(data)
#         emb = mu + torch.randn_like(logstd) * torch.exp(logstd)
#         num_nodes = maybe_num_nodes(data.edge_index)
#         loss = self.decoder.loss(emb=emb,
#                   data=data.pos_edge_labels,
#                   discriminator=self.decoder,
#                   mu=mu,
#                   logstd=logstd,
#                   num_nodes=num_nodes)
#         loss.backward()
#         self.encoder.step()

#         if not isinstance(loss, float):
#             return float(loss)

#         return loss
