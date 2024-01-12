from collections import namedtuple
from typing import Union

import torch
from learning_types import LearningTypes, OptimizerType, SuperUnsuperModel
from module import (
    SUPREME,
    Discriminator,
    EncoderEntireInput,
    EncoderInnerProduct,
    SupremeClassification,
    SupremeClusteringLink,
)
from torch.nn import Module

DEVICE = torch.device("cpu")
EPS = 1e-15


def model_loss_output(self, model_choice: str) -> int:
    if model_choice == LearningTypes.regression.name:
        return 1
    elif model_choice == LearningTypes.classification.name:
        return len(self.labels.value_counts().unique())
    return self.new_x.shape[-1]


def select_optimizer(
    optimizer_type: str, model: Module, learning_rate: float
) -> torch.optim:
    """
    This function selects the optimizer

    Parameters:
    -----------
    optimizer_type:
        Name of the optimizer
    model:
        Our model, supervised or unsupervised GCN
    learning_rate:
        Learning rate

    Return:
        Torch optimizer
    """
    if isinstance(model, EncoderInnerProduct):
        return torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)

    if isinstance(model, EncoderEntireInput):
        losses = namedtuple("losses", ["encoder_loss", "decoder_loss"])
        encoder_loss = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
        decoder_loss = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)
        return losses(encoder_loss=encoder_loss, decoder_loss=decoder_loss)
    if optimizer_type == OptimizerType.sgd.name:
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9
        )
    elif optimizer_type == OptimizerType.adam.name:
        return torch.optim.Adam(
            model.model.parameters(), lr=learning_rate, weight_decay=0.001
        )
    elif optimizer_type == OptimizerType.sparse_adam.name:
        return torch.optim.SparseAdam(list(model.parameters()), lr=learning_rate)
    else:
        raise NotImplementedError


def select_model(
    super_unsuper_model: str, in_size: int, hid_size: int, out_size: int
) -> Union[
    SupremeClassification,
    SupremeClusteringLink,
    EncoderInnerProduct,
    EncoderEntireInput,
]:
    """
    This function selects the return of the model

    Parameters:
    ----------
    in_size:
        Input size of the model
    hid_size:
        hidden size
    out_size:
        output size of the model

    Return:
        Models, whether original SUPREME, or encoder-decoder model
    """
    if super_unsuper_model in [
        LearningTypes.classification.name,
        LearningTypes.regression.name,
    ]:
        model = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
        return SupremeClassification(
            model=model, super_unsuper_model=super_unsuper_model
        )
    else:
        if super_unsuper_model == SuperUnsuperModel.linkprediction.name:
            model = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
            return SupremeClusteringLink(model=model)
        elif super_unsuper_model == SuperUnsuperModel.encoderinproduct.name:
            encoder = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
            return EncoderInnerProduct(encoder=encoder)
        elif super_unsuper_model == SuperUnsuperModel.entireinput.name:
            encoder = SUPREME(in_size=in_size, hid_size=hid_size, out_size=out_size)
            decoder = Discriminator(
                in_size=in_size, hid_size=hid_size, out_size=out_size
            )
            return EncoderEntireInput(encoder=encoder, decoder=decoder)
