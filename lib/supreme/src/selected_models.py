
from torch_geometric.data import Data
import torch
DEVICE = torch.device("cpu")
from learning_types import LearningTypes


class GCNSupervised:
    def __init__(self, learning) -> None:
        self.learning = learning

    def prepare_data(self, new_x, edge_index, labels, col):
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
            y=torch.tensor(labels[col].values, dtype=torch.float32),
        )
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
    def prepare_data(new_x, edge_index, labels, col):
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
    def select_model():
        pass
        # return criterion, out_size

    def train(self, model, optimizer, data, criterion):
        model.train()
        optimizer.zero_grad()
        emb1 = model(data)
        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask],
        )
        loss.backward()
        optimizer.step()
        return emb1


def load_model(learning):
    if learning in [LearningTypes.classification.name, LearningTypes.regression.name]:
        return GCNSupervised(learning=learning)
    return GCNUnsupervised()