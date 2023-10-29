
from torch_geometric.data import Data
import torch


class RegresisionGCN:
    def __init__(self):
        pass


    def prepare_data():
        Data(
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