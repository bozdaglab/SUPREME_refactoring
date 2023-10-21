import torch
import numpy as np

def ratio(new_x:torch)-> int:
    shape=new_x.shape[0]
    train_idx = round(shape * 0.75)
    return [train_idx, shape-train_idx]

def masking_indexes(data, indexes):
    return np.array(
                    [i in set(indexes) for i in range(data.x.shape[0])]
                )
