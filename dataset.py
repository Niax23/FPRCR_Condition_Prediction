import torch
import numpy as np
import torch_geometric
from numpy import concatenate as npcat

class FPRCRDataset(torch.utils.data.Dataset):
    def __init__(self, reacs, prods, labels):
        super(FPRCRDataset, self).__init__()
        self.reac_fps = reacs
        self.prod_fps = prods
        self.labels = labels

    def __len__(self):
        return len(self.reac_fps)

    def __getitem__(self, idx):
        return self.reac_fps[idx],self.prod_fps[idx], self.labels[idx]


def FPRCR_collate_fn(batch):
    reac_fps = torch.stack([x[0] for x in batch])
    prod_fps = torch.stack([x[1] for x in batch])
    labels = torch.LongTensor([x[2] for x in batch])
    return reac_fps, prod_fps, labels

