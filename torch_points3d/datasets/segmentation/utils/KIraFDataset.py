from torch.utils.data import Dataset
import numpy as np
import torch

class bhatkale45Dataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.labels = y
        if x is None or y is None:
            self.num_classes = 0
        else:
            self.num_classes = len(list(torch.unique(y)))
            
    def __len__(self):
        if self.labels is None: #Remove it, here because of the real data is not there right now, the label which is returned is always None.
            return 0
        return self.labels.shape[0] 

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]

    def extend(self, new_x: torch.Tensor, new_y: torch.Tensor):
        assert type(new_x) == torch.Tensor
        if new_x.shape[0:1] == new_y.shape[0:1]:
            self.x = torch.cat((self.x, new_x))
            self.labels = torch.cat((self.labels, new_y))
        else:
            raise Exception('No matching x and y shape')
