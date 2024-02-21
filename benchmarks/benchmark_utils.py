import os
import pickle
from typing import Union, List

import torch
from PIL import Image
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import Dataset


class FileListDataset(Dataset):
    def __init__(self, filelist: List[str], targets: List[int]):
        self.filelist = filelist
        self.targets = targets
        assert len(filelist) == len(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        img = Image.open(self.filelist[item])
        target = self.targets[item]
        return img, target


class UnlabelledDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0],  # just return Data needs to be tuple otherwise TransformGroups unpack is not working!


class UnlabelledAvalancheDataset(AvalancheDataset):
    def __getitem__(self, idx: Union[int, slice]):
        res = super().__getitem__(idx)
        if len(res) == 1:
            return res[0]  # unpack!
        return res
