################################################################
# MODIFICATIONS TO THIS FILES ARE NOT PERMITTED
################################################################

import os
import pickle
import torch
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import TransformGroups, make_classification_dataset
from torch.utils.data import Subset
from torchvision.transforms.v2 import Compose, Normalize, ToImage, ToDtype, CenterCrop

from benchmarks.benchmark_utils import UnlabelledAvalancheDataset, UnlabelledDataset, FileListDataset
from utils.data_transforms import get_unlabelled_transform, get_train_transform
from utils.generic import set_random_seed


def _get_test_transform():
    return Compose([ToImage(),
                    ToDtype(torch.float32, scale=True),
                    CenterCrop((224, 224)),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])


def generate_benchmark(benchmark_config: str):
    with open(os.path.join("scenario_configs", benchmark_config), "rb") as f:
        config = pickle.load(f)

    # set random seed for all sampling
    set_random_seed()
    with open(os.path.join("data", "clvision2024-data", "clvision2024-splits", "train.txt"), "r") as f:
        train_lines = f.readlines()

        file_list = []
        target_list = []
        for line in train_lines:
            line_parts = line.strip().split(" ")
            file_list.append(os.path.join("data", "clvision2024-data", "clvision2024-imgs", line_parts[0].strip()))
            target_list.append(int(line_parts[1].strip()))

    with open(os.path.join("data", "clvision2024-data", "clvision2024-splits", "test.txt"), "r") as f:
        test_lines = f.readlines()
        test_lines = [os.path.join("data", "clvision2024-data", "clvision2024-imgs", t.strip()) for t in test_lines]

    train_ds = FileListDataset(file_list, target_list)
    test_ds = FileListDataset(test_lines, [0] * len(test_lines))
    train_datasets = []
    unlabeled_datasets = []

    for indices_train, indices_unl in zip(config["img_indices_per_exp"], config["unlabelled_indices_per_exp"]):
        train_datasets.append(Subset(train_ds, indices_train))

        unlabelled_ds = UnlabelledAvalancheDataset([UnlabelledDataset(Subset(train_ds, indices_unl))],
                                                   transform_groups=TransformGroups(
                                                       {"train": (get_unlabelled_transform(), None),
                                                        "eval": (_get_test_transform(), None)}))
        unlabeled_datasets.append(unlabelled_ds)

    scenario = dataset_benchmark(train_datasets=train_datasets,
                                 test_datasets=[test_ds],
                                 train_transform=get_train_transform(),
                                 eval_transform=_get_test_transform())

    scenario.unlabelled_stream = unlabeled_datasets
    # ensure that each training stream has and corresponding unlabelled part
    assert len(scenario.train_stream) == len(scenario.unlabelled_stream)
    return scenario
