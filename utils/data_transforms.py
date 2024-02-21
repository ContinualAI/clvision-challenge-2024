import torch
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize, RandomCrop


def get_train_transform():
    train_transform = Compose([ToImage(),
                               ToDtype(torch.float32, scale=True),
                               RandomCrop((224, 224)),
                               Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return train_transform


def get_unlabelled_transform():
    train_transform = Compose([ToImage(),
                               ToDtype(torch.float32, scale=True),
                               RandomCrop((224, 224)),
                               Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return train_transform
