from __future__ import annotations

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from datasets.transforms import build_transforms


def build_mnist_loaders(cfg):
    dc = cfg["dataset"]
    train_tf = build_transforms("mnist", dc["image_size"], train=True)
    test_tf = build_transforms("mnist", dc["image_size"], train=False)

    full_train = MNIST(root=dc["root"], train=True, download=True, transform=train_tf)
    test_set = MNIST(root=dc["root"], train=False, download=True, transform=test_tf)

    val_len = int(len(full_train) * dc["val_ratio"])
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=dc["batch_size"], shuffle=True, num_workers=dc["num_workers"], pin_memory=dc.get("pin_memory", False))
    val_loader = DataLoader(val_set, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"], pin_memory=dc.get("pin_memory", False))
    test_loader = DataLoader(test_set, batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"], pin_memory=dc.get("pin_memory", False))
    return train_loader, val_loader, test_loader
