from datasets.cifar10_dataset import build_cifar10_loaders
from datasets.mnist_dataset import build_mnist_loaders


def build_dataloaders(cfg):
    name = cfg["dataset"]["name"].lower()
    if name == "cifar10":
        return build_cifar10_loaders(cfg)
    if name == "mnist":
        return build_mnist_loaders(cfg)
    raise ValueError(f"Unsupported dataset: {name}")
