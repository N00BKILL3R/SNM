from torchvision import transforms


def build_transforms(dataset_name: str, image_size: int, train: bool = True):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor()]
    if dataset_name.lower() == "cifar10":
        ops += [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    else:
        ops += [transforms.Normalize((0.1307,), (0.3081,))]
    return transforms.Compose(ops)
