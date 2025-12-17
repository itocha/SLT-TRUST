import datetime
import os
import uuid

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid

MNIST_mean = (0,)
MNIST_std = (1,)
CIFAR10_mean = (0.4914, 0.4822, 0.4465)
CIFAR10_std = (0.2023, 0.1994, 0.2010)

def random_noise_data(dataset):
    """
    Generate random noise dataset with the same statistics as the original dataset.

    Args:
        dataset (str): Name of the dataset ("mnist", "cifar10", or "cora")

    Returns:
        torch.Tensor: Random noise data with the same shape and statistics as the original dataset
    """
    if dataset == "mnist":
        x_test = torch.normal(mean=MNIST_mean[0], std=MNIST_std[0], size=(10000, 1, 28, 28))
        return x_test.float()

    elif dataset == "cifar10":
        x_test = torch.normal(mean=CIFAR10_mean[0], std=CIFAR10_std[0], size=(10000, 3, 32, 32))
        return x_test.float()

    elif dataset == "cora":
        data = Planetoid(root='/work/Shared/Datasets', name='Cora')[0]
        features = data.x
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        x_test = torch.normal(mean=mean, std=std).repeat(2708, 1)
        return x_test.float()

    elif dataset == "ogbn-arxiv":
        data = PygNodePropPredDataset(name='ogbn-arxiv', root='/work/Shared/Datasets')[0]
        features = data.x
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        x_test = torch.normal(mean=mean, std=std).repeat(169343, 1)
        return x_test.float()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_checkpoint_path(model_name="lenet", iteration=0, base_dir="checkpoints"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    uid = uuid.uuid4().hex[:6]
    checkpoint_path = f"{base_dir}/{model_name}_best_{timestamp}_pid{pid}_{uid}_iter{iteration}.pth"
    return checkpoint_path
