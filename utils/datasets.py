import os

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch.utils.data import DataLoader, random_split
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from torchvision import datasets, transforms

### Change the root to the path of the dataset
datasets_path = os.environ.get('DATASETS_PATH', '/ldisk/Shared/Datasets')

# Global cache for datasets to avoid repeated downloads
_dataset_cache = {}


def _loader_kwargs(args, eval_mode=False):
    nw = getattr(args, "num_workers", max(1, os.cpu_count() // 8))
    bs = getattr(args, "batch_size", 256)
    kw = dict(
        num_workers=nw if not eval_mode else max(1, nw // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 if nw > 0 else None,
    )
    return {k: v for k, v in kw.items() if v is not None}, min(bs * 2, 512)


def get_mnist_loaders(args):
    # Check cache first
    cache_key = f"mnist_{args.seed}"
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Suppress download messages by setting download=False if dataset exists
    download = not os.path.exists(os.path.join(datasets_path, 'MNIST'))
    train_dataset = datasets.MNIST(
        root=datasets_path,
        train=True,
        download=download,
        transform=transform
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    test_dataset = datasets.MNIST(
        root=datasets_path,
        train=False,
        download=download,
        transform=transform
    )

    train_kwargs, eval_bs = _loader_kwargs(args, eval_mode=False)
    val_kwargs, _ = _loader_kwargs(args, eval_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **train_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=eval_bs,
                            shuffle=False, **val_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs,
                            shuffle=False, **val_kwargs)

    # Cache the loaders
    _dataset_cache[cache_key] = (train_loader, val_loader, test_loader)
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(args):
    # Check cache first
    cache_key = f"cifar10_{args.seed}"
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Suppress download messages by setting download=False if dataset exists
    download = not os.path.exists(os.path.join(datasets_path, 'cifar-10-batches-py'))
    full_train_dataset = datasets.CIFAR10(root=datasets_path, train=True, download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=datasets_path, train=False, download=download, transform=transform_test)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_dataset.dataset.transform = transform_test

    train_kwargs, eval_bs = _loader_kwargs(args, eval_mode=False)
    val_kwargs, _ = _loader_kwargs(args, eval_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **train_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=eval_bs,
                            shuffle=False, **val_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs,
                            shuffle=False, **val_kwargs)

    # Cache the loaders
    _dataset_cache[cache_key] = (train_loader, val_loader, test_loader)
    return train_loader, val_loader, test_loader


def get_cora_loaders(args):
    # Load Cora dataset
    dataset = Planetoid(root=datasets_path, name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Move data to device
    if args is not None:
        device = torch.device("cuda" if args.cuda else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device, non_blocking=True)

    # Create train, validation, and test masks
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # print(f"Train: {train_mask.sum().item()} samples")
    # print(f"Val: {val_mask.sum().item()} samples")
    # print(f"Test: {test_mask.sum().item()} samples")

    return data, train_mask, val_mask, test_mask


def get_ogbn_arxiv_loaders(args):
    device = torch.device("cuda" if getattr(args, "cuda", False) else "cpu")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=datasets_path)
    data = dataset[0]

    data.y = data.y.view(-1).to(torch.long)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx  = split_idx["test"]

    x = data.x
    mu = x[train_idx].mean(dim=0, keepdim=True)
    std = x[train_idx].std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-12)
    data.x = (x - mu) / std

    N = data.num_nodes
    edge_index = to_undirected(data.edge_index, num_nodes=N)
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(N, N)).t()
    adj_t = adj_t.set_diag()
    data.adj_t = adj_t

    train_mask = torch.zeros(N, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[valid_idx] = True
    test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[test_idx] = True

    data = data.to(device, non_blocking=True)
    data.adj_t = data.adj_t.to(device, non_blocking=True)

    return data, train_mask.to(device, non_blocking=True), val_mask.to(device, non_blocking=True), test_mask.to(device, non_blocking=True)

