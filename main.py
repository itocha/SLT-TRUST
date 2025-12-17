import datetime
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from args import get_args
from models import GAT, GCN, LeNet, ResNet18
from utils.data_utils import create_checkpoint_path, random_noise_data
from utils.datasets import (
    get_cifar10_loaders,
    get_cora_loaders,
    get_mnist_loaders,
    get_ogbn_arxiv_loaders,
)
from utils.magnitude_pruning import MagnitudePruning
from utils.metrics import (
    calculate_ece_mc,
    calculate_flops_cnn,
    calculate_flops_gnn,
    calculate_memory_footprint,
    compute_ape,
)
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.set_kthvalue import set_kthvalue
from utils.slt_modules import (
    PartialFrozenLinear,
    calc_global_sparsity,
    get_threshold,
    initialize_params,
)

warnings.filterwarnings("ignore")

def train(args, model, device, train_loader, optimizer, epoch, pruner=None, scaler=None):
    model.train()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for _, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if args.amp:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
        else:
            output = model(data)
            loss = criterion(output, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.magnitude_pruning:
            model = pruner.prune_model(model)
            pruner.step()

        if args.partial_frozen_slt:
            set_kthvalue(model, args.ep_algo, device)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return 100.0 * correct / total

def evaluate(args, model, device, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def train_iteration_cnn(args, device, train_loader, val_loader, test_loader, iteration, verbose=True):
    if verbose and iteration == 0:
        print("Args:")
        print("-" * 40)
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    if args.model == "lenet":
        model = LeNet(args).to(device, non_blocking=True)
    elif args.model == "resnet18":
        model = ResNet18(args).to(device, non_blocking=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.partial_frozen_slt:
        initialize_params(
            model=model,
            w_init_method=args.init_mode_weight,
            s_init_method=args.init_mode_score,
            m_init_method=args.m_init_method,
            p_ratio=args.p_ratio,
            r_ratio=args.r_ratio,
            r_method='sparsity_distribution',
            nonlinearity='relu',
            algo=args.ep_algo
        )

    optimizer = get_optimizer(
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=float(args.weight_decay),
        model=model,
        filter_bias_and_bn=args.filter_bias_and_bn
    )

    scheduler = get_scheduler(
        scheduler_name = args.scheduler_name,
        optimizer      = optimizer,
        milestones     = args.milestones,
        gamma          = args.gamma,
        max_epoch      = args.epochs,
        min_lr         = args.min_lr,
        warmup_lr_init = args.warmup_lr_init,
        warmup_t       = args.warmup_t,
        warmup_prefix  = args.warmup_prefix
    )

    best_val_acc = 0
    patience_counter = 0
    patience = 5
    min_delta = 0

    if verbose:
        print("-" * 40)
        print("Training Results:")
        print("Epoch, train_acc, val_acc, test_acc")
        print("-" * 40)

    checkpoint_path = create_checkpoint_path(model_name=args.model, iteration=iteration)

    if args.magnitude_pruning:
        total_steps = len(train_loader) * args.epochs
        if hasattr(args.pruning_rate, 'item'):  # It's a tensor
            pruning_rate = args.pruning_rate.item()
        else:
            pruning_rate = args.pruning_rate[0] if isinstance(args.pruning_rate, (list, tuple)) else args.pruning_rate
        pruner = MagnitudePruning(pruning_rate=pruning_rate, total_steps=total_steps)
    else:
        pruner = None

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.amp else None

    if args.m_init_method == 'layer_wise' and args.ep_algo == 'global_ep':
        total_weight_elements = 0
        weighted_pruning_sum = 0
        layer_index = 0
        for _, module in enumerate(model.modules()):
            if isinstance(module, PartialFrozenLinear):
                weight_elements = module.weight.numel()
                total_weight_elements += weight_elements
                # Handle both tensor and list cases for pruning_rate
                if hasattr(args.pruning_rate, 'item'):  # It's a tensor
                    weighted_pruning_sum += weight_elements * args.pruning_rate.item()
                else:
                    weighted_pruning_sum += weight_elements * args.pruning_rate[layer_index]
                layer_index += 1
        new_pruning_rate = weighted_pruning_sum / total_weight_elements
        new_pruning_rate = torch.tensor(new_pruning_rate)
        for _, module in enumerate(model.modules()):
            if isinstance(module, PartialFrozenLinear):
                module.global_sparsity = new_pruning_rate
        args.pruning_rate = new_pruning_rate

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch, pruner, scaler)
        scheduler.step()
        val_acc = evaluate(args, model, device, val_loader)

        if verbose:
            print(f"{epoch}\t{train_acc:.2f}\t{val_acc:.2f}")

        # Early stopping logic
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            if args.magnitude_pruning:
                if epoch > args.epochs//2:
                    torch.save(model.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        # Check for early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best validation accuracy: {best_val_acc:.2f}")
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    if args.partial_frozen_slt:
        set_kthvalue(model, args.ep_algo, device)
    if args.slt:
        threshold = get_threshold(model, args.epochs, args)
    else:
        threshold = None

    all_probs_list, all_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            # Collect individual MC samples instead of averaging
            mc_outputs = []
            for _ in range(args.mc_samples):
                if args.slt:
                    output = model(data, threshold)
                else:
                    output = model(data)
                # Convert logits to probabilities for ECE calculation
                probs = torch.softmax(output, dim=1)
                mc_outputs.append(probs.detach().cpu())
            all_probs_list.append(mc_outputs)
            all_labels.append(target.cpu().numpy())

    # Concatenate all batches
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate mean accuracy using averaged predictions
    all_mean_probs = []
    for batch_probs in all_probs_list:
        batch_mean = torch.stack(batch_probs).mean(0)  # Average across MC samples for this batch
        all_mean_probs.append(batch_mean)

    # Concatenate all batches
    mean_probs = torch.cat(all_mean_probs, dim=0).numpy()
    mean_acc = accuracy_score(all_labels, np.argmax(mean_probs, axis=1)) * 100

    # Calculate ECE using MC samples properly
    # Reorganize MC samples by sample index instead of batch
    n_samples = len(all_labels)
    n_classes = all_probs_list[0][0].size(-1)

    # Initialize tensor to store all MC samples
    mc_probs_reshaped = torch.zeros(args.mc_samples, n_samples, n_classes)

    # Fill the tensor with MC samples
    sample_idx = 0
    for batch_probs in all_probs_list:
        batch_size = batch_probs[0].size(0)
        for mc_idx in range(args.mc_samples):
            mc_probs_reshaped[mc_idx, sample_idx:sample_idx + batch_size] = batch_probs[mc_idx]
        sample_idx += batch_size

    mc_probs_list = [mc_probs_reshaped[i] for i in range(args.mc_samples)]

    ece = calculate_ece_mc(mc_probs_list, all_labels, num_bins=args.num_bins) * 100

    x_test_random = random_noise_data(args.dataset).to(device, non_blocking=True)
    random_probs = []
    with torch.no_grad():
        for _ in range(args.mc_samples):
            if args.slt:
                output = model(x_test_random, threshold)
            else:
                output = model(x_test_random)
            # Convert logits to probabilities for APE calculation
            probs = torch.softmax(output, dim=1)
            random_probs.append(probs.detach().cpu().numpy())
    random_probs = np.concatenate(random_probs, axis=0)
    ape = compute_ape(random_probs)

    if args.magnitude_pruning:
        if hasattr(args.pruning_rate, 'item'):  # It's a tensor
            global_sparsity = args.pruning_rate.item()
        else:
            global_sparsity = args.pruning_rate[0] if isinstance(args.pruning_rate, list) else args.pruning_rate
    else:
        global_sparsity = calc_global_sparsity(model, args)

    flops = calculate_flops_cnn(model, args, device, global_sparsity)
    memory_mb = calculate_memory_footprint(model, args)

    if verbose:
        print("-" * 40)
        print("Final Results")
        print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6), Memory(MB)")
        print("-" * 40)
        print(f"{float(ece):.2f}\t{ape:.4f}\t{mean_acc:.2f}\t{flops:.2f}\t{memory_mb:.4f}")
        print("-" * 40)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return mean_acc, ece, ape, flops, memory_mb

def train_iteration_gnn(args, device, data, train_mask, val_mask, test_mask, iteration, verbose=True):
    if verbose and iteration == 0:
        print("Args:")
        print("-" * 40)
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    if args.model == "gcn":
        model = GCN(args).to(device, non_blocking=True)
    elif args.model == "gat":
        model = GAT(args).to(device, non_blocking=True)
    else:
        raise ValueError(f"Unknown GNN model: {args.model}")

    if args.partial_frozen_slt:
        initialize_params(
            model=model,
            w_init_method=args.init_mode_weight,
            s_init_method=args.init_mode_score,
            m_init_method=args.m_init_method,
            p_ratio=args.p_ratio,
            r_ratio=args.r_ratio,
            r_method='sparsity_distribution',
            nonlinearity='relu',
            algo=args.ep_algo
        )

    optimizer = get_optimizer(
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=float(args.weight_decay),
        model=model,
        filter_bias_and_bn=args.filter_bias_and_bn
    )

    scheduler = get_scheduler(
        scheduler_name=args.scheduler_name,
        optimizer=optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        max_epoch=args.epochs,
        min_lr=args.min_lr,
        warmup_lr_init=args.warmup_lr_init,
        warmup_t=args.warmup_t,
        warmup_prefix=args.warmup_prefix
    )

    best_val_acc = 0
    patience_counter = 0
    patience = 10
    min_delta = 0

    if verbose:
        print("-" * 40)
        print("Training Results:")
        print("Epoch, train_acc, val_acc, test_acc")
        print("-" * 40)

    checkpoint_path = create_checkpoint_path(model_name=args.model, iteration=iteration)

    if args.magnitude_pruning:
        pruning_rate = args.pruning_rate[0] if isinstance(args.pruning_rate, list) else args.pruning_rate
        pruner = MagnitudePruning(
            pruning_rate=pruning_rate,
            total_steps=args.epochs
        )

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if args.amp else None

    if args.m_init_method == 'layer_wise' and args.ep_algo == 'global_ep':
        total_weight_elements = 0
        weighted_pruning_sum = 0
        layer_index = 0
        for _, module in enumerate(model.modules()):
            if isinstance(module, PartialFrozenLinear):
                weight_elements = module.weight.numel()
                total_weight_elements += weight_elements
                # Handle both tensor and list cases for pruning_rate
                if hasattr(args.pruning_rate, 'item'):  # It's a tensor
                    weighted_pruning_sum += weight_elements * args.pruning_rate.item()
                else:
                    weighted_pruning_sum += weight_elements * args.pruning_rate[layer_index]
                layer_index += 1
        new_pruning_rate = weighted_pruning_sum / total_weight_elements
        new_pruning_rate = torch.tensor(new_pruning_rate)
        for _, module in enumerate(model.modules()):
            if isinstance(module, PartialFrozenLinear):
                module.global_sparsity = new_pruning_rate
        args.pruning_rate = new_pruning_rate

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        if args.amp:
            with autocast():
                # Use adj_t if available (for ogbn-arxiv with SparseTensor), otherwise use edge_index
                if hasattr(data, 'adj_t'):
                    out = model(data.x, data.adj_t)
                else:
                    out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[train_mask], data.y[train_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Use adj_t if available (for ogbn-arxiv with SparseTensor), otherwise use edge_index
            if hasattr(data, 'adj_t'):
                out = model(data.x, data.adj_t)
            else:
                out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        scheduler.step()

        if args.magnitude_pruning:
            model = pruner.prune_model(model)
            pruner.step()

        model.eval()
        with torch.no_grad():
            # Use adj_t if available (for ogbn-arxiv with SparseTensor), otherwise use edge_index
            if hasattr(data, 'adj_t'):
                out = model(data.x, data.adj_t)
            else:
                out = model(data.x, data.edge_index)
            train_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().mean() * 100
            val_acc = (out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean() * 100
            test_acc = (out[test_mask].argmax(dim=1) == data.y[test_mask]).float().mean() * 100

        if verbose:
            print(f"{epoch}\t{train_acc:.2f}\t{val_acc:.2f}\t{test_acc:.2f}")

        # Early stopping logic
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            if args.magnitude_pruning:
                if epoch > args.epochs//2:
                    torch.save(model.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        # Check for early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best validation accuracy: {best_val_acc:.2f}")
            break

        if args.partial_frozen_slt:
            set_kthvalue(model, args.ep_algo, device)

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    if args.partial_frozen_slt:
        set_kthvalue(model, args.ep_algo, device)

    all_probs_list = []
    with torch.no_grad():
        for _ in range(args.mc_samples):
            # Use adj_t if available (for ogbn-arxiv with SparseTensor), otherwise use edge_index
            if hasattr(data, 'adj_t'):
                out = model(data.x, data.adj_t)
            else:
                out = model(data.x, data.edge_index)
            probs = torch.exp(out)
            all_probs_list.append(probs[test_mask].detach().cpu())

    # Get labels only once
    all_labels = data.y[test_mask].cpu().numpy()

    # Calculate mean accuracy using averaged predictions
    mean_probs = torch.stack(all_probs_list).mean(0).numpy()
    mean_acc = accuracy_score(all_labels, np.argmax(mean_probs, axis=1)) * 100

    ece = calculate_ece_mc(all_probs_list, all_labels, num_bins=args.num_bins) * 100

    # Calculate APE using x_test_random and mc_samples
    x_test_random = random_noise_data(args.dataset).to(device, non_blocking=True)
    random_probs = []
    with torch.no_grad():
        for _ in range(args.mc_samples):
            # Use adj_t if available (for ogbn-arxiv with SparseTensor), otherwise use edge_index
            if hasattr(data, 'adj_t'):
                out = model(x_test_random, data.adj_t)
            else:
                out = model(x_test_random, data.edge_index)
            probs = torch.exp(out)
            random_probs.append(probs.detach().cpu().numpy())
    random_probs = np.concatenate(random_probs, axis=0)
    ape = compute_ape(random_probs)

    if args.magnitude_pruning:
        if hasattr(args.pruning_rate, 'item'):  # It's a tensor
            global_sparsity = args.pruning_rate.item()
        else:
            global_sparsity = args.pruning_rate[0] if isinstance(args.pruning_rate, list) else args.pruning_rate
    else:
        global_sparsity = calc_global_sparsity(model, args)

    flops = calculate_flops_gnn(model, data, args, global_sparsity)
    memory_mb = calculate_memory_footprint(model, args)

    if np.isnan(ece):
        ece = 99.99

    if np.isnan(ape):
        ape = 0.0000

    if verbose:
        print("-" * 40)
        print("Final Results")
        print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6), Memory(MB)")
        print("-" * 40)
        print(f"{float(ece):.2f}\t{ape:.4f}\t{mean_acc:.2f}\t{flops:.2f}\t{memory_mb:.6f}")
        print("-" * 40)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return mean_acc, ece, ape, flops, memory_mb

def main():
    start_time = time.time()
    start_datetime = datetime.datetime.now()

    args = get_args()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.dataset == "mnist":
        train_loader, val_loader, test_loader = get_mnist_loaders(args)
    elif args.dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(args)
    elif args.dataset == "cora":
        data, train_mask, val_mask, test_mask = get_cora_loaders(args)
    elif args.dataset == "ogbn-arxiv":
        data, train_mask, val_mask, test_mask = get_ogbn_arxiv_loaders(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    all_results = []
    for i in range(args.num_repeats):
        print(f"\nIteration {i+1}/{args.num_repeats}")

        if args.dataset in ["mnist", "cifar10"]:
            test_acc, ece, ape, flops, memory = train_iteration_cnn(
                args, device, train_loader, val_loader, test_loader, i
            )
        elif args.dataset in ["cora", "ogbn-arxiv"]:
            test_acc, ece, ape, flops, memory = train_iteration_gnn(
                args, device, data, train_mask, val_mask, test_mask, i
            )

        all_results.append((test_acc, ece, ape, flops, memory))

    # Calculate means and standard deviations
    acc_values = [r[0] for r in all_results]
    ece_values = [r[1] for r in all_results]
    ape_values = [r[2] for r in all_results]
    flops_values = [r[3] for r in all_results]
    memory_values = [r[4] for r in all_results]

    avg_acc = sum(acc_values) / len(acc_values)
    avg_ece = sum(ece_values) / len(ece_values)
    avg_ape = sum(ape_values) / len(ape_values)
    avg_flops = sum(flops_values) / len(flops_values)
    avg_memory = sum(memory_values) / len(memory_values)

    std_acc = np.std(acc_values)
    std_ece = np.std(ece_values)
    std_ape = np.std(ape_values)
    std_flops = np.std(flops_values)
    std_memory = np.std(memory_values)

    print("\nAverage Final Results (mean ± std)")
    print("ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6), Memory(MB)")
    print("-" * 50)
    print(f"{avg_ece:.2f}±{std_ece:.2f}\t{avg_ape:.4f}±{std_ape:.4f}\t{avg_acc:.2f}±{std_acc:.2f}\t{avg_flops:.2f}±{std_flops:.2f}\t{avg_memory:.6f}±{std_memory:.6f}")
    print("-" * 50)

    end_time = time.time()
    execution_time = end_time - start_time
    print("\nExecution Summary:")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
