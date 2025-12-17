import numpy as np
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

from utils.flops_calc import estimate_gcn_flops


def calculate_ece(probs, labels, num_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        probs (torch.Tensor or np.ndarray): Probability predictions of shape (n_samples, n_classes)
        labels (torch.Tensor or np.ndarray): True labels of shape (n_samples,)
        num_bins (int): Number of bins for ECE calculation

    Returns:
        float: Expected Calibration Error
    """
    # Convert to torch tensors if they are numpy arrays
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Get confidence scores and predictions
    confidences = probs.max(dim=1)[0]
    predictions = torch.argmax(probs, dim=1)
    errors = predictions.eq(labels)

    # Create bin boundaries
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=probs.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = errors[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def calculate_ece_mc(probs_list, labels, num_bins=10, debug=False):
    """
    Calculate Expected Calibration Error (ECE) for Monte Carlo predictions.

    Args:
        probs_list (list): List of probability predictions from MC samples,
                          each of shape (n_samples, n_classes)
        labels (torch.Tensor or np.ndarray): True labels of shape (n_samples,)
        num_bins (int): Number of bins for ECE calculation
        debug (bool): Whether to print debug information

    Returns:
        float: Expected Calibration Error
    """
    # Convert labels to torch tensor if needed
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Stack all MC samples
    probs_stack = torch.stack(probs_list, dim=0)  # (mc_samples, n_samples, n_classes)

    if debug:
        print(f"Debug ECE calculation:")
        print(f"  MC samples shape: {probs_stack.shape}")
        print(f"  Probs_stack range: {probs_stack.min():.4f} - {probs_stack.max():.4f}")
        print(f"  Any NaN in probs_stack: {torch.isnan(probs_stack).any()}")

    # Calculate mean probabilities across MC samples
    mean_probs = probs_stack.mean(dim=0)  # (n_samples, n_classes)

    if debug:
        print(f"  Mean probs range: {mean_probs.min():.4f} - {mean_probs.max():.4f}")
        print(f"  Any NaN in mean_probs: {torch.isnan(mean_probs).any()}")

    # Calculate epistemic uncertainty (variance across MC samples)
    # Add small epsilon to avoid division by zero
    eps = 1e-8

    # Check for NaN or infinite values in probs_stack before calculating variance
    if torch.isnan(probs_stack).any() or torch.isinf(probs_stack).any():
        print(f"  WARNING: NaN or Inf detected in probs_stack, replacing with zeros")
        probs_stack = torch.where(torch.isnan(probs_stack) | torch.isinf(probs_stack),
                                 torch.zeros_like(probs_stack),
                                 probs_stack)

    epistemic_uncertainty = probs_stack.var(dim=0, unbiased=False).mean(dim=1)  # (n_samples,)

    # Handle NaN values in epistemic uncertainty
    if torch.isnan(epistemic_uncertainty).any():
        print(f"  WARNING: NaN detected in epistemic uncertainty, replacing with zeros")
        epistemic_uncertainty = torch.where(torch.isnan(epistemic_uncertainty),
                                           torch.zeros_like(epistemic_uncertainty),
                                           epistemic_uncertainty)
        # Return a reasonable ECE value instead of 99 to avoid breaking Bayesian optimization
        return 0.5

    # Get confidence scores and predictions from mean probabilities
    confidences = mean_probs.max(dim=1)[0]
    predictions = torch.argmax(mean_probs, dim=1)
    errors = predictions.eq(labels)

    # Adjust confidence scores by epistemic uncertainty
    # Higher epistemic uncertainty should reduce confidence
    adjusted_confidences = torch.clamp(confidences - epistemic_uncertainty, 0, 1)

    if debug:
        print(f"  Original confidences range: {confidences.min():.4f} - {confidences.max():.4f}")
        print(f"  Epistemic uncertainty range: {epistemic_uncertainty.min():.4f} - {epistemic_uncertainty.max():.4f}")
        print(f"  Adjusted confidences range: {adjusted_confidences.min():.4f} - {adjusted_confidences.max():.4f}")
        print(f"  Number of samples: {len(adjusted_confidences)}")

    # Create bin boundaries
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=mean_probs.device)

    if debug:
        print(f"  Bin analysis:")

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        in_bin = adjusted_confidences.gt(bin_lower.item()) * adjusted_confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if debug:
            print(f"    Bin {i}: [{bin_lower:.2f}, {bin_upper:.2f}] - {prop_in_bin.item():.4f} samples")

        if prop_in_bin.item() > 0:
            accuracy_in_bin = errors[in_bin].float().mean()
            avg_confidence_in_bin = adjusted_confidences[in_bin].mean()
            bin_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            ece += bin_ece

            if debug:
                print(f"      Accuracy: {accuracy_in_bin:.4f}, Confidence: {avg_confidence_in_bin:.4f}, ECE: {bin_ece.item():.4f}")

    if debug:
        print(f"  Total ECE: {ece.item():.4f}")

    return ece.item()

def compute_ape(probs):
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    entropy = -np.sum(probs * log_probs, axis=1)
    ape = np.mean(entropy)
    return ape

def calculate_memory_footprint(model, args):
    """Calculate memory footprint of the model in MB"""
    total_memory = 0
    if args.m_init_method == 'layer_wise':
        index = 0
        for name, param in model.named_parameters():
            if args.slt or args.partial_frozen_slt:
                if 'score' in name:
                    total_memory += param.numel() * 1 * (1 - args.r_ratio[index] - args.p_ratio[index])
                    index += 1
            else:
                total_memory += param.numel() * 32 * (1 - args.r_ratio[index] - args.p_ratio[index])
                index += 1
    else:
        for name, param in model.named_parameters():
            if args.slt or args.partial_frozen_slt:
                if 'score' in name:
                    total_memory += param.numel() * 1
            else:
                total_memory += param.numel() * 32
        # Check if r_ratio and p_ratio are lists, if so use first element, otherwise use as is
        r_ratio = args.r_ratio[0] if isinstance(args.r_ratio, list) else args.r_ratio
        p_ratio = args.p_ratio[0] if isinstance(args.p_ratio, list) else args.p_ratio
        total_memory = total_memory * (1 - r_ratio - p_ratio)

    memory_mb = total_memory / (8 * 1024 * 1024)  # Convert bits to MB
    return memory_mb

def calculate_flops_cnn(model, args, device, global_sparsity):
    """Calculate FLOPS for CNN models"""
    # Create temporary model for FLOPS calculation
    temp_args = type("Args", (), vars(args))()
    temp_args.slt = False
    temp_args.partial_frozen_slt = False

    if args.model == "lenet":
        from models import LeNet
        temp_model = LeNet(temp_args).to(device, non_blocking=True)
    elif args.model == "resnet18":
        from models import ResNet18
        temp_model = ResNet18(temp_args).to(device, non_blocking=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    temp_model.eval()

    # Set input shape based on dataset
    if args.dataset.lower() == "mnist":
        input_shape = (1, 28, 28)
    elif args.dataset.lower() == "cifar10":
        input_shape = (3, 32, 32)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Calculate MACs and convert to FLOPS
    macs, _ = get_model_complexity_info(
        temp_model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False
    )
    flops = 2 * macs

    # Apply pruning rate if using SLT or partial frozen SLT
    if args.slt or args.partial_frozen_slt or args.magnitude_pruning:
        flops = flops * args.mc_samples * (1 - global_sparsity)

    # Convert to millions of FLOPS
    flops = flops / 1e6
    return flops

def calculate_flops_gnn(model, data, args, global_sparsity):
    """Calculate FLOPS for GNN models"""
    flops = estimate_gcn_flops(data, model)

    # Apply pruning rate if using SLT or partial frozen SLT
    if args.slt or args.partial_frozen_slt or args.magnitude_pruning:
        flops = flops * args.mc_samples * (1 - global_sparsity)

    return flops

