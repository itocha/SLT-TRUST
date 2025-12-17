"""
Magnitude-based pruning implementation for structured pruning of neural networks.
This module provides functionality to prune neural network weights based on their magnitudes,
following the same approach as train_bayesian_opt.py but implemented in PyTorch.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import prune


class PolynomialDecay:
    """
    Polynomial decay schedule for pruning rate.
    Implements the same schedule as tfmot.sparsity.keras.PolynomialDecay.
    """
    def __init__(self, initial_sparsity, final_sparsity, begin_step, end_step, frequency, power=3):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        self.power = power

    def __call__(self, step):
        if step < self.begin_step:
            return self.initial_sparsity
        elif step >= self.end_step:
            return self.final_sparsity
        else:
            progress = (step - self.begin_step) / (self.end_step - self.begin_step)
            return self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (progress ** self.power)


class MagnitudePruning:
    """
    A class to perform magnitude-based pruning on neural network models.
    This implementation follows the same approach as train_bayesian_opt.py,
    but implemented in PyTorch.
    """

    def __init__(self, pruning_rate: float, total_steps: int):
        """
        Initialize the MagnitudePruning class.

        Args:
            pruning_rate (float): The target sparsity rate (0.0 to 1.0)
            total_steps (int): Total number of steps (batches) for training
        """
        self.pruning_rate = pruning_rate
        self.total_steps = total_steps
        self.current_step = 0
        self.schedule = PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=pruning_rate,
            begin_step=0,  # Start pruning at 25% of training
            end_step=total_steps // 2,    # Complete pruning at 50% of training
            frequency=1,                  # Update every step
            power=3
        )

    def get_current_sparsity(self):
        """
        Get the current sparsity rate based on the polynomial decay schedule.

        Returns:
            float: Current sparsity rate
        """
        return self.schedule(self.current_step)

    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Apply magnitude-based pruning to the model.

        Args:
            model (nn.Module): The model to be pruned

        Returns:
            nn.Module: The pruned model
        """
        current_sparsity = self.get_current_sparsity()

        # Get all parameters for threshold calculation
        all_params = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and 'fc2' not in name:  # Exclude output layer
                all_params.append(module.weight.data.abs().view(-1))

        if not all_params:
            return model

        # Concatenate all parameters
        all_params = torch.cat(all_params)

        # Calculate threshold
        k = max(1, int(len(all_params) * current_sparsity))
        threshold = torch.kthvalue(all_params, k).values

        # Apply pruning to each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and 'fc2' not in name:  # Exclude output layer
                # Create pruning mask
                mask = module.weight.data.abs() > threshold
                # Apply pruning by directly setting values to 0
                module.weight.data[~mask] = 0.0

        return model

    def step(self):
        """
        Update the pruning step counter.
        Should be called after each training step.
        """
        self.current_step += 1

    def get_pruning_info(self):
        """
        Get current pruning information.

        Returns:
            dict: Current pruning information including step and sparsity
        """
        return {
            'step': self.current_step,
            'current_sparsity': self.get_current_sparsity(),
            'target_sparsity': self.pruning_rate
        }
