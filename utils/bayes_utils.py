import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_bayes_config(config_path: str) -> Dict[str, Any]:
    """Load bayes configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_parameter_spaces(ctx, config: Dict[str, Any]):
    """Setup parameter spaces based on configuration"""
    params_config = config.get('params', {})
    fixed_params = config.get('fixed_params', {})

    # Ensure num_layers is set in ctx.params - use fixed_params or default
    if not hasattr(ctx.params, 'num_layers'):
        # Get num_layers from fixed_params first, then default
        num_layers_value = fixed_params.get('num_layers', 4)

        # Add num_layers field to ContextParams dynamically
        if hasattr(ctx.params, '__fields__'):
            ctx.params.__fields__['num_layers'] = (int, num_layers_value)
        setattr(ctx.params, 'num_layers', num_layers_value)

    if ctx.params.m_init_method == "layer_wise":
        # Setup parameter spaces for each layer
        for layer in range(1, ctx.params.num_layers + 1):
            layer_key = f"layer_{layer}"
            layer_params = params_config.get(layer_key, {})

            for param_name in ['pruned_frozen', 'locked_frozen', 'prune_nfrozen', 'lock_nfrozen']:
                space_key = f"{param_name}_space"
                type_key = f"{param_name}_type"

                if space_key in layer_params and type_key in layer_params:
                    param_type = layer_params[type_key]
                    space = layer_params[space_key]

                    if param_type == 'discrete':
                        # For discrete parameters, space should be a list of values
                        ctx.params.__fields__[f"{layer_key}_{param_name}"] = (list, space)
                        setattr(ctx.params, f"{layer_key}_{param_name}", space)
                    elif param_type == 'continuous':
                        # For continuous parameters, space should be a tuple (min, max)
                        if isinstance(space, (list, tuple)) and len(space) == 2:
                            ctx.params.__fields__[f"{layer_key}_{param_name}"] = (tuple, tuple(space))
                            setattr(ctx.params, f"{layer_key}_{param_name}", tuple(space))
                        else:
                            raise ValueError(f"Continuous parameter {param_name} must have space as [min, max] or (min, max), got: {space} (type: {type(space)})")
                    elif param_type == 'random':
                        # For random parameters, space should be a list of values to choose from randomly
                        if isinstance(space, (list, tuple)):
                            ctx.params.__fields__[f"{layer_key}_{param_name}"] = (list, space)
                            setattr(ctx.params, f"{layer_key}_{param_name}", space)
                        else:
                            raise ValueError(f"Random parameter {param_name} must have space as a list of values, got: {space} (type: {type(space)})")

                    # Debug output to verify parameter setting
                    print(f"Setting {layer_key}_{param_name} with space: {space} and type: {param_type}")
    else:
        # Setup parameter spaces for unconstrained parameters
        for param_name in ['pruned_frozen', 'locked_frozen', 'prune_nfrozen', 'lock_nfrozen', 'scaling_rate', 'dropout_rate', 'mc_samples']:
            space_key = f"{param_name}_space"
            type_key = f"{param_name}_type"

            if space_key in params_config and type_key in params_config:
                param_type = params_config[type_key]
                space = params_config[space_key]

                if param_type == 'discrete':
                    # For discrete parameters, space should be a list of values
                    setattr(ctx.params, space_key, space)
                elif param_type == 'continuous':
                    # For continuous parameters, space should be a tuple (min, max)
                    if isinstance(space, (list, tuple)) and len(space) == 2:
                        setattr(ctx.params, space_key, tuple(space))
                    else:
                        raise ValueError(f"Continuous parameter {param_name} must have space as [min, max] or (min, max), got: {space} (type: {type(space)})")
                elif param_type == 'random':
                    # For random parameters, space should be a list of values to choose from randomly
                    if isinstance(space, (list, tuple)):
                        setattr(ctx.params, space_key, space)
                    else:
                        raise ValueError(f"Random parameter {param_name} must have space as a list of values, got: {space} (type: {type(space)})")

    # Special handling for num_bayes_layers (integer parameter)
    if 'num_bayes_layers_space' in params_config and 'num_bayes_layers_type' in params_config:
        param_type = params_config['num_bayes_layers_type']
        space = params_config['num_bayes_layers_space']

        if param_type == 'discrete':
            # For discrete integer parameters, space should be a list of integers
            setattr(ctx.params, 'num_bayes_layers_space', space)
        elif param_type == 'continuous':
            # For continuous integer parameters, space should be a tuple (min, max)
            if isinstance(space, (list, tuple)) and len(space) == 2:
                setattr(ctx.params, 'num_bayes_layers_space', tuple(space))
            else:
                raise ValueError(f"Continuous parameter num_bayes_layers must have space as [min, max] or (min, max)")
        elif param_type == 'random':
            # For random integer parameters, space should be a list of integers to choose from randomly
            if isinstance(space, (list, tuple)):
                setattr(ctx.params, 'num_bayes_layers_space', space)
            else:
                raise ValueError(f"Random parameter num_bayes_layers must have space as a list of integers")

    # Special handling for mc_samples (integer parameter)
    if 'mc_samples_space' in params_config and 'mc_samples_type' in params_config:
        param_type = params_config['mc_samples_type']
        space = params_config['mc_samples_space']

        if param_type == 'discrete':
            # For discrete integer parameters, space should be a list of integers
            setattr(ctx.params, 'mc_samples_space', space)
        elif param_type == 'continuous':
            # For continuous integer parameters, space should be a tuple (min, max)
            if isinstance(space, (list, tuple)) and len(space) == 2:
                setattr(ctx.params, 'mc_samples_space', tuple(space))
            else:
                raise ValueError(f"Continuous parameter mc_samples must have space as [min, max] or (min, max)")
        elif param_type == 'random':
            # For random integer parameters, space should be a list of integers to choose from randomly
            if isinstance(space, (list, tuple)):
                setattr(ctx.params, 'mc_samples_space', space)
            else:
                raise ValueError(f"Random parameter mc_samples must have space as a list of integers")

    # Initialize parameters as lists for layer-wise optimization
    if ctx.params.m_init_method == "layer_wise":
        num_layers = ctx.params.num_layers
        for param_name in ['pruned_frozen', 'locked_frozen', 'prune_nfrozen', 'lock_nfrozen']:
            if not isinstance(getattr(ctx.params, param_name, None), list):
                ctx.params.__fields__[param_name] = (list, [0.0] * num_layers)
                setattr(ctx.params, param_name, [0.0] * num_layers)


def calculate_constrained_parameters(ctx):
    """Calculate constrained parameters from unconstrained parameters using softmax"""
    if ctx.params.m_init_method == "layer_wise":
        num_layers = ctx.params.num_layers
        pruning_rates = []
        p_ratios = []
        r_ratios = []

        # Initialize lists to store parameters for each layer
        pruned_frozen_list = []
        locked_frozen_list = []
        prune_nfrozen_list = []
        lock_nfrozen_list = []

        for layer in range(1, num_layers + 1):

            pruned_frozen = getattr(ctx.params, f"layer_{layer}_pruned_frozen")
            locked_frozen = getattr(ctx.params, f"layer_{layer}_locked_frozen")
            prune_nfrozen = getattr(ctx.params, f"layer_{layer}_prune_nfrozen")
            lock_nfrozen = getattr(ctx.params, f"layer_{layer}_lock_nfrozen")

            # Apply softmax to convert to probabilities for each layer
            exp_vals = np.exp([pruned_frozen, locked_frozen, prune_nfrozen, lock_nfrozen])
            softmax_vals = exp_vals / np.sum(exp_vals)

            # Constrained parameters for each layer
            _pruned_frozen = softmax_vals[0]  # 0..1
            _locked_frozen = softmax_vals[1]  # 0..1
            _prune_nfrozen = softmax_vals[2]  # 0..1
            _lock_nfrozen = softmax_vals[3]  # 0..1

            # Calculate derived parameters for each layer
            pruning_rate = _pruned_frozen + _prune_nfrozen  # sum of pruned parameters
            p_ratio = _pruned_frozen  # pruned_frozen component
            r_ratio = _locked_frozen  # locked_frozen component

            # Convert np.float64 to Python float
            pruning_rates.append(float(pruning_rate))
            p_ratios.append(float(p_ratio))
            r_ratios.append(float(r_ratio))

            # Store the constrained parameters for each layer
            pruned_frozen_list.append(float(_pruned_frozen))
            locked_frozen_list.append(float(_locked_frozen))
            prune_nfrozen_list.append(float(_prune_nfrozen))
            lock_nfrozen_list.append(float(_lock_nfrozen))

        # Set the derived parameters as lists
        ctx.params.pruning_rate = pruning_rates
        ctx.params.p_ratio = p_ratios
        ctx.params.r_ratio = r_ratios

        # Set the constrained parameters for each layer
        ctx.params.pruned_frozen = pruned_frozen_list
        ctx.params.locked_frozen = locked_frozen_list
        ctx.params.prune_nfrozen = prune_nfrozen_list
        ctx.params.lock_nfrozen = lock_nfrozen_list

        return {
            'pruning_rate': pruning_rates,
            'p_ratio': p_ratios,
            'r_ratio': r_ratios,
            '_pruned_frozen': pruned_frozen_list,
            '_locked_frozen': locked_frozen_list,
            '_prune_nfrozen': prune_nfrozen_list,
            '_lock_nfrozen': lock_nfrozen_list
        }
    else:
        # Default behavior for non-layer-wise
        # Get unconstrained parameters
        pruned_frozen = ctx.params.pruned_frozen
        locked_frozen = ctx.params.locked_frozen
        prune_nfrozen = ctx.params.prune_nfrozen
        lock_nfrozen = ctx.params.lock_nfrozen

        # Apply softmax to convert to probabilities
        exp_vals = np.exp([pruned_frozen, locked_frozen, prune_nfrozen, lock_nfrozen])
        softmax_vals = exp_vals / np.sum(exp_vals)

        # Constrained parameters, the sum is always 1
        _pruned_frozen = softmax_vals[0]  # 0..1
        _locked_frozen = softmax_vals[1]  # 0..1
        _prune_nfrozen = softmax_vals[2]  # 0..1
        _lock_nfrozen = softmax_vals[3]  # 0..1

        # Calculate derived parameters
        pruning_rate = _pruned_frozen + _prune_nfrozen  # sum of pruned parameters
        p_ratio = _pruned_frozen  # pruned_frozen component
        r_ratio = _locked_frozen  # locked_frozen component

        # Set the derived parameters
        ctx.params.pruning_rate = pruning_rate
        ctx.params.p_ratio = p_ratio
        ctx.params.r_ratio = r_ratio

        return {
            'pruning_rate': pruning_rate,
            'p_ratio': p_ratio,
            'r_ratio': r_ratio,
            '_pruned_frozen': _pruned_frozen,
            '_locked_frozen': _locked_frozen,
            '_prune_nfrozen': _prune_nfrozen,
            '_lock_nfrozen': _lock_nfrozen
        }


def select_random_parameters(ctx, config: Dict[str, Any]):
    """Select random values for parameters with type 'random'"""
    params_config = config.get('params', {})

    # Set random seed for reproducibility, but include iteration number for variety
    base_seed = ctx.seed if hasattr(ctx, 'seed') else 42
    iteration = ctx.bayes_opt.iteration if hasattr(ctx.bayes_opt, 'iteration') else 0
    random.seed(base_seed + iteration)

    if ctx.params.m_init_method == "layer_wise" and ctx.params.ep_algo == "local_ep":
        # Handle layer-wise parameters
        for layer in range(1, ctx.params.num_layers + 1):
            layer_key = f"layer_{layer}"
            for param_name in ['pruned_frozen', 'locked_frozen', 'prune_nfrozen', 'lock_nfrozen']:
                type_key = f"{param_name}_type"
                space_key = f"{param_name}_space"

                # Check if this layer parameter exists in config
                if type_key in params_config and params_config[type_key] == 'random':
                    if space_key in params_config:
                        space = params_config[space_key]
                        if isinstance(space, (list, tuple)) and len(space) > 0:
                            # Select random value from the space
                            random_value = random.choice(space)
                            setattr(ctx.params, f"{layer_key}_{param_name}", random_value)
                # Also check for layer-specific parameters (e.g., layer_1_pruned_frozen_type)
                layer_type_key = f"{layer_key}_{param_name}_type"
                layer_space_key = f"{layer_key}_{param_name}_space"
                if layer_type_key in params_config and params_config[layer_type_key] == 'random':
                    if layer_space_key in params_config:
                        space = params_config[layer_space_key]
                        if isinstance(space, (list, tuple)) and len(space) > 0:
                            # Select random value from the space
                            random_value = random.choice(space)
                            setattr(ctx.params, f"{layer_key}_{param_name}", random_value)
    else:
        # Select random values for parameters with type 'random' (non-layer-wise)
        for param_name in ['pruned_frozen', 'locked_frozen', 'prune_nfrozen', 'lock_nfrozen', 'scaling_rate', 'dropout_rate', 'mc_samples']:
            type_key = f"{param_name}_type"
            space_key = f"{param_name}_space"

            if type_key in params_config and params_config[type_key] == 'random':
                if space_key in params_config:
                    space = params_config[space_key]
                    if isinstance(space, (list, tuple)) and len(space) > 0:
                        # Select random value from the space
                        random_value = random.choice(space)
                        setattr(ctx.params, param_name, random_value)

    # Special handling for num_bayes_layers
    if 'num_bayes_layers_type' in params_config and params_config['num_bayes_layers_type'] == 'random':
        if 'num_bayes_layers_space' in params_config:
            space = params_config['num_bayes_layers_space']
            if isinstance(space, (list, tuple)) and len(space) > 0:
                # Select random integer value from the space
                random_value = random.choice(space)
                setattr(ctx.params, 'num_bayes_layers', random_value)

    # Special handling for mc_samples
    if 'mc_samples_type' in params_config and params_config['mc_samples_type'] == 'random':
        if 'mc_samples_space' in params_config:
            space = params_config['mc_samples_space']
            if isinstance(space, (list, tuple)) and len(space) > 0:
                # Select random integer value from the space
                random_value = random.choice(space)
                setattr(ctx.params, 'mc_samples', random_value)


def print_iteration_table_header(available_params):
    """Print the table header for iteration results"""
    # Create dynamic header based on available parameters
    header_parts = ["Iter"]
    for param in available_params:
        if param == 'pruning_rate':
            header_parts.append("pruning_rate".ljust(22))  # Adjust width
        elif param == 'p_ratio':
            header_parts.append("p_ratio".ljust(22))  # Adjust width
        elif param == 'r_ratio':
            header_parts.append("r_ratio".ljust(22))  # Adjust width
        elif param == 'scaling_rate':
            header_parts.append("scaling_rate".ljust(10))
        elif param == 'num_bayes_layers':
            header_parts.append("num_bayes_layers".ljust(16))
        elif param == 'dropout_rate':
            header_parts.append("dropout_rate".ljust(10))
        elif param == 'mc_samples':
            header_parts.append("mc_samples".ljust(10))

    header_parts.extend(["accuracy(%)".ljust(15), "ece(%)".ljust(13), "aPE".ljust(11), "flops".ljust(7), "memory(MB)".ljust(10), "score".ljust(4)])
    header = " | ".join(header_parts)
    separator = "-" * len(header)

    print(header)
    print(separator)
    return header, separator


def print_iteration_result(iteration, current_params, eval_results, score, available_params):
    """Print a single iteration result in table format"""
    print_parts = [f"{iteration:4d}"]
    for param in available_params:
        value = current_params[param]
        if value is None:
            print_parts.append(f"{'N/A':>11}")
        elif param == 'num_bayes_layers':
            print_parts.append(f"{value:16d}")
        elif param == 'mc_samples':
            print_parts.append(f"{value:10d}")
        elif isinstance(value, list):
            # Convert list to a string representation
            value_str = ", ".join(f"{v:.2f}" for v in value)
            print_parts.append(f"{value_str:>11}")
        else:
            print_parts.append(f"{value:11.2f}")

    # Handle None values in eval_results
    accuracy = eval_results.get('accuracy', 0.0) or 0.0
    accuracy_std = eval_results.get('accuracy_std', 0.0) or 0.0
    ece = eval_results.get('ece', 0.0) or 0.0
    ece_std = eval_results.get('ece_std', 0.0) or 0.0
    ape = eval_results.get('ape', 0.0) or 0.0
    ape_std = eval_results.get('ape_std', 0.0) or 0.0
    flops = eval_results.get('flops', 0.0) or 0.0
    memory = eval_results.get('memory', 0.0) or 0.0

    print_parts.extend([
        f"{accuracy:5.2f}±{accuracy_std:4.2f}",
        f"{ece:5.2f}±{ece_std:4.2f}",
        f"{ape:4.4f}±{ape_std:4.4f}",
        f"{flops:5.2f}",
        f"{memory:10.6f}",
        f"{score:5.4f}"
    ])

    print(" | ".join(print_parts))


def print_top_results_log_format(ctx, available_params, dataset):
    """Print top 3 results in the same format as the log"""
    if not hasattr(ctx, '_detailed_results') or not ctx._detailed_results:
        return

    results = ctx._detailed_results

    # If dataset is mnist, filter results to accuracy >= 95%
    if dataset == "mnist":
        results = [r for r in results if r['accuracy'] >= 95.0]
    elif dataset == "cora":
        results = [r for r in results if r['accuracy'] >= 75.0]

    # Sort by different metrics
    score_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    accuracy_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    ece_sorted = sorted(results, key=lambda x: x['ece'])  # Lower is better
    ape_sorted = sorted(results, key=lambda x: x['ape'], reverse=True)  # Higher is better

    # Create dynamic header based on available parameters
    header_parts = ["Iter"]
    for param in available_params:
        if param == 'pruning_rate':
            header_parts.append("pruning_rate")
        elif param == 'p_ratio':
            header_parts.append("p_ratio")
        elif param == 'r_ratio':
            header_parts.append("r_ratio")
        elif param == 'scaling_rate':
            header_parts.append("scaling_rate")
        elif param == 'num_bayes_layers':
            header_parts.append("num_bayes_layers")
        elif param == 'dropout_rate':
            header_parts.append("dropout_rate")
        elif param == 'mc_samples':
            header_parts.append("mc_samples")

    header_parts.extend(["accuracy(%)", "ece(%)", "aPE", "flops", "memory(MB)", "score"])
    header = " | ".join(header_parts)
    separator = "-" * len(header)

    print("\n" + "=" * 80)
    print("TOP 3 Balanced Results:")
    print("=" * 80)
    print(header)
    print(separator)
    for i, result in enumerate(score_sorted[:3], 1):
        print_parts = [f"{result['iteration']:4d}"]
        for param in available_params:
            value = result[param]
            if value is None:
                print_parts.append(f"{'N/A':>11}")
            elif param == 'num_bayes_layers':
                print_parts.append(f"{value:16d}")
            elif param == 'mc_samples':
                print_parts.append(f"{value:10d}")
            elif isinstance(value, list):
                # Convert list to a string representation
                value_str = ", ".join(f"{v:.2f}" for v in value)
                print_parts.append(f"{value_str:>11}")
            else:
                print_parts.append(f"{value:11.2f}")

        # Handle None values in result
        accuracy = result.get('accuracy', 0.0) or 0.0
        ece = result.get('ece', 0.0) or 0.0
        ape = result.get('ape', 0.0) or 0.0
        flops = result.get('flops', 0.0) or 0.0
        memory = result.get('memory', 0.0) or 0.0
        score = result.get('score', 0.0) or 0.0

        print_parts.extend([
            f"{accuracy:10.2f}",
            f"{ece:6.2f}",
            f"{ape:4.4f}",
            f"{flops:5.2f}",
            f"{memory:10.6f}",
            f"{score:5.4f}"
        ])

        print(" | ".join(print_parts))

    print("\n" + "=" * 80)
    print("TOP 3 Accuracy Results:")
    print("=" * 80)
    print(header)
    print(separator)
    for i, result in enumerate(accuracy_sorted[:3], 1):
        print_parts = [f"{result['iteration']:4d}"]
        for param in available_params:
            value = result[param]
            if value is None:
                print_parts.append(f"{'N/A':>11}")
            elif param == 'num_bayes_layers':
                print_parts.append(f"{value:16d}")
            elif param == 'mc_samples':
                print_parts.append(f"{value:10d}")
            elif isinstance(value, list):
                # Convert list to a string representation
                value_str = ", ".join(f"{v:.2f}" for v in value)
                print_parts.append(f"{value_str:>11}")
            else:
                print_parts.append(f"{value:11.2f}")

        # Handle None values in result
        accuracy = result.get('accuracy', 0.0) or 0.0
        ece = result.get('ece', 0.0) or 0.0
        ape = result.get('ape', 0.0) or 0.0
        flops = result.get('flops', 0.0) or 0.0
        memory = result.get('memory', 0.0) or 0.0
        score = result.get('score', 0.0) or 0.0

        print_parts.extend([
            f"{accuracy:10.2f}",
            f"{ece:6.2f}",
            f"{ape:4.4f}",
            f"{flops:5.2f}",
            f"{memory:10.6f}",
            f"{score:5.4f}"
        ])

        print(" | ".join(print_parts))

    print("\n" + "=" * 80)
    print("TOP 3 ECE Results:")
    print("=" * 80)
    print(header)
    print(separator)
    for i, result in enumerate(ece_sorted[:3], 1):
        print_parts = [f"{result['iteration']:4d}"]
        for param in available_params:
            value = result[param]
            if value is None:
                print_parts.append(f"{'N/A':>11}")
            elif param == 'num_bayes_layers':
                print_parts.append(f"{value:16d}")
            elif param == 'mc_samples':
                print_parts.append(f"{value:10d}")
            elif isinstance(value, list):
                # Convert list to a string representation
                value_str = ", ".join(f"{v:.2f}" for v in value)
                print_parts.append(f"{value_str:>11}")
            else:
                print_parts.append(f"{value:11.2f}")

        # Handle None values in result
        accuracy = result.get('accuracy', 0.0) or 0.0
        ece = result.get('ece', 0.0) or 0.0
        ape = result.get('ape', 0.0) or 0.0
        flops = result.get('flops', 0.0) or 0.0
        memory = result.get('memory', 0.0) or 0.0
        score = result.get('score', 0.0) or 0.0

        print_parts.extend([
            f"{accuracy:10.2f}",
            f"{ece:6.2f}",
            f"{ape:4.4f}",
            f"{flops:5.2f}",
            f"{memory:10.6f}",
            f"{score:5.4f}"
        ])

        print(" | ".join(print_parts))

    print("\n" + "=" * 80)
    print("TOP 3 aPE Results:")
    print("=" * 80)
    print(header)
    print(separator)
    for i, result in enumerate(ape_sorted[:3], 1):
        print_parts = [f"{result['iteration']:4d}"]
        for param in available_params:
            value = result[param]
            if value is None:
                print_parts.append(f"{'N/A':>11}")
            elif param == 'num_bayes_layers':
                print_parts.append(f"{value:16d}")
            elif param == 'mc_samples':
                print_parts.append(f"{value:10d}")
            elif isinstance(value, list):
                # Convert list to a string representation
                value_str = ", ".join(f"{v:.2f}" for v in value)
                print_parts.append(f"{value_str:>11}")
            else:
                print_parts.append(f"{value:11.2f}")

        # Handle None values in result
        accuracy = result.get('accuracy', 0.0) or 0.0
        ece = result.get('ece', 0.0) or 0.0
        ape = result.get('ape', 0.0) or 0.0
        flops = result.get('flops', 0.0) or 0.0
        memory = result.get('memory', 0.0) or 0.0
        score = result.get('score', 0.0) or 0.0

        print_parts.extend([
            f"{accuracy:10.2f}",
            f"{ece:6.2f}",
            f"{ape:4.4f}",
            f"{flops:5.2f}",
            f"{memory:10.6f}",
            f"{score:5.4f}"
        ])

        print(" | ".join(print_parts))

    print("=" * 80)


def print_fixed_parameters(config: Dict[str, Any]):
    """Print all fixed parameters that are not being optimized by Bayesian optimization"""
    print("=" * 80)
    print("FIXED PARAMETERS (Not optimized by Bayesian optimization):")
    print("=" * 80)

    # Fixed parameters from config
    fixed_params = config.get('fixed_params', {})
    print("Fixed parameters from config:")
    for key, value in fixed_params.items():
        print(f"  {key}: {value}")

    # Default training parameters (not in Bayesian optimization)
    default_params = {
        'batch_size': 128,
        'epochs': 100,
        'optimizer_name': 'sgd',
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'filter_bias_and_bn': False,
        'scheduler_name': 'cosine_lr',
        'milestones': [50, 75],
        'gamma': 0.1,
        'min_lr': None,
        'warmup_lr_init': None,
        'warmup_t': None,
        'warmup_prefix': False,
        'init_mode_weight': 'signed_kaiming_constant',
        'init_mode_score': 'kaiming_normal',
        'init_scale_weight': 1.0,
        'init_scale_score': 1.0,
        'm_init_method': 'epl',
        'post_pruning_ratio': 0.0,
        'num_post_pruning_layers': 0,
        'num_workers': 4,
        'seed': 110,
        'no_cuda': False,
        'num_repeats': 1,
        'mc_samples': 10,
        'num_bins': 10,
        'hidden_channels': 256,
        'num_layers': 4,
        'slt': False,
        'partial_frozen_slt': False,
        'magnitude_pruning': False,
        'ep_algo': 'global_ep',
    }

    # Merge fixed_params with default_params (fixed_params override defaults)
    merged_params = default_params.copy()
    merged_params.update(fixed_params)

    print("\nFinal training parameters (config overrides defaults):")
    for key, value in merged_params.items():
        if key in fixed_params:
            print(f"  {key}: {value} (from config)")
        else:
            print(f"  {key}: {value} (default)")

    print("=" * 80)
