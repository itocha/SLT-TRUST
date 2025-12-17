import datetime
import os.path as osp
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

root_path = osp.dirname(osp.abspath(__file__))
sys.path.append(root_path)
from args import get_args

# Import functions from main.py
from main import train_iteration_cnn, train_iteration_gnn
from nautic import Context
from utils.bayes_utils import (
    calculate_constrained_parameters,
    load_bayes_config,
    print_fixed_parameters,
    print_iteration_result,
    print_iteration_table_header,
    print_top_results_log_format,
    select_random_parameters,
    setup_parameter_spaces,
)
from utils.datasets import (
    get_cifar10_loaders,
    get_cora_loaders,
    get_mnist_loaders,
    get_ogbn_arxiv_loaders,
)


def model_evaluation(ctx, params: Dict[str, Any], num_runs: int = 3) -> Dict[str, float]:
    """Actually train and evaluate model with given parameters (multiple runs for stability)"""
    # Get fixed parameters from context
    fixed_params = ctx.fixed_params if hasattr(ctx, 'fixed_params') else {}
    model_name = fixed_params.get('model', 'lenet')
    dataset = fixed_params.get('dataset', 'mnist')

    # Store results from multiple runs
    all_results = []

    # Create args object with the current parameters using overrides (only once)
    overrides = {
        'model': model_name,
        'dataset': dataset,
        'pruning_rate': params.get('pruning_rate', 0.5) if not isinstance(params.get('pruning_rate', 0.5), list) else params.get('pruning_rate', 0.5),
        'p_ratio': params.get('p_ratio', 0.25) if not isinstance(params.get('p_ratio', 0.25), list) else params.get('p_ratio', 0.25),
        'r_ratio': params.get('r_ratio', 0.25) if not isinstance(params.get('r_ratio', 0.25), list) else params.get('r_ratio', 0.25),
        'scaling_rate': params.get('scaling_rate', 1.0),
        'num_bayes_layers': int(params.get('num_bayes_layers', 2)),
        'dropout_rate': params.get('dropout_rate', 0.15),
        'slt': False,  # Enable SLT for Bayesian optimization
        'partial_frozen_slt': fixed_params.get('partial_frozen_slt', False),  # Get from fixed params
        'magnitude_pruning': fixed_params.get('magnitude_pruning', False),  # Get from fixed params
        'epochs': 100,  # Reduce epochs for faster optimization
        'num_repeats': 1,  # Single run per iteration
        'mc_samples': 5,
        'num_bins': 10,
        'num_layers': fixed_params.get('num_layers', 4),
        'seed': ctx.seed if hasattr(ctx, 'seed') else 42,  # Base seed
        # Add required attributes for training (using defaults from args.py)
        'optimizer_name': 'adamw',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'momentum': 0.9,
        'scheduler_name': 'cosine_lr',
        'milestones': [50, 75],
        'gamma': 0.1,
        'min_lr': None,
        'warmup_lr_init': None,
        'warmup_t': None,
        'warmup_prefix': False,
        'filter_bias_and_bn': False,
        'init_mode_weight': 'signed_kaiming_constant',
        'init_mode_score': 'kaiming_normal',
        'init_scale_weight': 1.0,
        'init_scale_score': 1.0,
        'm_init_method': 'epl',
        'ep_algo': 'global_ep',
        'post_pruning_ratio': 0.0,
        'num_post_pruning_layers': 0,
        'no_cuda': False,
        'amp': False,
        'hidden_channels': 256,
        'gat_heads': 8,
        'gat_out_heads': 1,
        'gat_concat': True,
        'gat_negative_slope': 0.2,
        'attn_dropout': 0.0
    }

    # Override with fixed parameters from config
    overrides.update(fixed_params)

    # Create args object manually instead of using get_args() to avoid argparse conflicts
    class Args:
        def __init__(self, **kwargs):
            # Set default values for required attributes (from args.py defaults)
            self.cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if self.cuda else "cpu")
            self.num_workers = 4
            self.pin_memory = True
            self.persistent_workers = False
            self.prefetch_factor = 2

            # Set default values from args.py
            self.model = 'lenet'
            self.dataset = 'mnist'
            self.batch_size = 128
            self.epochs = 100
            self.scaling_rate = 1.0
            self.optimizer_name = 'sgd'
            self.lr = 0.1
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.filter_bias_and_bn = False
            self.scheduler_name = 'cosine_lr'
            self.milestones = [50, 75]
            self.gamma = 0.1
            self.min_lr = None
            self.warmup_lr_init = None
            self.warmup_t = None
            self.warmup_prefix = False
            self.slt = False
            self.partial_frozen_slt = False
            self.pruning_rate = [0.0]
            self.p_ratio = [0.0]
            self.r_ratio = [0.0]
            self.init_mode_weight = 'signed_kaiming_constant'
            self.init_mode_score = 'kaiming_normal'
            self.init_scale_weight = 1.0
            self.init_scale_score = 1.0
            self.m_init_method = 'epl'
            self.ep_algo = 'global_ep'
            self.magnitude_pruning = False
            self.post_pruning_ratio = 0.0
            self.num_post_pruning_layers = 0
            self.seed = 42
            self.no_cuda = False
            self.num_repeats = 1
            self.amp = False
            self.mc_samples = 5
            self.num_bins = 10
            self.num_bayes_layers = 1
            self.dropout_rate = 0.05
            self.hidden_channels = 256
            self.num_layers = 4
            self.gat_heads = 8
            self.gat_out_heads = 1
            self.gat_concat = True
            self.gat_negative_slope = 0.2
            self.attn_dropout = 0.0

            # Set attributes from kwargs (overrides defaults)
            for key, value in kwargs.items():
                setattr(self, key, value)

    args = Args(**overrides)
    # print(f"Using parameters: {overrides}")

    # Set device
    device = torch.device("cuda" if args.cuda else "cpu")

    # Load data based on dataset (only once, not per run)
    if dataset == "mnist":
        train_loader, val_loader, test_loader = get_mnist_loaders(args)
    elif dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(args)
    elif dataset == "cora":
        data, train_mask, val_mask, test_mask = get_cora_loaders(args)
    elif dataset == "ogbn-arxiv":
        data, train_mask, val_mask, test_mask = get_ogbn_arxiv_loaders(args)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for run_idx in range(num_runs):
        # Update seed for each run to ensure different results
        args.seed = ctx.seed if hasattr(ctx, 'seed') else 42 + run_idx

        # print(f"Using parameters: {args}")

        # Train and evaluate model
        try:
            if dataset in ["mnist", "cifar10"]:
                test_acc, ece, ape, flops, memory = train_iteration_cnn(
                    args, device, train_loader, val_loader, test_loader,
                    ctx.bayes_opt.iteration, verbose=False
                )
            elif dataset in ["cora", "ogbn-arxiv"]:
                test_acc, ece, ape, flops, memory = train_iteration_gnn(
                    args, device, data, train_mask, val_mask, test_mask,
                    ctx.bayes_opt.iteration, verbose=False
                )

            run_result = {
                'accuracy': test_acc,
                'ece': ece,
                'ape': ape,
                'flops': flops,
                'memory': memory
            }
            all_results.append(run_result)

        except Exception as e:
            print(f"Error during model training (run {run_idx + 1}): {e}")
            # Return fallback values if training fails
            run_result = {
                'accuracy': 0.0,
                'ece': 0.0,
                'ape': 0.0,
                'flops': 0.0,
                'memory': 0.0
            }
            all_results.append(run_result)

    # Calculate average results across all runs
    if not all_results:
        return {
            'accuracy': 0.0,
            'ece': 0.0,
            'ape': 0.0,
            'flops': 0.0,
            'memory': 0.0
        }

    # Calculate means and standard deviations
    metrics = ['accuracy', 'ece', 'ape', 'flops', 'memory']
    std_metrics = ['accuracy', 'ece', 'ape']  # Only these metrics need std
    avg_results = {}

    for metric in metrics:
        values = [result[metric] for result in all_results]
        mean_val = np.mean(values)
        avg_results[metric] = round(mean_val, 4)

        # Only calculate std for specific metrics
        if metric in std_metrics:
            std_val = np.std(values)
            avg_results[f'{metric}_std'] = round(std_val, 4)

    return avg_results


def run_single_optimization(config_path: str, optimization_run: int = 3) -> Dict[str, Any]:
    """Run a single Bayesian optimization and return results"""
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RUN {optimization_run}")
    print(f"{'='*80}")

    # Load bayes configuration from specified file
    config = load_bayes_config(config_path)

    # # Print fixed parameters first
    # print_fixed_parameters(config)

    # Create context using the same config file
    ctx = Context.create(config_path,
                         path=osp.join(root_path, 'nautic', 'tasks'))

    # Ensure fields from fixed_params are added to ctx.params as BaseModel fields
    for key, value in config.get('fixed_params', {}).items():
        if not hasattr(ctx.params, key):
            # Add as BaseModel field
            ctx.params.__fields__[key] = (type(value), value)
            setattr(ctx.params, key, value)

    # Setup parameter spaces
    setup_parameter_spaces(ctx, config)

    # Set seed (different for each optimization run)
    base_seed = config.get('seed', 42)
    ctx.seed = base_seed + optimization_run * 1000  # Different seed for each run
    ctx.bayes_opt.num_iter = config.get('bayes_opt', {}).get('num_iter', 100)

    # Get fixed parameters and store in context
    fixed_params = config.get('fixed_params', {})
    ctx.fixed_params = fixed_params
    model = fixed_params.get('model', 'unknown')
    dataset = fixed_params.get('dataset', 'unknown')

    # Initialize detailed results storage
    ctx._detailed_results = []
    available_params = []

    # Get available parameters from config (including derived parameters)
    for param_name in ['pruning_rate', 'p_ratio', 'r_ratio', 'scaling_rate', 'num_bayes_layers', 'dropout_rate', 'mc_samples']:
        if param_name in config.get('params', {}):
            available_params.append(param_name)

    engine = ctx.engine

    # Check if any parameters are of type 'random'
    params_config = config.get('params', {})
    has_random_params = any(
        params_config.get(f"{param_name}_type") == 'random'
        for param_name in ['pruned_frozen', 'locked_frozen', 'prune_nfrozen', 'lock_nfrozen', 'scaling_rate', 'dropout_rate', 'num_bayes_layers', 'mc_samples']
    )

    print("SCORE CALCULATION FORMULA:")
    score_weights = config.get('bayes_opt', {}).get('score_weights', {})
    formula_parts = []
    for metric, weight_config in score_weights.items():
        weight = weight_config.get('weight', 0)
        base = weight_config.get('base', 1.0)
        # ece, flops, memory are better when lower, so use -{metric}
        formula_parts.append(f"{weight:.2f} * ({metric} / {base:.4f})")
    print(f"Score = " + " + ".join(formula_parts))
    print("=" * 80)
    # print(f"Model: {model}, Dataset: {dataset}")

    # Print table header
    print_iteration_table_header(available_params)

    # Initialize iteration
    ctx.bayes_opt.iteration = 0

    while True:
        engine.dse.bayesian_opt()
        if ctx.bayes_opt.terminate:
            break

        prev_score = ctx.bayes_opt.score
        if prev_score is not None:
            prev_score = round(prev_score, 4)

        # Select random parameters if any parameters are of type 'random'
        if has_random_params:
            select_random_parameters(ctx, config)

        # Determine if this is an SLT or IMP file based on fixed parameters
        is_slt = fixed_params.get('partial_frozen_slt', False)
        is_imp = fixed_params.get('magnitude_pruning', False)

        # Calculate constrained parameters from unconstrained parameters (only for SLT)
        if is_slt:
            constrained_params = calculate_constrained_parameters(ctx)
        else:
            constrained_params = {}

        # Get current parameters
        current_params = {}
        for param in available_params:
            if param in ['pruning_rate', 'p_ratio', 'r_ratio']:
                if is_slt:
                    # Use calculated constrained parameters for SLT
                    current_params[param] = constrained_params[param]
                else:
                    # Use direct parameter values for IMP
                    value = getattr(ctx.params, param)
                    current_params[param] = value
            else:
                value = getattr(ctx.params, param)
                if param == 'num_bayes_layers':
                    # Ensure integer for num_bayes_layers
                    if value is None:
                        current_params[param] = 2  # Default value
                        print(f"WARNING: {param} was None, setting to default value 2")
                    else:
                        current_params[param] = int(round(value))
                elif param == 'mc_samples':
                    # Ensure integer for mc_samples
                    if value is None:
                        current_params[param] = 5  # Default value
                        print(f"WARNING: {param} was None, setting to default value 10")
                    else:
                        current_params[param] = int(round(value))
                else:
                    if value is None:
                        current_params[param] = 0.0  # Default value
                        print(f"WARNING: {param} was None, setting to default value 0.0")
                    else:
                        current_params[param] = value

        # Actually train and evaluate model
        eval_results = model_evaluation(ctx, current_params)

        # Set evaluation results (only set fields that exist in ContextEval)
        for key, value in eval_results.items():
            if hasattr(ctx.eval, key):
                setattr(ctx.eval, key, value)

        # Calculate score for current parameters
        score_weights = config.get('bayes_opt', {}).get('score_weights', {})
        score = 0.0
        for metric, weight_config in score_weights.items():
            weight = weight_config.get('weight', 0)
            base = weight_config.get('base', 1.0)
            metric_value = eval_results.get(metric, 0.0) or 0.0

            # Handle NaN or infinite values in metric_value
            if np.isnan(metric_value) or np.isinf(metric_value):
                print(f"WARNING: NaN or Inf detected in {metric}, using default value 0.0")
                metric_value = 0.0

            score += weight * (metric_value / base)

        # Handle NaN or infinite values in final score
        if np.isnan(score) or np.isinf(score):
            print(f"WARNING: NaN or Inf detected in final score, using default value 0.0")
            score = 0.0

        ctx.bayes_opt.score = score

        # Print iteration result in table format
        print_iteration_result(ctx.bayes_opt.iteration, current_params, eval_results, score, available_params)

        # Store detailed results
        result_entry = {
            'iteration': ctx.bayes_opt.iteration,
            'score': score,
            **current_params,
            **eval_results
        }
        ctx._detailed_results.append(result_entry)

    # Return results for this optimization run
    return {
        'detailed_results': ctx._detailed_results,
        'available_params': available_params,
        'dataset': dataset,
        'model': model,
        'has_random_params': has_random_params,
        'config': config
    }


def main():
    start_time = time.time()
    start_datetime = datetime.datetime.now()

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Bayesian Optimization for SLT')
    parser.add_argument('config_path', help='Path to the YAML configuration file')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of optimization runs')
    args = parser.parse_args()

    config_path = args.config_path

    # Number of optimization runs
    num_optimization_runs = 3
    all_optimization_results = []

    # Run multiple optimization runs
    for opt_run in range(1, num_optimization_runs + 1):
        opt_results = run_single_optimization(config_path, opt_run)
        all_optimization_results.append(opt_results)

        # Print summary for this run
        print(f"\nOptimization Run {opt_run} Summary:")
        print(f"Best score: {max([r['score'] for r in opt_results['detailed_results']]):.4f}")
        print(f"Number of iterations: {len(opt_results['detailed_results'])}")

    # Calculate average results across all optimization runs
    print(f"\n{'='*80}")
    print("FINAL AVERAGED RESULTS ACROSS ALL OPTIMIZATION RUNS")
    print(f"{'='*80}")

    # Find best results from each run
    best_results = []
    for opt_run, opt_results in enumerate(all_optimization_results):
        best_result = max(opt_results['detailed_results'], key=lambda x: x['score'])
        best_results.append(best_result)
        print(f"Run {opt_run + 1} best: score={best_result['score']:.4f}, accuracy={best_result['accuracy']:.4f}")

    # Calculate statistics across all runs
    if best_results:
        scores = [r['score'] for r in best_results]
        accuracies = [r['accuracy'] for r in best_results]

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"\nAverage best score: {avg_score:.4f} ± {std_score:.4f}")
        print(f"Average best accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

        # Find overall best result
        overall_best = max(best_results, key=lambda x: x['score'])
        print(f"\nOverall best result:")
        print(f"  Score: {overall_best['score']:.4f}")
        print(f"  Accuracy: {overall_best['accuracy']:.4f}")
        print(f"  Parameters: {overall_best}")

    # Print final summary
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    print(f"Average time per optimization run: {execution_time/num_optimization_runs:.2f} seconds")


if __name__ == "__main__":
    main()
