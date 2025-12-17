import argparse

import torch


def get_args(overrides: dict = None):
    parser = argparse.ArgumentParser(description='LeNet with SLT training')

    # Model parameters
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'resnet18', 'gcn', 'gat'],
                        help='model architecture (default: lenet)')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cora', 'ogbn-arxiv'],
                        help='dataset to use (default: mnist)')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--scaling_rate', type=float, default=1.0,
                        help='scale factor for model size (default: 1.0)')

    # Optimizer parameters
    parser.add_argument('--optimizer_name', type=str, default='adamw',
                        choices=['sgd', 'adamw'],
                        help='optimizer (default: adamw)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for SGD optimizer (default: 1e-4)')
    parser.add_argument('--filter_bias_and_bn', action='store_true', default=False,
                        help='filter bias and batch normalization parameters (default: False)')

    # Scheduler parameters
    parser.add_argument('--scheduler_name', type=str, default='cosine_lr',
                        choices=['cosine_lr', 'cosine_lr_warmup', 'multi_step_lr'],
                        help='scheduler (default: cosine_lr)')
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75],
                        help='milestones for scheduler (default: [50, 75])')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma for scheduler (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='minimum learning rate (default: None)')
    parser.add_argument('--warmup_lr_init', type=float, default=None,
                        help='warmup learning rate (default: None)')
    parser.add_argument('--warmup_t', type=int, default=None,
                        help='warmup epochs (default: None)')
    parser.add_argument('--warmup_prefix', action='store_true', default=False,
                        help='warmup prefix (default: False)')

    # SLT parameters
    parser.add_argument('--slt', action='store_true',
                        help='use SLT (Sparse Learning)')
    parser.add_argument('--partial_frozen_slt', action='store_true',
                        help='use Partial Frozen SLT')
    parser.add_argument('--pruning_rate', type=float, nargs='+', default=[0.0],
                        help='sparsity for linear layers (default: [0.0])')
    parser.add_argument('--p_ratio', type=float, nargs='+', default=[0.0],
                        help='frozen pruned ratio (default: [0.0])')
    parser.add_argument('--r_ratio', type=float, nargs='+', default=[0.0],
                        help='frozen locked ratio (default: [0.0])')
    parser.add_argument('--init_mode_weight', type=str, default='signed_kaiming_constant',
                        help='weight initialization mode (default: signed_kaiming_constant)')
    parser.add_argument('--init_mode_score', type=str, default='kaiming_normal',
                        help='score initialization mode (default: kaiming_normal)')
    parser.add_argument('--init_scale_weight', type=float, default=1.0,
                        help='weight initialization scale (default: 1.0)')
    parser.add_argument('--init_scale_score', type=float, default=1.0,
                        help='score initialization scale (default: 1.0)')
    parser.add_argument('--m_init_method', type=str, default='epl',
                        help='m_init_method (default: epl)')
    parser.add_argument('--ep_algo', type=str, default='global_ep',
                        help='ep_algo (default: global_ep)')
    parser.add_argument('--magnitude_pruning', action='store_true',
                        help='use magnitude-based pruning instead of SLT')

    # Post pruning parameters
    parser.add_argument('--post_pruning_ratio', type=float, default=0.0,
                        help='post pruning ratio (default: 0.0)')
    parser.add_argument('--num_post_pruning_layers', type=int, default=0,
                        help='number of post pruning layers (default: 0)')

    # Miscellaneous parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--num_repeats', type=int, default=1,
                        help='number of repeats (default: 1)')
    parser.add_argument('--amp', action='store_true',
                        help='use automatic mixed precision training (default: False)')

    # Bayesian optimization parameters
    parser.add_argument('--mc_samples', type=int, default=5,
                        help='number of MC samples for inference (default: 5)')
    parser.add_argument('--num_bins', type=int, default=10,
                        help='number of bins for ECE calculation (default: 10)')
    parser.add_argument('--num_bayes_layers', type=int, default=1,
                        help='number of Bayesian layers (default: 1)')
    parser.add_argument('--dropout_rate', type=float, default=0.05,
                        help='dropout rate for MC dropout (default: 0.05)')

    # GNN parameters
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='number of hidden channels for GNN (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of GNN layers (default: 4)')

    # GAT parameters
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='number of attention heads for GAT (default: 8)')
    parser.add_argument('--gat_out_heads', type=int, default=1,
                        help='number of output attention heads for GAT (default: 1)')
    parser.add_argument('--gat_concat', action='store_true', default=True,
                        help='concatenate attention heads in GAT (default: True)')
    parser.add_argument('--gat_negative_slope', type=float, default=0.2,
                        help='negative slope for LeakyReLU in GAT (default: 0.2)')
    parser.add_argument('--attn_dropout', type=float, default=0.0,
                        help='attention dropout rate for GAT (default: 0.0)')

    if overrides:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    args.cuda = not getattr(args, 'no_cuda', False) and torch.cuda.is_available()

    if overrides:
        for k, v in overrides.items():
            # Get the argument type from parser
            for action in parser._actions:
                if action.dest == k:
                    if hasattr(action, 'type') and action.type is not None:
                        # Convert to the appropriate type
                        try:
                            if action.type == int:
                                v = int(v)
                            elif action.type == float:
                                v = float(v)
                            elif action.type == bool:
                                v = bool(v)
                        except (ValueError, TypeError):
                            # If conversion fails, keep original value
                            pass
                    break
            setattr(args, k, v)

    return args
