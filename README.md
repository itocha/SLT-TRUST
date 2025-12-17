## Project Overview

This repository contains code for the paper **"Memory-Efficient and Trustworthy Neural
Networks via Random Seed-Based Design"** submitted to IEEE Access.


There are codes on image classification (MNIST / CIFAR10 with LeNet / ResNet18) and graph neural networks (Cora / OGBN-Arxiv with GCN / GAT), together with their **Bayesian optimization**.

- **`main.py`**: Standard training and evaluation (single configuration, possibly repeated runs)
- **`main_bayes.py`**: Runs Bayesian optimization over hyperparameters
- **`models.py`**: Definitions of LeNet / ResNet18 / GCN / GAT and SLT/IMP-aware layers
- **`args.py`**: Command-line arguments (training hyperparameters)
- **`bayes_yaml/`**: Configuration files for Bayesian optimization

## Environment Setup
 ```bash
 conda env create -f requirements.yml
 conda activate slt_trust
 ```

## Training Script Usage (`main.py`)

### Basic Usage

```bash
python main.py --dataset <dataset> --model <model> [other arguments]
```

`main.py` performs:

- Dataset loading (`utils/datasets.py`)
- Model construction (`models.py`)
- Optimizer / scheduler setup (`utils/optimizer.py`, `utils/scheduler.py`)
- SLT / IMP initialization and updates (`utils/slt_modules.py`, `utils/magnitude_pruning.py`)
- Train / validation / test loop
- Computation of ECE, aPE, FLOPs, memory footprint (`utils/metrics.py`, `utils/flops_calc.py`, etc.)

For each repeat, it prints the final metrics, and at the end it prints mean ± std across repeats.


### Example: MNIST + LeNet + SLT

```bash
python main.py \
  --dataset mnist \
  --model lenet \
  --batch_size 256 \
  --partial_frozen_slt \
  --pruning_rate 0.6 \
  --p_ratio 0.0 \
  --r_ratio 0.0 \
  --scaling_rate 1.0 \
  --dropout_rate 0.05 \
  --num_bayes_layers 1 \
  --num_repeats 1 \
  --optimizer_name adamw \
  --epochs 3
```

### Example: Cora + GCN + SLT

```bash
python main.py \
  --dataset cora \
  --model gcn \
  --num_layers 4 \
  --partial_frozen_slt \
  --pruning_rate 0.6 \
  --p_ratio 0.25 \
  --r_ratio 0.25 \
  --scaling_rate 1.0 \
  --dropout_rate 0.05 \
  --ep_algo global_ep \
  --num_bayes_layers 1 \
  --optimizer_name adamw \
  --lr 0.01 \
  --weight_decay 5e-4 \
  --epochs 200 \
  --num_repeats 5 \
  --mc_samples 10
```

## Bayesian Optimization (`main_bayes.py` + `bayes_yaml/`)

### How to Run

```bash
python main_bayes.py bayes_yaml/<path/to/config>.yml
```

- `config_path`: YAML under `bayes_yaml/continuous/` or `bayes_yaml/discrete/`

`main_bayes.py` workflow:

1. Load the YAML (`load_bayes_config`) and register **fixed parameters (`fixed_params`)** and
   **search parameters (`params`)** into a `Context`.
2. Use the `nautic` engine to perform Bayesian optimization.
3. For each iteration:
   - Call `model_evaluation`, which internally calls `train_iteration_cnn` / `train_iteration_gnn`
     with the current hyperparameters.
   - Obtain `accuracy`, `ece`, `ape`, `flops`, `memory`.
   - Compute a scalar **score** from these metrics using `score_weights` defined in the YAML.
   - Log and store the result in a table-like format.
4. After each optimization run, collect the best score and print summary statistics.


### Example: Cora + GCN + SLT (discrete search)

```bash
python main_bayes.py bayes_yaml/discrete/gcn_cora_discrete_slt_global_epl.yml
```

### Example: MNIST + LeNet + SLT (continuous layer-wise)

```bash
python main_bayes.py bayes_yaml/continuous/lenet_mnist_continuous_slt_global_layerwise.yml
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
