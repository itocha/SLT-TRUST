import random

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

from nautic import taskx


class BayesOpt:
    @taskx
    def bayesian_opt(ctx):
        bo = ctx.bayes_opt
        log = ctx.log

        # Initialize iteration if not already set
        if not hasattr(bo, 'iteration') or bo.iteration is None:
            bo.iteration = 0

        if bo.engine is None:
            bo.summary = []

            pbounds = { }
            tune_vals = { }
            tune_space = { }
            tune_type = { }
            bo.control.params = { }
            for key in bo.tunable.model_fields:
                opt = getattr(bo.tunable, key)
                param_type = opt.type.get()

                if param_type == 'discrete':
                    # For discrete parameters, use list indices
                    pbounds[key] = (0, len(opt.space.get()) - 0.001)
                    tune_type[key] = 'discrete'
                elif param_type == 'continuous':
                    # For continuous parameters, use the space directly
                    space = opt.space.get()
                    if isinstance(space, (list, tuple)) and len(space) == 2:
                        # Convert list to tuple if necessary
                        pbounds[key] = tuple(space)
                        tune_type[key] = 'continuous'
                    else:
                        raise ValueError(f"Continuous parameter {key} must have space with 2 values (list or tuple), got: {space}")
                elif param_type == 'random':
                    # For random parameters, we don't need pbounds as they will be selected randomly
                    # We'll handle this in the parameter setting phase
                    tune_type[key] = 'random'
                else:
                    raise ValueError(f"Unsupported parameter type for {key}: {param_type}")

                tune_vals[key] = opt.value
                tune_space[key] = opt.space

            bo.control.params['values'] = tune_vals
            bo.control.params['space'] = tune_space
            bo.control.params['type'] = tune_type

            bo.control.metrics = {}
            metrics_values = {}
            for key in bo.metrics.model_fields:
                metrics_values[key] = getattr(bo.metrics, key)
            bo.control.metrics['values'] = metrics_values

            score_weights = { }
            for key in bo.score_weights.model_fields:
                score_weights[key] = getattr(bo.score_weights, key)
            bo.control.metrics['score_weights'] = score_weights

            # Check if any parameters are of type 'random'
            has_random_params = any(tune_type[key] == 'random' for key in tune_type)

            if has_random_params:
                # For random search, we don't need BayesianOptimization engine
                bo.engine = None
                bo.control.suggests = {}
            else:
                # For Bayesian optimization, create the engine
                bo.engine = BayesianOptimization(
                    f = None,
                    pbounds=pbounds,
                    random_state=bo.seed.get(),
                    allow_duplicate_points=False,
                    verbose=0
                )

                # Initial random points
                for _ in range(5):
                    bo.control.suggests = dict(zip(pbounds.keys(),
                                                   bo.engine._space.random_sample()))

            # For the first iteration, we don't have results yet, so score is None
            bo.score = None
        else:
            engine = bo.engine

            # Calculate score from the previous experiment results
            score = 0
            for key in bo.control.metrics['values']:
                metric_value = bo.control.metrics['values'][key].get()
                base_value = bo.control.metrics['score_weights'][key].base
                weight_value = bo.control.metrics['score_weights'][key].weight

                # Handle NaN or infinite values
                if np.isnan(metric_value) or np.isinf(metric_value):
                    print(f"WARNING: NaN or Inf detected in metric {key}, using default value 0.0")
                    metric_value = 0.0

                score += float(metric_value / base_value) * float(weight_value)

            # Handle NaN or infinite values in final score
            if np.isnan(score) or np.isinf(score):
                print(f"WARNING: NaN or Inf detected in final score, using default value 0.0")
                score = 0.0

            bo.score = score

            # Register the results with the engine (only for Bayesian optimization)
            if engine is not None:
                engine.register(params=bo.control.suggests,
                                target=bo.score)

                # Get next suggestion
                bo.control.suggests = bo.engine.suggest()
            else:
                # For random search, we don't need to register or suggest
                bo.control.suggests = {}

        # Set the parameters for other tasks
        if bo.control.suggests:  # Only process suggests if it's not empty
            for key, value in bo.control.suggests.items():
                param_type = bo.control.params['type'][key]
                if param_type == 'discrete':
                    idx = int(value)
                    metric_val = bo.control.params['space'][key].get()[idx]
                elif param_type == 'continuous':
                    metric_val = value
                elif param_type == 'random':
                    # For random parameters, select a random value from the space
                    space = bo.control.params['space'][key].get()
                    if isinstance(space, (list, tuple)) and len(space) > 0:
                        # Set random seed based on base seed and iteration for variety
                        base_seed = bo.seed.get()
                        iteration = bo.iteration
                        random.seed(base_seed + iteration)
                        metric_val = random.choice(space)
                    else:
                        raise ValueError(f"Random parameter {key} must have a non-empty list/tuple space")
                else:
                    raise ValueError(f"Unsupported parameter type for {key}: {param_type}")

                bo.control.params['values'][key].set(metric_val)

        # Handle random parameters separately (they are not in suggests)
        for key in bo.control.params['type']:
            if bo.control.params['type'][key] == 'random':
                space = bo.control.params['space'][key].get()
                if isinstance(space, (list, tuple)) and len(space) > 0:
                    # Set random seed based on base seed and iteration for variety
                    base_seed = bo.seed.get()
                    iteration = bo.iteration
                    random.seed(base_seed + iteration)
                    metric_val = random.choice(space)
                    bo.control.params['values'][key].set(metric_val)

        bo.terminate = not (bo.iteration < bo.num_iter)

        # Ensure iteration is incremented for all cases
        bo.iteration += 1
