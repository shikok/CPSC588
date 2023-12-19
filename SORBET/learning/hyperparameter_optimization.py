import os
from typing import List, Dict, Union, Any, Optional
from dataclasses import dataclass
from functools import partial
from tqdm import tqdm as tqdm

import numpy as np
from sklearn.metrics import roc_auc_score

import optuna
import ray
from ray import tune
from ray.air import ScalingConfig, RunConfig
from ray.tune.search.optuna import OptunaSearch

from optuna.samplers import RandomSampler, TPESampler, CmaEsSampler 

from .dataset import TorchOmicsDataset
from .train import train_model, _training_hyperparamters_types 
from .hyperparameter_optimization_utils import _serialize_layer_structure, _deserialize_layer_structure, _parse_hyperparameter_type
from .hyperparameter_optimization_utils import get_model_and_training_specifications 

# TODO: more hyperparameter to consider: percentile,normalization, morphological features
# TODO: ModelParamters structure is brittle. Improve. 

def objective(config, data_split, root_fpath, metadata_files, model_type, in_data_dimension, random_seed, tensorboard):
    train_params, model_structure = get_model_and_training_specifications(config, model_type, in_data_dimension, random_seed) 

    labs, preds = list(), list()
    # The third argument is the test set -- ignored.
    for train_ids, val_ids, _ in data_split:
        train_ds = TorchOmicsDataset(root_fpath, metadata_files, train_ids)
        test_ds = TorchOmicsDataset(root_fpath, metadata_files, val_ids)
        print(f"Training on {len(train_ds)} samples, validating on {len(test_ds)} samples.")
        print(root_fpath)
        if len(train_ds) == 0 or len(test_ds) == 0:
            raise ValueError(f"Empty dataset. Check that the dataset is not empty. check paths: {metadata_files}")
        _, (_preds, _labs), acc, tl, vl = train_model(model_type, model_structure, [train_ds, test_ds], print_lvl=0, tensorboard_dir=tensorboard, **train_params)
        preds.extend(_preds)
        labs.extend(_labs)
    
    auroc = roc_auc_score(labs, preds)
    tune.report(auroc=auroc) 

def hyperparameter_optimization(data_split, root_fpath, metadata_files, input_data_dimension,  
        model_type, model_hyperparameters, set_model: bool = False, 
        tensorboard_dir: str = None, ray_run_config_kwargs: dict = {'verbose': 1},
        num_model_evals: int = 100, n_cpus: int = 12, n_gpus: int = 6, allow_fractional: float = 0, random_seed: Optional[int] = None):
    
    # Identify available resources and infer concurrent jobs:
    # NOTE: Assumes that n. GPUs <= n. CPUs. This seems like the right assumption generally, but may need to be addressed in future iterations. 
    if allow_fractional != 0 and allow_fractional < 1:
        n_concurrent = int(n_gpus // allow_fractional)
        per_trial = {'gpu': n_gpus / n_concurrent, 'cpu': n_cpus / n_concurrent}
    else:
        n_concurrent = min(n_gpus, n_cpus)
        per_trial = {'gpu': n_gpus // n_concurrent, 'cpu': n_cpus // n_concurrent}
    
    # Initialize Ray instance:
    ray.init()

    # Set-up hyperparamter search space functions:
    model_specification = model_type.get_model_specification()
    hparam_suggestion_fn = partial(_define_by_run_func, model_hyperparameters=model_hyperparameters, model_specification=model_specification, pre_defined_model = set_model) 
    search = OptunaSearch(
            space = hparam_suggestion_fn,
            metric = "auroc",
            mode = "max",
    )

    # Run tuning.
    tuner = tune.Tuner(
            trainable = tune.with_resources(
                tune.with_parameters(
                    objective,
                    data_split = data_split,
                    root_fpath = root_fpath,
                    metadata_files = metadata_files,
                    model_type = model_type,
                    in_data_dimension = input_data_dimension,
                    random_seed = random_seed,
                    tensorboard = tensorboard_dir
                    ),
                resources = per_trial, 
            ),
            tune_config = tune.TuneConfig(
                search_alg = search,
                num_samples = num_model_evals,
                max_concurrent_trials = n_concurrent 
            ),
            run_config = RunConfig(**ray_run_config_kwargs)
    )

    results = tuner.fit()
    
    # Shutdown Ray instance:
    ray.shutdown()

    return results

def _define_by_run_func(trial, model_hyperparameters, model_specification: Dict[str, Union[Any, List[Any]]], pre_defined_model: bool = False):
    # Following example defined in: https://docs.ray.io/en/latest/tune/examples/optuna_example.html
    config_constants = dict()
   
    # Parse model layers (potentially variable number of layers):
    for sparam in model_specification.model_serializable_parameters:
        if pre_defined_model:
            layers = model_hyperparameters[sparam]
            config_constants[f'{sparam}_n_layers'] = len(layers)
            for idx, size in enumerate(layers):
                config_constants[f'{sparam}_{idx}'] = size
        else:
            _spec = model_hyperparameters[sparam]
            if len(_spec) == 3:
                n_low, n_high, _pspace = _spec
                cat = True
            else:
                n_low, n_high, min_size, max_size = _spec
                _pspace = [min_size, max_size]
                cat = False
                
            n_layers = trial.suggest_int(f'{sparam}_n_layers', n_low, n_high) 
            _serialize_layer_structure(trial, sparam, n_layers, _pspace, cat)  
    
    # Defines over training hyperparameters (_training_hyperparamters_types) and model specification hyperparameters.
    parameter_definitions = [*_training_hyperparamters_types, *model_specification.model_non_serializable_parameters] 
    for param_name, param_type, param_log_space in filter(lambda t: t[0] in model_hyperparameters, parameter_definitions):
        param_space = model_hyperparameters[param_name]
        if len(param_space) == 1: # Set as constant if length of space is 1
            config_constants[param_name] = param_space[0]
        else:
            _parse_hyperparameter_type(trial, param_space, param_name, param_type, param_log_space)

    if len(config_constants) > 0:
        return config_constants
