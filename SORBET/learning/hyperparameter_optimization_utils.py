from typing import List, Dict, Any, Union
from functools import partial

from optuna.trial import Trial

from .train import _training_hyperparamters_types 

def get_model_and_training_specifications(config, model_type, in_data_dimension, random_seed = None):
    """Converts a ray / optuna config directory into a structure accessible for training.
    """
    params = dict()
    for hparam_type in filter(lambda t: t in config, map(lambda t: t[0], _training_hyperparamters_types)):
        params[hparam_type] = config[hparam_type]
    
    model_specification = model_type.get_model_specification()
    
    model_structure = dict()
    model_structure[model_specification.model_input_specifier] = in_data_dimension
    model_structure[model_specification.random_seed_specifier] = random_seed

    for param_key in model_specification.model_serializable_parameters:
        model_structure[param_key] = _deserialize_layer_structure(param_key, config)

    for param_key, _, _ in filter(lambda k: k[0] in config, model_specification.model_non_serializable_parameters):
        model_structure[param_key] = config[param_key]
    
    return params, model_structure


def _parse_hyperparameter_type(trial: Trial, space: List[Any], hparam_name: str, hparam_type: Any, hparam_log_space: bool): 
    """Parses model specification and suggests the appropriate values for a given trial.
    
    If the size of the search space, <space>, is two, assumes that the values represent bounds on the range.
    Otherwise, treats <space> as a categorical list to suggest from.
    """
    # Implements assumption of categorical choice or range depending on the length of <space>
    _choice_or_range = lambda f: f(hparam_name, *space) if len(space) == 2 else trial.suggest_categorical(hparam_name, space)

    if hparam_type == int:
        _choice_or_range(trial.suggest_int)
    elif hparam_type == float:
        _choice_or_range(partial(trial.suggest_float, log = hparam_log_space))
    elif hparam_type == 'choice':
        trial.suggest_categorical(hparam_name, space)
    else:
        raise ValueError(f'Could not parse specification for {hparam_name}')


def _serialize_layer_structure(trial: Trial, search_param: str, n_layers: int, param_space: List[int], categorical: bool): 
    """Converts a search space over variable layer depth and width to a config for use with Optuna 
    """
    for idx in range(n_layers):
        if categorical:
            trial.suggest_categorical(f'{search_param}_{idx}', param_space)
        else:
            trial.suggest_int(f'{search_param}_{idx}', *param_space) 

def _deserialize_layer_structure(search_param: str, config: Dict[str, Union[float, int]]):
    """Converts serialized data from _serialize_layer_structure into a (correctly-ordered) list
    """
    unordered_param_list = list()
     
    for k, v in filter(lambda t: search_param in t[0], config.items()):
        if 'n_layers' in k: continue
        idx = int(k.split("_")[-1])
        unordered_param_list.append((idx, v))
    
    sorted_layers = sorted(unordered_param_list, key=lambda x: x[0])
    param_list = [val for _, val in sorted_layers]
    
    return param_list
