from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass 
class StructureSpecification:
    model_serializable_parameters: List[List[Union[bool, str, int]]]
    model_non_serializable_parameters: List[List[Union[bool, str, int]]]
    model_input_specifier: str = "in_channel"
    random_seed_specifier: str = "random_seed" 
