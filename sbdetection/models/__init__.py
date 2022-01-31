from typing import Union

from .RuleBased import Baseline, baseline
from .MLBased import Model

def call_model(kind:str, path:str = None)->Union[Model, Baseline]:
    if kind=='baseline':
        return baseline
    elif kind == 'ML':
        if path is not None:
            model = Model(model_path=path)
        return model
    
        
