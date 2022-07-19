from abc import ABCMeta, abstractmethod
from typing import Dict, Any

from torch import nn

_REGISTER: Dict = {}


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        del kwargs
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, img):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @classmethod
    def register(cls, model_name: str):
        def add_model_class(model_class: Any):
            if model_name not in _REGISTER:
                _REGISTER[model_name] = model_class
            return model_class

        return add_model_class

    @classmethod
    def get_model(cls, model_name: str, args):
        if model_name not in _REGISTER:
            raise KeyError(f"{model_name} is not registered in the register.")
        return _REGISTER[model_name](**args)
