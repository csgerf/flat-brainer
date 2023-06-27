from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from ._params import scheduler_dict


params_t = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


def clean_scheduler_params(name, scheduler_params) -> dict:
    scheduler_key_list = scheduler_dict[name]
    return {k: v for k, v in scheduler_params.items() if k in scheduler_key_list}


def get_optimizer(parameters: params_t, optimizer_name: str, optimizer_params) -> Optimizer:
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(parameters, **optimizer_params)
    return optimizer


def get_scheduler(optimizer: Optimizer, scheduler_name: str, scheduler_params) -> LRScheduler:
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
    scheduler_params = clean_scheduler_params(scheduler_name, scheduler_params)
    scheduler = scheduler_class(optimizer, **scheduler_params)
    return scheduler


# def get_model(model_name: str, model_params: dict) -> nn.Module:
#     # model_class = getattr(torch.nn, model_name)
#     print("model name: ", model_name, model_params)
#     model = TinyModel(**model_params)
#     return model

# def get_loss_function(loss_function_name: str, loss_function_params: dict) -> nn.Module:
#     loss_function_class = getattr(torch.nn, loss_function_name)
#     loss_function = loss_function_class(**loss_function_params)
#     return loss_function


