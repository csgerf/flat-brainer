import torch
import torch.nn as nn

__ALL__ = ["get_loss_fn"]


def get_loss_fn(loss_name: str, loss_params: dict) -> nn.Module:
    loss_fn = None
    if loss_name == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss(**loss_params)
    elif loss_name == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss(**loss_params)
    elif loss_name == "MSELoss":
        loss_fn = nn.MSELoss(**loss_params)
    else:
        raise NotImplementedError(f"Loss {loss_name} not implemented")
    return loss_fn
