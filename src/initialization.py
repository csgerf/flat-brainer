import os
from omegaconf import DictConfig, OmegaConf

from src.core.forge import component_factory as factory
from src.models import get_model as get_model_from_config
from src.core.forge.losses import get_loss_fn
from src.data.datasets.factory import get_dataloader as get_dataloader_from_config

from src.core.utils.config import ex


@ex.capture
def get_model(model_config):
    print("get_model")
    model = get_model_from_config(model_config)
    return model


@ex.capture
def get_optimizer(config, model):
    optimizer = factory.get_optimizer(model.parameters(), config.name, config.params)
    optimizer_params = optimizer.defaults
    print(optimizer_params)
    return optimizer


@ex.capture
def get_scheduler(config, optimizer):
    scheduler = factory.get_scheduler(optimizer, config.name, config.params)
    return scheduler


@ex.capture
def get_dataloader(config):
    data_loaders = get_dataloader_from_config(config)
    return data_loaders


@ex.capture
def get_criterion(config):
    criterion = get_loss_fn(config.name, config.params)
    return criterion


def build_trainer_from_config(config: DictConfig):
    model = get_model(config.model)
    optimizer = get_optimizer(config.optimizer, model=model)

    scheduler = get_scheduler(config.scheduler, optimizer=optimizer)
    criterion = get_criterion(config.loss)

    data_loaders = get_dataloader(config)

    # config_dict = OmegaConf.to_container(conf)


    return model, optimizer, scheduler, criterion, data_loaders

