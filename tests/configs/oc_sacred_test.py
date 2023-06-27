from src.core.utils import config as c
from src.core.utils.config import ex
from src.core.forge import component_factory as factory
from sacred.observers import FileStorageObserver

TEST_CONFIG_PATH = "C:\\dev\\working\\cv-train\\configs\\sample_yaml_config.yaml"
ex.observers.append(FileStorageObserver("logs"))


@ex.config
def get_config():
    cfg = c.read_config_file(TEST_CONFIG_PATH)
    print(cfg)


@ex.capture
def get_model(name, params):
    model = factory.get_model(name, params)
    return model


@ex.capture
def get_optimizer(name, params, model):
    optimizer = factory.get_optimizer(model.parameters(), name, params)
    optimizer_params = optimizer.defaults
    print(optimizer_params)
    return optimizer


@ex.capture
def get_scheduler(name, params, optimizer):
    scheduler = factory.get_scheduler(optimizer, name, params)
    return scheduler


@ex.automain
def main(_run, _config):
    model = get_model(_config["cfg"].model.name, _config["cfg"].model.params)
    optimizer = get_optimizer(_config["cfg"].optimizer.name, _config["cfg"].optimizer.params, model=model)
    scheduler = get_scheduler(_config["cfg"].scheduler.name, _config["cfg"].scheduler.params, optimizer=optimizer)

    _run.log_scalar("optimizer_defaults", optimizer.defaults)
    _run.log_scalar("scheduler_params", scheduler.state_dict())
    print('------------------------------------------------')