import pprint

from src.core.utils import config as c
from src.core.utils.config import ex
from src.apis.train import run as run_train
from omegaconf import DictConfig, OmegaConf

from sacred.observers import FileStorageObserver, MongoObserver
# ex.observers.append(FileStorageObserver("logs"))
# ex.observers.append(MongoObserver(db_name='experiments', collection_prefix="runs", url='mongodb://localhost:27017'))
ex.observers.append(MongoObserver(db_name='experiments', url='mongodb://localhost:27017'))
# def main():
#     pprint.PrettyPrinter(indent=2).pprint(config)
#
# def train(_run, _config):
#     # config = edict(_config)
#     print('------------------------------------------------')
#     print('train')
#     # pprint.PrettyPrinter(indent=2).pprint(config)
#     # run_train(config)
#
#
# TEST_CONFIG_PATH = "C:\\dev\\working\\cv-train\\configs\\sample_yaml_config.yaml"
TEST_CONFIG_PATH = "C:\\dev\\working\\cv-train\\data\\configs\\hubmap\\sample_hubmap_config.yaml"

global_cfg = OmegaConf.to_container(c.read_config_file(TEST_CONFIG_PATH))

@ex.config
def get_config():
    cfg = OmegaConf.to_container(c.read_config_file(TEST_CONFIG_PATH))
    print(cfg)


@c.model_ingredient.config
def model_config(cfg):
    print("model_config")

    return cfg["model"]


@ex.command
def train(_run, _config, _log):
    print("train")
    # config = _config["cfg"]
    config = OmegaConf.create(_run.config["cfg"])
    print("\t", config)

    # model = get_model(_config["cfg"].model)
    # optimizer = get_optimizer(_config["cfg"].optimizer.name, _config["cfg"].optimizer.params, model=model)
    # scheduler = get_scheduler(_config["cfg"].scheduler.name, _config["cfg"].scheduler.params, optimizer=optimizer)
    # criterion = get_criterion(_config["cfg"].loss.name, _config["cfg"].loss.params)
    # data_loaders = get_dataloader(_config["cfg"].dataset.name, _config["cfg"].dataset.params)
    print('------------------------------------------------')
    print('train')
    pprint.PrettyPrinter(indent=2).pprint(config)

    # run_train(_config["cfg"], model, optimizer, scheduler, criterion, data_loaders)
    run_train(config)


# @ex.main
# def main(_run, _config):
#     print("main")
#     config = _config["cfg"]
#     pprint.PrettyPrinter(indent=2).pprint(config)


if __name__ == '__main__':
    ex.run_commandline()
