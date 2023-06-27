import argparse
import yaml

from src.core.utils import config as c

# from src.core.utils.config import ex
# from src.apis.train import run as run_train
from omegaconf import DictConfig, OmegaConf
from src.core.utils import fs


TEST_CONFIG_PATH = "C:\\dev\\working\\cv-train\\data\\configs\\hubmap\\sample_hubmap_config.yaml"


def train(config_path):
    print("Training with config:")
    config = c.read_config_file(config_path)

    output_dir = config.trainer.get("output_dir", "runs")
    run_directory = fs.prepare_directories(output_dir)
    config.trainer.output_dir = run_directory

    print(config)
    # run_train(config)


def inference(config):
    print("Running inference with config:")
    print(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Config file path", default=TEST_CONFIG_PATH)

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    inference_parser = subparsers.add_parser("inference")

    args = parser.parse_args()

    # Load the config file

    config = None
    # Execute the appropriate function based on the provided command
    # if args.command == "train":
    #     train(args.config)
    # elif args.command == "inference":
    #     inference(config)
    # else:
    #     print("Invalid command. Please use 'train' or 'inference'.")
    #     parser.print_help()


if __name__ == "__main__":
    main()