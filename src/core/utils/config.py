from dataclasses import dataclass
from omegaconf import OmegaConf
from sacred import Experiment, Ingredient

model_ingredient = Ingredient('model')
optimizer_ingredient = Ingredient('optimizer')
loss_function_ingredient = Ingredient('loss_function')
scheduler_ingredient = Ingredient('scheduler')
training_ingredient = Ingredient('training')

ex = Experiment('pytorch_experiment',
                ingredients=[
                    model_ingredient,
                    optimizer_ingredient,
                    loss_function_ingredient,
                    scheduler_ingredient,
                    training_ingredient],
                )


@dataclass
class OptimizerConfig:
    name: str
    params: dict


@dataclass
class SchedulerConfig:
    name: str
    params: dict


@dataclass
class LossConfig:
    name: str
    params: dict


@dataclass
class ModelConfig:
    name: str
    params: dict


@dataclass
class DatasetConfig:
    name: str
    params: dict


@dataclass
class TrainerConfig:
    name: str
    batch_size: int
    num_epochs: int
    output_dir: str
    params: dict


@dataclass
class ValidationConfig:
    name: str
    batch_size: int
    params: dict


@dataclass
class ExperimentConfig:
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    model: ModelConfig | None = None
    dataset: DatasetConfig | None = None
    trainer: TrainerConfig | None = None


def default_config():
    batch_size: int = 32
    num_epochs: int = 100
    output_dir: str = "C:\\dev\\working\\cv-train\\output\\runs"

    optimizer = OptimizerConfig(name='Adam', params={'lr': 0.001, 'weight_decay': 0.0005})
    scheduler = SchedulerConfig(name='StepLR', params={'step_size': 30, 'gamma': 0.1})
    loss = LossConfig(name='CrossEntropyLoss', params={})
    model = ModelConfig(name='ResNet18', params={})
    dataset = DatasetConfig(name='medmnist', params={})
    trainer = TrainerConfig(name='Trainer',
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            output_dir=output_dir,
                            params={"batch_size": 32, "num_epochs": 100, "num_workers": 4, "device": "cuda"})
    default_experiment = ExperimentConfig(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        model=model,
        dataset=dataset,
        trainer=trainer
    )

    return default_experiment


def read_config_file(yaml_path: str):
    default_conf = default_config()
    schema = OmegaConf.structured(default_conf)

    conf = OmegaConf.load(yaml_path)
    conf = OmegaConf.merge(schema, conf)

    return conf
