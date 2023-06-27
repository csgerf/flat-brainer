from typing import Optional, Union
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
import mlflow

ValueType = Union[int, float, str]
NumberType = Union[int, float]


class BaseExperimentTracker:
    def log_parameter(self, name: str, value: ValueType) -> None:
        pass

    def log_result(self, name: str, value: NumberType) -> None:
        pass


class SacredTracker(BaseExperimentTracker):
    def __init__(self, ex: Experiment):
        self.ex = ex

    def log_parameter(self, name: str, value: ValueType) -> None:
        self.ex.log_scalar(name, value)

    def log_result(self, name: str, value: ValueType) -> None:
        self.ex.log_scalar(name, value)


class TensorboardTracker(BaseExperimentTracker):
    def __init__(self, writer: Optional[SummaryWriter] = None):
        if writer is None:
            writer = SummaryWriter() # Writer will output to ./runs/ directory by default.
        self.writer = writer

    def log_parameter(self, name: str, value: ValueType) -> None:
        self.writer.add_scalar(name, value)

    def log_result(self, name: str, value: ValueType) -> None:
        self.writer.add_scalar(name, value)


class MLflowTracker(BaseExperimentTracker):
    def __init__(self, run_name):
        mlflow.start_run(run_name=run_name)

    def log_parameter(self, name, value):
        mlflow.log_param(name, value)

    def log_result(self, name, value):
        mlflow.log_metric(name, value)

    def __del__(self):
        mlflow.end_run()
