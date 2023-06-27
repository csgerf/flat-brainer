import logging
import os
import re
import tempfile
from argparse import Namespace
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from torch.utils.tensorboard import SummaryWriter
from src.core.logging.common import BaseLogger


class TensorBoardLogger(BaseLogger):
    def __init__(
        self,
        experiment_name: str,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        root_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.root_dir = root_dir
        self.log_dir = log_dir

        self.writer = SummaryWriter(log_dir)

    @property
    def name(self) -> Optional[str]:
        return self.experiment_name

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self.mlflow.active_run().info.run_id

    def log(self, key, value):
        self.writer.add_scalar(key, value)

    def log_config_params(self, config):
        self.writer.add_hparams(config)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: A dictionary of metric names and values.
            step: The step number at which the metrics should be recorded. Defaults to
                global_step.
        """
        if step is None:
            step = self.global_step

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Logs an artifact (file or directory) of the run.

        Args:
            local_path: Path to the artifact to log.
            artifact_path: If provided, the directory in ``artifact_uri`` to write to.
                Otherwise ``artifact_path`` defaults to the basename of ``local_path``.
        """
        # self.mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Logs the contents of a directory as an artifact.

        Args:
            local_dir: Path to the directory to log.
            artifact_path: If provided, the directory in ``artifact_uri`` to write to.
                Otherwise ``artifact_path`` defaults to the basename of ``local_dir``.
        """