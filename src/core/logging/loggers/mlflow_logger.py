import logging
import os
import re
import tempfile
from argparse import Namespace
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from src.core.logging.common import BaseLogger


class MLFlowLogger(BaseLogger):
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

        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if run_id is None:
            mlflow.start_run()
        else:
            mlflow.start_run(run_id=run_id)

        self.mlflow = mlflow

    @property
    def name(self) -> Optional[str]:
        return self.experiment_name

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self.mlflow.active_run().info.run_id

    def log(self, key, value):
        self.mlflow.log_metric(key, value)

    def log_config_params(self, config):
        self.mlflow.log_params(config)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        for key, value in metrics.items():
            self.log(key, value)
