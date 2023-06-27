from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import logging
import os

import sacred

from src.core.logging.common import BaseLogger


class SacredLogger(BaseLogger):
    def __init__(self,
                 experiment: sacred.Experiment,
                 experiment_id: Optional[str] = None,
                 run_id: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 root_dir: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 **kwargs,
                 ):
        self.ex = experiment
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.root_dir = root_dir
        self.log_dir = log_dir

    @property
    def name(self) -> Optional[str]:
        return self.experiment_name
