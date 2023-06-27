import abc
from typing import Dict, Optional, Union

__ALL__ = ['BaseLogger']


class BaseLogger(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        """Return the experiment name."""

    @property
    @abc.abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""

    @property
    def root_dir(self) -> Optional[str]:
        """Return the root directory where all versions of an experiment get saved, or `None` if the logger does
        not save data locally."""
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Return directory the current version of the experiment gets saved, or `None` if the logger does not save
        data locally."""
        return None

    @abc.abstractmethod
    def log(self, key, value):
        pass

    @abc.abstractmethod
    def log_config_params(self, config):
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass
