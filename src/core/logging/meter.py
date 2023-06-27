from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np


__ALL__ = ["AverageValueMeter", "MeterGroup"]


class Meter(ABC):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    @abstractmethod
    def reset(self) -> None:
        """Resets the meter to default settings."""
        pass

    @abstractmethod
    def add(self, value) -> None:
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    @abstractmethod
    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self) -> None:
        super(AverageValueMeter, self).__init__()
        self.n = 0
        self.sum = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.reset()

    def add(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value
        self.n += n

        if self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.sum / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self) -> Tuple[float, float]:
        return self.mean, self.std
    
    def get_mean(self) -> float:
        return self.mean
    
    def get_std(self) -> float:
        return self.std

    def reset(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class MeterGroup:
    def __init__(self) -> None:
        self.meters: Dict[str, Meter] = {}

    def add_meter(self, name: str, meter: Meter) -> None:
        """Add a new meter to the group."""
        self.meters[name] = meter

    def add(self, name: str, value: Union[float, Tuple[float, ...]]) -> None:
        """Add a new value to the specified meter."""
        if name not in self.meters:
            raise KeyError(f"Meter {name} not found in the group")
        self.meters[name].add(value)

    def reset(self) -> None:
        """Reset all meters in the group."""
        for meter in self.meters.values():
            meter.reset()

    def value(self) -> Dict[str, Tuple[float, ...]]:
        """Get the values from all meters in the group."""
        return {name: meter.value() for name, meter in self.meters.items()}

