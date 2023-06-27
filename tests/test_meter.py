import pytest
import numpy as np
from src.core.logging.meter import AverageValueMeter, MeterGroup


def test_average_value_meter():
    # Instantiate AverageValueMeter
    meter = AverageValueMeter()

    # Add a single value and check mean and standard deviation
    meter.add(5.0)
    mean, std = meter.value()
    assert mean == 5.0
    assert std == np.inf

    # Add another value
    meter.add(7.0)
    mean, std = meter.value()
    assert mean == 6.0
    assert std == np.sqrt(2.0)

    # Reset the meter and check values
    meter.reset()
    mean, std = meter.value()
    assert np.isnan(mean)
    assert np.isnan(std)

    # Add values in bulk and check mean and standard deviation
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for val in values:
        meter.add(val)
    mean, std = meter.value()
    assert mean == np.mean(values)
    assert std == np.std(values, ddof=1)


def test_meter_group():
    group = MeterGroup()

    # Add meters to the group
    group.add_meter("meter1", AverageValueMeter())
    group.add_meter("meter2", AverageValueMeter())

    # Add values
    group.add("meter1", 5.0)
    group.add("meter2", 7.0)

    # Get values
    values = group.value()
    assert values["meter1"][0] == 5.0
    assert values["meter2"][0] == 7.0

    # Reset meters
    group.reset()
    values = group.value()
    assert np.isnan(values["meter1"][0])
    assert np.isnan(values["meter2"][0])


if __name__ == "__main__":
    pytest.main([__file__])
