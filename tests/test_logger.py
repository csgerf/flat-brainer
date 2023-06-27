import pytest
from src.core.logging import LoggerManager, TensorBoardLogger, PythonLogger

class MockLogger:
    def __init__(self):
        self.data = {}

    def log(self, key, value):
        self.data[key] = value

@pytest.fixture
def logger_manager():
    tensorboard_logger = TensorBoardLogger()
    python_logger = PythonLogger()
    mock_logger = MockLogger()
    
    return LoggerManager(tensorboard_logger, python_logger, mock_logger), mock_logger

def test_logger_manager(logger_manager):
    logger, mock_logger = logger_manager
    key = 'test_key'
    value = 'test_value'

    logger.log(key, value)
    assert mock_logger.data[key] == value
