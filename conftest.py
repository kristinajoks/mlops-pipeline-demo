import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skipif: skip test based on condition"
    )