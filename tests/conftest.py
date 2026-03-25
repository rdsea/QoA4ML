import pytest

from qoa4ml.config.configs import ClientConfig
from qoa4ml.lang.common_models import Condition, Metric
from qoa4ml.lang.datamodel_enum import OperatorEnum


@pytest.fixture
def sample_client_config_dict():
    """Minimal valid ClientConfig dictionary with a debug connector."""
    return {
        "client": {
            "name": "qoa_client_test",
            "username": "aaltosea",
            "user_id": "1",
            "instance_name": "aaltosea_instance_test1",
            "instance_id": "b6f83293-cf67-44dd-a7b5-77229d384012",
            "stage_id": "gateway",
            "functionality": "REST",
            "application_name": "test",
            "role": "ml",
            "run_id": "test1",
            "custom_info": {"your_custom_info": 1},
        },
        "connector": [
            {
                "name": "debug_connector",
                "connector_class": "Debug",
                "config": {"silence": True},
            }
        ],
    }


@pytest.fixture
def sample_client_config_with_probes_dict():
    """ClientConfig dictionary that includes probe definitions."""
    return {
        "client": {
            "name": "qoa_client_test",
            "username": "aaltosea",
            "user_id": "1",
            "instance_name": "aaltosea_instance_test1",
            "instance_id": "b6f83293-cf67-44dd-a7b5-77229d384012",
            "stage_id": "gateway",
            "functionality": "REST",
            "application_name": "test",
            "role": "ml",
            "run_id": "test1",
            "custom_info": {"your_custom_info": 1},
        },
        "connector": [
            {
                "name": "debug_connector",
                "connector_class": "Debug",
                "config": {"silence": True},
            }
        ],
        "probes": [
            {
                "probe_type": "docker",
                "frequency": 1,
                "require_register": False,
                "log_latency_flag": False,
                "environment": "Edge",
                "container_list": ["test"],
            },
            {
                "probe_type": "system",
                "frequency": 1,
                "require_register": False,
                "log_latency_flag": False,
                "environment": "Edge",
                "node_name": "Edge1",
            },
            {
                "probe_type": "process",
                "frequency": 1,
                "require_register": False,
                "log_latency_flag": False,
                "environment": "Edge",
            },
        ],
    }


@pytest.fixture
def sample_metric():
    """A simple Metric instance for reuse in tests."""
    return Metric(metric_name="accuracy", records=[0.95, 0.97])


@pytest.fixture
def sample_condition():
    """A simple Condition instance for reuse in tests."""
    return Condition(operator=OperatorEnum.geq, value=0.9)


@pytest.fixture
def sample_client_config(sample_client_config_dict):
    """A parsed ClientConfig model from the sample dict."""
    return ClientConfig(**sample_client_config_dict)
