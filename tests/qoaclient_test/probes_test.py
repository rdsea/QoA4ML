import os
import time

import pytest

from qoa4ml.qoa_client import QoaClient

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.requires_docker
def test_probes_reporting():
    qoa_client = QoaClient(config_path=f"{dir_path}/config/client_with_probes.yaml")
    qoa_client.start_all_probes()
    time.sleep(1)
