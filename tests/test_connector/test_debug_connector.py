import json

import pytest

from qoa4ml.config.configs import DebugConnectorConfig
from qoa4ml.connector.debug_connector import DebugConnector


@pytest.fixture()
def silent_connector():
    config = DebugConnectorConfig(silence=True)
    return DebugConnector(config)


@pytest.fixture()
def verbose_connector():
    config = DebugConnectorConfig(silence=False)
    return DebugConnector(config)


class TestDebugConnectorCreation:
    def test_create_with_silence_true(self):
        config = DebugConnectorConfig(silence=True)
        connector = DebugConnector(config)
        assert connector.silence is True

    def test_create_with_silence_false(self):
        config = DebugConnectorConfig(silence=False)
        connector = DebugConnector(config)
        assert connector.silence is False


class TestSendReport:
    def test_send_report_silent_no_output(self, silent_connector, capsys):
        message = json.dumps({"metric": "accuracy", "value": 0.95})
        silent_connector.send_report(message)
        captured = capsys.readouterr()
        assert "accuracy" not in captured.out

    def test_send_report_verbose_produces_output(self, verbose_connector, capsys):
        message = json.dumps({"metric": "latency", "value": 1.5})
        verbose_connector.send_report(message)
        captured = capsys.readouterr()
        assert "latency" in captured.err or "latency" in captured.out

    def test_send_report_with_nested_data(self, verbose_connector, capsys):
        message = json.dumps({"outer": {"inner": [1, 2, 3]}})
        verbose_connector.send_report(message)
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "inner" in output


class TestCheckConnection:
    def test_check_connection_returns_true(self, silent_connector):
        assert silent_connector.check_connection() is True

    def test_check_connection_verbose(self, verbose_connector):
        assert verbose_connector.check_connection() is True
