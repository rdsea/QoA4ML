"""Regression tests for MqttConnector instantiation and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qoa4ml.collector.host_object import HostObject
from qoa4ml.config.configs import MQTTConnectorConfig


@pytest.fixture
def mqtt_config() -> MQTTConnectorConfig:
    # Per config docstrings: in_queue = subscribe topic, out_queue = publish topic.
    return MQTTConnectorConfig(
        in_queue="sub_topic",
        out_queue="pub_topic",
        broker_url="localhost",
        broker_port=1883,
        broker_keepalive=60,
        client_id="test-client",
    )


@pytest.fixture
def host_object() -> HostObject:
    return MagicMock(spec=HostObject)


class TestMqttConnectorMissingPaho:
    def test_init_raises_importerror_when_paho_missing(
        self, monkeypatch, mqtt_config, host_object
    ):
        # Regression: previously the class set `mqtt = None` on ImportError
        # and then crashed with a confusing `AttributeError: 'NoneType'
        # object has no attribute 'Client'` inside `__init__`. Now the
        # guard raises a clear ImportError instead.
        from qoa4ml.connector import mqtt_connector

        monkeypatch.setattr(mqtt_connector, "mqtt", None)

        with pytest.raises(ImportError, match="paho-mqtt"):
            mqtt_connector.MqttConnector(host_object, mqtt_config)


class TestMqttConnectorSendReport:
    def test_send_report_delegates_to_publish(
        self, monkeypatch, mqtt_config, host_object
    ):
        # Regression: MqttConnector must implement BaseConnector.send_report
        # so it can be instantiated (rule: LSP — subclasses fulfill the
        # parent contract).
        from qoa4ml.connector import mqtt_connector

        mock_mqtt = MagicMock()
        mock_client = MagicMock()
        mock_mqtt.Client.return_value = mock_client
        mock_mqtt.CallbackAPIVersion.VERSION2 = object()
        monkeypatch.setattr(mqtt_connector, "mqtt", mock_mqtt)

        connector = mqtt_connector.MqttConnector(host_object, mqtt_config)
        connector.send_report("hello")

        # Regression: pub_queue must be out_queue (publish destination),
        # not in_queue; previous code had these inverted.
        mock_client.publish.assert_called_once_with("pub_topic", "hello")
        assert connector.sub_queue == "sub_topic"
