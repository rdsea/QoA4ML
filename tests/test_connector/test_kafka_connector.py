import sys
from unittest.mock import MagicMock, patch

import pytest

from qoa4ml.config.configs import KafkaConnectorConfig

mock_confluent_kafka = MagicMock()
sys.modules["confluent_kafka"] = mock_confluent_kafka

from qoa4ml.connector.kafka_connector import (  # noqa: E402
    KafkaConnector,
    kafka_delivery_error,
)


@pytest.fixture
def kafka_config():
    return KafkaConnectorConfig(
        topic="test-topic",
        broker_url="localhost:9092",
    )


@pytest.fixture(autouse=True)
def reset_producer_mock():
    mock_confluent_kafka.Producer.reset_mock()
    mock_confluent_kafka.Producer.return_value = MagicMock()


class TestKafkaConnectorInit:
    def test_init_creates_producer(self, kafka_config):
        connector = KafkaConnector(kafka_config)

        mock_confluent_kafka.Producer.assert_called_once_with(
            bootstrap_servers="localhost:9092",
        )
        assert connector.topic == "test-topic"
        assert connector.conf is kafka_config
        assert connector.log_flag is False

    def test_init_with_log_flag(self, kafka_config):
        connector = KafkaConnector(kafka_config, log=True)
        assert connector.log_flag is True


class TestKafkaConnectorSendReport:
    def test_send_report_produces_message(self, kafka_config):
        mock_producer = MagicMock()
        mock_confluent_kafka.Producer.return_value = mock_producer

        connector = KafkaConnector(kafka_config)
        connector.send_report("test message")

        mock_producer.poll.assert_called_once_with(0)
        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args
        assert call_kwargs[0][0] == "test-topic"
        assert call_kwargs[0][1] == b"test message"
        mock_producer.flush.assert_called_once()

    def test_send_report_with_logging(self, kafka_config):
        mock_producer = MagicMock()
        mock_confluent_kafka.Producer.return_value = mock_producer

        connector = KafkaConnector(kafka_config, log=True)

        with patch("qoa4ml.connector.kafka_connector.qoa_logger") as mock_logger:
            connector.send_report("logged message")
            mock_logger.info.assert_called_once()

    def test_send_report_without_logging(self, kafka_config):
        mock_producer = MagicMock()
        mock_confluent_kafka.Producer.return_value = mock_producer

        connector = KafkaConnector(kafka_config, log=False)

        with patch("qoa4ml.connector.kafka_connector.qoa_logger") as mock_logger:
            connector.send_report("quiet message")
            mock_logger.info.assert_not_called()


class TestKafkaConnectorGet:
    def test_get_returns_config(self, kafka_config):
        connector = KafkaConnector(kafka_config)
        assert connector.get() is kafka_config


class TestKafkaDeliveryCallback:
    def test_delivery_callback_logs_error_when_err(self):
        with patch("qoa4ml.connector.kafka_connector.qoa_logger") as mock_logger:
            kafka_delivery_error("some error", MagicMock())
            mock_logger.error.assert_called_once()

    def test_delivery_callback_no_log_when_no_err(self):
        with patch("qoa4ml.connector.kafka_connector.qoa_logger") as mock_logger:
            kafka_delivery_error(None, MagicMock())
            mock_logger.error.assert_not_called()
