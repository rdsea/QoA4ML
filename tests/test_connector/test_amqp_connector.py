from unittest.mock import MagicMock, patch

import pika
import pika.exceptions
import pytest

from qoa4ml.config.configs import AMQPConnectorConfig
from qoa4ml.connector.amqp_connector import AmqpConnector


@pytest.fixture
def amqp_config():
    return AMQPConnectorConfig(
        end_point="localhost",
        exchange_name="test_exchange",
        exchange_type="topic",
        out_routing_key="test.routing.key",
    )


@pytest.fixture
def amqps_config():
    return AMQPConnectorConfig(
        end_point="amqps://user:pass@rabbitmq.example.com:5671/vhost",
        exchange_name="test_exchange",
        exchange_type="topic",
        out_routing_key="test.routing.key",
    )


@pytest.fixture
def mock_pika():
    with patch("qoa4ml.connector.amqp_connector.pika") as mock:
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_connection.is_closed = False
        mock_connection.is_open = True
        mock_channel.is_closed = False
        mock_channel.is_open = True
        mock.BlockingConnection.return_value = mock_connection
        mock.ConnectionParameters.return_value = MagicMock()
        mock.URLParameters.return_value = MagicMock()
        mock.BasicProperties = pika.BasicProperties
        mock.exceptions = pika.exceptions
        yield mock, mock_connection, mock_channel


class TestAmqpConnectorInit:
    def test_init_creates_connection_with_hostname(self, amqp_config, mock_pika):
        mock, mock_conn, mock_ch = mock_pika
        AmqpConnector(amqp_config)

        mock.ConnectionParameters.assert_called_once_with(
            host="localhost", heartbeat=600
        )
        mock.BlockingConnection.assert_called_once()
        mock_conn.channel.assert_called_once()
        mock_ch.exchange_declare.assert_called_once_with(
            exchange="test_exchange", exchange_type="topic"
        )

    def test_init_creates_connection_with_amqps_url(self, amqps_config, mock_pika):
        mock, _mock_conn, _mock_ch = mock_pika
        AmqpConnector(amqps_config)

        mock.URLParameters.assert_called_once_with(amqps_config.end_point)
        mock.BlockingConnection.assert_called_once()

    def test_init_stores_config(self, amqp_config, mock_pika):
        connector = AmqpConnector(amqp_config)

        assert connector.config is amqp_config
        assert connector.exchange_name == "test_exchange"
        assert connector.exchange_type == "topic"
        assert connector.out_routing_key == "test.routing.key"
        assert connector.log_flag is False

    def test_init_with_log_enabled(self, amqp_config, mock_pika):
        connector = AmqpConnector(amqp_config, log=True)
        assert connector.log_flag is True

    def test_init_health_check_disabled_sets_heartbeat_zero(self, mock_pika):
        config = AMQPConnectorConfig(
            end_point="localhost",
            exchange_name="test_exchange",
            exchange_type="topic",
            out_routing_key="test.key",
            health_check_disable=True,
        )
        mock, _, _ = mock_pika
        AmqpConnector(config)

        mock.ConnectionParameters.assert_called_once_with(host="localhost", heartbeat=0)

    def test_init_amqps_url_health_check_disabled(self, mock_pika):
        config = AMQPConnectorConfig(
            end_point="amqps://user:pass@host:5671/vhost",
            exchange_name="ex",
            exchange_type="topic",
            out_routing_key="key",
            health_check_disable=True,
        )
        mock, _, _ = mock_pika
        AmqpConnector(config)

        url_params = mock.URLParameters.return_value
        assert url_params.heartbeat == 0


class TestAmqpConnectorSendReport:
    def test_send_report_publishes_message(self, amqp_config, mock_pika):
        _mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        connector.send_report(
            "test message", corr_id="abc123", routing_key="custom.key"
        )

        mock_ch.basic_publish.assert_called_once()
        call_kwargs = mock_ch.basic_publish.call_args
        assert call_kwargs[1]["exchange"] == "test_exchange"
        assert call_kwargs[1]["routing_key"] == "custom.key"
        assert call_kwargs[1]["body"] == "test message"
        assert call_kwargs[1]["properties"].correlation_id == "abc123"

    def test_send_report_default_corr_id_generates_uuid(self, amqp_config, mock_pika):
        _mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        with patch("qoa4ml.connector.amqp_connector.uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = "fake-uuid-1234"
            connector.send_report("msg")

        props = mock_ch.basic_publish.call_args[1]["properties"]
        assert props.correlation_id == "fake-uuid-1234"

    def test_send_report_default_routing_key_uses_config(self, amqp_config, mock_pika):
        _mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        connector.send_report("msg", corr_id="id1")

        call_kwargs = mock_ch.basic_publish.call_args[1]
        assert call_kwargs["routing_key"] == "test.routing.key"

    def test_send_report_custom_routing_key(self, amqp_config, mock_pika):
        _mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        connector.send_report("msg", routing_key="other.key")

        call_kwargs = mock_ch.basic_publish.call_args[1]
        assert call_kwargs["routing_key"] == "other.key"

    def test_send_report_sets_expiration(self, amqp_config, mock_pika):
        _mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        connector.send_report("msg", expiration=5000)

        props = mock_ch.basic_publish.call_args[1]["properties"]
        assert props.expiration == "5000"

    def test_send_report_failure_triggers_reconnect_and_retry(
        self, amqp_config, mock_pika
    ):
        mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        # First publish raises, second succeeds
        mock_ch.basic_publish.side_effect = [
            pika.exceptions.AMQPConnectionError("connection lost"),
            None,
        ]

        # create_connection will produce a new channel
        new_channel = MagicMock()
        new_conn = MagicMock()
        new_conn.channel.return_value = new_channel
        new_conn.is_closed = False
        new_conn.is_open = True
        mock.BlockingConnection.return_value = new_conn

        connector.send_report("retry msg", corr_id="id1")

        # After reconnect, exchange should be redeclared and message re-published
        new_channel.exchange_declare.assert_called_once_with(
            exchange="test_exchange", exchange_type="topic"
        )
        new_channel.basic_publish.assert_called_once()

    def test_send_report_channel_error_triggers_reconnect(self, amqp_config, mock_pika):
        mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        mock_ch.basic_publish.side_effect = [
            pika.exceptions.AMQPChannelError("channel error"),
            None,
        ]

        new_channel = MagicMock()
        new_conn = MagicMock()
        new_conn.channel.return_value = new_channel
        new_conn.is_closed = False
        mock.BlockingConnection.return_value = new_conn

        connector.send_report("msg")

        new_channel.exchange_declare.assert_called_once()
        new_channel.basic_publish.assert_called_once()


class TestAmqpConnectorEnsureConnection:
    def test_ensure_connection_reconnects_when_connection_closed(
        self, amqp_config, mock_pika
    ):
        mock, mock_conn, _mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        # Simulate closed connection
        mock_conn.is_closed = True

        new_conn = MagicMock()
        new_ch = MagicMock()
        new_conn.channel.return_value = new_ch
        new_conn.is_closed = False
        mock.BlockingConnection.return_value = new_conn

        connector._ensure_connection()

        assert connector.out_connection is new_conn
        assert connector.out_channel is new_ch
        new_ch.exchange_declare.assert_called_once_with(
            exchange="test_exchange", exchange_type="topic"
        )

    def test_ensure_connection_reconnects_when_channel_closed(
        self, amqp_config, mock_pika
    ):
        mock, mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        mock_conn.is_closed = False
        mock_ch.is_closed = True

        new_conn = MagicMock()
        new_ch = MagicMock()
        new_conn.channel.return_value = new_ch
        mock.BlockingConnection.return_value = new_conn

        connector._ensure_connection()

        assert connector.out_channel is new_ch

    def test_ensure_connection_does_nothing_when_open(self, amqp_config, mock_pika):
        mock, _mock_conn, _mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        initial_call_count = mock.BlockingConnection.call_count

        connector._ensure_connection()

        assert mock.BlockingConnection.call_count == initial_call_count


class TestAmqpConnectorCheckConnection:
    def test_check_connection_returns_true_when_open(self, amqp_config, mock_pika):
        _mock, _mock_conn, _mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        assert connector.check_connection() is True

    def test_check_connection_returns_false_when_connection_closed(
        self, amqp_config, mock_pika
    ):
        _mock, mock_conn, _mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        mock_conn.is_open = False

        assert connector.check_connection() is False

    def test_check_connection_returns_false_when_channel_closed(
        self, amqp_config, mock_pika
    ):
        _mock, _mock_conn, mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        mock_ch.is_open = False

        assert connector.check_connection() is False


class TestAmqpConnectorReconnect:
    def test_reconnect_creates_new_connection(self, amqp_config, mock_pika):
        mock, _mock_conn, _mock_ch = mock_pika
        connector = AmqpConnector(amqp_config)

        new_conn = MagicMock()
        new_ch = MagicMock()
        new_conn.channel.return_value = new_ch
        mock.BlockingConnection.return_value = new_conn

        connector.reconnect()

        assert connector.out_connection is new_conn
        assert connector.out_channel is new_ch
        new_ch.exchange_declare.assert_called_once_with(
            exchange="test_exchange", exchange_type="topic"
        )


class TestAmqpConnectorGet:
    def test_get_returns_config(self, amqp_config, mock_pika):
        connector = AmqpConnector(amqp_config)
        assert connector.get() is amqp_config
