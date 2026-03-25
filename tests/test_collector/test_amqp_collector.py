from unittest.mock import MagicMock, patch

import pytest

from qoa4ml.config.configs import AMQPCollectorConfig


@pytest.fixture
def amqp_collector_config():
    return AMQPCollectorConfig(
        end_point="localhost",
        exchange_name="test_exchange",
        exchange_type="topic",
        in_routing_key="test.routing.key",
        in_queue="test_queue",
    )


@pytest.fixture
def amqps_collector_config():
    return AMQPCollectorConfig(
        end_point="amqps://user:pass@rabbitmq.example.com:5671/vhost",
        exchange_name="test_exchange",
        exchange_type="topic",
        in_routing_key="test.routing.key",
        in_queue="test_queue",
    )


@pytest.fixture
def mock_pika():
    with patch("qoa4ml.collector.amqp_collector.pika") as mock:
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel

        mock_queue_result = MagicMock()
        mock_queue_result.method.queue = "test_queue"
        mock_channel.queue_declare.return_value = mock_queue_result

        mock.BlockingConnection.return_value = mock_connection
        mock.ConnectionParameters.return_value = MagicMock()
        mock.URLParameters.return_value = MagicMock()

        yield mock, mock_connection, mock_channel


class TestAmqpCollectorInit:
    def test_init_with_hostname(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        mock, _mock_conn, _mock_ch = mock_pika

        AmqpCollector(amqp_collector_config)

        mock.ConnectionParameters.assert_called_once_with(
            host="localhost", heartbeat=600
        )
        mock.BlockingConnection.assert_called_once()

    def test_init_with_amqps_url(self, amqps_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        mock, _mock_conn, _mock_ch = mock_pika

        AmqpCollector(amqps_collector_config)

        mock.URLParameters.assert_called_once_with(amqps_collector_config.end_point)
        url_params = mock.URLParameters.return_value
        assert url_params.heartbeat == 600

    def test_init_creates_channel(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, mock_conn, _mock_ch = mock_pika

        AmqpCollector(amqp_collector_config)

        mock_conn.channel.assert_called_once()

    def test_init_declares_exchange(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, mock_ch = mock_pika

        AmqpCollector(amqp_collector_config)

        mock_ch.exchange_declare.assert_called_once_with(
            exchange="test_exchange", exchange_type="topic"
        )

    def test_init_declares_queue(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, mock_ch = mock_pika

        AmqpCollector(amqp_collector_config)

        mock_ch.queue_declare.assert_called_once_with(
            queue="test_queue", exclusive=False
        )

    def test_init_binds_queue(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, mock_ch = mock_pika

        AmqpCollector(amqp_collector_config)

        mock_ch.queue_bind.assert_called_once_with(
            exchange="test_exchange",
            queue="test_queue",
            routing_key="test.routing.key",
        )

    def test_init_stores_attributes(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, _mock_ch = mock_pika

        collector = AmqpCollector(amqp_collector_config)

        assert collector.exchange_name == "test_exchange"
        assert collector.exchange_type == "topic"
        assert collector.in_routing_key == "test.routing.key"
        assert collector.host_object is None

    def test_init_with_host_object(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, _mock_ch = mock_pika

        host_obj = MagicMock()
        collector = AmqpCollector(amqp_collector_config, host_object=host_obj)

        assert collector.host_object is host_obj


class TestAmqpCollectorOnRequest:
    def test_on_request_with_host_object(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, _mock_ch = mock_pika
        host_obj = MagicMock()

        collector = AmqpCollector(amqp_collector_config, host_object=host_obj)

        ch = MagicMock()
        method = MagicMock()
        props = MagicMock()
        body = b'{"key": "value"}'

        collector.on_request(ch, method, props, body)

        host_obj.message_processing.assert_called_once_with(ch, method, props, body)

    def test_on_request_without_host_object_logs_message(
        self, amqp_collector_config, mock_pika
    ):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, _mock_ch = mock_pika

        collector = AmqpCollector(amqp_collector_config)

        ch = MagicMock()
        method = MagicMock()
        props = MagicMock()
        body = b'{"metric": "cpu", "value": 42}'

        with patch("qoa4ml.collector.amqp_collector.qoa_logger") as mock_logger:
            collector.on_request(ch, method, props, body)
            mock_logger.info.assert_called_once_with({"metric": "cpu", "value": 42})


class TestAmqpCollectorStartCollecting:
    def test_start_collecting_sets_qos_and_consumes(
        self, amqp_collector_config, mock_pika
    ):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, mock_ch = mock_pika

        collector = AmqpCollector(amqp_collector_config)
        collector.start_collecting()

        mock_ch.basic_qos.assert_called_once_with(prefetch_count=1)
        mock_ch.basic_consume.assert_called_once_with(
            queue="test_queue",
            on_message_callback=collector.on_request,
            auto_ack=True,
        )
        mock_ch.start_consuming.assert_called_once()


class TestAmqpCollectorStop:
    def test_stop_closes_channel(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, mock_ch = mock_pika

        collector = AmqpCollector(amqp_collector_config)
        collector.stop()

        mock_ch.stop_consuming.assert_called_once()
        mock_ch.close.assert_called_once()


class TestAmqpCollectorGetQueue:
    def test_get_queue_returns_queue_name(self, amqp_collector_config, mock_pika):
        from qoa4ml.collector.amqp_collector import AmqpCollector

        _mock, _mock_conn, _mock_ch = mock_pika

        collector = AmqpCollector(amqp_collector_config)
        assert collector.get_queue() == "test_queue"
