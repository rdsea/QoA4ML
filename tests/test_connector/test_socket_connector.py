from unittest.mock import MagicMock, mock_open, patch

import pytest

from qoa4ml.config.configs import SocketConnectorConfig
from qoa4ml.connector.socket_connector import SocketConnector


@pytest.fixture
def socket_config():
    return SocketConnectorConfig(host="127.0.0.1", port=9999)


class TestSocketConnectorInit:
    def test_init_stores_config(self, socket_config):
        connector = SocketConnector(socket_config)

        assert connector.config is socket_config
        assert connector.host == "127.0.0.1"
        assert connector.port == 9999


class TestSocketConnectorSendReport:
    @patch("qoa4ml.connector.socket_connector.socket")
    def test_send_report_serializes_and_sends(self, mock_socket_module, socket_config):
        mock_sock = MagicMock()
        mock_socket_module.socket.return_value = mock_sock
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        connector = SocketConnector(socket_config)
        connector.send_report("hello world")

        mock_socket_module.socket.assert_called_once_with(2, 1)
        mock_sock.connect.assert_called_once_with(("127.0.0.1", 9999))

        expected_data = b"hello world"
        mock_sock.sendall.assert_called_once_with(expected_data)
        mock_sock.close.assert_called_once()

    @patch("qoa4ml.connector.socket_connector.time")
    @patch("qoa4ml.connector.socket_connector.socket")
    def test_send_report_with_log_path_writes_latency(
        self, mock_socket_module, mock_time, socket_config
    ):
        mock_sock = MagicMock()
        mock_socket_module.socket.return_value = mock_sock
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        mock_time.time.side_effect = [1000.0, 1000.05]

        connector = SocketConnector(socket_config)

        m = mock_open()
        with patch("builtins.open", m):
            connector.send_report("data", log_path="/tmp/latency.log")

        m.assert_called_once_with("/tmp/latency.log", "a", encoding="utf-8")
        handle = m()
        handle.write.assert_called_once_with("50.00 ms\n")

    @patch("qoa4ml.connector.socket_connector.socket")
    def test_send_report_without_log_path_no_file_write(
        self, mock_socket_module, socket_config
    ):
        mock_sock = MagicMock()
        mock_socket_module.socket.return_value = mock_sock
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        connector = SocketConnector(socket_config)

        with patch("builtins.open", mock_open()) as m:
            connector.send_report("data")
            m.assert_not_called()

    @patch("qoa4ml.connector.socket_connector.socket")
    def test_send_report_connection_refused_handled(
        self, mock_socket_module, socket_config
    ):
        mock_sock = MagicMock()
        mock_socket_module.socket.return_value = mock_sock
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1
        mock_sock.connect.side_effect = ConnectionRefusedError("refused")

        connector = SocketConnector(socket_config)

        # Should not raise
        connector.send_report("data")

    @patch("qoa4ml.connector.socket_connector.socket")
    def test_send_report_serializes_complex_data(
        self, mock_socket_module, socket_config
    ):
        mock_sock = MagicMock()
        mock_socket_module.socket.return_value = mock_sock
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        connector = SocketConnector(socket_config)
        data = '{"key": "value", "number": 42}'
        connector.send_report(data)

        expected = data.encode("utf-8")
        mock_sock.sendall.assert_called_once_with(expected)
