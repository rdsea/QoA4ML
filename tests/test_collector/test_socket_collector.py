import json
from unittest.mock import MagicMock, patch

import pytest

from qoa4ml.collector.socket_collector import SocketCollector
from qoa4ml.config.configs import SocketCollectorConfig


@pytest.fixture
def socket_collector_config():
    return SocketCollectorConfig(
        host="0.0.0.0",
        port=8888,
        backlog=5,
        bufsize=4096,
    )


@pytest.fixture
def mock_process_report():
    return MagicMock()


class TestSocketCollectorInit:
    def test_init_stores_config(self, socket_collector_config, mock_process_report):
        collector = SocketCollector(socket_collector_config, mock_process_report)

        assert collector.config is socket_collector_config
        assert collector.host == "0.0.0.0"
        assert collector.port == 8888
        assert collector.backlog == 5
        assert collector.bufsize == 4096
        assert collector.process_report is mock_process_report
        assert collector.execution_flag is True


class TestSocketCollectorStartCollecting:
    @patch("qoa4ml.collector.socket_collector.socket")
    def test_start_collecting_binds_and_listens(
        self, mock_socket_module, socket_collector_config, mock_process_report
    ):
        mock_server_socket = MagicMock()
        mock_socket_module.socket.return_value = mock_server_socket
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        collector = SocketCollector(socket_collector_config, mock_process_report)

        # Make accept raise after first iteration to stop the loop
        mock_client_socket = MagicMock()
        test_data = json.dumps({"key": "value"}).encode("utf-8")
        mock_client_socket.recv.side_effect = [test_data, b""]
        mock_server_socket.accept.side_effect = [
            (mock_client_socket, ("127.0.0.1", 12345)),
            StopIteration,
        ]

        # Set execution_flag to stop after processing
        def stop_after_first(*args, **kwargs):
            collector.execution_flag = False

        mock_process_report.side_effect = stop_after_first

        # Re-configure accept to not raise after flag is set
        mock_server_socket.accept.side_effect = [
            (mock_client_socket, ("127.0.0.1", 12345)),
        ]

        collector.start_collecting()

        mock_socket_module.socket.assert_called_once_with(2, 1)
        mock_server_socket.bind.assert_called_once_with(("0.0.0.0", 8888))
        mock_server_socket.listen.assert_called_once_with(5)

    @patch("qoa4ml.collector.socket_collector.socket")
    def test_start_collecting_receives_and_deserializes(
        self, mock_socket_module, socket_collector_config, mock_process_report
    ):
        mock_server_socket = MagicMock()
        mock_socket_module.socket.return_value = mock_server_socket
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        collector = SocketCollector(socket_collector_config, mock_process_report)

        original_data = {"metric": "cpu", "value": 95.5}
        serialized = json.dumps(original_data).encode("utf-8")

        mock_client_socket = MagicMock()
        # Return data in chunks, then empty bytes to signal end
        mock_client_socket.recv.side_effect = [serialized[:10], serialized[10:], b""]

        def stop_loop(*args, **kwargs):
            collector.execution_flag = False

        mock_process_report.side_effect = stop_loop

        mock_server_socket.accept.return_value = (
            mock_client_socket,
            ("127.0.0.1", 5555),
        )

        collector.start_collecting()

        mock_process_report.assert_called_once_with(original_data)
        mock_client_socket.close.assert_called_once()

    @patch("qoa4ml.collector.socket_collector.socket")
    def test_start_collecting_handles_multiple_messages(
        self, mock_socket_module, socket_collector_config, mock_process_report
    ):
        mock_server_socket = MagicMock()
        mock_socket_module.socket.return_value = mock_server_socket
        mock_socket_module.AF_INET = 2
        mock_socket_module.SOCK_STREAM = 1

        collector = SocketCollector(socket_collector_config, mock_process_report)

        msg1 = json.dumps("message_1").encode("utf-8")
        msg2 = json.dumps("message_2").encode("utf-8")

        mock_client1 = MagicMock()
        mock_client1.recv.side_effect = [msg1, b""]
        mock_client2 = MagicMock()
        mock_client2.recv.side_effect = [msg2, b""]

        call_count = 0

        def stop_after_two(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                collector.execution_flag = False

        mock_process_report.side_effect = stop_after_two

        mock_server_socket.accept.side_effect = [
            (mock_client1, ("127.0.0.1", 1111)),
            (mock_client2, ("127.0.0.1", 2222)),
        ]

        collector.start_collecting()

        assert mock_process_report.call_count == 2
        mock_process_report.assert_any_call("message_1")
        mock_process_report.assert_any_call("message_2")

    def test_execution_flag_defaults_true(
        self, socket_collector_config, mock_process_report
    ):
        collector = SocketCollector(socket_collector_config, mock_process_report)
        assert collector.execution_flag is True

    def test_execution_flag_can_be_set_false(
        self, socket_collector_config, mock_process_report
    ):
        collector = SocketCollector(socket_collector_config, mock_process_report)
        collector.execution_flag = False
        assert collector.execution_flag is False
