import socket
import time

from ..config.configs import SocketConnectorConfig
from ..utils.logger import qoa_logger
from .base_connector import BaseConnector


class SocketConnector(BaseConnector):
    """TCP socket connector that publishes reports as UTF-8 bytes.

    Any network error (connection refused, timeout, reset, ...) is caught,
    logged, and swallowed so callers (typically `Probe.reporting` running in
    a `RepeatedTimer` thread) do not die on transient aggregator outages.
    """

    # Socket connect / send timeout (seconds). A send that takes longer than
    # this is almost certainly stuck against a dead aggregator; dropping the
    # frame is better than hanging the probe thread indefinitely.
    _DEFAULT_TIMEOUT = 5.0

    def __init__(self, config: SocketConnectorConfig):
        self.config = config
        self.host = config.host
        self.port = config.port
        self.timeout = self._DEFAULT_TIMEOUT

    def send_report(self, body_message: str, log_path: str | None = None) -> None:
        """Send ``body_message`` to the configured host:port.

        Parameters
        ----------
        body_message : str
            Report body, sent as UTF-8 bytes.
        log_path : str, optional
            If set, append the round-trip time (ms) to this file.
        """
        start = time.time()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(self.timeout)
        try:
            client_socket.connect((self.host, self.port))
            client_socket.sendall(body_message.encode("utf-8"))
        except OSError as error:
            # Covers ConnectionRefusedError, TimeoutError, BrokenPipeError,
            # ConnectionResetError, socket.gaierror, and any other OSError.
            error_type = type(error).__name__
            qoa_logger.error(f"SocketConnector send failed ({error_type}): {error}")
            return
        finally:
            try:
                client_socket.close()
            except OSError:
                pass

        if log_path:
            with open(log_path, "a", encoding="utf-8") as file:
                file.write(f"{(time.time() - start) * 1000:.2f} ms\n")
