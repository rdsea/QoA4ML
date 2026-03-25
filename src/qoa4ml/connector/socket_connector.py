import socket
import time

from ..config.configs import SocketConnectorConfig
from ..utils.logger import qoa_logger
from .base_connector import BaseConnector


class SocketConnector(BaseConnector):
    """
    SocketConnector handles the connection to a TCP socket for sending serialized messages.

    Parameters
    ----------
    config : SocketConnectorConfig
        Configuration settings for the socket connector.

    Attributes
    ----------
    config : SocketConnectorConfig
        The socket connector configuration.
    host : str
        The hostname or IP address to connect to.
    port : int
        The port number to connect to on the host.

    Methods
    -------
    send_report(body_message: str, log_path: Optional[str] = None) -> None
        Send a serialized message over the socket and optionally log the round-trip time.
    """

    def __init__(self, config: SocketConnectorConfig):
        """
        Initialize an instance of SocketConnector.

        Parameters
        ----------
        config : SocketConnectorConfig
            Configuration settings for the socket connector.
        """
        self.config = config
        self.host = config.host
        self.port = config.port

    def send_report(self, body_message: str, log_path: str | None = None) -> None:
        """
        Send a serialized message over the socket and optionally log the round-trip time.

        Parameters
        ----------
        body_message : str
            The message body to be serialized and sent.
        log_path : str, optional
            The path to the log file where round-trip time will be recorded, default is None.

        Notes
        -----
        - This method encodes the `body_message` as UTF-8 bytes.
        - It then sends the encoded message to the configured host and port.
        - If `log_path` is provided, the round-trip time in milliseconds will be recorded in the specified log file.

        Raises
        ------
        ConnectionRefusedError
            If the connection to the host is refused.
        """
        try:
            start = time.time()
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))
            serialized_message = body_message.encode("utf-8")
            client_socket.sendall(serialized_message)
            client_socket.close()

            if log_path:
                with open(log_path, "a", encoding="utf-8") as file:
                    file.write(f"{(time.time() - start) * 1000:.2f} ms\n")
        except ConnectionRefusedError:
            qoa_logger.error("Connection to aggregator refused")
