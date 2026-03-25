import socket
from collections.abc import Callable

from ..config.configs import SocketCollectorConfig
from .base_collector import BaseCollector


class SocketCollector(BaseCollector):
    """
    SocketCollector handles the collection of data over a TCP socket.

    Parameters
    ----------
    config : SocketCollectorConfig
        Configuration settings for the socket collector.
    process_report : Callable
        A callable function to process incoming reports.

    Attributes
    ----------
    config : SocketCollectorConfig
        The socket collector configuration.
    host : str
        The hostname or IP address to bind the socket.
    port : int
        The port number to bind the socket.
    backlog : int
        The maximum number of queued connections.
    bufsize : int
        The maximum size of data to be received at once.
    process_report : Callable
        A function to process the received report.
    execution_flag : bool
        Flag to control the execution loop.

    Methods
    -------
    start_collecting()
        Start the socket server to collect and process incoming data.
    """

    def __init__(self, config: SocketCollectorConfig, process_report: Callable) -> None:
        """
        Initialize an instance of SocketCollector.

        Parameters
        ----------
        config : SocketCollectorConfig
            Configuration settings for the socket collector.
        process_report : Callable
            A callable function to process incoming reports.
        """
        self.config = config
        self.host = config.host
        self.port = config.port
        self.backlog = config.backlog
        self.bufsize = config.bufsize
        self.process_report = process_report
        self.execution_flag = True

    def start_collecting(self) -> None:
        """
        Start the socket server to collect and process incoming data.

        Notes
        -----
        - This method starts a TCP socket server that listens for incoming connections.
        - Data received from clients is decoded from UTF-8 and parsed as JSON, then processed using the `process_report` function.
        - The server runs indefinitely until the `execution_flag` is set to False.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.backlog)

        while self.execution_flag:
            client_socket, _ = server_socket.accept()
            data = b""
            while True:
                packet = client_socket.recv(self.bufsize)
                if not packet:
                    break
                data += packet

            report = data.decode("utf-8")
            self.process_report(report)
            client_socket.close()
