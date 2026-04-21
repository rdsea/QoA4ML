import socket
from collections.abc import Callable

from ..config.configs import SocketCollectorConfig
from ..utils.logger import qoa_logger
from .base_collector import BaseCollector


class SocketCollector(BaseCollector):
    """TCP socket collector that forwards each received frame to ``process_report``.

    Provides ``start_collecting()`` for running the server (blocking) and
    ``stop()`` for a clean shutdown. The accept loop uses a short timeout so
    shutdown requests are honoured promptly even with no pending connection.
    """

    # Accept timeout (seconds). Small enough that `stop()` feels responsive,
    # large enough that the loop doesn't busy-spin.
    _ACCEPT_TIMEOUT = 0.5
    # Per-client recv timeout (seconds). Prevents a half-closed sender from
    # hanging the worker thread forever.
    _CLIENT_TIMEOUT = 5.0

    def __init__(self, config: SocketCollectorConfig, process_report: Callable) -> None:
        self.config = config
        self.host = config.host
        self.port = config.port
        self.backlog = config.backlog
        self.bufsize = config.bufsize
        self.process_report = process_report
        self.execution_flag = True
        self._server_socket: socket.socket | None = None

    def start_collecting(self) -> None:
        """Run the TCP server until ``stop()`` is called.

        Each incoming connection is read to EOF, decoded as UTF-8, and passed
        to ``process_report``. Errors on a single connection are logged and
        do not terminate the server.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(self.backlog)
        server_socket.settimeout(self._ACCEPT_TIMEOUT)
        self._server_socket = server_socket

        try:
            while self.execution_flag:
                try:
                    client_socket, _ = server_socket.accept()
                except TimeoutError:
                    continue
                except OSError as error:
                    # Socket was closed from stop(); exit loop cleanly.
                    if not self.execution_flag:
                        break
                    qoa_logger.exception(
                        f"SocketCollector accept failed ({type(error).__name__})"
                    )
                    continue

                self._handle_client(client_socket)
        finally:
            try:
                server_socket.close()
            finally:
                self._server_socket = None

    def _handle_client(self, client_socket: socket.socket) -> None:
        client_socket.settimeout(self._CLIENT_TIMEOUT)
        data = b""
        try:
            while True:
                packet = client_socket.recv(self.bufsize)
                if not packet:
                    break
                data += packet
            report = data.decode("utf-8")
            self.process_report(report)
        except (OSError, UnicodeDecodeError) as error:
            qoa_logger.exception(
                f"SocketCollector client frame dropped ({type(error).__name__})"
            )
        except Exception as error:
            qoa_logger.exception(
                f"SocketCollector process_report raised ({type(error).__name__})"
            )
        finally:
            try:
                client_socket.close()
            except OSError:
                pass

    def stop(self) -> None:
        """Signal the accept loop to exit and close the listening socket."""
        self.execution_flag = False
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
