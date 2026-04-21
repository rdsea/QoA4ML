import json
from urllib.parse import urlparse

import pika

from ..config.configs import AMQPCollectorConfig
from ..utils.logger import qoa_logger
from .base_collector import BaseCollector
from .host_object import HostObject

# Drop frames larger than this so a misbehaving producer cannot OOM the
# consumer thread. 1 MiB is generous for a single QoA report.
_MAX_FRAME_BYTES = 1 * 1024 * 1024


class AmqpCollector(BaseCollector):
    """
    AmqpCollector handles the connection to an AMQP server for collecting and processing messages.

    Parameters
    ----------
    configuration : AMQPCollectorConfig
        Configuration settings for connecting to the AMQP server.
    host_object : Optional[HostObject], optional
        An optional HostObject to process incoming messages, default is None.

    Attributes
    ----------
    host_object : Optional[HostObject]
        The host object responsible for processing messages.
    exchange_name : str
        The name of the exchange to connect to.
    exchange_type : str
        The type of the exchange (e.g., 'direct', 'topic').
    in_routing_key : str
        The routing key for incoming messages.
    in_connection : pika.BlockingConnection
        The connection to the RabbitMQ server.
    in_channel : pika.channel.Channel
        The channel for communication with RabbitMQ.
    queue : pika.spec.Queue.DeclareOk
        The queue to receive prediction responses.
    queue_name : str
        The name of the queue.

    Methods
    -------
    on_request(ch, method, props, body)
        Process incoming request messages.
    start_collecting()
        Start collecting messages from the queue.
    stop()
        Stop collecting messages and close the connection.
    get_queue() -> str
        Get the name of the queue.
    """

    def __init__(
        self,
        configuration: AMQPCollectorConfig,
        host_object: HostObject | None = None,
    ):
        """
        Initialize an instance of AmqpCollector.

        Parameters
        ----------
        configuration : AMQPCollectorConfig
            Configuration settings for connecting to the AMQP server.
        host_object : Optional[HostObject], optional
            An optional HostObject to process incoming messages, default is None.
        """
        self.host_object = host_object
        self.exchange_name = configuration.exchange_name
        self.exchange_type = configuration.exchange_type
        self.in_routing_key = configuration.in_routing_key

        if urlparse(configuration.end_point).scheme in {"amqp", "amqps"}:
            parameters = pika.URLParameters(configuration.end_point)
            parameters.heartbeat = 600
            self.in_connection = pika.BlockingConnection(parameters)
        else:
            self.in_connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=configuration.end_point, heartbeat=600)
            )

        self.in_channel = self.in_connection.channel()
        self.in_channel.exchange_declare(
            exchange=self.exchange_name, exchange_type=self.exchange_type
        )

        self.queue = self.in_channel.queue_declare(
            queue=configuration.in_queue, exclusive=False
        )
        self.queue_name = self.queue.method.queue

        self.in_channel.queue_bind(
            exchange=self.exchange_name,
            queue=self.queue_name,
            routing_key=self.in_routing_key,
        )

    def on_request(self, ch, method, props, body) -> None:
        """
        Process incoming request messages.

        Parameters
        ----------
        ch : pika.channel.Channel
            The channel object for the communication.
        method : pika.spec.Basic.Deliver
            The method frame object containing delivery parameters.
        props : pika.spec.BasicProperties
            The properties frame object containing message properties.
        body : bytes
            The message body sent from the producer.

        Notes
        -----
        If ``host_object`` is provided, it will handle message processing.
        Otherwise, the message is decoded and logged. Malformed payloads
        are logged and dropped instead of crashing the consumer thread.
        """
        if len(body) > _MAX_FRAME_BYTES:
            qoa_logger.error(
                f"AmqpCollector dropping oversize frame ({len(body)} > {_MAX_FRAME_BYTES} bytes)"
            )
            return

        if self.host_object is not None:
            try:
                self.host_object.message_processing(ch, method, props, body)
            except Exception as error:
                qoa_logger.exception(
                    f"AmqpCollector host_object raised ({type(error).__name__}); dropping frame"
                )
            return

        try:
            mess = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            qoa_logger.error(
                f"AmqpCollector dropping malformed frame ({type(error).__name__}): {error}"
            )
            return
        # The decoded payload may include user/instance identifiers; keep at DEBUG.
        qoa_logger.debug(f"AmqpCollector received {len(body)} bytes")
        qoa_logger.debug(mess)

    def start_collecting(self) -> None:
        """
        Start collecting messages from the queue.

        Notes
        -----
        This method starts the RabbitMQ consumer to collect messages from the queue and process them.
        The method will block and run indefinitely until `stop` is called.
        """
        self.in_channel.basic_qos(prefetch_count=1)
        self.in_channel.basic_consume(
            queue=self.queue_name, on_message_callback=self.on_request, auto_ack=True
        )
        self.in_channel.start_consuming()

    def stop(self) -> None:
        """Stop collecting and close both the channel and the connection.

        Previously only the channel was closed, leaking the underlying
        ``pika.BlockingConnection`` across restart cycles.
        """
        try:
            self.in_channel.stop_consuming()
        finally:
            try:
                self.in_channel.close()
            finally:
                self.in_connection.close()

    def get_queue(self) -> str:
        """
        Get the name of the queue.

        Returns
        -------
        str
            The name of the queue.
        """
        return self.queue.method.queue
