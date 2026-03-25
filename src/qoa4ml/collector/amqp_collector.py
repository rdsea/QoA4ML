import json

import pika

from ..config.configs import AMQPCollectorConfig
from ..utils.logger import qoa_logger
from .base_collector import BaseCollector
from .host_object import HostObject


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

        if "amqps://" in configuration.end_point:
            self.in_connection = pika.BlockingConnection(
                pika.URLParameters(configuration.end_point)
            )
        else:
            self.in_connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=configuration.end_point)
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
        If `host_object` is provided, it will handle message processing. Otherwise, the raw message will be logged.
        """
        if self.host_object is not None:
            self.host_object.message_processing(ch, method, props, body)
        else:
            mess = json.loads(str(body.decode("utf-8")))
            qoa_logger.info(mess)

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
        """
        Stop collecting messages and close the connection.

        Notes
        -----
        This method stops the consumer from collecting messages and closes the channel.
        """
        self.in_channel.stop_consuming()
        self.in_channel.close()

    def get_queue(self) -> str:
        """
        Get the name of the queue.

        Returns
        -------
        str
            The name of the queue.
        """
        return self.queue.method.queue
