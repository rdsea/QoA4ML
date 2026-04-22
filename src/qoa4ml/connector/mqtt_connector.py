from ..collector.host_object import HostObject
from ..config.configs import MQTTConnectorConfig
from ..utils.logger import qoa_logger
from .base_connector import BaseConnector

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None  # type: ignore[assignment]


class MqttConnector(BaseConnector):
    """Publish reports through MQTT and dispatch incoming messages to ``host_object``.

    ``host_object.message_processing(client, userdata, msg)`` is invoked
    whenever the broker delivers a message on the subscribed topic.
    Callers must call :meth:`start` after construction so paho's network
    loop is actually running; otherwise publishes will queue locally
    until the loop drains them.
    """

    def __init__(self, host_object: HostObject, configuration: MQTTConnectorConfig):
        if mqtt is None:
            raise ImportError(
                "paho-mqtt is required for MqttConnector; install qoa4ml[ml]"
            )
        # Init the host object to return message
        self.host_object = host_object
        # Config field semantics (per MQTTConnectorConfig):
        #   in_queue  = topic to subscribe to for incoming messages
        #   out_queue = topic to publish outgoing messages to
        # Earlier code had these swapped; align publish-to-out, subscribe-to-in.
        self.pub_queue = configuration.out_queue
        self.sub_queue = configuration.in_queue
        # Create the mqtt client
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=configuration.client_id,
            clean_session=False,
            userdata=None,
            transport="tcp",
        )
        # Set some functional method
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        # Connect to mqtt broker
        self.client.connect(
            configuration.broker_url,
            configuration.broker_port,
            configuration.broker_keepalive,
        )

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        # paho >= 2.x CallbackAPIVersion.VERSION2 passes (reason_code,
        # properties) instead of the v1 single rc int. Matching this
        # signature is required or paho raises TypeError on every connect
        # and the subscription below never runs.
        qoa_logger.debug(f"Connected with reason_code {reason_code}")
        # Subscribing in on_connect() means that if we lose the connection
        # and reconnect then subscriptions will be renewed.
        client.subscribe(self.sub_queue)

    def on_message(self, client, userdata, msg):
        # Pass the data to the host object
        self.host_object.message_processing(client, userdata, msg)

    def stop(self):
        # stop the connection
        self.client.disconnect()

    def start(self):
        # Start looking for data from broker
        self.client.loop_start()

    def send_data(self, body_message: str):
        # Send data in form of text message
        self.client.publish(self.pub_queue, body_message)

    def send_report(self, body_message: str):
        # Satisfies the BaseConnector contract; MQTT publish is the report channel.
        self.send_data(body_message)
