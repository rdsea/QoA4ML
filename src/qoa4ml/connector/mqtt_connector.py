from typing import TYPE_CHECKING

import lazy_import

from ..collector.host_object import HostObject
from ..config.configs import MQTTConnectorConfig
from ..utils.logger import qoa_logger
from .base_connector import BaseConnector

if TYPE_CHECKING:
    import paho.mqtt.client as mqtt
else:
    mqtt = lazy_import.lazy_module("paho.mqtt")


# TODO: this client handle both connector and collector
class MqttConnector(BaseConnector):
    # This class will handle all the mqtt connection for each client application
    # FIX: what is host object?
    def __init__(self, host_object: HostObject, configuration: MQTTConnectorConfig):
        # from paho.mqtt.client import Client as MqttClient
        # from paho.mqtt.enums import CallbackAPIVersion
        # Init the host object to return message
        self.host_object = host_object
        # Init the send/receive queue
        self.pub_queue = configuration.in_queue
        self.sub_queue = configuration.out_queue
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

    def on_connect(self, client, userdata, flags, rc):
        qoa_logger.debug("Connected with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
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
