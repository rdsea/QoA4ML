import json

from confluent_kafka import Consumer

from ..config.configs import KafkaCollectorConfig
from ..utils.logger import qoa_logger
from .base_collector import BaseCollector
from .host_object import HostObject


class KafkaCollector(BaseCollector):
    def __init__(
        self,
        config: KafkaCollectorConfig,
        host_object: HostObject | None = None,
    ):
        self.config = config
        self.host_object = host_object
        self.running = False
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.config.broker_url,
                "group.id": self.config.group_id,
                "auto.offset.reset": self.config.auto_offset_reset,
            }
        )

    def on_request(self, ch, method, props, body):
        if self.host_object is not None:
            self.host_object.message_processing(ch, method, props, body)
        else:
            mess = json.loads(str(body.decode("utf-8")))
            qoa_logger.info(mess)

    def start_collecting(self):
        self.running = True
        while self.running:
            msg = self.consumer.poll(self.config.poll_interval)

            if msg is None:
                continue
            if msg.error():
                qoa_logger.error("Consumer error: %s", msg.error())
                continue

    def stop(self):
        self.running = False
