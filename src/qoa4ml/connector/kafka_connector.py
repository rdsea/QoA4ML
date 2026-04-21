from confluent_kafka import Producer

from ..config.configs import KafkaConnectorConfig
from ..utils.logger import qoa_logger
from .base_connector import BaseConnector


def kafka_delivery_error(err, msg):
    if err is not None:
        topic = getattr(msg, "topic", lambda: None)()
        key = getattr(msg, "key", lambda: None)()
        qoa_logger.error(f"Kafka delivery failed on topic={topic!r} key={key!r}: {err}")


class KafkaConnector(BaseConnector):
    def __init__(self, config: KafkaConnectorConfig, log: bool = False):
        self.conf = config
        self.topic = config.topic
        self.log_flag = log
        self.producer: Producer = Producer({"bootstrap.servers": config.broker_url})

    def send_report(
        self,
        body_message: str,
    ):
        self.producer.poll(0)

        self.producer.produce(
            self.topic,
            body_message.encode("utf-8"),
            callback=kafka_delivery_error,
        )
        self.producer.flush()

        if self.log_flag:
            qoa_logger.info(f"Sent message to topic {self.topic}: {body_message}")

    def get(self):
        return self.conf
