import json

from confluent_kafka import Consumer

from ..config.configs import KafkaCollectorConfig
from ..utils.logger import qoa_logger
from .base_collector import BaseCollector
from .host_object import HostObject

# Drop frames larger than this so a misbehaving producer cannot OOM the
# consumer thread. Matches the cap applied in amqp_collector.
_MAX_FRAME_BYTES = 1 * 1024 * 1024


class KafkaCollector(BaseCollector):
    def __init__(
        self,
        config: KafkaCollectorConfig,
        host_object: HostObject | None = None,
    ) -> None:
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

    def on_request(self, ch, method, props, body) -> None:
        # confluent-kafka's msg.value() is None for tombstone / keyed-null
        # records; skip them rather than raising TypeError on len(None).
        if body is None:
            return
        if len(body) > _MAX_FRAME_BYTES:
            qoa_logger.error(
                f"KafkaCollector dropping oversize frame "
                f"({len(body)} > {_MAX_FRAME_BYTES} bytes)"
            )
            return
        if self.host_object is not None:
            self.host_object.message_processing(ch, method, props, body)
            return
        try:
            mess = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            qoa_logger.error(
                f"KafkaCollector dropping malformed frame "
                f"({type(error).__name__}): {error}"
            )
            return
        # Payload may include user/instance identifiers; keep at DEBUG.
        qoa_logger.debug(f"KafkaCollector received {len(body)} bytes")
        qoa_logger.debug(mess)

    def start_collecting(self) -> None:
        """Subscribe to the configured topic and dispatch each message.

        Previously the loop polled without subscribing and discarded every
        received message, so the collector silently dropped all traffic.
        """
        self.consumer.subscribe([self.config.topic])
        self.running = True
        try:
            while self.running:
                msg = self.consumer.poll(self.config.poll_interval)
                if msg is None:
                    continue
                if msg.error():
                    qoa_logger.error(f"Kafka consumer error: {msg.error()}")
                    continue
                self.on_request(None, None, None, msg.value())
        finally:
            self.consumer.close()

    def stop(self) -> None:
        self.running = False
