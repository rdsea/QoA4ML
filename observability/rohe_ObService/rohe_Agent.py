import json
from threading import Thread

import pymongo

from qoa4ml.collector.amqp_collector import AmqpCollector
from qoa4ml.utils.logger import qoa_logger


class Rohe_Agent:  # noqa: N801 - preserved external class name
    def __init__(self, configuration, mg_db=False):
        self.conf = configuration
        collector_conf = self.conf["collector"]
        self.collector = AmqpCollector(
            collector_conf["amqp_collector"]["conf"], host_object=self
        )
        db_conf = self.conf["database"]
        self.mongo_client = pymongo.MongoClient(db_conf["url"])
        self.db = self.mongo_client[db_conf["db_name"]]
        self.metric_collection = self.db[db_conf["metric_collection"]]
        self.sub_thread: Thread | None = None
        self.insert_db = mg_db

    def reset_db(self):
        self.metric_collection.drop()

    def start_consuming(self):
        qoa_logger.info("Rohe_Agent start consuming")
        self.collector.start_collecting()

    def start(self):
        self.sub_thread = Thread(target=self.start_consuming, daemon=True)
        self.sub_thread.start()
        qoa_logger.info("Rohe_Agent consumer thread started")

    def message_processing(self, ch, method, props, body):
        try:
            mess = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            qoa_logger.error(
                f"Rohe_Agent dropping malformed frame ({type(error).__name__}): {error}"
            )
            return
        qoa_logger.debug(f"Rohe_Agent received QoA report: {mess}")
        if self.insert_db:
            insert_id = self.metric_collection.insert_one(mess)
            qoa_logger.debug(f"Rohe_Agent inserted {insert_id.inserted_id}")

    def stop(self):
        """Stop the AMQP consumer and close the channel/connection."""
        self.insert_db = False
        try:
            self.collector.stop()
        except Exception as error:
            qoa_logger.exception(
                f"Rohe_Agent collector stop failed ({type(error).__name__})"
            )
        if self.sub_thread is not None:
            self.sub_thread.join(timeout=5)
            self.sub_thread = None

    def restart(self):
        """Re-enable DB inserts and re-start the consumer if it was stopped."""
        self.insert_db = True
        if self.sub_thread is None or not self.sub_thread.is_alive():
            self.start()
