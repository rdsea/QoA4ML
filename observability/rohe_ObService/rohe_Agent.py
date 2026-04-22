import json
from threading import Thread

import pymongo
from pymongo.errors import PyMongoError

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
        # Guard double-start so we don't leak the prior consumer thread.
        if self.sub_thread is not None and self.sub_thread.is_alive():
            qoa_logger.warning("Rohe_Agent.start() called while already running")
            return
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
            # Wrap the DB boundary so a transient Mongo outage can't kill
            # the consumer thread (project "Error Resilience" rule).
            try:
                insert_id = self.metric_collection.insert_one(mess)
            except PyMongoError as error:
                qoa_logger.exception(
                    f"Rohe_Agent mongo insert failed ({type(error).__name__})"
                )
                return
            qoa_logger.debug(f"Rohe_Agent inserted {insert_id.inserted_id}")

    def stop(self):
        """Pause the AMQP consumer and reset its worker thread.

        Notes
        -----
        This is a *pause*: the Mongo client and AMQP collector are kept
        alive so :meth:`restart` can resume without rebuilding them.
        Call :meth:`shutdown` for a terminal close that releases those
        resources.
        """
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

    def shutdown(self):
        """Terminal teardown: pause, then close Mongo connections.

        Use at service shutdown — a subsequent :meth:`restart` would need
        to reconstruct ``mongo_client`` first because ``close()`` marks
        the MongoClient permanently unusable.
        """
        self.stop()
        try:
            self.mongo_client.close()
        except Exception as error:
            qoa_logger.exception(
                f"Rohe_Agent mongo close failed ({type(error).__name__})"
            )

    def restart(self):
        """Re-enable DB inserts and re-start the consumer if it was stopped."""
        self.insert_db = True
        if self.sub_thread is None or not self.sub_thread.is_alive():
            self.start()
