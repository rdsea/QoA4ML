import math
import time
from datetime import datetime
from pathlib import Path

from tinyflux import Point, TimeQuery, TinyFlux
from tinyflux.storages import CSVStorage


class EmbeddedDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db = TinyFlux(db_path, flush_on_insert=False, storage=CSVStorage)

    def insert(self, timestamp: float, tags: dict, fields: dict):
        timestamp_datetime = datetime.fromtimestamp(timestamp)
        datapoint = Point(time=timestamp_datetime, tags=tags, fields=fields)
        self.db.insert(datapoint, compact_key_prefixes=True)

    def get_lastest_timestamp(self):
        time_query = TimeQuery()
        timestamp = datetime.fromtimestamp(math.floor(time.time()))
        return self.db.search(time_query >= timestamp)
