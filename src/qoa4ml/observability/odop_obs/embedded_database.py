import math
import time
from datetime import datetime

from tinyflux import Point, TimeQuery, TinyFlux


class EmbeddedDatabase:
    def __init__(self, db_path: str) -> None:
        self.db = TinyFlux(db_path)

    def insert(self, timestamp: float, tags: dict, fields: dict):
        timestamp_datetime = datetime.fromtimestamp(timestamp)
        datapoint = Point(time=timestamp_datetime, tags=tags, fields=fields)
        self.db.insert(datapoint, compact_key_prefixes=True)

    def get_lastest_timestamp(self):
        time_query = TimeQuery()
        timestamp = datetime.fromtimestamp(math.floor(time.time()))
        return self.db.search(time_query >= timestamp)
