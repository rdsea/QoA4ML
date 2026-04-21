import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tinyflux import Point, TimeQuery, TinyFlux
from tinyflux.storages import CSVStorage


class EmbeddedDatabase:
    # Window (in seconds) to scan when looking for the latest datapoint.
    # Bounded so `get_latest_timestamp` stays O(window) regardless of DB size.
    DEFAULT_LOOKBACK_SECONDS = 60

    def __init__(self, db_path: Path) -> None:
        self.db = TinyFlux(db_path, flush_on_insert=False, storage=CSVStorage)

    def insert(self, timestamp: float, tags: dict, fields: dict):
        # TinyFlux internally stores Point times as UTC-aware datetimes;
        # inserting a naive localtime worked accidentally on UTC machines
        # but silently broke queries everywhere else.
        timestamp_datetime = datetime.fromtimestamp(timestamp, tz=UTC)
        datapoint = Point(time=timestamp_datetime, tags=tags, fields=fields)
        self.db.insert(datapoint, compact_key_prefixes=True)

    def get_latest_timestamp(self, lookback_seconds: int | None = None):
        """
        Return the most recent datapoint as a one-element list (or [] if none).

        The previous implementation returned ``results[-1:]`` after a
        ``search(time <= now)`` — implicitly depending on row ordering and
        using a naive datetime comparison that was silently empty on
        non-UTC hosts. The query now uses UTC-aware datetimes and picks
        the max-timestamp point explicitly.
        """
        window = lookback_seconds or self.DEFAULT_LOOKBACK_SECONDS
        now = datetime.fromtimestamp(time.time(), tz=UTC)
        time_query = TimeQuery()
        results = self.db.search(
            (time_query <= now) & (time_query > now - timedelta(seconds=window))
        )
        if not results:
            results = self.db.search(time_query <= now)
        if not results:
            return []
        return [max(results, key=lambda point: point.time)]
