import time
from datetime import UTC, datetime

import pytest

from qoa4ml.observability.odop_obs.embedded_database import EmbeddedDatabase


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test_db.csv"
    return EmbeddedDatabase(db_path=str(db_path))


class TestEmbeddedDatabaseInit:
    def test_init_creates_database(self, tmp_path):
        db_path = tmp_path / "init_test.csv"
        db = EmbeddedDatabase(db_path=str(db_path))
        assert db.db is not None

    def test_init_with_path(self, db):
        assert db.db is not None


class TestInsert:
    def test_insert_single_datapoint(self, db):
        ts = time.time()
        db.insert(timestamp=ts, tags={"type": "node"}, fields={"cpu": 45.0})
        all_points = db.db.all()
        assert len(all_points) == 1
        assert all_points[0].fields["cpu"] == 45.0

    def test_insert_multiple_datapoints(self, db):
        base_ts = time.time() - 10
        for i in range(5):
            db.insert(
                timestamp=base_ts + i,
                tags={"type": "process", "id": str(i)},
                fields={"cpu": 10.0 + i, "memory": 200.0 + i * 10},
            )
        all_points = db.db.all()
        assert len(all_points) == 5

    def test_insert_preserves_tags(self, db):
        ts = time.time()
        db.insert(
            timestamp=ts,
            tags={"type": "node", "node_name": "edge1"},
            fields={"cpu": 55.0},
        )
        all_points = db.db.all()
        assert all_points[0].tags["type"] == "node"
        assert all_points[0].tags["node_name"] == "edge1"

    def test_insert_preserves_fields(self, db):
        ts = time.time()
        db.insert(
            timestamp=ts,
            tags={"type": "node"},
            fields={"cpu": 72.5, "memory": 1024.0},
        )
        all_points = db.db.all()
        assert all_points[0].fields["cpu"] == 72.5
        assert all_points[0].fields["memory"] == 1024.0

    def test_insert_converts_timestamp_to_datetime(self, db):
        ts = time.time()
        db.insert(timestamp=ts, tags={"type": "test"}, fields={"val": 1.0})
        all_points = db.db.all()
        assert all_points[0].time is not None
        assert isinstance(all_points[0].time, datetime)

    def test_insert_ordering(self, db):
        base_ts = time.time() - 10
        db.insert(timestamp=base_ts, tags={"order": "first"}, fields={"val": 1.0})
        db.insert(timestamp=base_ts + 5, tags={"order": "second"}, fields={"val": 2.0})
        all_points = db.db.all()
        assert all_points[0].tags["order"] == "first"
        assert all_points[1].tags["order"] == "second"


class TestGetLatestTimestamp:
    def test_empty_database_returns_empty_list(self, db):
        result = db.get_latest_timestamp()
        assert result == []

    def test_returns_list_when_data_within_range(self, db):
        utc_now = datetime.now(UTC)
        ts = utc_now.timestamp() - 5
        db.insert(timestamp=ts, tags={"type": "test"}, fields={"val": 42.0})
        result = db.get_latest_timestamp()
        assert isinstance(result, list)
        if len(result) > 0:
            assert result[0].fields["val"] == 42.0
