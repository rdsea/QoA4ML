import json
from unittest.mock import MagicMock, patch

import pytest

from qoa4ml.config.configs import NodeAggregatorConfig, SocketCollectorConfig
from qoa4ml.lang.datamodel_enum import EnvironmentEnum
from qoa4ml.observability.odop_obs.node_aggregator import NodeAggregator


@pytest.fixture
def socket_config():
    return SocketCollectorConfig(host="127.0.0.1", port=5000, backlog=5, bufsize=4096)


@pytest.fixture
def unit_conversion():
    return {
        "frequency": {"GHz": 1.0, "MHz": 0.001},
        "mem": {"GB": 1.0, "MB": 0.001},
        "cpu": {"usage": {"%": 1.0}},
        "gpu": {"usage": {"%": 1.0}},
    }


@pytest.fixture
def aggregator_config(socket_config, unit_conversion):
    return NodeAggregatorConfig(
        socket_collector_config=socket_config,
        environment=EnvironmentEnum.hpc,
        query_method="GET",
        data_separator=".",
        unit_conversion=unit_conversion,
    )


@pytest.fixture
def aggregator(aggregator_config, tmp_path):
    with (
        patch(
            "qoa4ml.observability.odop_obs.node_aggregator.SocketCollector"
        ) as mock_collector,
        patch("qoa4ml.observability.odop_obs.node_aggregator.make_folder"),
        patch(
            "qoa4ml.observability.odop_obs.node_aggregator.EmbeddedDatabase"
        ) as mock_db,
    ):
        mock_collector.return_value = MagicMock()
        mock_db.return_value = MagicMock()
        agg = NodeAggregator(config=aggregator_config, odop_path=tmp_path)
        return agg


class TestNodeAggregatorInit:
    def test_init_sets_config(self, aggregator, aggregator_config):
        assert aggregator.config is aggregator_config

    def test_init_sets_unit_conversion(self, aggregator, unit_conversion):
        assert aggregator.unit_conversion == unit_conversion

    def test_init_sets_environment(self, aggregator):
        assert aggregator.environment == EnvironmentEnum.hpc

    def test_init_creates_router(self, aggregator):
        assert aggregator.router is not None

    def test_init_creates_embedded_database(self, aggregator):
        assert aggregator.embedded_database is not None

    def test_init_creates_node_name(self, aggregator):
        assert isinstance(aggregator.node_name, str)
        assert len(aggregator.node_name) > 0


class TestConvertUnit:
    def test_convert_frequency(self, aggregator):
        report = {"cpu.frequency": "GHz"}
        result = aggregator.convert_unit(report)
        assert result["cpu.frequency"] == 1.0

    def test_convert_mem(self, aggregator):
        report = {"system.mem.usage": "GB"}
        result = aggregator.convert_unit(report)
        assert result["system.mem.usage"] == 1.0

    def test_convert_cpu_usage(self, aggregator):
        report = {"cpu.usage.unit": "%"}
        result = aggregator.convert_unit(report)
        assert result["cpu.usage.unit"] == 1.0

    def test_convert_gpu_usage(self, aggregator):
        report = {"gpu.usage.unit": "%"}
        result = aggregator.convert_unit(report)
        assert result["gpu.usage.unit"] == 1.0

    def test_numeric_values_unchanged(self, aggregator):
        report = {"cpu.percent": 45.0, "memory.used": 2048}
        result = aggregator.convert_unit(report)
        assert result["cpu.percent"] == 45.0
        assert result["memory.used"] == 2048

    def test_empty_report(self, aggregator):
        result = aggregator.convert_unit({})
        assert result == {}


class TestRevertUnit:
    def test_revert_frequency(self, aggregator):
        converted = {"cpu.frequency.unit": 1.0}
        result = aggregator.revert_unit(converted)
        assert result["cpu.frequency.unit"] == "GHz"

    def test_revert_mem(self, aggregator):
        converted = {"system.mem.unit": 1.0}
        result = aggregator.revert_unit(converted)
        assert result["system.mem.unit"] == "GB"

    def test_revert_cpu_usage(self, aggregator):
        converted = {"cpu.usage.unit": 1.0}
        result = aggregator.revert_unit(converted)
        assert result["cpu.usage.unit"] == "%"

    def test_revert_gpu_usage(self, aggregator):
        converted = {"gpu.usage.unit": 1.0}
        result = aggregator.revert_unit(converted)
        assert result["gpu.usage.unit"] == "%"

    def test_no_unit_keys_unchanged(self, aggregator):
        converted = {"cpu.percent": 45.0}
        result = aggregator.revert_unit(converted)
        assert result["cpu.percent"] == 45.0


class TestProcessReport:
    @pytest.fixture
    def dot_aggregator(self, aggregator_config, tmp_path):
        aggregator_config.data_separator = "dot"
        with (
            patch(
                "qoa4ml.observability.odop_obs.node_aggregator.SocketCollector"
            ) as mock_collector,
            patch("qoa4ml.observability.odop_obs.node_aggregator.make_folder"),
            patch(
                "qoa4ml.observability.odop_obs.node_aggregator.EmbeddedDatabase"
            ) as mock_db,
        ):
            mock_collector.return_value = MagicMock()
            mock_db.return_value = MagicMock()
            agg = NodeAggregator(config=aggregator_config, odop_path=tmp_path)
            return agg

    def test_process_hpc_system_report(self, dot_aggregator):
        dot_aggregator.embedded_database = MagicMock()
        report = {
            "type": "system",
            "metadata": {"node_name": "edge1"},
            "timestamp": 1700000000.0,
            "cpu": {"percent": 45.0},
        }
        dot_aggregator.process_report(json.dumps(report))
        dot_aggregator.embedded_database.insert.assert_called_once()
        call_args = dot_aggregator.embedded_database.insert.call_args
        assert call_args[0][0] == 1700000000.0
        assert call_args[0][1]["type"] == "node"

    def test_process_hpc_process_report(self, dot_aggregator):
        dot_aggregator.embedded_database = MagicMock()
        report = {
            "type": "process",
            "metadata": {"pid": 1234},
            "timestamp": 1700000000.0,
            "cpu": {"percent": 12.0},
        }
        dot_aggregator.process_report(json.dumps(report))
        dot_aggregator.embedded_database.insert.assert_called_once()
        call_args = dot_aggregator.embedded_database.insert.call_args
        assert call_args[0][1]["type"] == "process"

    def test_process_unknown_type_logs_error(self, dot_aggregator):
        dot_aggregator.embedded_database = MagicMock()
        report = {
            "type": "unknown",
            "metadata": {},
            "timestamp": 1700000000.0,
        }
        dot_aggregator.process_report(json.dumps(report))
        dot_aggregator.embedded_database.insert.assert_not_called()


class TestGetLatestTimestamp:
    @pytest.fixture
    def dot_aggregator(self, aggregator_config, tmp_path):
        aggregator_config.data_separator = "dot"
        with (
            patch(
                "qoa4ml.observability.odop_obs.node_aggregator.SocketCollector"
            ) as mock_collector,
            patch("qoa4ml.observability.odop_obs.node_aggregator.make_folder"),
            patch(
                "qoa4ml.observability.odop_obs.node_aggregator.EmbeddedDatabase"
            ) as mock_db,
        ):
            mock_collector.return_value = MagicMock()
            mock_db.return_value = MagicMock()
            agg = NodeAggregator(config=aggregator_config, odop_path=tmp_path)
            return agg

    def test_get_latest_timestamp_returns_list(self, dot_aggregator):
        mock_point = MagicMock()
        mock_point.tags = {"type": "node"}
        mock_point.fields = {"cpu": 45.0}
        from datetime import datetime

        mock_point.time = datetime(2024, 1, 1)
        dot_aggregator.embedded_database = MagicMock()
        dot_aggregator.embedded_database.get_latest_timestamp.return_value = [
            mock_point
        ]
        result = dot_aggregator.get_latest_timestamp()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_latest_timestamp_empty(self, aggregator):
        aggregator.embedded_database = MagicMock()
        aggregator.embedded_database.get_latest_timestamp.return_value = []
        result = aggregator.get_latest_timestamp()
        assert result == []
