import sys
from unittest.mock import MagicMock

import pytest

mock_prometheus_client = MagicMock()
sys.modules["prometheus_client"] = mock_prometheus_client

from qoa4ml.connector.prom_connector import PromConnector  # noqa: E402


@pytest.fixture
def prom_info():
    return {
        "port": 8000,
        "metric": {
            "cpu_usage": {
                "Type": "Gauge",
                "Prom_name": "cpu_usage_gauge",
                "Description": "CPU usage percentage",
            },
            "request_count": {
                "Type": "Counter",
                "Prom_name": "request_count_total",
                "Description": "Total request count",
            },
            "request_latency": {
                "Type": "Summary",
                "Prom_name": "request_latency_summary",
                "Description": "Request latency",
            },
            "response_size": {
                "Type": "Histogram",
                "Prom_name": "response_size_hist",
                "Description": "Response sizes",
                "Buckets": [10, 50, 100, 500, 1000],
            },
        },
    }


@pytest.fixture(autouse=True)
def reset_prom_mocks():
    mock_prometheus_client.reset_mock()
    mock_prometheus_client.Gauge.return_value = MagicMock()
    mock_prometheus_client.Counter.return_value = MagicMock()
    mock_prometheus_client.Summary.return_value = MagicMock()
    mock_prometheus_client.Histogram.return_value = MagicMock()


class TestPromConnectorInit:
    def test_init_creates_gauge_metric(self, prom_info):
        PromConnector(prom_info)

        mock_prometheus_client.Gauge.assert_called_once_with(
            "cpu_usage_gauge", "CPU usage percentage"
        )

    def test_init_creates_counter_metric(self, prom_info):
        PromConnector(prom_info)

        mock_prometheus_client.Counter.assert_any_call(
            "request_count_total", "Total request count"
        )

    def test_init_creates_summary_metric(self, prom_info):
        PromConnector(prom_info)

        mock_prometheus_client.Summary.assert_called_once_with(
            "request_latency_summary", "Request latency"
        )

    def test_init_creates_histogram_metric(self, prom_info):
        PromConnector(prom_info)

        mock_prometheus_client.Histogram.assert_called_once_with(
            "response_size_hist",
            "Response sizes",
            buckets=(10, 50, 100, 500, 1000),
        )

    def test_init_creates_violation_counters(self, prom_info):
        PromConnector(prom_info)

        violation_calls = [
            c
            for c in mock_prometheus_client.Counter.call_args_list
            if "violation" in str(c)
        ]
        assert len(violation_calls) == 4

    def test_init_starts_http_server(self, prom_info):
        PromConnector(prom_info)

        mock_prometheus_client.start_http_server.assert_called_once_with(8000)

    def test_init_stores_port_and_info(self, prom_info):
        connector = PromConnector(prom_info)

        assert connector.port == 8000
        assert connector.info == prom_info["metric"]


class TestPromConnectorOperations:
    def _make_connector(self, prom_info):
        return PromConnector(prom_info)

    def test_set_gauge(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.set("cpu_usage", 75.5)
        connector.metrics["cpu_usage"]["metric"].set.assert_called_once_with(75.5)

    def test_set_counter_calls_inc(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.set("request_count", 5)
        connector.metrics["request_count"]["metric"].inc.assert_called_with(5)

    def test_set_summary_calls_observe(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.set("request_latency", 0.5)
        connector.metrics["request_latency"]["metric"].observe.assert_called_with(0.5)

    def test_set_histogram_calls_observe(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.set("response_size", 250)
        connector.metrics["response_size"]["metric"].observe.assert_called_with(250)

    def test_inc_gauge(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.inc("cpu_usage", 10)
        connector.metrics["cpu_usage"]["metric"].inc.assert_called_with(10)

    def test_inc_counter(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.inc("request_count", 3)
        connector.metrics["request_count"]["metric"].inc.assert_called_with(3)

    def test_inc_default_value(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.inc("cpu_usage")
        connector.metrics["cpu_usage"]["metric"].inc.assert_called_with(1)

    def test_dec_gauge(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.dec("cpu_usage", 5)
        connector.metrics["cpu_usage"]["metric"].dec.assert_called_with(5)

    def test_dec_non_gauge_does_nothing(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.dec("request_count", 5)
        connector.metrics["request_count"]["metric"].dec.assert_not_called()

    def test_observe_summary(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.observe("request_latency", 1.5)
        connector.metrics["request_latency"]["metric"].observe.assert_called_with(1.5)

    def test_observe_histogram(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.observe("response_size", 42)
        connector.metrics["response_size"]["metric"].observe.assert_called_with(42)

    def test_inc_violation(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.inc_violation("cpu_usage", 2)
        connector.metrics["cpu_usage"]["violation"].inc.assert_called_with(2)

    def test_update_violation_count(self, prom_info):
        connector = self._make_connector(prom_info)
        connector.update_violation_count()
        assert mock_prometheus_client.generate_latest.call_count == len(
            prom_info["metric"]
        )
