import time
from unittest.mock import MagicMock, patch

import pytest

from qoa4ml.config.configs import (
    AMQPConnectorConfig,
    ClientInfo,
    ConnectorConfig,
    DebugConnectorConfig,
)
from qoa4ml.connector.amqp_connector import AmqpConnector
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.connector.debug_connector import DebugConnector
from qoa4ml.lang.attributes import DataQualityEnum, ServiceQualityEnum
from qoa4ml.lang.datamodel_enum import ServiceAPIEnum
from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_reports import MLReport


def _make_config_dict(**overrides):
    base = {
        "client": {
            "name": "test_client",
            "username": "testuser",
            "user_id": "42",
            "instance_id": "b6f83293-cf67-44dd-a7b5-77229d384012",
            "instance_name": "test_instance",
            "stage_id": "gateway",
            "functionality": "REST",
            "application_name": "unit_test_app",
            "role": "ml",
            "run_id": "run1",
            "custom_info": {"key": "value"},
        },
        "connector": [
            {
                "name": "debug_connector",
                "connector_class": "Debug",
                "config": {"silence": True},
            }
        ],
    }
    base.update(overrides)
    return base


def _make_client(**overrides):
    return QoaClient(report_cls=MLReport, config_dict=_make_config_dict(**overrides))


class TestQoaClientInit:
    def test_init_from_dict_config(self):
        client = _make_client()
        assert client.client_config.name == "test_client"
        assert client.stage_id == "gateway"
        assert client.functionality == "REST"
        assert client.timer_flag is False
        assert client.inference_flag is False

    def test_init_creates_debug_connector(self):
        client = _make_client()
        assert "debug_connector" in client.connector_list
        assert isinstance(client.connector_list["debug_connector"], DebugConnector)

    def test_init_sets_default_connector(self):
        client = _make_client()
        assert client.default_connector == "debug_connector"

    def test_init_no_connector_sets_default_to_none(self):
        config = _make_config_dict()
        config.pop("connector")
        client = QoaClient(report_cls=MLReport, config_dict=config)
        assert client.default_connector is None
        assert client.connector_list == {}

    def test_init_instance_id_from_config(self):
        client = _make_client()
        assert "b6f83293-cf67-44dd-a7b5-77229d384012" in str(client.instance_id)

    def test_init_instance_id_from_env(self):
        env_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch.dict("os.environ", {"INSTANCE_ID": env_id}):
            client = _make_client()
        assert str(client.instance_id) == env_id

    def test_init_auto_generates_instance_id(self):
        config = _make_config_dict()
        config["client"]["instance_id"] = ""
        with patch.dict("os.environ", {}, clear=True):
            client = QoaClient(report_cls=MLReport, config_dict=config)
        assert client.instance_id is not None
        assert len(str(client.instance_id)) > 0


class TestInitConnector:
    def test_init_debug_connector(self):
        client = _make_client()
        config = ConnectorConfig(
            name="test_debug",
            connector_class=ServiceAPIEnum.debug,
            config=DebugConnectorConfig(silence=True),
        )
        connector = client.init_connector(config)
        assert isinstance(connector, DebugConnector)

    @patch("qoa4ml.qoa_client.AmqpConnector")
    def test_init_amqp_connector(self, mock_amqp_cls):
        mock_amqp_cls.return_value = MagicMock(spec=BaseConnector)
        client = _make_client()
        amqp_config = AMQPConnectorConfig(
            end_point="localhost",
            exchange_name="test_exchange",
            exchange_type="topic",
            out_routing_key="test.key",
        )
        config = ConnectorConfig(
            name="test_amqp",
            connector_class=ServiceAPIEnum.amqp,
            config=amqp_config,
        )
        connector = client.init_connector(config)
        mock_amqp_cls.assert_called_once_with(amqp_config)
        assert connector is mock_amqp_cls.return_value

    def test_init_connector_unsupported_raises(self):
        client = _make_client()
        config = ConnectorConfig(
            name="bad",
            connector_class=ServiceAPIEnum.rest,
            config=None,
        )
        with pytest.raises(
            RuntimeError, match="Connector config is not of correct type"
        ):
            client.init_connector(config)


class TestObserveMetric:
    def test_observe_metric_category_0_service(self):
        client = _make_client()
        client.observe_metric(ServiceQualityEnum.RESPONSE_TIME, 1.5, category=0)
        report = client.report(reset=False)
        assert "gateway" in report["service"]

    def test_observe_metric_category_1_data(self):
        client = _make_client()
        client.observe_metric(DataQualityEnum.ACCURACY, 0.95, category=1)
        report = client.report(reset=False)
        assert "gateway" in report["data"]

    def test_observe_metric_category_2_security(self):
        client = _make_client()
        with pytest.raises(ValueError, match="Can't handle report type"):
            client.observe_metric("some_metric", 1.0, category=2)

    def test_observe_metric_invalid_category_raises(self):
        client = _make_client()
        with pytest.raises(RuntimeError, match="Report type not supported"):
            client.observe_metric("metric", 1.0, category=99)

    def test_observe_metric_with_description(self):
        client = _make_client()
        client.observe_metric(
            ServiceQualityEnum.AVAILABILITY, 99.9, category=0, description="uptime"
        )
        report = client.report(reset=False)
        assert "gateway" in report["service"]


class TestObserveInference:
    def test_observe_inference(self):
        client = _make_client()
        client.observe_inference({"class": "cat", "confidence": 0.9})
        report = client.report(reset=False)
        assert len(report["ml_inference"]) == 1

    def test_observe_inference_metric(self):
        client = _make_client()
        client.observe_inference_metric(ServiceQualityEnum.RESPONSE_TIME, 0.5)
        report = client.report(reset=False)
        assert len(report["ml_inference"]) == 1


class TestTimer:
    def test_timer_start(self):
        client = _make_client()
        result = client.timer()
        assert result == {}
        assert client.timer_flag is True

    def test_timer_stop(self):
        client = _make_client()
        client.timer()
        time.sleep(0.01)
        result = client.timer()
        assert client.timer_flag is False
        assert "startTime" in result
        assert "responseTime" in result
        assert result["responseTime"] > 0

    def test_timer_records_response_time_metric(self):
        client = _make_client()
        client.timer()
        client.timer()
        report = client.report(reset=False)
        assert "gateway" in report["service"]


class TestImportPreviousReport:
    def _make_previous_report(self):
        return {
            "service": {},
            "data": {},
            "ml_inference": {},
            "metadata": {},
        }

    def test_import_single_report(self):
        client = _make_client()
        prev = self._make_previous_report()
        client.import_previous_report(prev)

    def test_import_list_of_reports(self):
        client = _make_client()
        reports = [self._make_previous_report(), self._make_previous_report()]
        client.import_previous_report(reports)


class TestReport:
    def test_report_returns_dict(self):
        client = _make_client()
        result = client.report()
        assert isinstance(result, dict)

    def test_report_with_reset_true_clears_state(self):
        client = _make_client()
        client.observe_metric(ServiceQualityEnum.AVAILABILITY, 99.0, category=0)
        report1 = client.report(reset=True)
        report2 = client.report(reset=True)
        assert "gateway" in report1["service"]
        assert report2["service"] == {}

    def test_report_with_reset_false_keeps_state(self):
        client = _make_client()
        client.observe_metric(ServiceQualityEnum.AVAILABILITY, 99.0, category=0)
        report1 = client.report(reset=False)
        report2 = client.report(reset=False)
        assert "gateway" in report1["service"]
        assert "gateway" in report2["service"]

    def test_report_with_submit_true_calls_connector(self):
        client = _make_client()
        mock_connector = MagicMock(spec=BaseConnector)
        client.connector_list["debug_connector"] = mock_connector
        client.observe_metric(ServiceQualityEnum.AVAILABILITY, 99.0, category=0)
        client.report(submit=True)
        time.sleep(0.1)
        mock_connector.send_report.assert_called_once()

    def test_report_with_submit_no_connector_logs_warning(self):
        config = _make_config_dict()
        config.pop("connector")
        client = QoaClient(report_cls=MLReport, config_dict=config)
        result = client.report(submit=True)
        assert isinstance(result, dict)

    def test_report_custom_report_dict(self):
        client = _make_client()
        custom = {"my_key": "my_value"}
        result = client.report(report=custom)
        assert result["report"] == custom

    def test_report_includes_metadata(self):
        client = _make_client()
        result = client.report()
        assert "metadata" in result


class TestAsynReport:
    def test_asyn_report_default_connector(self):
        client = _make_client()
        mock_connector = MagicMock(spec=BaseConnector)
        client.connector_list["debug_connector"] = mock_connector
        client.asyn_report('{"test": 1}')
        mock_connector.send_report.assert_called_once_with('{"test": 1}')

    def test_asyn_report_explicit_connectors(self):
        client = _make_client()
        mock_conn1 = MagicMock(spec=BaseConnector)
        mock_conn2 = MagicMock(spec=BaseConnector)
        client.asyn_report('{"data": "ok"}', connectors=[mock_conn1, mock_conn2])
        mock_conn1.send_report.assert_called_once_with('{"data": "ok"}')
        mock_conn2.send_report.assert_called_once_with('{"data": "ok"}')

    def test_asyn_report_amqp_checks_connection(self):
        mock_amqp = MagicMock(spec=AmqpConnector)
        mock_amqp.check_connection.return_value = True
        client = _make_client()
        client.asyn_report('{"msg": 1}', connectors=[mock_amqp])
        mock_amqp.check_connection.assert_called_once()
        mock_amqp.send_report.assert_called_once()

    def test_asyn_report_amqp_reconnects_if_disconnected(self):
        mock_amqp = MagicMock(spec=AmqpConnector)
        mock_amqp.check_connection.return_value = False
        client = _make_client()
        client.asyn_report('{"msg": 1}', connectors=[mock_amqp])
        mock_amqp.reconnect.assert_called_once()
        mock_amqp.send_report.assert_called_once()

    def test_asyn_report_no_connector_no_crash(self):
        config = _make_config_dict()
        config.pop("connector")
        client = QoaClient(report_cls=MLReport, config_dict=config)
        client.asyn_report('{"test": 1}')


class TestGetClientConfig:
    def test_returns_client_info(self):
        client = _make_client()
        config = client.get_client_config()
        assert isinstance(config, ClientInfo)
        assert config.name == "test_client"

    def test_returns_correct_stage(self):
        client = _make_client()
        config = client.get_client_config()
        assert config.stage_id == "gateway"
