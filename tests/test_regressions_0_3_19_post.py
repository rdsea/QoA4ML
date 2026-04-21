"""Regression tests for post-0.3.19 review fixes.

Each test pins behaviour that was wrong before this patch series — see the
top-level review report for context. Tests are grouped by area and named
after the bug they prevent from coming back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qoa4ml.config.configs import (
    ClientInfo,
    DockerProbeConfig,
    ProcessProbeConfig,
    SystemProbeConfig,
)
from qoa4ml.lang.attributes import ServiceQualityEnum
from qoa4ml.lang.datamodel_enum import EnvironmentEnum
from qoa4ml.qoa_client import (
    _ALLOWED_REGISTRATION_SCHEMES,
    _validate_registration_url,
)
from qoa4ml.reports.ml_reports import MLReport

# Make the sibling rohe_ObService package importable so ``importorskip`` can
# see ``rohe_Agent``; without this the rohe HTTP tests skip even when flask
# is installed because the companion module is not on sys.path.
_ROHE_DIR = Path(__file__).resolve().parent.parent / "observability" / "rohe_ObService"
if _ROHE_DIR.is_dir() and str(_ROHE_DIR) not in sys.path:
    sys.path.insert(0, str(_ROHE_DIR))

# --- Critical: pyproject version ---


def test_package_version_matches_changelog() -> None:
    """Regression: 0.3.19 was in the CHANGELOG but pyproject still pinned 0.3.18."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'version = "0.3.19"' in text


# --- High: ProbeConfig defaults ---


def test_process_probe_config_defaults_optional_flags() -> None:
    """Regression: ``require_register`` / ``log_latency_flag`` had no defaults."""
    cfg = ProcessProbeConfig(probe_type="process", frequency=1, pid=None)
    assert cfg.require_register is False
    assert cfg.log_latency_flag is False


def test_system_probe_config_defaults_optional_flags() -> None:
    cfg = SystemProbeConfig(probe_type="system", frequency=1, node_name="local")
    assert cfg.require_register is False
    assert cfg.log_latency_flag is False


def test_docker_probe_config_defaults_optional_flags() -> None:
    cfg = DockerProbeConfig(probe_type="docker", frequency=1, container_name="x")
    assert cfg.require_register is False
    assert cfg.log_latency_flag is False


# --- High: Memory unit labels ---


@patch("qoa4ml.probes.system_monitoring_probe.get_sys_gpu_metadata", return_value={})
@patch("qoa4ml.probes.system_monitoring_probe.find_igpu", return_value={})
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_cpu_metadata",
    return_value={"cores": 1},
)
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_mem",
    return_value={"total": 1024 * 1024 * 1024, "used": 256 * 1024 * 1024},
)
def test_system_mem_unit_label_is_si(_mem, _cpu, _igpu, _gpu) -> None:
    from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

    config = SystemProbeConfig(probe_type="system", frequency=1, node_name="local")
    connector = MagicMock()
    probe = SystemMonitoringProbe(config, connector)
    assert probe.get_mem_metadata()["mem"]["unit"] == "GB"
    assert probe.get_mem_usage()["unit"] == "MB"


# --- High: MLReport handles security ---


def _client_info() -> ClientInfo:
    return ClientInfo(
        name="t",
        username="u",
        user_id="1",
        instance_id="b6f83293-cf67-44dd-a7b5-77229d384012",
        instance_name="i",
        stage_id="s",
        functionality="REST",
        application_name="app",
        role="ml",
    )


def test_ml_report_observe_metric_security_branch() -> None:
    """Regression: ReportTypeEnum.security crashed MLReport with ValueError."""
    report = MLReport(_client_info())
    from qoa4ml.lang.common_models import Metric
    from qoa4ml.lang.datamodel_enum import ReportTypeEnum

    report.observe_metric(
        ReportTypeEnum.security,
        "stage_a",
        Metric(metric_name=ServiceQualityEnum.RELIABILITY, records=[0.99]),
    )
    snapshot = report.generate_report(reset=False)
    assert "stage_a" in snapshot.security


def test_ml_report_combine_stage_report_does_not_mutate_input() -> None:
    """Regression: combine_stage_report() mutated its current_stage_report arg."""
    from uuid import UUID

    from qoa4ml.lang.common_models import Metric
    from qoa4ml.reports.ml_report_model import StageReport

    report = MLReport(_client_info())
    instance = UUID("b6f83293-cf67-44dd-a7b5-77229d384012")
    metric = Metric(metric_name=ServiceQualityEnum.RESPONSE_TIME, records=[0.1])
    current: dict[str, StageReport] = {}
    previous: dict[str, StageReport] = {
        "stage_x": StageReport(
            name="stage_x",
            metrics={ServiceQualityEnum.RESPONSE_TIME: {instance: metric}},
        )
    }
    combined = report.combine_stage_report(current, previous)
    # The input ``current`` dict must not have been touched.
    assert current == {}
    assert "stage_x" in combined


# --- High: NodeAggregator.stop joins cleanly ---


def test_node_aggregator_stop_signals_collector(tmp_path: Path) -> None:
    """Regression: stop() flipped a flag the collector loop never read."""
    from qoa4ml.config.configs import (
        NodeAggregatorConfig,
        SocketCollectorConfig,
    )
    from qoa4ml.observability.odop_obs.node_aggregator import NodeAggregator

    config = NodeAggregatorConfig(
        socket_collector_config=SocketCollectorConfig(
            host="127.0.0.1", port=0, backlog=1, bufsize=1024
        ),
        environment=EnvironmentEnum.edge,
        unit_conversion={
            "frequency": {"Hz": "Hz"},
            "mem": {"MB": "MB"},
            "cpu": {"usage": {"percentage": "%"}},
            "gpu": {"usage": {"percentage": "%"}},
        },
        query_method="GET",
        data_separator=".",
    )
    aggregator = NodeAggregator(config, tmp_path)
    aggregator.collector = MagicMock()
    aggregator.server_thread = MagicMock()
    aggregator.stop()
    aggregator.collector.stop.assert_called_once()
    aggregator.server_thread.join.assert_called_once()


# --- High: registration URL validation ---


def test_validate_registration_url_rejects_file_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported registration URL scheme"):
        _validate_registration_url("file:///etc/passwd")


def test_validate_registration_url_rejects_metadata_address() -> None:
    with pytest.raises(ValueError, match="cloud-metadata"):
        _validate_registration_url("http://169.254.169.254/latest/meta-data/")


def test_validate_registration_url_accepts_https() -> None:
    _validate_registration_url("https://example.com/register")
    assert "https" in _ALLOWED_REGISTRATION_SCHEMES


# --- Medium: PromConnector strict misuse ---


@pytest.fixture
def _prom_module_mock(monkeypatch):
    import sys

    mod = MagicMock()
    monkeypatch.setitem(sys.modules, "prometheus_client", mod)
    # Force re-import so the connector picks up the mock.
    import importlib

    import qoa4ml.connector.prom_connector as pc

    importlib.reload(pc)
    yield mod
    importlib.reload(pc)


def test_prom_connector_rejects_dec_on_non_gauge(_prom_module_mock):
    from qoa4ml.connector.prom_connector import PromConnector

    info = {
        "port": 8000,
        "metric": {
            "x": {
                "Type": "Counter",
                "Prom_name": "x",
                "Description": "x",
            }
        },
    }
    connector = PromConnector(info)
    with pytest.raises(ValueError, match="only Gauge supports dec"):
        connector.dec("x", 1)


def test_prom_connector_rejects_unknown_type(_prom_module_mock):
    from qoa4ml.connector.prom_connector import PromConnector

    with pytest.raises(ValueError, match="unknown metric type"):
        PromConnector(
            {
                "port": 8000,
                "metric": {
                    "x": {"Type": "Mystery", "Prom_name": "x", "Description": "x"}
                },
            }
        )


# --- Medium: AMQP collector size cap ---


def test_amqp_collector_drops_oversize_payload():
    """Regression: unbounded body could OOM consumer thread."""
    import qoa4ml.collector.amqp_collector as ac

    # Build a collector skeleton without touching pika.
    collector = ac.AmqpCollector.__new__(ac.AmqpCollector)
    collector.host_object = None
    body = b"x" * (ac._MAX_FRAME_BYTES + 1)
    with patch.object(ac, "qoa_logger") as logger:
        collector.on_request(MagicMock(), MagicMock(), MagicMock(), body)
        logger.error.assert_called_once()


def test_amqp_collector_size_cap_applies_with_host_object():
    """Regression: the cap was only enforced on the no-host_object path."""
    import qoa4ml.collector.amqp_collector as ac

    host_obj = MagicMock()
    collector = ac.AmqpCollector.__new__(ac.AmqpCollector)
    collector.host_object = host_obj
    body = b"x" * (ac._MAX_FRAME_BYTES + 1)
    with patch.object(ac, "qoa_logger") as logger:
        collector.on_request(MagicMock(), MagicMock(), MagicMock(), body)
        host_obj.message_processing.assert_not_called()
        logger.error.assert_called_once()


# --- Medium: socket collector size cap ---


def test_socket_collector_drops_oversize_payload():
    """Regression: unbounded recv loop allowed memory exhaustion."""
    from qoa4ml.collector.socket_collector import SocketCollector
    from qoa4ml.config.configs import SocketCollectorConfig

    process_report = MagicMock()
    collector = SocketCollector(
        SocketCollectorConfig(host="127.0.0.1", port=0, backlog=1, bufsize=128),
        process_report,
    )

    # Fake a client socket that streams more than the cap in one recv.
    fake_socket = MagicMock()
    payload = b"x" * (collector._MAX_FRAME_BYTES + 16)
    fake_socket.recv.side_effect = [payload, b""]
    collector._handle_client(fake_socket)
    process_report.assert_not_called()


# --- Medium: Kafka connector logs payload at DEBUG only ---


def test_kafka_connector_logs_payload_at_debug(monkeypatch):
    """Regression: full message body was logged at INFO when log=True."""
    fake_module = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "confluent_kafka", fake_module)
    import importlib

    import qoa4ml.connector.kafka_connector as kc

    importlib.reload(kc)

    from qoa4ml.config.configs import KafkaConnectorConfig

    connector = kc.KafkaConnector(
        KafkaConnectorConfig(broker_url="localhost:9092", topic="t"),
        log=True,
    )
    with patch.object(kc, "qoa_logger") as logger:
        connector.send_report('{"x": 1}')
        info_calls = [c for c in logger.info.call_args_list if "x" in str(c)]
        assert info_calls == []
        assert any('{"x": 1}' in str(c) for c in logger.debug.call_args_list)
    importlib.reload(kc)


# --- Medium: rohe_ObService HTTP error codes ---


@pytest.fixture(scope="module")
def _rohe_app():
    """Import rohe_ObService once and register its resource exactly once.

    Flask raises AssertionError if ``add_url_rule`` is called after the first
    request has been served, so tests cannot each register the resource.
    """
    pytest.importorskip("flask", reason="rohe_ObService HTTP tests need flask")
    pytest.importorskip(
        "flask_restful", reason="rohe_ObService HTTP tests need flask_restful"
    )
    pytest.importorskip("rohe_Agent", reason="rohe sibling package not on sys.path")

    from importlib import import_module

    rohe = import_module("rohe_ObService")
    rohe.app.config["TESTING"] = True
    if "rohe_obs" not in {r.endpoint for r in rohe.api.resources}:
        rohe.api.add_resource(
            rohe.Rohe_ObService,
            "/registration",
            endpoint="rohe_obs",
            resource_class_kwargs={
                "database": {},
                "connector": {"c1": {"conf": {}}},
                "collector": {"c1": {"conf": {}}},
            },
        )
    return rohe


@pytest.fixture
def _rohe_client(_rohe_app):
    """Flask test client with module-level registration state cleared each test."""
    _rohe_app.application_list.clear()
    _rohe_app.agent_list.clear()
    return _rohe_app, _rohe_app.app.test_client()


def test_rohe_obs_service_error_response_uses_400(_rohe_client):
    """Regression: missing application_name returned 200 OK."""
    _, client = _rohe_client
    resp = client.post("/registration", data="not-json", content_type="text/plain")
    assert resp.status_code == 400


# --- Medium: load_config returns None on error path (no dead branch) ---


def test_load_config_returns_none_on_missing_file(tmp_path):
    """Regression: dead unreachable ``return None`` line removed."""
    from qoa4ml.utils.qoa_utils import load_config

    assert load_config(str(tmp_path / "nope.json")) is None


def test_load_config_warns_on_unsupported_format(tmp_path):
    from qoa4ml.utils.qoa_utils import load_config

    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    assert load_config(str(f)) is None


# --- Medium: subprocess uses shutil.which ---


def test_get_cgroup_version_uses_shutil_which(monkeypatch):
    """Regression: relying on PATH lookup of ``mount`` is a CWE-427 footgun."""
    import qoa4ml.utils.qoa_utils as qu

    # Clear lru_cache and force which() to return None — should default v1.
    qu.get_cgroup_version.cache_clear()
    monkeypatch.setattr(qu.shutil, "which", lambda _: None)
    assert qu.get_cgroup_version() == "v1"
    qu.get_cgroup_version.cache_clear()


# --- mlquality: numpy cast ---


def test_classification_confidence_returns_python_float():
    """Regression: numpy scalars broke downstream JSON serialization."""
    np = pytest.importorskip("numpy")
    from qoa4ml.probes.mlquality import classification_confidence

    arr = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    out = classification_confidence(arr, score=True)
    assert isinstance(out["confidence"], float)
    json.dumps(out)  # must not raise


# --- Embedded DB exponential backoff bound ---


def test_embedded_database_lookback_is_bounded(tmp_path):
    """Regression: previous fallback was an unbounded full-table scan."""
    from qoa4ml.observability.odop_obs.embedded_database import EmbeddedDatabase

    db = EmbeddedDatabase(tmp_path / "metrics.csv")
    # No data inserted: backoff should terminate, not loop forever.
    out = db.get_latest_timestamp()
    assert out == []
    assert db.MAX_LOOKBACK_SECONDS == 24 * 60 * 60


# --- merge_report immutability ---


def test_merge_report_does_not_mutate_inputs():
    """Regression: merge_report mutated both arguments and swallowed errors."""
    from qoa4ml.utils.qoa_utils import merge_report

    a = {"x": 1, "nested": {"a": 1, "b": 2}}
    b = {"y": 2, "nested": {"b": 99, "c": 3}}
    a_snapshot = json.loads(json.dumps(a))
    b_snapshot = json.loads(json.dumps(b))

    merged = merge_report(a, b, prio=True)
    # f_report wins on conflict
    assert merged == {"x": 1, "y": 2, "nested": {"a": 1, "b": 2, "c": 3}}
    # Inputs unchanged
    assert a == a_snapshot
    assert b == b_snapshot


def test_merge_report_prio_false_picks_i_report_value():
    from qoa4ml.utils.qoa_utils import merge_report

    merged = merge_report({"x": 1}, {"x": 2}, prio=False)
    assert merged == {"x": 2}


# --- QoaClient.set_config allow-list ---


def test_set_config_rejects_unknown_field():
    """Regression: arbitrary attribute injection on ClientInfo."""
    from qoa4ml.qoa_client import QoaClient

    client = QoaClient(
        config_dict={
            "client": {
                "name": "t",
                "username": "u",
                "user_id": "1",
                "instance_id": "b6f83293-cf67-44dd-a7b5-77229d384012",
                "instance_name": "i",
                "stage_id": "s",
                "functionality": "REST",
                "application_name": "app",
                "role": "ml",
            },
            "connector": [
                {
                    "name": "debug",
                    "connector_class": "Debug",
                    "config": {"silence": True},
                }
            ],
        },
    )
    with pytest.raises(AttributeError, match="not a known ClientInfo field"):
        client.set_config("totally_made_up_field", "x")
    # Known fields still work
    client.set_config("username", "renamed")
    assert client.client_config.username == "renamed"


# --- rohe_ObService input validation + token ---


def test_rohe_obs_service_rejects_unsafe_application_name(_rohe_client):
    """Regression: unsanitized identifiers flowed into AMQP routing keys / Mongo names."""
    _, client = _rohe_client
    bad_names = ["foo.#", "../etc", "name with spaces", ""]
    for name in bad_names:
        resp = client.post("/registration", json={"application_name": name})
        assert resp.status_code == 400, f"expected 400 for {name!r}"


def test_rohe_obs_service_token_required_when_set(monkeypatch, _rohe_client):
    """Regression: registration was unauthenticated."""
    monkeypatch.setenv("ROHE_OBS_TOKEN", "secret-token")
    _, client = _rohe_client

    # No token: 401
    resp = client.post("/registration", json={"application_name": "good_name"})
    assert resp.status_code == 401

    # Wrong token: 401
    resp = client.post(
        "/registration",
        json={"application_name": "good_name"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


# --- merge_report dict-vs-non-dict asymmetric merge ---


def test_merge_report_dict_vs_non_dict_respects_prio():
    """Pin asymmetric merge contract — prio decides which side wins."""
    from qoa4ml.utils.qoa_utils import merge_report

    f = {"x": {"a": 1}}
    i = {"x": 5}

    # prio=True: f wins → keeps the nested dict
    assert merge_report(f, i, prio=True)["x"] == {"a": 1}
    # prio=False: i wins → returns the scalar
    assert merge_report(f, i, prio=False)["x"] == 5
    # Inputs untouched
    assert f == {"x": {"a": 1}}
    assert i == {"x": 5}


# --- ClientInfo.set_config validates assignment ---


def test_set_config_rejects_invalid_value_type():
    """Regression: set_config bypassed Pydantic validators."""
    from pydantic import ValidationError

    from qoa4ml.qoa_client import QoaClient

    client = QoaClient(
        config_dict={
            "client": {
                "name": "t",
                "username": "u",
                "user_id": "1",
                "instance_id": "b6f83293-cf67-44dd-a7b5-77229d384012",
                "instance_name": "i",
                "stage_id": "s",
                "functionality": "REST",
                "application_name": "app",
                "role": "ml",
            },
            "connector": [
                {
                    "name": "debug",
                    "connector_class": "Debug",
                    "config": {"silence": True},
                }
            ],
        },
    )
    # logging_level is typed int — a non-numeric string must be rejected.
    with pytest.raises(ValidationError):
        client.set_config("logging_level", "not-a-number")


# --- rohe_ObService bearer scheme is case-insensitive (RFC 7235) ---


def test_rohe_obs_token_scheme_case_insensitive(monkeypatch, _rohe_client):
    monkeypatch.setenv("ROHE_OBS_TOKEN", "secret-token")
    rohe, client = _rohe_client
    # Stub Rohe_Agent so the handler doesn't actually touch Mongo/AMQP.
    with patch.object(rohe, "Rohe_Agent", MagicMock(return_value=MagicMock())):
        # Lowercase scheme must work the same as `Bearer`.
        resp = client.post(
            "/registration",
            json={"application_name": "good_name"},
            headers={"Authorization": "bearer secret-token"},
        )
    # Auth accepted, so definitely not 401.
    assert resp.status_code != 401


# --- rohe_ObService 1:1 application→agent invariant ---


def test_rohe_obs_service_reuses_agent_per_application(_rohe_client):
    """Regression: registration for an existing application must reuse, not respawn.

    Pins the 1:1 ``application_id → agent`` invariant.
    """
    rohe, client = _rohe_client

    fake_agent_factory = MagicMock(return_value=MagicMock())
    with patch.object(rohe, "Rohe_Agent", fake_agent_factory):
        for _ in range(5):
            resp = client.post("/registration", json={"application_name": "my_app"})
            assert resp.status_code == 200

    assert fake_agent_factory.call_count == 1
    application_id = rohe.application_list["my_app"]["id"]
    assert len(rohe.agent_list[application_id]) == 1


def test_rohe_obs_service_rolls_back_on_agent_start_failure(_rohe_client):
    """Regression: a failed agent.start() used to brick the application_id."""
    rohe, client = _rohe_client

    failing_agent = MagicMock()
    failing_agent.start.side_effect = RuntimeError("AMQP down")

    with patch.object(rohe, "Rohe_Agent", return_value=failing_agent):
        resp = client.post("/registration", json={"application_name": "flaky_app"})
        assert resp.status_code == 500

    application_id = rohe.application_list["flaky_app"]["id"]
    assert application_id not in rohe.agent_list
