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
    cfg = DockerProbeConfig(probe_type="docker", frequency=1, container_list=["x"])
    assert cfg.require_register is False
    assert cfg.log_latency_flag is False
    assert cfg.container_list == ["x"]


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


# ============================================================
# Second-round post-review fixes (High + Medium + Low)
# ============================================================


# --- High: kafka_collector subscribes + dispatches + caps ---


def test_kafka_collector_subscribes_and_dispatches(monkeypatch):
    """Regression: previous loop polled without subscribing and dropped all traffic."""
    fake_confluent = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "confluent_kafka", fake_confluent)
    import importlib

    import qoa4ml.collector.kafka_collector as kc

    importlib.reload(kc)

    from qoa4ml.config.configs import KafkaCollectorConfig

    collector = kc.KafkaCollector(
        KafkaCollectorConfig(
            topic="t1",
            broker_url="localhost:9092",
            group_id="g1",
        )
    )

    # Drive one poll returning a valid message, then stop the loop.
    sample = MagicMock()
    sample.error.return_value = None
    sample.value.return_value = b'{"x": 1}'

    def poll(_timeout):
        collector.running = False  # exit after first iteration
        return sample

    collector.consumer.poll = poll
    collector.consumer.subscribe = MagicMock()
    collector.consumer.close = MagicMock()

    with patch.object(kc, "qoa_logger") as logger:
        collector.start_collecting()
        collector.consumer.subscribe.assert_called_once_with(["t1"])
        collector.consumer.close.assert_called_once()
        # Payload logged at DEBUG only.
        assert not any('{"x": 1}' in str(call) for call in logger.info.call_args_list)
    importlib.reload(kc)


def test_kafka_collector_drops_oversize_frame(monkeypatch):
    fake_confluent = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "confluent_kafka", fake_confluent)
    import importlib

    import qoa4ml.collector.kafka_collector as kc

    importlib.reload(kc)

    from qoa4ml.config.configs import KafkaCollectorConfig

    collector = kc.KafkaCollector(
        KafkaCollectorConfig(topic="t", broker_url="x", group_id="g")
    )
    body = b"x" * (kc._MAX_FRAME_BYTES + 1)
    with patch.object(kc, "qoa_logger") as logger:
        collector.on_request(None, None, None, body)
        logger.error.assert_called_once()
    importlib.reload(kc)


# --- High: general_application_report guard + observe_inference wrap ---


def _general_report_harness():
    from qoa4ml.reports.general_application_report import GeneralApplicationReport

    info = ClientInfo(
        name="t",
        username="u",
        user_id="1",
        instance_id="b6f83293-cf67-44dd-a7b5-77229d384012",
        instance_name="i",
        stage_id="inference",
        functionality="REST",
        application_name="app",
        role="ml",
    )
    return GeneralApplicationReport(info)


def test_general_report_process_previous_empty_metrics_no_crash():
    """Regression: metrics[-1] on empty list raised IndexError."""
    report = _general_report_harness()
    report.process_previous_report(
        {"metadata": {}, "metrics": []}
    )  # no metrics key — must not raise


def test_general_report_observe_inference_scalar_is_wrapped():
    """Regression: observe_inference with a non-list value ValidationError'd."""
    report = _general_report_harness()
    report.observe_inference(0.42)
    fm = report.report.metrics[-1]
    assert fm.records == [0.42]


# --- High: NONE_RATIO inverted formula ---


def test_eva_none_ratio_is_actually_the_none_fraction():
    """Regression: NONE_RATIO used to return the VALID fraction."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("pandas", reason="dataquality_utils needs pandas")
    from qoa4ml.lang.attributes import DataQualityEnum
    from qoa4ml.utils.dataquality_utils import eva_none

    arr = np.array([1.0, 2.0, np.nan, np.nan])
    out = eva_none(arr)
    assert out is not None
    # 2/4 are NaN → 50%.
    assert out[DataQualityEnum.NONE_RATIO] == pytest.approx(50.0)


def test_eva_none_ratio_empty_dataset_is_zero():
    np = pytest.importorskip("numpy")
    pytest.importorskip("pandas", reason="dataquality_utils needs pandas")
    from qoa4ml.lang.attributes import DataQualityEnum
    from qoa4ml.utils.dataquality_utils import eva_none

    out = eva_none(np.array([], dtype=float))
    assert out is not None
    assert out[DataQualityEnum.NONE_RATIO] == 0.0


# --- High: get_process_allowed_memory divide-by-zero ---


def test_get_process_allowed_memory_no_tasks_returns_raw_limit(monkeypatch, tmp_path):
    """Regression: dividing by zero killed the probe each tick."""
    import qoa4ml.utils.qoa_utils as qu

    # Force v2 path into a fake cgroup tree we control.
    qu.get_cgroup_version.cache_clear()
    monkeypatch.setattr(qu, "get_cgroup_version", lambda: "v2")
    fake_proc = tmp_path / "proc_cgroup"
    fake_proc.write_text("0::/unit.slice\n", encoding="utf-8")
    fake_sys = tmp_path / "sys" / "fs" / "cgroup" / "unit.slice"
    fake_sys.mkdir(parents=True)
    (fake_sys / "memory.max").write_text("12345", encoding="utf-8")

    real_open = open

    def fake_open(path, *args, **kwargs):
        path_str = str(path)
        if path_str == "/proc/self/cgroup":
            return real_open(fake_proc, *args, **kwargs)
        if path_str == "/sys/fs/cgroup/unit.slice/memory.max":
            return real_open(fake_sys / "memory.max", *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(qu.glob, "glob", lambda _: [])  # zero tasks
    # Must return the raw limit rather than raising ZeroDivisionError.
    assert qu.get_process_allowed_memory() == 12345.0


# --- High: AmqpConnector publish is thread-safe / raises on final failure ---


def test_amqp_connector_raises_on_final_publish_failure():
    """Regression: silent drop was misleading for user-initiated publishes."""
    import qoa4ml.connector.amqp_connector as ac

    # Build a connector skeleton without touching pika.
    connector = ac.AmqpConnector.__new__(ac.AmqpConnector)
    connector.config = MagicMock()
    connector.exchange_name = "ex"
    connector.exchange_type = "topic"
    connector.out_routing_key = "rk"
    connector.log_flag = False
    connector.health_check_disable = True
    import threading as _threading

    connector._publish_lock = _threading.Lock()

    # Fake pika channel that fails then fails again on retry.
    import pika.exceptions as pe

    fake_channel = MagicMock()
    fake_channel.basic_publish.side_effect = pe.AMQPConnectionError()
    fake_channel.exchange_declare = MagicMock()
    connector.out_channel = fake_channel
    connector.out_connection = MagicMock()
    connector.out_connection.is_closed = False

    # Skip the reconnect's real work.
    connector.create_connection = MagicMock()

    with pytest.raises(ac.AmqpPublishError):
        connector.send_report("payload")


# --- High: validate_probe_type tolerates ProbeConfig instances ---


def test_validate_probe_type_accepts_probe_instances():
    """Regression: .get() on a ProbeConfig instance raised AttributeError."""
    from qoa4ml.config.configs import ClientConfig, ProcessProbeConfig

    probe = ProcessProbeConfig(probe_type="process", frequency=1, pid=None)
    # Mixed inputs: a raw dict and a ProbeConfig instance. No crash.
    cfg = ClientConfig.model_validate(
        {
            "client": {"name": "c"},
            "probes": [
                {"probe_type": "system", "frequency": 1},
                probe,
            ],
        }
    )
    assert cfg.probes is not None
    assert len(cfg.probes) == 2


# --- Medium: Probe double-start stops the old timer ---


def test_probe_double_start_stops_prior_timer():
    """Regression: start_reporting() twice leaked the prior daemon thread."""
    from qoa4ml.config.configs import SystemProbeConfig
    from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

    with (
        patch("qoa4ml.probes.system_monitoring_probe.get_sys_cpu_metadata") as cpu,
        patch("qoa4ml.probes.system_monitoring_probe.get_sys_mem") as mem,
        patch("qoa4ml.probes.system_monitoring_probe.find_igpu", return_value={}),
        patch(
            "qoa4ml.probes.system_monitoring_probe.get_sys_gpu_metadata",
            return_value={},
        ),
    ):
        cpu.return_value = {"cores": 1}
        mem.return_value = {"total": 1, "used": 1}
        probe = SystemMonitoringProbe(
            SystemProbeConfig(probe_type="system", frequency=1, node_name="n"),
            MagicMock(),
        )
        with patch("qoa4ml.probes.probe.RepeatedTimer") as timer_cls:
            first = MagicMock()
            second = MagicMock()
            timer_cls.side_effect = [first, second]
            probe.start_reporting()
            probe.start_reporting()  # must stop `first` before overwriting
            first.stop.assert_called_once()
        probe.stop_reporting()


# --- Medium: MLReport.combine_stage_report deep-copies current stages ---


def test_combine_stage_report_deep_copies_current():
    from uuid import UUID

    from qoa4ml.lang.common_models import Metric
    from qoa4ml.reports.ml_report_model import StageReport

    report = MLReport(
        ClientInfo(
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
    )
    instance = UUID("b6f83293-cf67-44dd-a7b5-77229d384012")
    metric = Metric(metric_name=ServiceQualityEnum.RESPONSE_TIME, records=[0.1])
    current: dict[str, StageReport] = {
        "s1": StageReport(
            name="s1",
            metrics={ServiceQualityEnum.RESPONSE_TIME: {instance: metric}},
        )
    }
    combined = report.combine_stage_report(current, {})
    # Mutating the returned stage must not touch ``current``.
    combined["s1"].metrics[ServiceQualityEnum.RESPONSE_TIME][instance].records.append(
        99
    )
    assert current["s1"].metrics[ServiceQualityEnum.RESPONSE_TIME][
        instance
    ].records == [0.1]


# --- Medium: observe_metric stores a deep copy of the caller's metric ---


def test_observe_metric_deep_copies_caller_metric():
    """Regression: caller mutations to the Metric leaked into stored report."""
    from qoa4ml.lang.common_models import Metric
    from qoa4ml.lang.datamodel_enum import ReportTypeEnum

    report = MLReport(
        ClientInfo(
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
    )
    m = Metric(metric_name=ServiceQualityEnum.RESPONSE_TIME, records=[0.1])
    report.observe_metric(ReportTypeEnum.service, "s1", m)
    # Mutate the caller's metric after observing; stored metric must be intact.
    m.records.append(99)
    stored = next(iter(report.report.service["s1"].metrics.values()))
    stored_metric = next(iter(stored.values()))
    assert stored_metric.records == [0.1]


# --- Medium: ClientInfo.set_config rejects unknown AND invalid values ---


def test_qoa_client_strict_mode_raises_when_no_connector_initiated():
    """Regression: silent degraded mode was project fail-fast violation."""
    from qoa4ml.qoa_client import QoaClient

    with pytest.raises(RuntimeError, match="no connectors"):
        QoaClient(
            config_dict={
                "client": {"name": "c"},
                # deliberately no connector, no registration_url
            },
            strict=True,
        )


# --- Medium: convert_unit tolerates unknown unit strings ---


def test_node_aggregator_convert_unit_tolerates_unknown_unit(tmp_path):
    from qoa4ml.config.configs import NodeAggregatorConfig, SocketCollectorConfig
    from qoa4ml.observability.odop_obs.node_aggregator import NodeAggregator

    cfg = NodeAggregatorConfig(
        socket_collector_config=SocketCollectorConfig(
            host="127.0.0.1", port=0, backlog=1, bufsize=256
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
    aggr = NodeAggregator(cfg, tmp_path)
    # Unknown unit passes through — no KeyError.
    out = aggr.convert_unit({"cpu.usage.unit": "UNKNOWN", "metadata.mem": "GB"})
    assert out["cpu.usage.unit"] == "UNKNOWN"
    # Metadata keys are skipped entirely.
    assert out["metadata.mem"] == "GB"


# --- Medium: dataquality_utils still imports without pandas/PIL extras ---


def test_dataquality_utils_imports_without_pandas():
    """Regression: unconditional pandas/PIL imports broke core installs.

    Manages sys.modules manually via try/finally so the teardown ordering
    guarantees we reload with the real pandas/PIL restored before any
    subsequent test runs.
    """
    import importlib
    import sys

    import qoa4ml.utils.dataquality_utils as dq

    saved_pd = sys.modules.get("pandas", ...)
    saved_pil = sys.modules.get("PIL", ...)
    try:
        sys.modules["pandas"] = None  # type: ignore[assignment]
        sys.modules["PIL"] = None  # type: ignore[assignment]
        importlib.reload(dq)
        # Module import must succeed even when extras are absent.
        assert dq.pd is None
        assert dq.Image is None
    finally:
        if saved_pd is ...:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = saved_pd  # type: ignore[assignment]
        if saved_pil is ...:
            sys.modules.pop("PIL", None)
        else:
            sys.modules["PIL"] = saved_pil  # type: ignore[assignment]
        importlib.reload(dq)  # restore real pd/Image for subsequent tests


# --- Medium: mqtt_connector v2 on_connect signature accepts the right args ---


def test_mqtt_on_connect_accepts_v2_signature(monkeypatch):
    """Regression: paho VERSION2 callback sig mismatch silently lost subscriptions."""
    import qoa4ml.connector.mqtt_connector as mc

    # Force the module-level ``mqtt`` alias to a MagicMock so the connector
    # can be constructed regardless of whether paho-mqtt is installed.
    fake_mqtt = MagicMock()
    fake_mqtt.Client.return_value = MagicMock()
    monkeypatch.setattr(mc, "mqtt", fake_mqtt)

    config = MagicMock()
    config.out_queue = "pub"
    config.in_queue = "sub"
    config.client_id = "c"
    config.broker_url = "localhost"
    config.broker_port = 1883
    config.broker_keepalive = 60

    host = MagicMock()
    connector = mc.MqttConnector(host, config)
    fake_client = MagicMock()
    # Drive the callback with the v2 arity (5 args).
    connector.on_connect(fake_client, None, {}, 0, None)
    fake_client.subscribe.assert_called_once_with("sub")


# --- Low: CHANGELOG test-count claim is plausible ---


def test_regression_file_has_many_tests():
    import re

    this_file = Path(__file__).read_text(encoding="utf-8")
    count = len(re.findall(r"^def test_", this_file, flags=re.MULTILINE))
    assert count >= 30, f"regression count regressed — only {count} tests found"


def test_kafka_collector_tolerates_none_body(monkeypatch):
    """Regression: len(None) raised TypeError in the consumer thread."""
    fake_confluent = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "confluent_kafka", fake_confluent)
    import importlib

    import qoa4ml.collector.kafka_collector as kc

    importlib.reload(kc)

    from qoa4ml.config.configs import KafkaCollectorConfig

    collector = kc.KafkaCollector(
        KafkaCollectorConfig(topic="t", broker_url="x", group_id="g")
    )
    # Must return cleanly without raising — tombstone / keyed-null records.
    collector.on_request(None, None, None, None)
    importlib.reload(kc)


def test_general_report_observe_inference_accepts_tuple():
    """Regression: narrowing to list-only dropped tuples silently."""
    report = _general_report_harness()
    report.observe_inference((0.1, 0.2, 0.3))
    fm = report.report.metrics[-1]
    assert fm.records == [0.1, 0.2, 0.3]


def test_rohe_agent_stop_restart_cycle_keeps_mongo_alive():
    """Regression: stop() used to close MongoClient, breaking restart()."""
    pytest.importorskip("pymongo", reason="rohe_Agent needs pymongo")
    pytest.importorskip("rohe_Agent", reason="rohe sibling package not on sys.path")

    from importlib import import_module
    from unittest.mock import patch as mpatch

    rohe_agent_module = import_module("rohe_Agent")

    fake_collector = MagicMock()
    fake_mongo = MagicMock()

    with mpatch.object(rohe_agent_module, "AmqpCollector", return_value=fake_collector):
        with mpatch("pymongo.MongoClient", return_value=fake_mongo):
            agent = rohe_agent_module.Rohe_Agent(
                {
                    "collector": {"amqp_collector": {"conf": MagicMock()}},
                    "database": {
                        "url": "mongodb://localhost",
                        "db_name": "db",
                        "metric_collection": "c",
                    },
                }
            )

    # stop() is a pause — mongo_client.close() MUST NOT be called.
    agent.stop()
    fake_mongo.close.assert_not_called()
    # shutdown() is the terminal teardown.
    agent.shutdown()
    fake_mongo.close.assert_called_once()
