import json
from unittest.mock import patch

from qoa4ml.config.configs import ClientInfo, DebugConnectorConfig, SystemProbeConfig
from qoa4ml.connector.debug_connector import DebugConnector
from qoa4ml.lang.datamodel_enum import EnvironmentEnum


def _make_probe_config(**overrides):
    defaults = {
        "probe_type": "system",
        "frequency": 1,
        "require_register": False,
        "log_latency_flag": False,
        "environment": EnvironmentEnum.hpc,
        "node_name": "test-node",
    }
    defaults.update(overrides)
    return SystemProbeConfig(**defaults)


def _make_client_info():
    return ClientInfo(
        name="test_client",
        username="tester",
        user_id="1",
        instance_id="b6f83293-cf67-44dd-a7b5-77229d384012",
        stage_id="gateway",
        functionality="REST",
        application_name="test_app",
        role="ml",
    )


def _make_connector():
    return DebugConnector(DebugConnectorConfig(silence=True))


@patch("qoa4ml.probes.system_monitoring_probe.get_sys_gpu_metadata", return_value={})
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_cpu_metadata",
    return_value={"cores": 4, "model": "TestCPU"},
)
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_mem",
    return_value={"total": 16 * 1024 * 1024 * 1024, "used": 8 * 1024 * 1024 * 1024},
)
class TestSystemMonitoringProbeInit:
    def test_init_sets_node_name(self, mock_mem, mock_cpu, mock_gpu):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        assert probe.node_name == "test-node"

    def test_init_auto_node_name(self, mock_mem, mock_cpu, mock_gpu):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        config = _make_probe_config(node_name=None)
        probe = SystemMonitoringProbe(config, _make_connector(), _make_client_info())
        assert isinstance(probe.node_name, str)
        assert len(probe.node_name) > 0

    def test_init_collects_cpu_metadata(self, mock_mem, mock_cpu, mock_gpu):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        assert probe.cpu_metadata == {"cores": 4, "model": "TestCPU"}

    def test_init_collects_gpu_metadata(self, mock_mem, mock_cpu, mock_gpu):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        assert probe.gpu_metadata == {}

    def test_init_collects_mem_metadata(self, mock_mem, mock_cpu, mock_gpu):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        assert "mem" in probe.mem_metadata
        assert probe.mem_metadata["mem"]["unit"] == "Gb"


@patch("qoa4ml.probes.system_monitoring_probe.get_sys_gpu_metadata", return_value={})
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_gpu_usage",
    return_value={"gpu_load": 50},
)
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_cpu_metadata",
    return_value={"cores": 8},
)
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_cpu_util",
    return_value=25.5,
)
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_mem",
    return_value={"total": 16 * 1024 * 1024 * 1024, "used": 4 * 1024 * 1024 * 1024},
)
class TestSystemMonitoringProbeCreateReportHPC:
    def test_create_report_returns_valid_json(
        self, mock_mem, mock_cpu_util, mock_cpu_meta, mock_gpu_usage, mock_gpu_meta
    ):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report_str = probe.create_report()
        report = json.loads(report_str)
        assert isinstance(report, dict)

    def test_create_report_hpc_structure(
        self, mock_mem, mock_cpu_util, mock_cpu_meta, mock_gpu_usage, mock_gpu_meta
    ):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert report["type"] == "system"
        assert "metadata" in report
        assert "cpu" in report
        assert "gpu" in report
        assert "mem" in report
        assert "timestamp" in report

    def test_create_report_cpu_usage_value(
        self, mock_mem, mock_cpu_util, mock_cpu_meta, mock_gpu_usage, mock_gpu_meta
    ):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert report["cpu"]["usage"]["value"] == 25.5
        assert report["cpu"]["usage"]["unit"] == "percentage"

    def test_create_report_mem_usage(
        self, mock_mem, mock_cpu_util, mock_cpu_meta, mock_gpu_usage, mock_gpu_meta
    ):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert report["mem"]["usage"]["unit"] == "Mb"
        assert report["mem"]["usage"]["value"] > 0

    def test_create_report_metadata_contains_node_name(
        self, mock_mem, mock_cpu_util, mock_cpu_meta, mock_gpu_usage, mock_gpu_meta
    ):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert report["metadata"]["node_name"] == "test-node"


@patch("qoa4ml.probes.system_monitoring_probe.get_sys_gpu_metadata", return_value={})
@patch("qoa4ml.probes.system_monitoring_probe.get_sys_gpu_usage", return_value={})
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_cpu_metadata",
    return_value={"cores": 4},
)
@patch("qoa4ml.probes.system_monitoring_probe.get_sys_cpu_util", return_value=10.0)
@patch(
    "qoa4ml.probes.system_monitoring_probe.get_sys_mem",
    return_value={"total": 8 * 1024 * 1024 * 1024, "used": 2 * 1024 * 1024 * 1024},
)
class TestSystemMonitoringProbeCreateReportEdge:
    def test_create_report_edge_uses_model(
        self, mock_mem, mock_cpu_util, mock_cpu_meta, mock_gpu_usage, mock_gpu_meta
    ):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.edge),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert "metadata" in report
        assert "cpu" in report
        assert "mem" in report
        assert "timestamp" in report


class TestSystemMonitoringProbeGpuEdge:
    @patch(
        "qoa4ml.probes.system_monitoring_probe.get_sys_mem",
        return_value={"total": 8 * 1024 * 1024 * 1024, "used": 1024 * 1024 * 1024},
    )
    @patch(
        "qoa4ml.probes.system_monitoring_probe.get_sys_cpu_metadata",
        return_value={"cores": 2},
    )
    @patch(
        "qoa4ml.probes.system_monitoring_probe.find_igpu",
        return_value={"type": "igpu", "path": "/dev/gpu0"},
    )
    def test_edge_environment_uses_find_igpu(self, mock_igpu, mock_cpu, mock_mem):
        from qoa4ml.probes.system_monitoring_probe import SystemMonitoringProbe

        probe = SystemMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.edge),
            _make_connector(),
            _make_client_info(),
        )
        assert probe.gpu_metadata == {"type": "igpu", "path": "/dev/gpu0"}
        mock_igpu.assert_called_once()
