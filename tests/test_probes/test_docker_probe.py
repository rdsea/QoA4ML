import json
from unittest.mock import MagicMock, patch

from qoa4ml.config.configs import ClientInfo, DebugConnectorConfig, DockerProbeConfig
from qoa4ml.connector.debug_connector import DebugConnector
from qoa4ml.reports.resources_report_model import (
    DockerContainerMetadata,
    DockerContainerReport,
    ResourceReport,
)


def _make_probe_config(**overrides):
    defaults = {
        "probe_type": "docker",
        "frequency": 1,
        "require_register": False,
        "log_latency_flag": False,
        "container_list": [],
    }
    defaults.update(overrides)
    return DockerProbeConfig(**defaults)


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


def _make_container_report(
    container_id="abc123", image="test:latest", cpu=25.0, mem=128.0
):
    return DockerContainerReport(
        metadata=DockerContainerMetadata(id=container_id, image=image),
        timestamp=1700000000.0,
        cpu=ResourceReport(usage={"cpu_percentage": cpu}),
        mem=ResourceReport(usage={"memory_usage": mem}),
    )


class TestDockerMonitoringProbeInit:
    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    def test_init_creates_docker_client(self, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        mock_docker.from_env.assert_called_once()
        assert probe.docker_client is not None

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    def test_init_with_container_list(self, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        config = _make_probe_config(container_list=["web", "db"])
        probe = DockerMonitoringProbe(config, _make_connector(), _make_client_info())
        assert probe.config.container_list == ["web", "db"]

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    def test_init_empty_container_list(self, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        assert probe.config.container_list == []


class TestDockerMonitoringProbeCreateReport:
    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch("qoa4ml.probes.docker_monitoring_probe.get_docker_stats")
    def test_create_report_returns_valid_json(self, mock_stats, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        mock_stats.return_value = [_make_container_report()]

        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        report_str = probe.create_report()
        report = json.loads(report_str)
        assert isinstance(report, dict)

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch("qoa4ml.probes.docker_monitoring_probe.get_docker_stats")
    def test_create_report_has_container_reports(self, mock_stats, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        mock_stats.return_value = [
            _make_container_report("c1", "app:v1", cpu=10.0, mem=64.0),
            _make_container_report("c2", "db:v2", cpu=20.0, mem=256.0),
        ]

        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        report = json.loads(probe.create_report())
        assert "container_reports" in report
        assert len(report["container_reports"]) == 2

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch("qoa4ml.probes.docker_monitoring_probe.get_docker_stats")
    def test_create_report_container_stats(self, mock_stats, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        mock_stats.return_value = [
            _make_container_report("abc", "myimage:latest", cpu=45.0, mem=512.0)
        ]

        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        report = json.loads(probe.create_report())
        container = report["container_reports"][0]
        assert container["metadata"]["id"] == "abc"
        assert container["metadata"]["image"] == "myimage:latest"
        assert container["cpu"]["usage"]["cpu_percentage"] == 45.0
        assert container["mem"]["usage"]["memory_usage"] == 512.0

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch("qoa4ml.probes.docker_monitoring_probe.get_docker_stats")
    def test_create_report_has_metadata(self, mock_stats, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        mock_stats.return_value = []

        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        report = json.loads(probe.create_report())
        assert "metadata" in report
        assert "timestamp" in report

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch("qoa4ml.probes.docker_monitoring_probe.get_docker_stats")
    def test_create_report_empty_containers(self, mock_stats, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        mock_stats.return_value = []

        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        report = json.loads(probe.create_report())
        assert report["container_reports"] == []

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch(
        "qoa4ml.probes.docker_monitoring_probe.get_docker_stats",
        side_effect=RuntimeError("Docker daemon not running"),
    )
    def test_create_report_runtime_error(self, mock_stats, mock_docker):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()

        probe = DockerMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        report_str = probe.create_report()
        report = json.loads(report_str)
        assert report == {"error": "RuntimeError"}

    @patch("qoa4ml.probes.docker_monitoring_probe.docker")
    @patch("qoa4ml.probes.docker_monitoring_probe.get_docker_stats")
    def test_create_report_passes_container_list_to_stats(
        self, mock_stats, mock_docker
    ):
        from qoa4ml.probes.docker_monitoring_probe import DockerMonitoringProbe

        mock_docker.from_env.return_value = MagicMock()
        mock_stats.return_value = []

        config = _make_probe_config(container_list=["web", "api"])
        probe = DockerMonitoringProbe(config, _make_connector(), _make_client_info())
        probe.create_report()
        mock_stats.assert_called_once_with(probe.docker_client, ["web", "api"])
