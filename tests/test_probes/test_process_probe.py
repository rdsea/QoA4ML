import json
import os
from unittest.mock import patch

import pytest

from qoa4ml.config.configs import ClientInfo, DebugConnectorConfig, ProcessProbeConfig
from qoa4ml.connector.debug_connector import DebugConnector
from qoa4ml.lang.datamodel_enum import EnvironmentEnum


def _make_probe_config(**overrides):
    defaults = {
        "probe_type": "process",
        "frequency": 1,
        "require_register": False,
        "log_latency_flag": False,
        "environment": EnvironmentEnum.hpc,
        "pid": None,
    }
    defaults.update(overrides)
    return ProcessProbeConfig(**defaults)


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


class TestProcessMonitoringProbeInit:
    def test_init_uses_current_pid(self):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        assert probe.pid == os.getpid()

    def test_init_uses_custom_pid(self):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        current_pid = os.getpid()
        probe = ProcessMonitoringProbe(
            _make_probe_config(pid=current_pid), _make_connector(), _make_client_info()
        )
        assert probe.pid == current_pid

    def test_init_invalid_pid_raises(self):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        with pytest.raises(RuntimeError, match="No process with pid"):
            ProcessMonitoringProbe(
                _make_probe_config(pid=999999999),
                _make_connector(),
                _make_client_info(),
            )

    def test_init_hpc_metadata_is_dict(self):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        assert isinstance(probe.metadata, dict)
        assert "pid" in probe.metadata
        assert "user" in probe.metadata


class TestProcessMonitoringProbeCreateReport:
    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_memory",
        return_value=8 * 1024 * 1024 * 1024,
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_cpus",
        return_value=[0, 1, 2, 3],
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_mem",
        return_value={"rss": 100 * 1024 * 1024, "vms": 200 * 1024 * 1024},
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_child_cpu",
        return_value={"main": 5.0, "children": [1.0, 2.0]},
    )
    def test_create_report_hpc_returns_valid_json(
        self, mock_cpu, mock_mem, mock_cpus, mock_memory
    ):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report_str = probe.create_report()
        report = json.loads(report_str)
        assert isinstance(report, dict)
        assert report["type"] == "process"
        assert "metadata" in report
        assert "cpu" in report
        assert "mem" in report
        assert "timestamp" in report

    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_memory",
        return_value=8 * 1024 * 1024 * 1024,
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_cpus",
        return_value=[0, 1],
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_mem",
        return_value={"rss": 50 * 1024 * 1024, "vms": 100 * 1024 * 1024},
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_child_cpu",
        return_value={"main": 10.0, "children": []},
    )
    def test_create_report_hpc_metadata(
        self, mock_cpu, mock_mem, mock_cpus, mock_memory
    ):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert "pid" in report["metadata"]
        assert "user" in report["metadata"]
        assert "allowed_cpu_list" in report["metadata"]
        assert "allowed_memory_size" in report["metadata"]

    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_memory",
        return_value=4 * 1024 * 1024 * 1024,
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_cpus",
        return_value=[0],
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_mem",
        return_value={"rss": 30 * 1024 * 1024, "vms": 60 * 1024 * 1024},
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_child_cpu",
        return_value={"main": 3.0, "children": [0.5]},
    )
    def test_create_report_cpu_usage(self, mock_cpu, mock_mem, mock_cpus, mock_memory):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        cpu_usage = report["cpu"]["usage"]
        assert cpu_usage["main"] == 3.0
        assert cpu_usage["children"] == [0.5]

    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_memory",
        return_value=4 * 1024 * 1024 * 1024,
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_cpus",
        return_value=[0],
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_mem",
        return_value={"rss": 50 * 1024 * 1024, "vms": 100 * 1024 * 1024},
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_child_cpu",
        return_value={"main": 1.0, "children": []},
    )
    def test_create_report_mem_usage(self, mock_cpu, mock_mem, mock_cpus, mock_memory):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.hpc),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        mem_usage = report["mem"]["usage"]
        assert "rss" in mem_usage
        assert "vms" in mem_usage
        assert mem_usage["rss"]["unit"] == "Mb"
        assert mem_usage["vms"]["unit"] == "Mb"

    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_memory",
        return_value=4 * 1024 * 1024 * 1024,
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.get_process_allowed_cpus",
        return_value=[0, 1],
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_mem",
        return_value={"rss": 100 * 1024 * 1024, "vms": 200 * 1024 * 1024},
    )
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_child_cpu",
        return_value={"main": 2.0, "children": []},
    )
    def test_create_report_edge_uses_model(
        self, mock_cpu, mock_mem, mock_cpus, mock_memory
    ):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(environment=EnvironmentEnum.edge),
            _make_connector(),
            _make_client_info(),
        )
        report = json.loads(probe.create_report())
        assert "metadata" in report
        assert "cpu" in report
        assert "mem" in report
        assert "timestamp" in report


class TestProcessMonitoringProbeGetMethods:
    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_child_cpu",
        return_value={"main": 7.5, "children": [1.2, 3.4]},
    )
    def test_get_cpu_usage(self, mock_cpu):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        usage = probe.get_cpu_usage()
        assert usage["main"] == 7.5
        assert len(usage["children"]) == 2

    @patch(
        "qoa4ml.probes.process_monitoring_probe.report_proc_mem",
        return_value={"rss": 256 * 1024 * 1024, "vms": 512 * 1024 * 1024},
    )
    def test_get_mem_usage(self, mock_mem):
        from qoa4ml.probes.process_monitoring_probe import ProcessMonitoringProbe

        probe = ProcessMonitoringProbe(
            _make_probe_config(), _make_connector(), _make_client_info()
        )
        usage = probe.get_mem_usage()
        assert "rss" in usage
        assert "vms" in usage
        assert usage["rss"]["value"] == 256.0
        assert usage["vms"]["value"] == 512.0
