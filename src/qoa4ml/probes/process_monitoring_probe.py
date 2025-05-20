from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

import lazy_import
import psutil

from qoa4ml.config.configs import ClientInfo, ProcessProbeConfig
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.lang.datamodel_enum import EnvironmentEnum
from qoa4ml.probes.probe import Probe
from qoa4ml.utils.qoa_utils import (
    convert_to_mbyte,
    get_process_allowed_cpus,
    get_process_allowed_memory,
    report_proc_child_cpu,
    report_proc_mem,
)

if TYPE_CHECKING:
    from ..reports import resources_report_model
else:
    resources_report_model = lazy_import.lazy_module(
        "qoa4ml.reports.resources_report_model"
    )
logging.basicConfig(
    format="%(asctime)s:%(levelname)s -- %(message)s", level=logging.INFO
)


class ProcessMonitoringProbe(Probe):
    """
    ProcessMonitoringProbe is responsible for monitoring system processes and creating reports based on usage statistics.

    Parameters
    ----------
    config : ProcessProbeConfig
        Configuration settings for the process monitoring probe.
    connector : BaseConnector
        Connector to send the report data.
    client_info : Optional[ClientInfo]
        Information about the client, default is None.

    Attributes
    ----------
    config : ProcessProbeConfig
        The process monitoring probe configuration.
    pid : int
        The process ID to monitor.
    environment : EnvironmentEnum
        The environment in which the process is running.
    process : psutil.Process
        The psutil Process object for the monitored process.
    obs_service_url : Optional[str]
        The URL of the observation service, if registration is required.
    metadata : Union[dict, resources_report_model.ProcessMetadata]
        Metadata related to the monitored process.

    Methods
    -------
    get_cpu_usage() -> dict
        Get the CPU usage of the process.
    get_mem_usage() -> dict
        Get the memory usage of the process.
    create_report() -> str
        Create a JSON report based on the process statistics.
    """

    def __init__(
        self,
        config: ProcessProbeConfig,
        connector: BaseConnector,
        client_info: ClientInfo | None = None,
    ) -> None:
        """
        Initialize an instance of ProcessMonitoringProbe.

        Parameters
        ----------
        config : ProcessProbeConfig
            Configuration settings for the process monitoring probe.
        connector : BaseConnector
            Connector to send the report data.
        client_info : Optional[ClientInfo]
            Information about the client, default is None.
        """
        super().__init__(config, connector, client_info)
        self.config = config
        self.pid = os.getpid() if self.config.pid is None else self.config.pid
        if not psutil.pid_exists(self.pid):
            raise RuntimeError(f"No process with pid {self.pid}")

        self.environment = config.environment
        self.process = psutil.Process(self.pid)
        if self.config.require_register:
            self.obs_service_url = self.config.obs_service_url

        if self.environment == EnvironmentEnum.hpc:
            self.metadata = {"pid": str(self.pid), "user": self.process.username()}
        else:
            self.metadata = resources_report_model.ProcessMetadata(
                pid=str(self.pid), user=self.process.username(), client_info=client_info
            )

    def get_cpu_usage(self) -> dict:
        """
        Get the CPU usage of the process.

        Returns
        -------
        dict
            Dictionary containing the CPU usage information.
        """
        process_usage = report_proc_child_cpu(self.process)
        return process_usage

    def get_mem_usage(self) -> dict:
        """
        Get the memory usage of the process.

        Returns
        -------
        dict
            Dictionary containing the memory usage in megabytes.
        """
        data = report_proc_mem(self.process)
        return {
            "rss": {"value": convert_to_mbyte(data["rss"]), "unit": "Mb"},
            "vms": {"value": convert_to_mbyte(data["vms"]), "unit": "Mb"},
        }

    def create_report(self) -> str:
        """
        Create a JSON report based on the process statistics.

        Returns
        -------
        str
            JSON-encoded report containing process statistics.

        Notes
        -----
        - This method collects CPU and memory usage stats for the specified process.
        - Reports are generated differently based on the environment (HPC or other).
        """
        timestamp = time.time()
        cpu_usage = self.get_cpu_usage()
        mem_usage = self.get_mem_usage()
        allowed_cpu_list = get_process_allowed_cpus()
        allowed_memory_size = get_process_allowed_memory()

        if self.environment == EnvironmentEnum.hpc:
            report = {
                "type": "process",
                "metadata": {
                    "pid": str(self.pid),
                    "user": self.process.username(),
                    "allowed_cpu_list": str(allowed_cpu_list),
                    "allowed_memory_size": str(allowed_memory_size),
                },
                "timestamp": round(timestamp),
                "cpu": {
                    "usage": cpu_usage,
                },
                "mem": {
                    "usage": mem_usage,
                },
            }
        else:
            report = resources_report_model.ProcessReport(
                metadata=self.metadata,
                timestamp=round(timestamp),
                cpu=resources_report_model.ResourceReport(usage=cpu_usage),
                mem=resources_report_model.ResourceReport(usage=mem_usage),
            ).model_dump()

        return json.dumps(report)
