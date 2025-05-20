from __future__ import annotations

import json
import socket
import time
from typing import TYPE_CHECKING

import lazy_import

from qoa4ml.config.configs import ClientInfo, SystemProbeConfig
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.lang.datamodel_enum import EnvironmentEnum
from qoa4ml.probes.probe import Probe
from qoa4ml.utils.gpu_utils import get_sys_gpu_metadata, get_sys_gpu_usage
from qoa4ml.utils.jetson_utils import find_igpu, get_gpu_load
from qoa4ml.utils.qoa_utils import (
    convert_to_gbyte,
    convert_to_mbyte,
    get_sys_cpu_metadata,
    get_sys_cpu_util,
    get_sys_mem,
)

if TYPE_CHECKING:
    from ..reports import resources_report_model
else:
    resources_report_model = lazy_import.lazy_module(
        "qoa4ml.reports.resources_report_model"
    )


class SystemMonitoringProbe(Probe):
    """
    SystemMonitoringProbe is responsible for monitoring system resources and creating reports based on usage statistics.

    Parameters
    ----------
    config : SystemProbeConfig
        Configuration settings for the system monitoring probe.
    connector : BaseConnector
        Connector to send the report data.
    client_info : Optional[ClientInfo]
        Information about the client, default is None.

    Attributes
    ----------
    config : SystemProbeConfig
        The system monitoring probe configuration.
    node_name : str
        The name of the node being monitored.
    environment : EnvironmentEnum
        The environment in which the node is running.
    cpu_metadata : dict
        Metadata about the CPU.
    gpu_metadata : dict
        Metadata about the GPU.
    mem_metadata : dict
        Metadata about the memory.
    metadata : dict
        General metadata about the node.

    Methods
    -------
    get_cpu_metadata() -> dict
        Get metadata about the CPU.
    get_cpu_usage() -> dict
        Get the CPU usage of the system.
    get_gpu_metadata() -> dict
        Get metadata about the GPU.
    get_gpu_usage() -> dict
        Get the GPU usage of the system.
    get_mem_metadata() -> dict
        Get metadata about the memory.
    get_mem_usage() -> dict
        Get the memory usage of the system.
    create_report() -> str
        Create a JSON report based on system resource usage statistics.
    """

    def __init__(
        self,
        config: SystemProbeConfig,
        connector: BaseConnector,
        client_info: ClientInfo | None = None,
    ) -> None:
        """
        Initialize an instance of SystemMonitoringProbe.

        Parameters
        ----------
        config : SystemProbeConfig
            Configuration settings for the system monitoring probe.
        connector : BaseConnector
            Connector to send the report data.
        client_info : Optional[ClientInfo]
            Information about the client, default is None.
        """
        super().__init__(config, connector, client_info)
        self.config = config
        self.node_name = (
            socket.gethostname().split(".")[0]
            if self.config.node_name is None
            else self.config.node_name
        )
        if self.config.require_register:
            self.obs_service_url = self.config.obs_service_url
        self.environment = config.environment
        self.cpu_metadata = self.get_cpu_metadata()
        self.gpu_metadata = self.get_gpu_metadata()
        self.mem_metadata = self.get_mem_metadata()
        self.metadata = {"node_name": self.node_name}

    def get_cpu_metadata(self) -> dict:
        """
        Get metadata about the CPU.

        Returns
        -------
        dict
            Dictionary containing metadata about the CPU.
        """
        return get_sys_cpu_metadata()

    def get_cpu_usage(self) -> dict:
        """
        Get the CPU usage of the system.

        Returns
        -------
        dict
            Dictionary containing the CPU usage information in percentage.
        """
        value = get_sys_cpu_util()
        return {"value": value, "unit": "percentage"}

    def get_gpu_metadata(self) -> dict:
        """
        Get metadata about the GPU.

        Returns
        -------
        dict
            Dictionary containing metadata about the GPU.
        """
        if self.environment == EnvironmentEnum.edge:
            report = find_igpu()
        else:
            report = get_sys_gpu_metadata()
        return report

    def get_gpu_usage(self) -> dict:
        """
        Get the GPU usage of the system.

        Returns
        -------
        dict
            Dictionary containing the GPU usage information.
        """
        if self.environment == EnvironmentEnum.edge:
            report = get_gpu_load(self.gpu_metadata)
        else:
            report = get_sys_gpu_usage()
        return report

    def get_mem_metadata(self) -> dict:
        """
        Get metadata about the memory.

        Returns
        -------
        dict
            Dictionary containing memory metadata in gigabytes.
        """
        mem = get_sys_mem()
        return {"mem": {"capacity": convert_to_gbyte(mem["total"]), "unit": "Gb"}}

    def get_mem_usage(self) -> dict:
        """
        Get the memory usage of the system.

        Returns
        -------
        dict
            Dictionary containing the memory usage in megabytes.
        """
        mem = get_sys_mem()
        return {"value": convert_to_mbyte(mem["used"]), "unit": "Mb"}

    def create_report(self) -> str:
        """
        Create a JSON report based on system resource usage statistics.

        Returns
        -------
        str
            JSON-encoded report containing system resource usage statistics.

        Notes
        -----
        - This method collects CPU, GPU, and memory usage stats for the system.
        - Reports are generated differently based on the environment (HPC or other).
        """
        timestamp = time.time()
        cpu_usage = self.get_cpu_usage()
        gpu_usage = self.get_gpu_usage()
        mem_usage = self.get_mem_usage()

        if self.environment == EnvironmentEnum.hpc:
            report = {
                "type": "system",
                "metadata": {**self.metadata},
                "timestamp": round(timestamp),
                "cpu": {
                    "metadata": self.cpu_metadata,
                    "usage": cpu_usage,
                },
                "gpu": {
                    "metadata": self.gpu_metadata,
                    "usage": gpu_usage,
                },
                "mem": {
                    "metadata": self.mem_metadata,
                    "usage": mem_usage,
                },
            }
        else:
            report = resources_report_model.SystemReport(
                metadata=resources_report_model.SystemMetadata(
                    node_name=self.node_name, client_info=self.client_info
                ),
                timestamp=round(timestamp),
                cpu=resources_report_model.ResourceReport(
                    metadata=self.cpu_metadata, usage=cpu_usage
                ),
                gpu=resources_report_model.ResourceReport(
                    metadata=self.gpu_metadata, usage=gpu_usage
                ),
                mem=resources_report_model.ResourceReport(
                    metadata=self.mem_metadata, usage=mem_usage
                ),
            ).model_dump()

        return json.dumps(report)
