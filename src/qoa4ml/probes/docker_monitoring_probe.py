import json
import time

import docker
import requests
from docker.errors import DockerException

from qoa4ml.config.configs import ClientInfo, DockerProbeConfig
from qoa4ml.connector.base_connector import BaseConnector
from qoa4ml.probes.probe import Probe
from qoa4ml.reports.resources_report_model import DockerReport
from qoa4ml.utils.docker_util import get_docker_stats
from qoa4ml.utils.logger import qoa_logger


class DockerMonitoringProbe(Probe):
    """
    DockerMonitoringProbe is responsible for monitoring Docker containers and creating reports.

    Parameters
    ----------
    config : DockerProbeConfig
        Configuration settings for the Docker monitoring probe.
    connector : BaseConnector
        Connector to send the report data.
    client_info : ClientInfo
        Information about the client.

    Attributes
    ----------
    config : DockerProbeConfig
        The Docker monitoring probe configuration.
    docker_client : docker.DockerClient
        The Docker client for communicating with Docker API.

    Methods
    -------
    create_report() -> str
        Create a report based on Docker container statistics.
    """

    def __init__(
        self,
        config: DockerProbeConfig,
        connector: BaseConnector,
        client_info: ClientInfo,
    ) -> None:
        """
        Initialize an instance of DockerMonitoringProbe.

        Parameters
        ----------
        config : DockerProbeConfig
            Configuration settings for the Docker monitoring probe.
        connector : BaseConnector
            Connector to send the report data.
        client_info : ClientInfo
            Information about the client.
        """
        super().__init__(config, connector, client_info)
        self.config: DockerProbeConfig = config
        self.docker_client = docker.from_env()

    def create_report(self) -> str:
        """
        Create a report based on Docker container statistics.

        Returns
        -------
        str
            JSON-encoded report containing Docker container statistics.

        Notes
        -----
        - This method collects statistics for the specified Docker containers.
        - If the report dictionary is empty, it adds a 2-second delay to prevent fast looping.
        - Docker/HTTP/OS failures are logged via qoa_logger and returned as a
          JSON error so the probe thread never dies silently (service boundary: log + return JSON error, never crash probe thread).
        """
        try:
            reports = get_docker_stats(self.docker_client, self.config.container_list)
            assert self.client_info is not None
            docker_report = DockerReport(
                metadata=self.client_info,
                timestamp=time.time(),
                container_reports=reports,
            )
            return json.dumps(docker_report.model_dump())
        except (DockerException, requests.RequestException, OSError) as error:
            error_type = type(error).__name__
            qoa_logger.exception(
                f"Docker probe failed ({error_type}); returning error payload"
            )
            return json.dumps({"error": error_type, "detail": str(error)})
